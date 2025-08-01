# coding:utf-8

import math, customKernel
import numpy as np
import ConfigParser
from abaqusConstants import *
from collections import defaultdict
from abaqus import getInput, getInputs
from odbAccess import openOdb, upgradeOdb
from scipy import spatial
import datetime
from scipy import ndimage
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.ndimage import label as nd_label
from scipy.spatial.distance import cdist

class Population:
    def __init__(
        self,
        decs = None,
        cons = None,
        objs = None,
        densities = None,
    ):
        assert (
            decs is None or decs.ndim == 2
        ), "Population initiate error, decs must be 2-D array"
        assert (
            cons is None or cons.ndim == 2
        ), "Population initiate error, cons must be 2-D array"
        assert (
            objs is None or objs.ndim == 2
        ), "Population initiate error, objs must be 2-D array"
        assert (
            densities is None or densities.ndim == 2
        ), "Population initiate error, objs must be 2-D array"
        self.decs = decs.copy()
        self.cons = cons.copy() if cons is not None else None
        self.objs = objs.copy() if objs is not None else None
        self.densities = densities.copy() if densities is not None else None

    def copy(self):
        new_decs = self.decs.copy()
        new_cons = self.cons.copy() if self.cons is not None else None
        new_objs = self.objs.copy() if self.objs is not None else None
        new_densities = self.densities.copy() if self.objs is not None else None
        pop = Population(decs=new_decs, cons=new_cons, objs=new_objs,densities=new_densities)
        return pop

    def __add__(self, pop):
        new_decs = np.vstack([self.decs, pop.decs])
        new_objs = np.vstack([self.objs, pop.objs])
        new_cons = np.vstack([self.cons, pop.cons])
        new_densities = np.vstack([self.densities, pop.densities])
        new_pop = Population(decs=new_decs, objs=new_objs, cons=new_cons, densities=new_densities)
        return new_pop

    def __getitem__(self, ind):
        if self.decs is None:
            raise RuntimeError("The population has not been initialized")
        if type(ind) == int:
            ind = [ind]
        if type(ind) == np.ndarray:
            assert ind.dtype in [np.int32, np.int64, np.bool8]
            assert ind.ndim == 1 or ind.ndim == 2
            if ind.ndim == 2:
                assert 1 in ind.shape
                ind = ind.flatten()

        new_decs = self.decs[ind]
        new_objs = self.objs[ind] if self.objs is not None else None
        new_cons = self.cons[ind] if self.cons is not None else None
        new_densities = self.densities[ind] if self.densities is not None else None
        new_pop = Population(decs=new_decs, objs=new_objs, cons=new_cons, densities=new_densities)
        return new_pop

    def __len__(self):
        return self.decs.shape[0]

    def __setitem__(self, item, pop):

        if self.decs is not None:
            self.decs[item] = pop.decs
        if self.cons is not None:
            self.cons[item] = pop.cons
        if self.objs is not None:
            self.objs[item] = pop.objs
        if self.densities is not None:
            self.densities[item] = pop.densities


def nd_sort(objs, *args):
    objs_temp = objs.copy()
    if len(objs_temp) == 0:
        return [], 0
    N, M = objs_temp.shape
    if len(args) == 1:
        count = args[0]
    else:
        con = args[0]
        count = args[1]
        Infeasible = np.any(con > 0, axis=1)
        objs_temp[Infeasible, :] = np.tile(
            np.amax(objs_temp, axis=0), (np.sum(Infeasible), 1)
        ) + np.tile(
            np.sum(np.where(con < 0, 0, con)[Infeasible, :], axis=1).reshape(
                np.sum(Infeasible), 1
            ),
            (1, M),
        )
    return ENS_SS(objs_temp, count)

def ENS_SS(objs, count):
    nsort = count
    objs, index, ind = np.unique(
        objs, return_index=True, return_inverse=True, axis=0
    )
    count, M = objs.shape
    frontno = np.full(count, np.inf)
    maxfront = 0
    Table, _ = np.histogram(ind, bins=np.arange(0, np.max(ind) + 2))
    while np.sum(Table[frontno < np.inf]) < np.min((nsort, len(ind))):
        maxfront += 1
        for i in range(count):
            if frontno[i] == np.inf:
                dominate = False
                for j in range(i - 1, -1, -1):
                    if frontno[j] == maxfront:
                        m = 1
                        while m < M and objs[i][m] >= objs[j][m]:
                            m += 1
                        dominate = m == M
                        if dominate or M == 2:
                            break
                if not dominate:
                    frontno[i] = maxfront
    frontno = frontno[ind]
    return [frontno, maxfront]

def Best(pop):
    feasible = np.where(np.all(pop.cons <= 0, axis=1))[0]
    if len(feasible) == 0:
        return None
    FrontNo, _ = nd_sort(pop.objs[feasible], pop.cons[feasible], len(feasible))
    best = np.where(FrontNo == 1)[0]
    if len(best) == 0:
        return None
    return pop[feasible[best]]



def Conversion(originMatrix, cellsize):
    coordinates = originMatrix.reshape(-1, 2)
    sub_coords_list = []
    seen = set()
    for x, y in coordinates:
        for dx in range(cellsize):
            for dy in range(cellsize):
                nx,ny = x + dx,y + dy
                coord_tuple = (nx, ny)
                if coord_tuple not in seen:
                    sub_coords_list.append(coord_tuple)
                    seen.add(coord_tuple)
    sub_coords_array = np.array(sub_coords_list)
    return sub_coords_array

def turntoIndices(sub_coords_array,col):
    indices = []
    for x, y in sub_coords_array:
        if 0 <= x <= 24:
            newrow = 24 - x
        else:
            newrow = x
        if 0 <= newrow <= 24:
            newcol = y
        else:
            newcol = col - 1 - y
        indices.append(int(newrow * col + newcol))
    return indices


def Cal(Pop, modelDatabase, cellsize, row, col, Gen, indexToElLabel):
    PopDec = Pop.decs
    N, _ = PopDec.shape
    batch_size = 3
    Popobj = np.zeros((N, 2))
    Popcon = np.zeros((N, 1))
    Popdensities = np.ones((N, row * col))

    for i in range(0, N, batch_size):
        batch_end = min(i + batch_size, N)
        batch_indices = list(range(i, batch_end))

        for j in batch_indices:
            if Gen >= 20:
                Popdensities[j, :] = PopDec[j, :]
            else:
                sub_coords_array = Conversion(PopDec[j, :], cellsize)
                indices = turntoIndices(sub_coords_array, col)
                Popdensities[j, indices] = 0
            Popcon[j] = calCON(Popdensities[j, :], row, col)

        ODBFileNames = multiFEA(len(batch_indices), modelDatabase, Popdensities[i:batch_end, :], indexToElLabel)

        for j, idx in enumerate(batch_indices):
            Popobj[idx, 0] = singleobjectiveFunctionFromODB(ODBFileNames[j])
            Popobj[idx, 1] = np.mean(Popdensities[idx, :] == 1)

    Pop.objs = Popobj
    Pop.cons = Popcon
    Pop.densities = Popdensities
    return Pop

def turntoOnedimensional(matrix):
    rows, cols = matrix.shape
    Onedimensional = np.zeros(rows * cols)

    for i in range(rows):
        if 0 <= i <= 24:
            newrow = 24 - i
        else:
            newrow = i

        for j in range(cols):
            if 0 <= newrow <= 24:
                newcol = j
            else:
                newcol = cols - 1 - j
            Onedimensional[i * cols + j] = matrix[newrow, newcol]

    return Onedimensional


def turntoMatrix(densities,row,col):
    matrix = np.zeros((row, col))
    for i in range(row):
        if 0 <= i <= 24:
            matrix[24-i,:] = densities[i*col:(i+1)*col]
        else:
            matrix[i,:] = densities[i*col:(i+1)*col][::-1]
    return matrix

def calCON(densities,row,col):
    matrix  = turntoMatrix(densities,row,col)
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    _, num_features = ndimage.label(matrix, structure=structure)
    if num_features == 1:
        return 0
    else:
        return 1


def sort_by_euclidean_distance(parent1, parent2, *args):
    num_individuals = parent1.shape[0]

    if args:
        num_coordinates = parent1.shape[1] // 4
        coord_stride = 4
    else:
        num_coordinates = parent1.shape[1] // 2
        coord_stride = 2

    sorted_parent1 = np.zeros_like(parent1)
    sorted_parent2 = np.zeros_like(parent2)

    for i in range(num_individuals):
        distances = []
        for j in range(num_coordinates):
            if args:
                coord1 = parent1[i, j * coord_stride:(j * coord_stride) + 2]
                coord2 = parent2[i, j * coord_stride:(j * coord_stride) + 2]
            else:
                coord1 = parent1[i, j * coord_stride:(j + 1) * coord_stride]
                coord2 = parent2[i, j * coord_stride:(j + 1) * coord_stride]

            distance = np.linalg.norm(coord1 - coord2)
            distances.append((j, distance))

        sorted_indices = [index for index, _ in sorted(distances, key=lambda x: x[1])]

        for k, idx in enumerate(sorted_indices):
            sorted_parent1[i, k * coord_stride:(k + 1) * coord_stride] = parent1[i,
                                                                         idx * coord_stride:(idx + 1) * coord_stride]
            sorted_parent2[i, k * coord_stride:(k + 1) * coord_stride] = parent2[i,
                                                                         idx * coord_stride:(idx + 1) * coord_stride]

    return sorted_parent1, sorted_parent2


def OperatorReal(ParentDec, lb, ub):
    Parent1Dec = ParentDec[0 : int(math.floor(ParentDec.shape[0] / 2)),]
    Parent2Dec = ParentDec[int(math.floor(ParentDec.shape[0] / 2)) : int(math.floor(ParentDec.shape[0] / 2)) * 2, :]
    N, D = Parent1Dec.shape

    sorted_Parent1Dec, sorted_Parent2Dec = sort_by_euclidean_distance(Parent1Dec, Parent2Dec)

    # Crossover operation
    beta = np.zeros((N, D))
    mu = np.random.random((N, D))
    disC = 20  # Distribution index for crossover
    proC = 1  # Crossover probability

    beta[mu <= 0.5] = (2 * mu[mu <= 0.5]) ** (1 / (disC + 1))
    beta[mu > 0.5] = (2 - 2 * mu[mu > 0.5]) ** (-1 / (disC + 1))
    beta = beta * (-1) ** np.random.randint(0, 2, (N, D))
    beta[np.random.random((N, D)) < 0.5] = 1
    beta[np.tile(np.random.random((N, 1)) > proC, (1, D))] = 1
    beta[:, 1::2] = beta[:, 0::2]

    OffDec = np.vstack(
        (
            (sorted_Parent1Dec + sorted_Parent2Dec) / 2 + beta * (sorted_Parent1Dec - sorted_Parent2Dec) / 2,
            (sorted_Parent1Dec + sorted_Parent2Dec) / 2 - beta * (sorted_Parent1Dec - sorted_Parent2Dec) / 2,
        )
    )


    # Mutation operation
    disM = 20  # Distribution index for mutation
    proM = 1  # Mutation probability
    Lower = np.tile(lb, (2 * N, 1))
    Upper = np.tile(ub, (2 * N, 1))
    Site = np.random.random((N * 2, D // 2)) < proM / (D // 2)
    Site = np.repeat(Site, 2, axis=1)


    mu = np.random.random((2 * N, D))

    # Mutation logic ensuring pairs are mutated together
    temp = np.logical_and(Site, mu <= 0.5)
    OffDec = np.minimum(np.maximum(OffDec, Lower), Upper)
    OffDec[temp] = OffDec[temp] + (Upper[temp] - Lower[temp]) * (
            (
                    2 * mu[temp]
                    + (1 - 2 * mu[temp])
                    * (
                            1
                            - (OffDec[temp] - Lower[temp])
                            / (Upper[temp] - Lower[temp])  # noqa
                    )
                    ** (disM + 1)
            )
            ** (1 / (disM + 1))
            - 1
    )  # noqa
    temp = np.logical_and(Site, mu > 0.5)  # noqa: E510
    OffDec[temp] = OffDec[temp] + (Upper[temp] - Lower[temp]) * (
            1
            - (
                    2 * (1 - mu[temp])
                    + 2
                    * (mu[temp] - 0.5)
                    * (
                            1
                            - (Upper[temp] - OffDec[temp])
                            / (Upper[temp] - Lower[temp])
                    )  # noqa
                    ** (disM + 1)
            )
            ** (1 / (disM + 1))
    )
    OffDec = np.round(OffDec).astype(int)
    return OffDec

def Operatorbinary(ParentDec,row,col):
    Parent1Dec = ParentDec[0 : int(math.floor(ParentDec.shape[0] / 2)),]
    Parent2Dec = ParentDec[int(math.floor(ParentDec.shape[0] / 2)) : int(math.floor(ParentDec.shape[0] / 2)) * 2, :]
    N, D = Parent1Dec.shape
    OffDec = np.empty_like(ParentDec)
    for i in range(N):
        matrix1 = turntoMatrix(Parent1Dec[i,:],row,col)
        matrix2 = turntoMatrix(Parent2Dec[i,:],row,col)
        offspring1, offspring2 = crossover(matrix1, matrix2)
        offspring1 = mutate(offspring1)
        offspring2 = mutate(offspring2)

        OffDec[i, :] = turntoOnedimensional(offspring1)
        OffDec[N + i, :] = turntoOnedimensional(offspring2)

    return OffDec

def crossover(matrix1, matrix2):
    labeled1, ncomponents1 = find_connected_components(matrix1)
    labeled2, ncomponents2 = find_connected_components(matrix2)

    offspring1 = np.copy(matrix1)
    offspring2 = np.copy(matrix2)

    matched_pairs = set()

    for region_label1 in range(1, ncomponents1 + 1):
        region1 = (labeled1 == region_label1)
        best_match = None
        max_overlap = 0

        for region_label2 in range(1, ncomponents2 + 1):
            region2 = (labeled2 == region_label2)
            overlap = np.sum(region1 & region2)

            if overlap > max_overlap:
                best_match = region_label2
                max_overlap = overlap

        if best_match is not None:
            region2 = (labeled2 == best_match)
            overlap = region1 & region2

            if np.all(region1 == overlap):
                process_containment(region1, region2, offspring1, offspring2)
            elif np.all(region2 == overlap):
                process_containment(region2, region1, offspring2, offspring1)
            else:
                process_intersection(region1, region2, offspring1, offspring2)

            matched_pairs.add((region_label1, best_match))

    for region_label1 in range(1, ncomponents1 + 1):
        if not any(region_label1 == pair[0] for pair in matched_pairs):
            region1 = (labeled1 == region_label1)
            process_non_overlapping(region1, offspring1)

    for region_label2 in range(1, ncomponents2 + 1):
        if not any(region_label2 == pair[1] for pair in matched_pairs):
            region2 = (labeled2 == region_label2)
            process_non_overlapping(region2, offspring2)

    return offspring1, offspring2

def mutate(offspring):
    rows, cols = offspring.shape
    labeled, ncomponents = find_connected_components(offspring)

    if ncomponents > 0:
        num_regions = np.random.randint(1, ncomponents + 1)
        selected_labels = np.random.choice(np.arange(1, ncomponents + 1), size=num_regions, replace=False)

        for region_label in selected_labels:
            region = (labeled == region_label)

            if np.random.rand() > 0.5:
                expanded_region = binary_dilation(region)
                expansion = expanded_region & ~region
                indices = np.argwhere(expansion)
                num_to_mutate = np.random.randint(int(len(indices) * 0.1), int(len(indices) * 0.2) + 1)
                if num_to_mutate > 0:
                    selected_indices = indices[np.random.choice(indices.shape[0], size=num_to_mutate, replace=False)]
                    offspring[tuple(selected_indices.T)] = 0
            else:
                eroded_region = binary_erosion(region)
                edge = region & ~eroded_region
                indices = np.argwhere(edge)
                num_to_mutate = np.random.randint(int(len(indices) * 0.1), int(len(indices) * 0.2) + 1)
                if num_to_mutate > 0:
                    selected_indices = indices[np.random.choice(indices.shape[0], size=num_to_mutate, replace=False)]
                    offspring[tuple(selected_indices.T)] = 1
    return offspring

def find_connected_components(matrix):
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    labeled, ncomponents = nd_label(matrix == 0, structure=structure)
    return labeled, ncomponents

def process_non_overlapping(region, offspring):
    offspring[region] = 0


def process_containment(region1, region2, offspring1, offspring2):
    overlap_part = region1 & region2
    surrounding_region2 = binary_dilation(overlap_part) & region2 & ~overlap_part
    eroded_region2 = binary_erosion(region2)

    offspring1[surrounding_region2] = 0
    offspring2[region2 & ~eroded_region2] = 1


def process_intersection(region1, region2, offspring1, offspring2):
    overlap_part = region1 & region2
    surrounding_region2 = binary_dilation(overlap_part) & region2 & ~overlap_part
    surrounding_region1 = binary_dilation(overlap_part) & region1 & ~overlap_part

    offspring1[surrounding_region2] = 0
    offspring2[surrounding_region1] = 0

    eroded_region1 = binary_erosion(region1)
    eroded_region2 = binary_erosion(region2)

    edge_region1 = region1 & ~eroded_region1 & ~overlap_part
    edge_region2 = region2 & ~eroded_region2 & ~overlap_part

    offspring1[edge_region1] = 1
    offspring2[edge_region2] = 1

def tournament_selection(K, N, *args):
    fitness_list = []
    for fitness_array in args:
        fitness_list.append(fitness_array.reshape(1, -1))
    fitness = np.vstack(fitness_list)
    _, rank = np.unique(fitness, return_inverse=True, axis=1)
    parents = np.random.randint(low=0, high=len(rank), size=(N, K))
    best = np.argmin(rank[parents], axis=1)
    index = parents[np.arange(N), best]
    return index


def singleFEA(i, modelDatabase):
    modelDatabase.Job('OutputDatabase_' + str(i), modelName, numDomains=int(myNumCpus),
                      numCpus=int(myNumCpus)).submit()
    modelDatabase.jobs['OutputDatabase_' + str(i)].waitForCompletion()

    return 'OutputDatabase_' + str(i) + '.odb'



def multiFEA(batch_size, modelDatabase, densities_batch, indexToElLabel):
    job_names = []

    for i in range(batch_size):
        job_name = 'OutputDatabase_' + str(i)
        EvoupdateModelDatabase(densities_batch[i], indexToElLabel)
        modelDatabase.Job(job_name, modelName, numDomains=int(myNumCpus), numCpus=int(myNumCpus)).submit()
        job_names.append(job_name)

    for job_name in job_names:
        modelDatabase.jobs[job_name].waitForCompletion()

    ODBFileNames = [job_name + '.odb' for job_name in job_names]
    return ODBFileNames


def singleobjectiveFunctionFromODB(ODBFileName):

    outputDatabase = openOdb(ODBFileName)


    objectiveFunction = 0.0
    for step in outputDatabase.steps.values():
        objectiveFunction += step.historyRegions['Assembly ASSEMBLY'].historyOutputs['ALLWK'].data[-1][1]


    outputDatabase.close()
    return objectiveFunction


def densityToYoungs(density,E0,Emin,penalty):

    E = Emin + math.pow(density,penalty)*(E0-Emin)
    return E


def EvosingleinitModelDatabase(designSpaceSet,E0,Emin,poisson):

    model.Material('Material0').Elastic(((Emin, poisson),))
    model.Material('Material1').Elastic(((E0, poisson),))
    model.HomogeneousSolidSection('Section0', 'Material0')
    model.HomogeneousSolidSection('Section1', 'Material1')

    part.SectionAssignment(designSpaceSet, 'Section1')

    for step in [k for k in model.steps.keys() if k != 'Initial']:
        model.FieldOutputRequest('StrainEnergies'+step,step,variables=('ELEDEN', ),frequency=LAST_INCREMENT)
        model.HistoryOutputRequest('Compliance'+step,step,variables=('ALLWK', ),frequency=LAST_INCREMENT)


def GrasingleinitModelDatabase(initDensity, designSpaceSet,densityPrecision,E0,Emin,penalty):

    for i in range(densityPrecision + 1):
        density = float(i) / float(densityPrecision)
        Ei = densityToYoungs(density,E0,Emin,penalty)
        model.Material('Material'+str(i)).Elastic(((Ei, poisson), ))
        model.HomogeneousSolidSection('Section'+str(i),'Material'+str(i))


    scaledDensity = int(initDensity * densityPrecision)
    sectionName = 'Section'+str(scaledDensity)
    part.SectionAssignment(designSpaceSet,sectionName)


    for step in [k for k in model.steps.keys() if k != 'Initial']:
        model.FieldOutputRequest('StrainEnergies'+step,step,variables=('ELEDEN', ),frequency=LAST_INCREMENT)
        model.HistoryOutputRequest('Compliance'+step,step,variables=('ALLWK', ),frequency=LAST_INCREMENT)

def EvoupdateModelDatabase(densities, indexToElLabel):

    scaledDensities = densities.astype('int')
    indexToDensitiesMap = dict(enumerate(scaledDensities,0))

    densitiesToElLabelsMap = defaultdict(list)
    for key, value in sorted(indexToDensitiesMap.iteritems()):
        elLabel, density = indexToElLabel[key], value
        densitiesToElLabelsMap[density].append(elLabel)

    for key, value in densitiesToElLabelsMap.iteritems():
        density, elLabels = key, value
        set = part.SetFromElementLabels('Set'+str(density),elLabels)
        part.SectionAssignment(set,'Section'+str(density))

def GraupdateModelDatabase(densities,densityPrecision,indexToElLabel):

    scaledDensities = (densities * densityPrecision).astype('int')
    indexToDensitiesMap = dict(enumerate(scaledDensities,0))


    densitiesToElLabelsMap = defaultdict(list)
    for key, value in sorted(indexToDensitiesMap.iteritems()):
        elLabel, density = indexToElLabel[key], value
        densitiesToElLabelsMap[density].append(elLabel)


    for key, value in densitiesToElLabelsMap.iteritems():
        density, elLabels = key, value
        set = part.SetFromElementLabels('Set'+str(density),elLabels)
        part.SectionAssignment(set,'Section'+str(density))


def writeToFile(filename,content):
    f = open(filename,'a')
    f.write(str(content))
    f.close()


def writeArrayToCSV(filename, array):
    for i in range(len(array)):
        line = str(i) + ',' + str(array[i]) + '\n'
        writeToFile(filename,line)


def write_list_to_csv(filename, list_data):
    with open(filename, 'w') as file:
        for item in list_data:
            if isinstance(item, np.ndarray):
                np.savetxt(file, item, delimiter=',', fmt='%s')
            else:
                file.write(str(item) + '\n')


def logMsg(msg):
    writeToFile('log.txt', msg + ' \r\n')
    print >> sys.__stdout__, msg



def zerosFloatArray(n):
    return np.zeros(n, 'f')

def cal_fitness(pop):
    N, M = pop.decs.shape
    dominate = np.zeros((N, N), dtype=bool)

    popcons = pop.cons

    for i in range(0, N - 1):
        for j in range(i + 1, N):
            if popcons[i] == 0 and popcons[j] == 1:
                dominate[i][j] = True
            elif popcons[i] == 1 and popcons[j] == 0:
                dominate[j][i] = True
            else:
                k = int(np.any(pop.objs[i] < pop.objs[j])) - int(
                    np.any(pop.objs[i] > pop.objs[j])
                )
                if k == 1:
                    dominate[i][j] = True
                elif k == -1:
                    dominate[j][i] = True

    s = np.sum(dominate, axis=1)

    r = np.zeros(N)
    for i in range(N):
        r[i] = np.sum(s[dominate[:, i]])

    distance = cdist(pop.objs, pop.objs)
    distance[np.eye(len(distance), dtype=bool)] = np.inf
    distance = np.sort(distance, axis=1)
    d = 1 / (distance[:, int(np.floor(np.sqrt(N)) - 1)] + 2)
    fitness = r + d

    return fitness

def environmental_selection(pop, N,Gen):
    if Gen < 20:
        coords = pop.decs.reshape(pop.decs.shape[0], -1, 2).copy()
        for i in range(coords.shape[0]):
            coords[i] = coords[i][np.lexsort((coords[i][:, 1], coords[i][:, 0]))]
        _, indices = np.unique(coords, axis=0, return_index=True)
        pop = pop[indices]

    fitness = cal_fitness(pop)

    next = fitness < 1
    if np.sum(next) < N:
        rank = np.argsort(fitness)
        next[rank[:N]] = True
    elif np.sum(next) > N:
        delete = truncation(pop.objs[next], np.sum(next) - N)
        temp = np.argwhere(next)
        next[temp[delete]] = False
    pop = pop[next]
    fitness = fitness[next]
    return pop, fitness

def truncation(objs, K):
    distance = cdist(objs, objs)
    distance[np.eye(len(distance), dtype=bool)] = np.inf
    delete = np.zeros(len(objs), dtype=bool)
    while np.sum(delete) < K:
        remain = np.argwhere(~delete).flatten()
        temp = distance[remain]
        temp = temp[:, remain]
        temp = np.sort(temp, axis=1)
        _, rank = np.unique(temp, return_index=True, axis=0)
        delete[remain[rank[0]]] = True
    return delete

def generateNewDec(Pop, pop_size, nvar, cellsize, *args):
    decs = Pop.decs
    if args:
        NewDec = np.ones((pop_size, nvar))
        for i in range(pop_size):
            sub_coords_array = Conversion(decs[i,:],cellsize)
            indices = turntoIndices(sub_coords_array,args[0])
            NewDec[i,indices] = 0
        return NewDec
    else:
        NewDec = np.zeros((pop_size, nvar))
        for i in range(pop_size):
            dec = decs[i, :]
            coordinates = dec.reshape(-1, 2)
            sub_coords_list = []
            for x, y in coordinates:
                sub_coords_list.append((x, y))
                sub_coords_list.append((x, y + cellsize))
                sub_coords_list.append((x + cellsize, y))
                sub_coords_list.append((x + cellsize, y + cellsize))
            sub_coords_array = np.array(sub_coords_list)
            NewDec[i, :] = sub_coords_array.ravel()
        return NewDec

def judgeSolution(best, objs, Flag):
    if best is None:
        if not Flag:
            logMsg("No solution found and Flag is False. Returning None for best densities.")
            return None
        else:
            logMsg("No solution found and Flag is True. Appending NaN to objs.")
            objs.append(0)
            objs.append(np.nan)
            return objs
    else:
        _, nonRepetition = np.unique(best.objs, axis=0, return_index=True)
        best = best[nonRepetition]
        if Flag:
            logMsg(np.array2string(best.objs, max_line_width=np.inf, threshold=np.inf))
            logMsg(str(best.objs.shape[0]))
            objs.append(best.objs.shape[0])
            objs.append(best.objs)
            return objs
        else:
            bestDensities = best.densities
            return bestDensities


def densityUpdateOC(densities, sensitivities,maxTotalDensity,maxFilterQ,filterQ):
    updatedDensities = []
    l1, l2, move = 0.0, 100000.0, 0.2


    if (i < 15):
        filterQ = 1.0
    else:
        filterQ = min(maxFilterQ,1.01*filterQ)


    while (l2-l1 > 1e-13):
        lmid = 0.5*(l2+l1)
        updatedDensities = np.maximum(0.001,np.maximum(densities-move,np.minimum(1.0,np.minimum(densities+move,np.power(densities*np.sqrt(sensitivities/lmid),filterQ)))))
        if (totalDensity(updatedDensities) - maxTotalDensity) > 0.0:
            l1 = lmid
        else:
            l2 = lmid
    return updatedDensities

def historyAverage(prevSensitivities, sensitivities):
    averagedSensitivities = (prevSensitivities + sensitivities)/2.0
    return averagedSensitivities

def filterMapFromModelDatabase(modelDatabase, elements, filterRadius):

    filterMap = {}

    coord = zerosFloatArray((len(elements), 3))
    for i in range(len(elements)):
        coord[i] = 0.0

        nds = elements[i].connectivity
        for nd in nds:
            coord[i] = np.add(coord[i], np.divide(nodes[nd].coordinates, len(nds)))

    points = zip(coord[:, 0], coord[:, 1], coord[:, 2])
    tree = spatial.KDTree(points)

    neighbours = tree.query_ball_tree(tree, filterRadius, 2.0)
    for i in range(len(neighbours)):
        elNeighbours = neighbours[i]
        filterMap[i] = [[], []]

        for j in range(len(elNeighbours)):
            k = int(elNeighbours[j])
            dis = np.sqrt(np.sum(np.power(np.subtract(coord[i], coord[k]), 2)))
            filterMap[i][0].append(k)
            filterMap[i][1].append(filterRadius - dis)

        filterMap[i][1] = np.divide(filterMap[i][1], np.sum(filterMap[i][1]))

    return filterMap

def filterSensitivities(sensitivities,filterMap):
    originalSensitivities = sensitivities.copy()
    for el in filterMap.keys():
        sensitivities[el] = 0.0
        for i in range(len(filterMap[el][0])):
            originalIndex = filterMap[el][0][i]
            sensitivities[el]+=originalSensitivities[originalIndex]*filterMap[el][1][i]
    return sensitivities

def sensitivitiesFromODB(ODBFileName, densities,elLabelToIndex,penalty,nrElements):

    outputDatabase = openOdb(ODBFileName)

    sensitivities = zerosFloatArray(nrElements)
    for step in outputDatabase.steps.values():
        abaqusElementsStrainEnergy = step.frames[-1].fieldOutputs['ESEDEN'].values
        for abaqusElementStrainEnergy in abaqusElementsStrainEnergy:

            elLabel = abaqusElementStrainEnergy.elementLabel
            if elLabel in elLabelToIndex:
                index = elLabelToIndex[elLabel]
                sensitivities[index] += penalty*((abaqusElementStrainEnergy.data)/(densities[index]))

    outputDatabase.close()

    return sensitivities

def totalDensity(densities):
    return np.sum(densities)

def gradientOptimization(density,designSpaceSet,indexToElLabel,nrElements):
    gravolumeFraction = np.mean(density == 1)
    maxTotalDensity = nrElements * gravolumeFraction
    density[density == 0] = 0.01
    density[density == 1] = 0.5
    filterRadius = configuration.getfloat('Optimization Config', 'FilterRadius')
    penalty = configuration.getfloat('Optimization Config', 'Penalty')
    maxFilterQ = configuration.getfloat('Optimization Config', 'GreyScaleFilter')
    historyAverageEnabled = configuration.getboolean('Optimization Config','HistoryAverage')
    convergenceCriterium = 0.01
    densityPrecision = 1000
    filterQ = 1.0

    elLabelToIndex = {}
    for i in range(nrElements):
        label = elements[i].label
        index = i
        elLabelToIndex[label] = index

    GrasingleinitModelDatabase(gravolumeFraction, designSpaceSet,densityPrecision,E0,Emin,penalty)

    if filterRadius > 0:
        filterMap = filterMapFromModelDatabase(modelDatabase, elements, filterRadius)

    change, i, objectiveFunction = 1, 0, 0
    sensitivities, prevSensitivities, objectiveFunctionHistory = zerosFloatArray(nrElements), zerosFloatArray(nrElements), []

    while change > convergenceCriterium:

        ODBFileName = singleFEA(0,modelDatabase)
        objectiveFunction = singleobjectiveFunctionFromODB(ODBFileName)
        objectiveFunctionHistory.append(objectiveFunction)
        prevSensitivities = sensitivities.copy()
        sensitivities = sensitivitiesFromODB(ODBFileName,density,elLabelToIndex,penalty,nrElements)

        if filterRadius > 0.0:
            sensitivities = filterSensitivities(sensitivities,filterMap)


        if i > 0 and historyAverageEnabled:
            sensitivities = historyAverage(prevSensitivities, sensitivities)

        density = densityUpdateOC(density, sensitivities,maxTotalDensity,maxFilterQ,filterQ)

        GraupdateModelDatabase(density,densityPrecision,indexToElLabel)
        if i > 10:
            change = math.fabs(
                (sum(objectiveFunctionHistory[i - 4:i + 1]) - sum(objectiveFunctionHistory[i - 9:i - 4])) / sum(
                    objectiveFunctionHistory[i - 9:i - 4]))
        if i > 30:
            break
        i += 1
    return density


def turnintoONE(densities, row, col):
    matrix = np.zeros((row, col))

    for i in range(row):
        if 0 <= i <= 24:
            matrix[24 - i, :] = densities[i * col:(i + 1) * col]
        else:
            matrix[i, :] = densities[i * col:(i + 1) * col][::-1]

    inverted_matrix = np.where(matrix == 0, 1, 0)

    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

    labeled_matrix, num_features = ndimage.label(inverted_matrix, structure=structure)

    object_slices = ndimage.find_objects(labeled_matrix)

    for obj_slice in object_slices:
        region = labeled_matrix[obj_slice]
        region_height, region_width = region.shape
        if region_height < 4 or region_width < 4:
            matrix[obj_slice] = np.where(region > 0, 1, matrix[obj_slice])
        else:
            has_4x4 = False
            for i in range(region_height - 3):
                for j in range(region_width - 3):
                    if np.all(region[i:i+4, j:j+4]):
                        has_4x4 = True
                        break
                if has_4x4:
                    break
            if not has_4x4:
                matrix[obj_slice] = np.where(region > 0, 1, matrix[obj_slice])

    final_densities = np.zeros_like(densities)
    for i in range(row):
        if 0 <= i <= 24:
            final_densities[i * col:(i + 1) * col] = matrix[24 - i, :]
        else:
            final_densities[i * col:(i + 1) * col] = matrix[i, :][::-1]

    return final_densities






if __name__ == '__main__':

    start = datetime.datetime.now()

    configuration = ConfigParser.ConfigParser()
    configuration.read('simp3d-config.ini')

    Emin = 0.001
    E0 = configuration.getfloat('Material Config', 'Youngs')
    poisson = configuration.getfloat('Material Config', 'Poisson')


    mdbName = configuration.get('Model Config', 'ModelDatabaseName')
    modelName = configuration.get('Model Config', 'ModelName')
    partName = configuration.get('Model Config', 'PartName')
    instanceName = configuration.get('Model Config', 'InstanceName')
    designSpaceSetName = configuration.get('Model Config', 'DesignSpaceSetName')
    myNumCpus = configuration.getint('Parallelization Config', 'NumCPUs')

    modelDatabase = openMdb(mdbName)
    model = modelDatabase.models[modelName]
    part = model.parts[partName]
    instance1 = model.rootAssembly.instances[instanceName]
    designSpaceSet = part.sets[designSpaceSetName]
    elements, nodes = designSpaceSet.elements, instance1.nodes
    nrElements = len(elements)


    indexToElLabel = {}

    for i in range(nrElements):
        label = elements[i].label
        index = i
        indexToElLabel[index] = label

    EvosingleinitModelDatabase(designSpaceSet,E0,Emin,poisson)
    objs = []
    objectiveFunctionHistory = []

    t = datetime.datetime.now()
    s = str("{:02d}:{:02d}".format(t.hour, t.minute))
    logMsg('Generate initial population start' + ' at ' + s)
    pop_size = 50
    col = 150
    row = 50
    cellsize = 16
    minExcavation = nrElements * 0.3
    signalExcavation = cellsize * cellsize
    nvar = int(np.ceil(minExcavation / signalExcavation ) * 2)
    varcolUpper = col - cellsize
    varcolLower = 0
    varrowUpper = row - cellsize
    varrowLower = 0
    lb = np.array([varrowLower, varcolLower] * (nvar // 2))
    ub = np.array([varrowUpper, varcolUpper] * (nvar // 2))
    Dec = np.random.uniform(np.tile(lb, (pop_size, 1)), np.tile(ub, (pop_size, 1)))

    Dec = np.round(Dec).astype(int)
    Pop = Population(decs=Dec)
    Pop = Cal(Pop, modelDatabase, cellsize, row, col, 0, indexToElLabel)
    _, fitness = environmental_selection(Pop, pop_size, 0)


    best = Best(Pop)
    Flag = True
    objs = judgeSolution(best, objs, Flag)
    objectiveFunctionHistory.append(min(best.objs[:, 0]) if best is not None else np.nan)

    MaxGen = 40
    Gen = 1
    while Gen <= MaxGen:
        t = datetime.datetime.now()
        s = str("{:02d}:{:02d}".format(t.hour, t.minute))
        logMsg(str(Gen) + 'th start' + ' at ' + s)

        if Gen % 10 == 0 and Gen < 21:
            if Gen == 20:
                logMsg("change encoding")
                nvar = nrElements
                NewDec = generateNewDec(Pop, pop_size, nvar, cellsize, col)
                Pop.decs = NewDec
                OffDec = Operatorbinary(NewDec, row, col)
            else:
                logMsg("change cellsize")
                cellsize = int(cellsize / 2)
                varcolUpper = col - cellsize
                varrowUpper = row - cellsize
                nvar = nvar * 4
                lb = np.array([varrowLower, varcolLower] * (nvar // 2))
                ub = np.array([varrowUpper, varcolUpper] * (nvar // 2))
                NewDec = generateNewDec(Pop, pop_size, nvar, cellsize)
                Pop.decs = NewDec
                OffDec = OperatorReal(NewDec, lb, ub)
            Offspring = Population(decs=OffDec)
            Offspring = Cal(Offspring, modelDatabase, cellsize, row, col, Gen, indexToElLabel)
            Pop, fitness = environmental_selection(Pop + Offspring, pop_size, Gen)
        else:
            MatingPool = tournament_selection(2, pop_size, fitness)
            if Gen >= 20:
                OffDec = Operatorbinary(Pop[MatingPool].decs, row, col)
            else:
                OffDec = OperatorReal(Pop[MatingPool].decs, lb, ub)
            Offspring = Population(decs=OffDec)
            Offspring = Cal(Offspring, modelDatabase, cellsize, row, col, Gen, indexToElLabel)
            Pop, fitness = environmental_selection(Pop + Offspring, pop_size, Gen)

        best = Best(Pop)
        objs = judgeSolution(best, objs, Flag)
        objectiveFunctionHistory.append(min(best.objs[:, 0]) if best is not None else np.nan)
        Gen += 1



    best = Best(Pop)
    Flag = False
    bestdensities = judgeSolution(best, objs, Flag)
    if bestdensities is not None:
        optimizationDensities = np.zeros((bestdensities.shape[0], nrElements))
        optimizationObjs = np.zeros((bestdensities.shape[0], 2))
        optimizationCons = np.zeros((bestdensities.shape[0], 1))
        optimizationDecs = np.zeros((bestdensities.shape[0], 1))

        for i in range(bestdensities.shape[0]):
            t = datetime.datetime.now()
            s = str("{:02d}:{:02d}".format(t.hour, t.minute))
            logMsg('Pop' + ' ' + str(i + 1) + 'th start' + ' at ' + s)

            optimizationDensities[i, :] = gradientOptimization(bestdensities[i, :], designSpaceSet, indexToElLabel, nrElements)
            optimizationDensities[i, :] = np.around(optimizationDensities[i, :])
            optimizationDensities[i, :] = turnintoONE(optimizationDensities[i, :], row, col)
            EvosingleinitModelDatabase(designSpaceSet, E0, Emin, poisson)
            EvoupdateModelDatabase(optimizationDensities[i, :], indexToElLabel)
            ODBFileName = singleFEA(0, modelDatabase)
            optimizationObjs[i, 0] = singleobjectiveFunctionFromODB(ODBFileName)
            optimizationObjs[i, 1] = np.mean(optimizationDensities[i, :] == 1)
            optimizationCons[i] = calCON(optimizationDensities[i, :], row, col)

        optimizationPop = Population(decs=optimizationDecs, densities=optimizationDensities, objs=optimizationObjs, cons=optimizationCons)
        best = Best(optimizationPop)
        Flag = True
        objs = judgeSolution(best, objs, Flag)
        objectiveFunctionHistory.append(min(best.objs[:, 0]) if best is not None else np.nan)

        Flag = False
        finalDensities = judgeSolution(best, objs, Flag)

        if finalDensities is not None:
            for i in range(finalDensities.shape[0]):
                EvoupdateModelDatabase(finalDensities[i], indexToElLabel)
                modelDatabase.saveAs('Optimal_Design_after_' + str(i + 1))
                ODBFileName = singleFEA(i, modelDatabase)
                upgradeOdb(existingOdbPath=ODBFileName, upgradedOdbPath='Optimal_ODB_after_' + str(i + 1) + '.odb')

    writeArrayToCSV('objectiveFunctionHistory.csv', objectiveFunctionHistory)
    write_list_to_csv('objs.csv', objs)

    end = datetime.datetime.now()
    spend = end - start
    logMsg('Total time consumption: ' + str(spend))







