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
from scipy.spatial.distance import cdist
from scipy.special import comb
from itertools import combinations
import random
from scipy.cluster.vq import kmeans2
from scipy.linalg import sqrtm

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


def turntocoordinate(PopDec, varrowUpper, varcolUpper):

    coordinateDec = np.zeros_like(PopDec)
    for j in range(PopDec.shape[1]):
        if j % 2 == 0:
            coordinateDec[:, j] = (PopDec[:, j] * varrowUpper)
        else:
            coordinateDec[:, j] = (PopDec[:, j] * varcolUpper)

    coordinateDec = np.around(coordinateDec).astype(int)

    return coordinateDec


def Cal(Pop, modelDatabase, cellsize, row, col, indexToElLabel, varrowUpper, varcolUpper):
    PopDec = Pop.decs
    coordinateDec = turntocoordinate(PopDec, varrowUpper, varcolUpper)
    N, _ = PopDec.shape

    batch_size = 3

    Popobj = np.zeros((N, 2))
    Popcon = np.zeros((N, 1))
    Popdensities = np.ones((N, row * col))

    for i in range(0, N, batch_size):
        batch_end = min(i + batch_size, N)
        batch_indices = range(i, batch_end)

        for j in batch_indices:
            sub_coords_array = Conversion(coordinateDec[j, :], cellsize)
            indices = turntoIndices(sub_coords_array, col)
            Popdensities[j, indices] = 0
            Popcon[j] = calCON(Popdensities[j, :], row, col)

        if len(batch_indices) == 1:
            ODBFileName = singleFEA(0, modelDatabase)
            ODBFileNames = [ODBFileName]
        else:
            ODBFileNames = multiFEA(len(batch_indices),
                                    modelDatabase,
                                    Popdensities[i:batch_end, :],
                                    indexToElLabel)

        for local_idx, idx in enumerate(batch_indices):
            Popobj[idx, 0] = singleobjectiveFunctionFromODB(ODBFileNames[local_idx])
            Popobj[idx, 1] = np.mean(Popdensities[idx, :] == 1)

    Pop.objs = Popobj
    Pop.cons = Popcon
    Pop.densities = Popdensities

    return Pop



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

def multiobjectiveFunctionFromODB(ODBFileName):
    objectiveFunction = np.zeros((1, 2))
    outputDatabase = openOdb(ODBFileName)

    for step in outputDatabase.steps.values():
        objectiveFunction[:,0] += step.historyRegions['Assembly ASSEMBLY'].historyOutputs['ALLWK'].data[-1][1]
        U = step.frames[-1].fieldOutputs['U'].values

    objectiveFunction[:,1] = -float('inf')

    for U1 in U:
        objectiveFunction[:,1] = max(objectiveFunction[:,1], U1.data[1])
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


def judgeSolution(best, objs, Flag):
    if best is None:
        if not Flag:
            logMsg("No solution found and Flag is False. Returning None for best densities.")
            return None
        else:
            logMsg("No solution found and Flag is True. Appending NaN to objs.")
            objs.append(0)
            objs.append(np.nan)  # 添加 NaN
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



def uniform_point(N, M):
    H1 = 1
    while comb(H1 + M, M - 1, exact=True) <= N:
        H1 += 1
    W = (
        np.array(list(combinations(range(1, H1 + M), M - 1)))
        - np.tile(np.arange(M - 1), (comb(H1 + M - 1, M - 1, exact=True), 1))
        - 1
    )
    W = (
        np.hstack((W, np.zeros((W.shape[0], 1)) + H1))
        - np.hstack((np.zeros((W.shape[0], 1)), W))
    ) / H1
    if H1 < M:
        H2 = 0
        while (
            comb(H1 + M - 1, M - 1, exact=True)
            + comb(H2 + M, M - 1, exact=True)
            <= N
        ):
            H2 += 1
        if H2 > 0:
            W2 = (
                np.array(list(combinations(range(1, H2 + M), M - 1)))
                - np.tile(
                    np.arange(M - 1), (comb(H2 + M - 1, M - 1, exact=True), 1)
                )
                - 1
            )
            W2 = (np.hstack((W2, np.zeros(
                (W2.shape[0], 1)))) - np.hstack((np.zeros((W2.shape[0], 1)), W2))) / H2  # noqa
            W = np.vstack((W, W2 / 2 + 1 / (2 * M)))
    W = np.maximum(W, 1e-6)
    N = W.shape[0]
    return W, N


def UpdateCMA(X, Sigma, gen):

    n = X.shape[1]
    mu = 4 + np.floor(3 * np.log(n))
    mu1 = int(np.floor(mu / 2))
    w = np.log((mu + 1) / 2) - np.log(np.arange(1, mu1 + 1))
    w = w / np.sum(w)
    w = np.array([w])
    mueff = 1 / np.sum(w ** 2)
    cs = (mueff + 2) / (n + mueff + 5)
    tmp = np.sqrt((mueff - 1) / (n + 1)) - 1
    ds = 1 + 2 * np.where(tmp < 0, 0, tmp) + cs
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = np.min(
        (1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff)))
    ENI = np.sqrt(n) * (1 - 1 / 4 / n + 1 / 21 / n ** 2)
    y = (X[0: mu1, :] - np.tile(Sigma['x'], (mu1, 1))) / Sigma['sigma']
    yw = np.dot(w, y)
    Sigma['x'] = Sigma['x'] + Sigma['sigma'] * yw
    Sigma['ps'] = (1 - cs) * Sigma['ps'] + np.dot(np.sqrt(cs *
                                                          (2 - cs) * mueff) * np.linalg.inv(sqrtm(Sigma['C'])), yw.T)  # noqa
    hs = np.linalg.norm(Sigma['ps']) / np.sqrt(1 - (1 - cs)
                                               ** (2 * (gen + 1))) < (1.4 + 2 / (n + 1)) * ENI  # noqa
    deltahs = 1 - hs
    Sigma['pc'] = (1 - cc) * Sigma['pc'] + hs * \
        np.sqrt(cc * (2 - cc) * mueff) * yw
    Sigma['sigma'] = Sigma['sigma'] * \
        np.exp(cs / ds * (np.linalg.norm(Sigma['ps']) / ENI - 1))
    Sigma["C"] = (1 - c1 - cmu) * Sigma['C'] + c1 * (np.dot(Sigma['pc']. T, Sigma['pc']) +  # noqa
                                                     deltahs * Sigma['C']) + cmu * np.dot(np.dot(y.T, np.diag(w.flatten())), y)  # noqa
    Sigma["C"] = np.triu(Sigma['C']) + np.triu(Sigma['C'], 1).T
    D, B = np.linalg.eigh(Sigma['C'])
    diagD = D
    diagC = np.diag(Sigma['C'])
    ConditionCov = np.max(diagD) > (1e+14) * np.min(diagD)
    NoEffectCoord = np.any(
        Sigma['x'] == Sigma['x'] + 0.2 * Sigma['sigma'] * np.sqrt(diagC))
    NoEffectAxis = np.all(Sigma['x'] == Sigma['x'] + 0.1 * Sigma['sigma']
                          * np.sqrt(diagD[int(np.mod(gen, n))]) * B[:, int(np.mod(gen, n))].T)  # noqa
    TolXUp = np.any(Sigma['sigma'] * np.sqrt(diagC) > (1e+4))
    if ConditionCov or NoEffectCoord or NoEffectAxis or TolXUp:
        t = [0] * n
        Sigma = {'s': [], 'x': np.array(
            t), 'sigma': 0.5, 'C': np.eye(n), 'pc': 0, 'ps': 0}
    return Sigma


def OperatorDE(Parent1Dec, Parent2Dec, Parent3Dec, lb, ub):
    CR = 1
    F  = 0.5
    N, D = Parent1Dec.shape

    # crossover
    Site = np.random.random((N, D)) < CR
    OffDec = Parent1Dec.copy()
    OffDec[Site] = OffDec[Site] + F * (Parent2Dec[Site] - Parent3Dec[Site])


    # Differental evolution
    disM = 20  # Distribution index for mutation
    proM = 1  # Mutation probability
    Lower = np.tile(lb, (N, 1))
    Upper = np.tile(ub, (N, 1))
    Site = np.random.random((N, D)) < proM / D
    mu = np.random.random((N, D))

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


def fix_decs(tmp, ub, lb):
    fixdecs = np.fmax(np.fmin(tmp, ub), lb)
    return fixdecs


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
    ss = str("{:02d}:{:02d}".format(t.hour, t.minute))
    logMsg('Generate initial population start' + ' at ' + ss)
    pop_size = 50
    col = 150
    row = 50
    cellsize = 8
    minExcavation = nrElements * 0.5
    signalExcavation = cellsize * cellsize
    nvar = int(np.ceil(minExcavation / signalExcavation ) * 2)
    varcolUpper = col - cellsize
    varcolLower = 0
    varrowUpper = row - cellsize
    varrowLower = 0



    K = 5
    W, pop_size = uniform_point(pop_size, 2)
    T = np.ceil(pop_size / 10)
    W = 1 / W / np.tile(np.sum(1 / W, axis=1, keepdims=True), (1, W.shape[1]))
    _, G = kmeans2(W, K, minit='++')
    t = [[] for i in range(K)]
    for i in range(K):
        t[i] = np.argwhere(G == i).flatten()
    G = t
    B = cdist(W, W)
    B = np.argsort(B, axis=1, kind="mergesort")
    B = B[:, 0:int(T)]

    lb = np.zeros(nvar)
    ub = np.ones(nvar)
    Dec = np.random.uniform(np.tile(lb, (pop_size, 1)), np.tile(ub, (pop_size, 1)))
    Pop = Population(decs=Dec)
    Pop = Cal(Pop, modelDatabase, cellsize, row, col, indexToElLabel, varrowUpper, varcolUpper)
    Z = np.min(Pop.objs, axis=0)
    sk = [[] for i in range(K)]
    for i in range(K):
        sk[i] = G[i][np.random.randint(len(G[i]))]
    xk = Pop[sk].decs
    Sigma = {'s': sk, 'x': xk, 'sigma': [0.5] * K, 'C': [np.eye(nvar) for i in range(K)], 'pc': [0] * K, 'ps': [0] * K}


    best = Best(Pop)
    Flag = True
    objs = judgeSolution(best, objs, Flag)
    objectiveFunctionHistory.append(min(best.objs[:, 0]) if best is not None else np.nan)

    MaxGen = 40
    Gen = 1
    while Gen <= MaxGen:
        t = datetime.datetime.now()
        ss = str("{:02d}:{:02d}".format(t.hour, t.minute))
        logMsg(str(Gen) + 'th start' + ' at ' + ss)

        for s in range(pop_size):
            if s in Sigma["s"]:
                k = Sigma["s"].index(s)
            else:
                k = -1
            if k != -1:
                P = B[s, random.sample(np.arange(B.shape[1]).tolist(), B.shape[1])]
                tmp = np.random.multivariate_normal(Sigma['x'][k, :], Sigma['sigma'][k] ** 2 * Sigma['C'][k], 4 + int(np.floor(3 * np.log(nvar))))
                tmp = fix_decs(tmp, ub, lb)
                Offspring = Population(decs=tmp)
                Offspring = Cal(Offspring, modelDatabase, cellsize, row, col, indexToElLabel, varrowUpper, varcolUpper)
                Combine = Offspring + Pop[s]
                rank = np.argsort(np.max(np.abs(Combine.objs - np.tile(Z, (Combine.decs.shape[0], 1))) * np.tile(W[s, :], (Combine.decs.shape[0], 1)), axis=1), kind='mergsort')
                Sigma_tmp = {'s': Sigma['s'][k], 'x': Sigma['x'][k], 'sigma': Sigma['sigma']
                [k], 'C': Sigma['C'][k], 'pc': Sigma['pc'][k], 'ps': Sigma['ps'][k]}
                tmp = UpdateCMA(Combine[rank].decs, Sigma_tmp, np.ceil(Gen / pop_size))
                Sigma['s'][k] = tmp['s']
                Sigma['x'][k] = tmp['x']
                Sigma['sigma'][k] = tmp['sigma']
                Sigma['C'][k] = tmp['C']
                Sigma['pc'][k] = tmp['pc']
                Sigma['ps'][k] = tmp['ps']
                if Sigma['s'][k] == []:
                    sk = G[k][np.random.randint(len(G[k]))]
                    Sigma['s'][k] = sk
                    Sigma['x'][k] = Pop[int(sk)].decs
            else:
                if np.random.random() < 0.9:
                    P = B[s, random.sample(np.arange(B.shape[1]).tolist(), B.shape[1])]
                else:
                    P = random.sample(np.arange(pop_size).tolist(), pop_size)
                    P = np.array(P)
                Offspring = OperatorDE(Pop[s].decs, Pop[int(P[0])].decs, Pop[int(P[1])].decs, lb, ub)
                Offspring = Population(decs=Offspring)
                Offspring = Cal(Offspring, modelDatabase, cellsize, row, col, indexToElLabel, varrowUpper, varcolUpper)

            for x in range(Offspring.decs.shape[0]):
                Z = np.where(Z < Offspring[x].objs, Z, Offspring[x].objs)
                g_old = np.max(np.abs(Pop[P].objs - np.tile(Z, (len(P), 1))) * W[P, :], axis=1)
                g_new = np.max(np.tile(np.abs(Offspring[x].objs - Z), (len(P), 1)) * W[P, :], axis=1)
                tmp = np.argwhere(g_old >= g_new)
                if len(tmp) >= 2:
                    t = tmp[[0, 1]]
                else:
                    if len(tmp) == 1:
                        t = tmp[0]
                    else:
                        t = []
                Pop[P[t]] = Offspring[x]

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
            ss = str("{:02d}:{:02d}".format(t.hour, t.minute))
            logMsg('Pop' + ' ' + str(i + 1) + 'th start' + ' at ' + ss)

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







