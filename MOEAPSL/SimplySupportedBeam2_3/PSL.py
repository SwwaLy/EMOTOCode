# coding:utf-8

import math, customKernel
import numpy as np
import ConfigParser
from abaqusConstants import *
from collections import defaultdict
from odbAccess import openOdb, upgradeOdb
from scipy import spatial
import datetime
from scipy import ndimage

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

class RBM:
    def __init__(self, nVisible=0, nHidden=0, Epoch=10, BatchSize=1, Penalty=0.01, Momentum=0.5, LearnRate=0.1):
        self.nVisible = nVisible
        self.nHidden = nHidden
        self.Epoch = Epoch
        self.BatchSize = BatchSize
        self.Penalty = Penalty
        self.Momentum = Momentum
        self.LearnRate = LearnRate
        self.Weight = 0.1 * np.random.randn(self.nVisible, self.nHidden)
        self.vBias = np.zeros(self.nVisible)
        self.hBias = np.zeros(self.nHidden)

    def sigmoid(self, x):
        # Clip values to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def train(self, X):
        vishidinc = np.zeros_like(self.Weight)
        hidbiasinc = np.zeros_like(self.hBias)
        visbiasinc = np.zeros_like(self.vBias)
        for epoch in range(self.Epoch):
            if epoch > 5:
                self.Momentum = 0.9
            kk = np.random.permutation(X.shape[0])
            for batch in range(X.shape[0] // self.BatchSize):
                batchdata = X[kk[batch * self.BatchSize: (batch + 1) * self.BatchSize], :]

                # Positive phase
                poshidprobs = self.sigmoid(np.dot(batchdata, self.Weight) + np.tile(self.hBias, (self.BatchSize, 1)))
                poshidstates = poshidprobs > np.random.rand(self.BatchSize, self.nHidden)

                # Negative phase
                negdataprobs = self.sigmoid(np.dot(poshidstates, self.Weight.T) + np.tile(self.vBias, (self.BatchSize, 1)))
                negdata = negdataprobs > np.random.rand(self.BatchSize, self.nVisible)
                neghidprobs = self.sigmoid(np.dot(negdata, self.Weight) + np.tile(self.hBias, (self.BatchSize, 1)))

                # Update weight
                posprods = np.dot(batchdata.T, poshidprobs)
                negprods = np.dot(negdataprobs.T, neghidprobs)
                poshidact = np.sum(poshidprobs, axis=0)
                posvisact = np.sum(batchdata, axis=0)
                neghidact = np.sum(neghidprobs, axis=0)
                negvisact = np.sum(negdata, axis=0)

                vishidinc = self.Momentum * vishidinc + self.LearnRate * ((posprods - negprods) / self.BatchSize - self.Penalty * self.Weight)
                visbiasinc = self.Momentum * visbiasinc + (self.LearnRate / self.BatchSize) * (posvisact - negvisact)
                hidbiasinc = self.Momentum * hidbiasinc + (self.LearnRate / self.BatchSize) * (poshidact - neghidact)
                self.Weight += vishidinc
                self.vBias += visbiasinc
                self.hBias += hidbiasinc

    def reduce(self, X):
        return self.sigmoid(np.dot(X, self.Weight) + np.tile(self.hBias, (X.shape[0], 1))) > np.random.rand(X.shape[0], self.nHidden)

    def recover(self, H):
        return self.sigmoid(np.dot(H, self.Weight.T) + np.tile(self.vBias, (H.shape[0], 1))) > np.random.rand(H.shape[0], self.nVisible)


def crowding_distance(pop, frontNo=None):
    if isinstance(pop, Population):
        objs = pop.objs
    else:
        objs = pop
    N, M = objs.shape
    if frontNo is None:
        frontNo = np.ones(N)
    # Initialize crowding distance as zero
    cd = np.zeros(N)
    fronts = np.setdiff1d(np.unique(frontNo), np.inf)
    for f in range(len(fronts)):
        front = np.argwhere(frontNo == fronts[f]).flatten()
        fmax = np.max(objs[front], axis=0).flatten()
        fmin = np.min(objs[front], axis=0).flatten()
        for i in range(M):
            rank = np.argsort(objs[front, i])
            cd[front[rank[0]]] = np.inf
            cd[front[rank[-1]]] = np.inf
            for j in range(1, len(front) - 1):
                cd[front[rank[j]]] += (
                    objs[front[rank[j + 1]]][i] - objs[front[rank[j - 1]]][i]
                ) / (fmax[i] - fmin[i])
    return cd

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



def Cal(Pop, modelDatabase, row, col, indexToElLabel):
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
            Popdensities[j, :] = PopDec[j, :]

        ODBFileNames = multiFEA(len(batch_indices), modelDatabase, Popdensities[i:batch_end, :], indexToElLabel)

        for j, idx in enumerate(batch_indices):
            Popobj[idx, 0] = singleobjectiveFunctionFromODB(ODBFileNames[j])
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


def BinaryCrossover(Parent1, Parent2):
    k = np.random.random(size=(Parent1.shape)) < 0.5
    Offspring1 = Parent1
    Offspring2 = Parent2
    Offspring1[k] = Parent2[k]
    Offspring2[k] = Parent1[k]
    Offspring = np.vstack((Offspring1, Offspring2))
    return Offspring

def BinaryMutation(Offspring):
    Site = np.random.random(size=(Offspring.shape)) < 1 / Offspring.shape[1]
    Offspring[Site] = ~Offspring[Site].astype(bool)
    return Offspring

def Operator(ParentDec, ParentMask, rbm, dae, Site, allZero, allOne):  # noqa
    Parent1Mask = ParentMask[0: int(ParentMask.shape[0] / 2), :]
    Parent2Mask = ParentMask[int(ParentMask.shape[0] / 2):, :]
    Parent1Dec = ParentDec[0: int(ParentDec.shape[0] / 2), :]
    Parent2Dec = ParentDec[int(ParentDec.shape[0] / 2):, :]

    if np.any(Site):
        other = np.logical_and(~allZero, ~allOne)
        OffTemp = BinaryCrossover(rbm.reduce(Parent1Mask[Site.flatten()][:, other]),  # noqa
                                  rbm.reduce(Parent2Mask[Site.flatten()][:, other]))  # noqa
        OffTemp = rbm.recover(OffTemp)
        OffMask = np.zeros((OffTemp.shape[0], Parent1Mask.shape[1]))
        OffMask[:, other] = OffTemp
        OffMask[:, allOne] = True
    else:
        OffMask = []
    if len(OffMask) == 0:
        OffMask = BinaryCrossover(
            Parent1Mask[~Site.flatten(), :], Parent2Mask[~Site.flatten(), :])
    else:
        OffMask = np.vstack((OffMask, BinaryCrossover(
            Parent1Mask[~Site.flatten(), :], Parent2Mask[~Site.flatten(), :])))
    OffMask = BinaryMutation(OffMask)

    offdec = np.ones(OffMask.shape)
    return offdec, OffMask

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

def EvoupdateModelDatabase(densities,indexToElLabel):

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


def EnvironmentalSelection(pop, dec, Mask, N, length, num):
    success = np.zeros(pop.decs.shape[0])
    _, uni, _ = np.unique(pop.objs, return_index=True,
                          return_inverse=True, axis=0)
    if len(uni) == 1:
        _, uni, _ = np.unique(pop.decs, return_index=True,
                              return_inverse=True, axis=0)
    pop = pop[uni]
    dec = dec[uni, :]
    Mask = Mask[uni, :]
    N = np.min((N, pop.decs.shape[0]))
    FrontNo, MaxFNo = nd_sort(pop.objs, pop.cons, N)
    Next = FrontNo < MaxFNo
    CrowDis = crowding_distance(pop.objs, FrontNo)
    last = np.argwhere(FrontNo == MaxFNo)
    rank = np.argsort(-CrowDis[last], axis=0)
    Next[last[rank[0:(N - np.sum(Next))]]] = True

    # Calculate the ratio of successful offsprings
    success[uni[Next]] = True
    s1 = np.sum(success[length + 1: length + num])
    s2 = np.sum(success[length + num:])
    sRatio = (s1 + (1e-6)) / (s1 + s2 + (1e-6))
    sRatio = np.min((np.max((sRatio, 0.1)), 0.9))
    pop = pop[Next]
    FrontNo = FrontNo[Next]
    CrowDis = CrowDis[Next]
    dec = dec[Next, :]
    Mask = Mask[Next, :]
    return pop, dec, Mask, FrontNo, CrowDis, sRatio



def judgeSolution(best, objs, Flag):
    if best is None:
        logMsg("No solution found.")
        return None
    else:
        _, nonRepetition = np.unique(best.objs, axis=0, return_index=True)
        best = best[nonRepetition]
        if Flag:
            logMsg(np.array2string(best.objs,max_line_width=np.inf,threshold=np.inf))
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
    W = np.argsort(np.random.random(size=(N, M)), axis=0)
    W = (np.random.random(size=(N, M)) + W - 1) / N
    return W, N


def ModelTraining(Mask, dec):
    # Determine the size of hidden layers
    allzero = np.all(~Mask, axis=0)
    allone = np.all(Mask, axis=0)
    other = np.logical_and(~allzero, ~allone)
    K = np.sum(
        np.mean(np.abs(Mask[:, other] * dec[:, other]) > (1e-6), axis=0) > np.random.random(size=(1, np.sum(other))))  # noqa
    K = np.min((np.max((K, 1)), Mask.shape[0]))

    #  Train RBM and DAE
    rbm = RBM(np.sum(other), K, 10, 1, 0, 0.5, 0.1)
    rbm.train(Mask[:, other])
    dae = []
    return rbm, dae, allzero, allone


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

    EvosingleinitModelDatabase(designSpaceSet, E0, Emin, poisson)
    objs = []
    objectiveFunctionHistory = []

    t = datetime.datetime.now()
    s = str("{:02d}:{:02d}".format(t.hour, t.minute))
    logMsg('Generate initial population start' + ' at ' + s)
    pop_size = 50
    col = 150
    row = 50
    nvar = nrElements
    Mask, _ = uniform_point(pop_size, nvar)
    Mask = Mask > 0.5
    Dec = np.ones((pop_size, nvar))

    Pop = Population(decs=Dec * Mask)
    Pop = Cal(Pop, modelDatabase, row, col, indexToElLabel)
    Pop, Dec, Mask, FrontNo, CrowdDis, _ = EnvironmentalSelection(Pop, Dec, Mask, pop_size, 0, 0)

    best = Best(Pop)
    Flag = True
    objs = judgeSolution(best, objs, Flag)
    objectiveFunctionHistory.append(min(best.objs[:, 0]))

    MaxGen = 40
    Gen = 1
    rho = 0.5
    while Gen <= MaxGen:
        t = datetime.datetime.now()
        s = str("{:02d}:{:02d}".format(t.hour, t.minute))
        logMsg(str(Gen) + 'th start' + ' at ' + s)

        site = rho > np.random.random(size=(1, np.ceil(pop_size / 2).astype(int)))
        if np.any(site):
            rbm, dae, allZero, allOne = ModelTraining(Mask, Dec)
        else:
            rbm, dae, allZero, allOne = [[], [], [], []]

        MatingPool = tournament_selection(2, np.ceil(pop_size / 2).astype(int) * 2, FrontNo, -CrowdDis)  # noqa
        Offdec, Offmask = Operator(Dec[MatingPool, :], Mask[MatingPool, :], rbm, dae, site, allZero, allOne)
        Offspring = Population(decs=Offdec * Offmask)
        Offspring = Cal(Offspring, modelDatabase, row, col, indexToElLabel)

        Pop, Dec, Mask, FrontNo, CrowdDis, sRatio = EnvironmentalSelection(Pop + Offspring, np.vstack((Dec, Offdec)),np.vstack((Mask, Offmask.astype(bool))),pop_size, Pop.decs.shape[0], 2 * np.sum(site))  # noqa
        rho = (rho + sRatio) / 2

        best = Best(Pop)
        objs = judgeSolution(best, objs, Flag)
        objectiveFunctionHistory.append(min(best.objs[:, 0]))
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
        finalDensities = best.densities

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






