# coding:utf-8
import math, customKernel
import numpy as np
import ConfigParser
from abaqusConstants import *
from collections import defaultdict
from abaqus import getInput,getInputs
from odbAccess import openOdb, upgradeOdb
from scipy import spatial, ndimage
import datetime

# 调用有限元计算模块
def FEA(i,modelDatabase):
    
    modelDatabase.Job('OutputDatabase_'+str(i),modelName,numDomains=int(myNumCpus),numCpus=int(myNumCpus)).submit()# 调用abaqus内部函数进行仿真计算
    modelDatabase.jobs['OutputDatabase_'+str(i)].waitForCompletion()


    return 'OutputDatabase_'+str(i)+'.odb'# 仿真结果储存在odb文件中


def sensitivitiesFromODB(ODBFileName, densities):

    outputDatabase = openOdb(ODBFileName)# 读取odb文件

    sensitivities = zerosFloatArray(nrElements)
    for step in outputDatabase.steps.values():
        abaqusElementsStrainEnergy = step.frames[-1].fieldOutputs['ESEDEN'].values# 读取整个模型所有网格的单元敏度
        for abaqusElementStrainEnergy in abaqusElementsStrainEnergy:

            elLabel = abaqusElementStrainEnergy.elementLabel
            if elLabel in elLabelToIndex: 
                index = elLabelToIndex[elLabel]
                sensitivities[index] += penalty*((abaqusElementStrainEnergy.data)/(densities[index]))# 获取设计域内网格的单元敏度
    

    outputDatabase.close()

    return sensitivities;

# 读取结构对应目标函数
def objectiveFunctionFromODB(ODBFileName):

    outputDatabase = openOdb(ODBFileName)


    objectiveFunction = 0.0
    for step in outputDatabase.steps.values():
        objectiveFunction += step.historyRegions['Assembly ASSEMBLY'].historyOutputs['ALLWK'].data[-1][1]


    outputDatabase.close()
    return objectiveFunction

# 优化设计域内网格单元密度。输入为上一循环得到的单元密度以及敏度，输出为更新后的单元密度
def densityUpdateOC(densities, sensitivities):
    updatedDensities = []
    l1, l2, move = 0.0, 100000.0, 0.2


    global filterQ
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


# 仿真准备
def initModelDatabase(modelDatabase, initDensity, designSpaceSet):

    for i in range(densityPrecision + 1):
        density = float(i) / float(densityPrecision)
        Ei = densityToYoungs(density)
        model.Material('Material'+str(i)).Elastic(((Ei, poisson), ))
        model.HomogeneousSolidSection('Section'+str(i),'Material'+str(i))# 定义材料（将单元密度导入cae文件的方式）


    scaledDensity = int(initDensity * densityPrecision)
    sectionName = 'Section'+str(scaledDensity)
    part.SectionAssignment(designSpaceSet,sectionName)


    for step in [k for k in model.steps.keys() if k != 'Initial']:# 调用abaqus内部函数，规定目标函数输出
        model.FieldOutputRequest('StrainEnergies'+step,step,variables=('ELEDEN', ),frequency=LAST_INCREMENT)
        model.HistoryOutputRequest('Compliance'+step,step,variables=('ALLWK', ),frequency=LAST_INCREMENT)

# 单元密度更新
def updateModelDatabase(modelDatabase, densities, designSpaceElements):

    scaledDensities = (densities * densityPrecision).astype('int')# 网格单元密度归类
    indexToDensitiesMap = dict(enumerate(scaledDensities,0))


    densitiesToElLabelsMap = defaultdict(list)
    for key, value in sorted(indexToDensitiesMap.iteritems()):
        elLabel, density = indexToElLabel[key], value
        densitiesToElLabelsMap[density].append(elLabel)


    for key, value in densitiesToElLabelsMap.iteritems():
        density, elLabels = key, value
        set = part.SetFromElementLabels('Set'+str(density),elLabels)
        part.SectionAssignment(set,'Section'+str(density))# 更新cae文件的材料参数，实现更新后单元密度的导入


def historyAverage(prevSensitivities, sensitivities):
    averagedSensitivities = (prevSensitivities + sensitivities)/2.0
    return averagedSensitivities;


def filterMapFromModelDatabase(modelDatabase, elements, filterRadius):# 以模型、网格、过滤半径为输入

    filterMap = {}


    coord = zerosFloatArray((len(elements),3))# 存储网格中心坐标


    for i in range(len(elements)):
        coord[i] = 0.0

        nds = elements[i].connectivity# 获取网格对应的节点编号
        for nd in nds:

            coord[i] = np.add(coord[i],np.divide(nodes[nd].coordinates,len(nds)))# 计算单元中心坐标


    points = zip(coord[:,0],coord[:,1],coord[:,2])
    tree = spatial.KDTree(points)# 引入k-d树搜索策略加快速度。搜索得到每一个过滤范围内包含的网格


    neighbours=tree.query_ball_tree(tree,filterRadius,2.0)
    for i in range(len(neighbours)):
        elNeighbours = neighbours[i]
        filterMap[i]=[[],[]]

        for j in range(len(elNeighbours)):
            k = int(elNeighbours[j])
            dis = np.sqrt(np.sum(np.power(np.subtract(coord[i],coord[k]),2)))# 得到过滤范围内的网格与中心网格的距离，基于此计算权重
            filterMap[i][0].append(k)# 得到过滤范围内的网格编号
            filterMap[i][1].append(filterRadius - dis)# 得到网格的权重


        filterMap[i][1] = np.divide(filterMap[i][1],np.sum(filterMap[i][1]))

    return filterMap

# 敏度过滤
def filterSensitivities(sensitivities,filterMap):
    originalSensitivities = sensitivities.copy()
    for el in filterMap.keys():
        sensitivities[el] = 0.0
        for i in range(len(filterMap[el][0])):
            originalIndex = filterMap[el][0][i]
            sensitivities[el]+=originalSensitivities[originalIndex]*filterMap[el][1][i]# 加权平均获取过滤后的敏度
    return sensitivities


def densityToYoungs(density):

    E = Emin + math.pow(density,penalty)*(E0-Emin)
    return E;

def writeToFile(filename,content):
    f = open(filename,'a')
    f.write(str(content))
    f.close()

def writeArrayToCSV(filename, array):
    for i in range(len(array)):
        line = str(i) + ',' + str(array[i]) + '\n'
        writeToFile(filename,line)


def logMsg(msg):# 在log.txt可以看到目标函数值
    writeToFile('log.txt',msg + ' \r\n')
    print >> sys.__stdout__, msg


def zerosFloatArray(n):
    return np.zeros(n,'f')

# 用totalDensity函数计算体积（质量）
def totalDensity(densities):
    return np.sum(densities)


def turnintoONE(densities):
    matrix = np.zeros((50, 150))

    for i in range(50):
        if 0 <= i <= 24:
            matrix[24 - i, :] = densities[i * 150:(i + 1) * 150]
        else:
            matrix[i, :] = densities[i * 150:(i + 1) * 150][::-1]

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
    for i in range(50):
        if 0 <= i <= 24:
            final_densities[i * 150:(i + 1) * 150] = matrix[24 - i, :]
        else:
            final_densities[i * 150:(i + 1) * 150] = matrix[i, :][::-1]

    return final_densities

# 主函数
if __name__ == '__main__':

    start = datetime.datetime.now()

    global myNumCpus 
    global writebackLog
    global optimizationTreshold
    global densityPrecision# 材料数量（这里单元密度在0-1的范围内划分了1000份）
    global densityMove
    global nrElements# 模型网格数（变量数）
    global maxTotalDensity


    global Emin
    global E0
    global poisson


    global volumeFraction # 体积分数（质量约束）
    global filterRadius# 过滤半径

    global filterQ
    global maxFilterQ
    global penalty


    configuration = ConfigParser.ConfigParser()
    configuration.read('simp3d-config.ini')# 读取配置文件，获取计算需要的参数


    Emin = 0.001
    E0 = configuration.getfloat('Material Config', 'Youngs')
    poisson = configuration.getfloat('Material Config', 'Poisson')


    volumeFraction = configuration.getfloat('Optimization Config', 'VolumeFraction')
    filterRadius = configuration.getfloat('Optimization Config', 'FilterRadius')

    penalty = configuration.getfloat('Optimization Config', 'Penalty')
    maxFilterQ = configuration.getfloat('Optimization Config', 'GreyScaleFilter')
    historyAverageEnabled = configuration.getboolean('Optimization Config','HistoryAverage')


    mdbName = configuration.get('Model Config', 'ModelDatabaseName')
    modelName = configuration.get('Model Config', 'ModelName')
    partName = configuration.get('Model Config', 'PartName')
    instanceName = configuration.get('Model Config', 'InstanceName')
    designSpaceSetName = configuration.get('Model Config', 'DesignSpaceSetName')


    myNumCpus = configuration.getint('Parallelization Config','NumCPUs')


    logMsg('SIMP3D.py will run with the following setup: ')
    logMsg('Volume Fraction: ' + str(volumeFraction))
    logMsg('Filter Radius: ' + str(filterRadius))
    logMsg('Penalty: '+ str(penalty))
    logMsg('Grey-scale Filter: '+ str(maxFilterQ))
    logMsg('History Average: ' + str(historyAverageEnabled))
    logMsg('Youngs Modulus: '+ str(E0))
    logMsg('Poisson: '+ str(poisson))

    convergenceCriterium = 0.01#（收敛判据）

    densityPrecision = 1000
    filterQ = 1.0


    modelDatabase = openMdb(mdbName)
    model = modelDatabase.models[modelName]
    part = model.parts[partName]
    instance = model.rootAssembly.instances[instanceName]
    designSpaceSet = part.sets[designSpaceSetName]
    elements, nodes = designSpaceSet.elements, instance.nodes

    nrElements = len(elements)# 设计域网格数量读取
    nrNodes = len(nodes)# 设计域节点数量读取
    maxTotalDensity = nrElements*volumeFraction# 质量约束

    elLabelToIndex, indexToElLabel = {}, {}
    for i in range(nrElements):
        label = elements[i].label
        index = i
        indexToElLabel[index] = label
        elLabelToIndex[label] = index


    initModelDatabase(modelDatabase,volumeFraction, designSpaceSet)
    densities, prevDensities=zerosFloatArray(nrElements), zerosFloatArray(nrElements)
    densities.fill(volumeFraction)


    if filterRadius > 0:
        filterMap = filterMapFromModelDatabase(modelDatabase, elements, filterRadius)


    change, i, objectiveFunction = 1, 0, 0
    sensitivities, prevSensitivities, objectiveFunctionHistory, volumeFractionHistory = zerosFloatArray(nrElements), zerosFloatArray(nrElements), [], []
    while change > convergenceCriterium:
        t=datetime.datetime.now()
        s=str("{:02d}:{:02d}".format(t.hour,t.minute))# 计时器
        logMsg('Starting FEA analysis ' + str(i) + ' at ' + s)
        ODBFileName = FEA(i,modelDatabase)
        objectiveFunction = objectiveFunctionFromODB(ODBFileName)
        objectiveFunctionHistory.append(objectiveFunction)
        logMsg('Objective function: ' + str(objectiveFunction))

        prevSensitivities = sensitivities.copy()
        sensitivities = sensitivitiesFromODB(ODBFileName, densities)
        t=datetime.datetime.now()
        s=str("{:02d}:{:02d}".format(t.hour,t.minute))
        logMsg('Sensitivities done'+ ' at ' + s)

        if filterRadius > 0.0:
            sensitivities = filterSensitivities(sensitivities,filterMap)
            t=datetime.datetime.now()
            s=str("{:02d}:{:02d}".format(t.hour,t.minute))
            logMsg('Filtering done' + ' at ' + s)

        
        if i > 0:
            if historyAverageEnabled:
                sensitivities = historyAverage(prevSensitivities,sensitivities)

        prevDensities = densities.copy()
        densities = densityUpdateOC(densities, sensitivities)
        t=datetime.datetime.now()
        s=str("{:02d}:{:02d}".format(t.hour,t.minute))
        logMsg('Density update done' + ' at ' + s)


        updateModelDatabase(modelDatabase, densities, elements)
        t=datetime.datetime.now()
        s=str("{:02d}:{:02d}".format(t.hour,t.minute))
        logMsg('UpdateModelDatabase done'+' at '+s)

        if i > 10:
            change=math.fabs((sum(objectiveFunctionHistory[i-4:i+1])-sum(objectiveFunctionHistory[i-9:i-4]))/sum(objectiveFunctionHistory[i-9:i-4]))
            logMsg('Change since last iteration: ' + str(change))

        if i > 30:
            break
            
        i += 1

    densities = np.around(densities)
    densities = turnintoONE(densities)
    vol = np.mean(densities == 1)
    updateModelDatabase(modelDatabase, densities, elements)
    ODBFileName = FEA(i, modelDatabase)
    objectiveFunction = objectiveFunctionFromODB(ODBFileName)
    objectiveFunctionHistory.append(objectiveFunction)


    # 下面四行是结果储存
    logMsg('Writing results')


    logMsg('vol:' + str(vol))


    writeArrayToCSV('Objective Function History.csv', objectiveFunctionHistory)


    modelDatabase.saveAs('Final_Design.cae')


    upgradeOdb(existingOdbPath=ODBFileName, upgradedOdbPath='Final_ODB.odb')# 最终优化结果储存

    end = datetime.datetime.now()
    spend = end - start
    logMsg('Total time consumption: ' + str(spend))
    
    # 优化后的目标函数值可以在log.txt中看到，优化后的单元密度分布通过displayResults.py读取