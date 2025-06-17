# coding:utf-8
import customKernel
from caeModules import *
from odbAccess import openOdb

def showFinalDesign():

    o = session.odbs.values()[0]
    viewPortFinalDesign = session.viewports.values()[0]
    viewPortFinalDesign.enableMultipleColors()
    viewPortFinalDesign.setValues(displayedObject=o)
    cmap = viewPortFinalDesign.colorMappings['Material']
    colorOverrides = {}
    densityPrecision = 1
    setColorInfo = (True, '#00e6e6', 'Default', '#00e6e6')
    colorOverrides['ELASTIC_MATERIAL'] = setColorInfo
    for i in range(densityPrecision + 1):
        density = i / float(densityPrecision)
        setName = 'MATERIAL'+str(i)
        setColorInfo = (True, densityToHex(density), 'Default', densityToHex(density))
        colorOverrides[setName] = setColorInfo

    cmap.updateOverrides(overrides=colorOverrides)
    viewPortFinalDesign.setColor(colorMapping=cmap)


def densityToHex(density):
    b = 0
    if(round(density,1) == 0.5):
        r = 255
        g = 255
    elif(density < 0.5):
        r = 255
        g = int(density * 2 * 255.0)
    else:
        r = int((1.0 - density) * 2 * 255.0)
        g = 255

    return rgbToHex(r,g,b)
 
 

def rgbToHex(r,g,b):
    triplet = (r, g, b)
    return '#'+''.join(map(chr, triplet)).encode('hex')
 

if __name__ == '__main__':
    global densityPrecision
    global mddb

    outputDatabase = openOdb(getInput('Input ODB file:',default='OutputDatabase_23.odb'))# 这里的odb文件选择最后一个循环得到的odb文件
    showFinalDesign()