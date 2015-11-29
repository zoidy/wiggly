##Buffer_distance=number 1
##total_area=output number

#Calculate the area of point buffers.
#This script is runnable from Processing inside QGIS and is also capable of being analyzed
#by the Wiggly framework.

import optparse,sys,os
import random
import numpy as np
import matplotlib.pyplot as plt
import wiggly as w
from puqutil import dump_hdf5
import wiggly.gis.wiggly_qgis as wqgs


from qgis.core import *
from qgis.gui import *


#***************************************************************************
#   This is our model
#***************************************************************************
#in order to use the test model with InteractiveHostMP, there needs to be a
#run function with the arguments named args and jobinfo as shown. The return value must be jobinfo
def run(args=None,jobinfo=None):
    #Required. Check to make sure everything is ok
    wqgs.wigglyModelStart()
     
    parser=optparse.OptionParser()
    parser.add_option("--paramsFile",type="string",dest="paramsFile")
    parser.add_option("--baseShapes",type="string",dest="baseshapes_json")

    paramValues={}
    
    if args==None:
        #we're running from Processing
        global Buffer_distance
        paramValues['buff_dist']={'value':Buffer_distance}
    else:
        #we're running from wiggly
        (options,args)=parser.parse_args(args)
        if options.paramsFile==None:
            raise Exception("paramsFile was not defined!")
        print(options.paramsFile)
        paramValues=w.utilities.puqParams_fromFile(options.paramsFile)
    
    if not 'buff_dist' in paramValues:
        raise Exception('this model requires a parameter named buff_dist')
        
    A=areaAllFeatures(paramValues['buff_dist']['value'])
    
    dump_hdf5('A',A,'Total area of all buffered features')
    
    #sett output value when run directly via Processing
    global total_area
    total_area=A
        
    #Required. Last thing to do before returning.
    wqgs.wigglyModelEnd()
    
    return jobinfo

def areaAllFeatures(buff):
    #calculates the area of all features in a memory layer.
    #also outputs an html file containing information about the layer (%TEMP%\processing folder)
    #example adapted from http://hub.qgis.org/issues/8955
    
    # create a memory layer and add a point to it
    layer = QgsVectorLayer("Point?crs=EPSG:27700", "temporary_points", "memory")
    feature = QgsFeature()
    feature.setGeometry(QgsGeometry.fromPoint(QgsPoint(42, 42)))
    layer.dataProvider().addFeatures([feature])
    layer.updateExtents()
    
    # add the layer to the map layer registry
    # required for memory layers which can't be accessed by uri alone
    QgsMapLayerRegistry.instance().addMapLayer(layer)
    
    # run the 'fixed distance buffer' algorithm, distance=buff, segments=100
    ret = processing.runalg('qgis:fixeddistancebuffer',layer, buff, 100, False, None)
    progress.setInfo('buffer: '+str(ret))
    
    # load the result
    output = processing.getObjectFromUri(ret['OUTPUT'])
    print(output)
    # calculate the total area of the features
    total_area = 0
    for feature in output.getFeatures():
        if feature.geometry()!=None: #in case we buffered 0 or negative
            total_area += feature.geometry().area()
            
    output=processing.runalg('gdalogr:vectorinfo', ret['OUTPUT'], None)
    f=open(output['OUTPUT'],'r')
    data=f.read()
    f.close()
    progress.setInfo('info:\n '+data)
    return total_area
#***************************************************************************
    

if __name__=='__builtin__':
    #run a test
    run()
    progress.setInfo('Total area: '+str(total_area))
    
    
