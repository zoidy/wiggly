#################################################################
# Example "model" which shows how to use Wiggly 
#
# The model consists of a mixture of shapes and non-shape
# parameters, both probabilistic and fuzzy.
#################################################################
import optparse
import sys
sys.path.append("P:\\dissertation\\4-research\\1.my work\\software")
import wiggly as w
from puqutil import dump_hdf5

#in order to use the test model with InteractiveHostMP, there needs to be a
#run function with the arguments named args and jobinfo as shown. The return value must be jobinfo
def run(args=None,jobinfo=None):
    #this function is the "model". It is called by the Wiggly UQ or SA engine

    #get the required model inputs from the args parameter
    #   paramsFile: file name to read model parameters from
    #   baseShapes: file containing the original shapes
    #Both of these files are generated by Wiggly and need to be parsed here.
    parser=optparse.OptionParser()
    parser.add_option("--paramsFile",type="string",dest="paramsFile")
    parser.add_option("--baseShapes",type="string",dest="baseshapes_json")

    if args==None:
        (options,args)=parser.parse_args()
    else:
        (options,args)=parser.parse_args(args)

    if options.paramsFile==None:
            raise Exception("paramsFile was not defined!")
    if options.baseshapes_json==None:
            raise Exception("baseShapes was not defined!")

    #extract all the parameters from the parameter file and convert them from the PUQ
    #format to a dictionary. Note that the dictionary contains ALL parameters, shape
    #and non-shape alike.
    #
    #paramvalues is a dictionary. Key is the parameter name. THe value is a dictionary
    #with keys 'value' and 'desc' whose values correspond to the parameter's value and
    #description.
    paramValues=w.utilities.puqParams_fromFile(options.paramsFile)
    for pname,data in paramValues.iteritems():
        print('{}\t{}\t{}'.format(pname,data['value'],data['desc']))

    #convert shape-related parameters back into shapely shapes. Shapes is a dictionary
    #with keys equal to the shape's name. See puq2hapes documentation for more details.
    shapes=w.objectmanager.ObjectManager.puq2Shapes(options.baseshapes_json,
                                                    paramsFileName=options.paramsFile)

    #do some validation to make sure we have the inputs we need
    nonShapeParams=[]
    for paramName in paramValues.keys():
        if not w.objectmanager.ObjectManager.isShapeParam(paramName):
            nonShapeParams.append(paramName)
    if not 'c' in paramValues:
        raise Exception('the constant "c" for the test_model was not found')
    if len(shapes)!=4:
        raise Exception('There needs to be 4 shapes for the test_model')
    if len(nonShapeParams)!=5:
        raise Exception('There needs to be 5 non-shape parameters for the test_model')

    #################
    totalArea=0
    totalPerim=0
    totalAreaF=0
    totalPerimF=0
    for shpName,shpData in shapes.iteritems():
        shp=shpData['shp']
        shpType=shpData['type']

        if shpType=='R' or shpType=='D':
            totalArea+=shp.area
            totalPerim+=shp.length
        elif shpType=='E' or shpType=='V':
            totalAreaF+=shp.area
            totalPerimF+=shp.length
        else:
            raise Exception('unrecognized shape type {}'.format(shpType))

    #sum fuzzy and probabilistic areas for simplicity
    totalArea+=totalAreaF
    totalPerim+=totalPerimF

    A=(totalArea + paramValues['x']['value']*paramValues['f']['value'] +
        paramValues['y']['value']*paramValues['g']['value']**2 + paramValues['c']['value'])
    P=(totalPerim*paramValues['x']['value']**2 + paramValues['c']['value']*paramValues['y']['value'] +
        paramValues['f']['value'] + paramValues['g']['value'])

    #output the results. using dump_hdf5 is required so the engine is able
    #to parse the results.
    dump_hdf5('A',A,'A_t + x*f + y*g^2 + c (A_t is total area)')
    dump_hdf5('P',P,'P_t*x^2 + c*y + f + g (P_t is total length)')

    #DO NOT FORGET TO RETURN jobinfo!!!!
    #If it is not returned, the engine will wait forever.
    return jobinfo
#################

#this allows running the test model using the normal InteractiveHost, if needed
if __name__ == '__main__':
    run()