"""
Adapter which allows using QGIS scripts (including Processing scripts) as analysis models in Wiggly.

Requirements:
    QGIS 2.4 or above
    A working Wiggly installation in the QGIS-bundled Python

QGIS setup:
    The current version of QGIS (2.4 as of this writing) ships with old versions of certain
    libraries, which must be updated prior to installing Wiggly into the QGIS Python.
    See the readme in the qgis example directory
    
Usage:
    See the example in the examples directory.

Notes:
    Don't name your Processing scripts 'qgis.py' or 'test.py' or anything too common as those
    kinds of names may interfere with QGIS and cause Wiggly not to work as expected.

    Your processing scripts must contain a function which has the operations you wish to analyze.
    In order for your processing script to be runnable by processing as well as this library,
    it is recommended to place all script code in a function called go() which takes in as parameters
    any required inputs. See the qgis example for more details
"""

import sys,os,multiprocessing


#class _ScriptRunnerProgress(processing.core.SilentProgress.SilentProgress):
class _ScriptRunnerProgress(object):
    """
    Output processing messages from progress.setInfo or progress.setPercentage
    """
    def setInfo(self,msg):
        print('setInfo: {}'.format(msg))

    def setPercentage(self,percent):
        print('setPercentage: {}'.format(percent))

def getProcessingUserScriptAsModule(scriptname):
    """
    Returns a reference to the module of the given Processing user script. A
    user script is one located in the Processing toolbox under Scripts -> User scripts.

    - *scriptname*: A string indicating the filename (no path or extension) of the script.
      E.g. 'myscript'
    """
    import importlib
    from qgis.core import *
    from qgis.gui import *
    from PyQt4.QtGui import *

    (fname,ext)=os.path.splitext(scriptname)
    if ext!='':
        raise Exception('Script name parameter must not have an extension: {}'.format(scriptname))

    #get the path of the user scripts dir and add it to the path
    userscript_path=os.path.join(QgsApplication.qgisSettingsDirPath(),r'processing\scripts')
    if not userscript_path in sys.path:
        sys.path.append(userscript_path)

    modexist=[k for k in sys.modules.keys() if scriptname in k]
    if len(modexist)>0:
        for m in modexist:
            del(sys.modules[m])

    return importlib.import_module(scriptname)

_initScriptFunc=None
def wigglyInit(initScriptFunc):
    """
    Initializing routine before model execution.

    - *initScriptFunc*: the function handle containing the control script which
      sets up the UQ/SA run.
    """

    import logging,inspect

    try:
        m=inspect.getmodule(initScriptFunc)
        mf=m.__file__
    except Exception,e:
        raise Exception('Could not get the file name of the initScriptFunc module. {}'.format(str(e)))

    ######################################################################    
    #http://gis.stackexchange.com/questions/35279/
    ######################################################################
    #tell multiprocessing to run new instances of python instead of new
    #instances of QGIS. Note pythonw is windows only. Would need to change
    #this in non-Windows os.
    path = os.path.abspath(os.path.join(sys.exec_prefix, '../../bin/pythonw.exe'))
    if not os.path.exists(path):
        raise Exception('Python executable not found: ' + path)
    multiprocessing.set_executable(path)

    #tell multiprocessing about the location of the module containing the
    #function we want to evaluate using Wiggly. Normally,
    #argv is set to the correct value  by the system but inside, QGIS,
    #this value is set to something else.
    sys.argv = [mf]
    ######################################################################
    
    #PUQ-wiggly uses the logging library which doesn't play nice with
    #QGIS ('IOError: [Errno 9] Bad file descriptor' errors)
    logging.disable(logging.CRITICAL)

    ######################################################################
    #the ordering of importing PyQT and qgis libraries matters for some reason
    #when using Processing. Check to make sure PyQT is imported after qgis.
    ######################################################################
    #get the filename of the calling script
    frame,filename,line_number,function_name,lines,index = inspect.stack()[1]
    f=open(filename,'r')
    fname=os.path.split(filename)[1]
    lines=f.readlines()
    idx_pyqt4=[]
    idx_qgis=[]
    idx_processing=[]
    for i,line in enumerate(lines):        
        if not line.startswith('#'):
            if 'PyQt4' in line and 'import' in line: idx_pyqt4.append(i)
            if 'qgis' in line and 'import' in line: idx_qgis.append(i)
            if 'processing' in line and 'import' in line and not (line.startswith(' ') or line.startswith('\t')): idx_processing.append(i)
    if len(idx_pyqt4)>0 and len(idx_qgis)>0:
        if min(idx_pyqt4)<min(idx_qgis):
            raise Exception('Error in {}:{}\n   You must import the PyQt4 libraries AFTER the qgis libraries'.format(fname,min(min(idx_pyqt4),min(idx_qgis))+1))
    if len(idx_pyqt4)>0 and len(idx_qgis)==0:
        raise Exception('Error in {}:{}\n  If you import the PyQt4 libraries, you MUST import the qgis libraries BEFORE'.format(fname,min(idx_pyqt4)+1))
    if len(idx_processing)>0:
        raise Exception ('Error in {}:{}\n  Do not import processing at the module level (it breaks things)'.format(fname,min(idx_processing)))
    ######################################################################
    
    global _initScriptFunc
    _initScriptFunc=initScriptFunc

def wigglyRun(*args,**kwargs):
    #args, and kwargs to pass to the function given in wigglyInit
    if _initScriptFunc==None:
        raise Exception('Wiggly wrapper was not initialized sucessfully')
    return _initScriptFunc(*args,**kwargs)

def wigglyCtrlInit():
    """
    Initializes QGIS libraries for the subprocesses created by Wiggly.

    Since each subprocess is it's own process, this function needs to be called from the file
    containing the control script using the command
    
        exec(wqgs.wigglyCtrlInit(),globals())
        
    placed at the end of the control sccript file.

    See the qgis example.
    """
    import inspect
    (srclines,lineno)=inspect.getsourcelines(_loadprocessing_code)
    actuallines=[l[4:] for l in srclines[1:]]
    
    #return a string containing the function contents (blank lines are removed)   
    return ''.join(actuallines)
   

_loadprocessing_errs={}
def wigglyModelStart():
    """
    Call this function on the first line of the analysis model's 
    
        run(args=None,jobinfo=None)
        
    function. If any exceptions occurred during the
    
        exec(wqgs.wigglyCtrlInit(),globals())
        
    in the control script's file, they will be thrown here.
    """
    for k,v in _loadprocessing_errs.iteritems():
        raise Exception('{}:{}'.format(k,v))

_app=None
def wigglyModelEnd():
    """
    Call this function on immmediately before the return statement of the analysis model's 
    
        run(args=None,jobinfo=None)
        
    function.
    """
    try:
        #Ideally, would like to exit QGIS gracefully but python crashes when I try. Disable this for now
        #_app.exitQgis()
        pass
    except:
        pass

def _loadprocessing_code():
    #Do Not Edit This Function!!!
    try:
        thename
    except:
        thename=None
    if thename==None:
        thename='__parents_main__'
    import traceback,multiprocessing,os,sys
    import wiggly.gis.wiggly_qgis as wqgs
    try:
        #if any call below raises an exception, progress and processing won't get defined
        #Errors in this function will manifest themselves as PUQ errors saying just that.
        #Also, since _processing_done doesn't currently work, you'll get a popup if there
        #was an error here.
        if __name__==thename:
            from qgis.gui import *
            from qgis.core import *            
            from PyQt4.QtGui import *   #must be imported after qgis
            
            #will throw exception if variable isn't found
            qgisprefix = os.environ['QGIS_PREFIX_PATH']
            if qgisprefix=='': raise Exception('QGIS_PREFIX_PATH is blank')

            wqgs._app = QgsApplication([], True)

            # configure QGIS paths
            QgsApplication.setPrefixPath(qgisprefix, True)

            # initalise QGIS
            QgsApplication.initQgis()

            providers = QgsProviderRegistry.instance().providerList()
            if len(providers)==0:
                s='No providers found! Did you set your paths correctly?\n\n'
                s+=QgsApplication.showSettings()
                raise Exception(s)

            # Prepare processing framework
            pluginspath=os.path.join(qgisprefix,'python/plugins')
            if not os.path.exists(pluginspath): raise Exception('Plugins path {} not found'.format(pluginspath))
            if not pluginspath in sys.path:
                sys.path.append(pluginspath)
            from processing.core.Processing import Processing
            Processing.initialize()
            import processing
            reload(sys) #extremely important. Without this, all print statements and exceptions will be eaten!
            
            #Processing framework scripts expect the progress and processing variables to be
            #globally available.
            try:
                __builtins__['progress']=wqgs._ScriptRunnerProgress()
                __builtins__['processing']=processing
            except:
                pass
                #eat this exception
    except Exception,e:
        try:
            print(traceback.format_exc())
            import tkMessageBox
            tkMessageBox.showinfo(title="Greetings", message=traceback.format_exc())
        except:
            print('fatal')
        wqgs._loadprocessing_errs[multiprocessing.current_process().name]='_loadprocessing_code: '+traceback.format_exc()