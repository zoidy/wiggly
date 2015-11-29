# Customize this starter script by adding code
# to the run_script function. See the Help for
# complete information on how to create a script
# and use Script Runner.

""" 
Add this script to the ScriptRunner plugin in QGIS.

Runs Wiggly with the Processing script named wiggly_processing_example.py 
as the analysis model.
"""

# Some commonly used imports
from qgis.core import *
from qgis.gui import *
from PyQt4.QtCore import *
from PyQt4.QtGui import *

import sys,os
import matplotlib.pyplot as plt
import random
import numpy as np

import wiggly.gis.wiggly_qgis as wqgs
import wiggly as w

def run_script(iface):
    """This function is called by ScriptRunner"""
    
    #initialize the Wiggly-QGIS adapter with the function containing the control script.
    #The control script is the one that sets up the UQ/SA run.
    wqgs.wigglyInit(control_script)
    wqgs.wigglyRun()
    
def control_script():
    w.utilities.MESSAGE_LEVEL = 3
    
    #get the module containing our model.
    themodule=wqgs.getProcessingUserScriptAsModule('buffer_area_dist')
    
    #output the analysis runs to this directory
    wdir=r'c:\temp'
    if not os.path.exists(wdir):
        os.makedirs(wdir)
    
    plt.close('all')
    
    seed=12345
    random.seed(seed)
    np.random.seed(seed)
    
    n_prob=100
    n_fuzzy=None
    acuts=[]
 
    probVars={}
    probVars['buff_dist']={'dist':'normal', 'kwargs':{'mean':0, 'dev':6}, 'desc':'normal, mu=0 sd=6'}
    fuzzyVars={}
    consts={}
    
    puqOm=None
 
    uq=w.uq.UQ(testProgFunc=themodule.run,workingDir=wdir,
               seed=seed,objMgr=puqOm,
               probVars=probVars,n_prob=n_prob,
               fuzzyVars=fuzzyVars,n_fuzzy=n_fuzzy,fuzzyVarACuts=acuts,
               consts=consts)    
    
    uq.run(dryrun=False,keep_files=False,parallel_jobs=4) 
    
    plt.show()

#This line is REQUIRED to be at the very end of the file where
#the control script function is located
exec(wqgs.wigglyCtrlInit(),globals())