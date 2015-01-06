import random,sys,os,traceback,time,inspect,glob,re,multiprocessing
import ctypes
import numpy as np
import matplotlib.pyplot as plt
import scipy
import utilities,objectmanager
import fuzz
import puq

""""
Uncertainty quantification module using Monte Carlo.
"""

class UQ(object):
    """
    Manages a UQ run in a simplified way.

    - *testProgScriptFile*: The model (e.g., .py script) to evaluate. This script must be able to
      take as input a command line argument

          --paramsFile xxx --baseShapes yyy

      where xxx is a file name consising of name-value pairs (one per line). All of the
      parameters (object and non-object) given in *objMgr, probVars, fuzzyVars, consts* are
      passed in xxx. The actual values of xxx and yyy are constructed by UQ automatically.

      The yyy value is the file name contains the base shapes. Use :func:`objectmanager.ObjectManager.isShapeParam`,
      :func:`objectmanager.ObjectManager.puq2Shapes` and :func:`utilities.puqParams_fromFile` inside
      of the *testProgScriptFile* to read the baseshapes.

      If testProgScriptFile is relative path, the current python
      *workingDir* is used as a base.
    - *testProgFunc*: A python function. Instead of passing a script file in *testProgScriptFile*,
      a Python function may be passed instead, giving significantly better performance for fast
      running models. Either this parameter or *testProgScriptFile* must be specified, but not both.

      The function must exist at the module level (not inside a class). The function must
      take two named arguments, args and jobinfo, and return the jobinfo argument without modifying
      it.  E.g., if the function is named f,::

          def f(args=None, jobinfo=None):
              <evaluate your model here>
              return jobinfo

      The args parameter is a list of strings that can be parsed by the optparse module. It contains
      at least two arguments: 'paramsFile' and 'baseshapes' which are described above in the
      *testProgScriptFile* parameter. Additional arguments, along with their values may be
      specifed in the *args* parameter of this method.
    - *args*: extra arguments to pass to *testProgScriptFile* or *testProgFunc*.
    - *workingDir*: the working directory where all scripts are located and where input and output
      files will be read/written.
      By default, its the current python working dir.
    - *objMgr*: An ObjectManager instance.
    - *baseShapesName*: The name (without path) of the base shapes file.
      See :func:`Objectmanager.crispObjects2PuqParams`.
    - *probVars*: additional scalar probabilistic variables. This is a dictionary with the structure

            {$varName$:{'dist':string, 'desc':string, 'kwargs':{$pname$:float, ... } }, ...  }

      The keys are the variable name. The value is a dictionaries which specifies the rest of the parameter.

          - Key 'dist': a string corresponding to some of the Parameter distribution types supported by puq.
            Valid values are: 'uniform', 'normal', 'triangular'.
          - Key 'desc': a string description of the variable.
          - Key 'kwargs' is a dictionary specifying the parameters of the distribution selected in 'dist'.
            See the puq reference for the Parameters class for the names of the kwargs for each distribution.

      E.g., to
      specify a NormalParameter, set dist to 'normal', and for the kwargs dictionary, set it to
      {'mean':the mean, 'dev':the standard dev}.

    - *fuzzyVars*: additional scalar fuzzy variables.

        {$varName$:{desc:string, kl:float, ku:float, sl:float, su:float}, ...}

      kl and ku are the lower and upper values of the kernel of the fuzzy number (can be the same value).
      sl and su are the lower and upper values of the support.
    - *fuzzyVarsACuts*: A list of float indicating the alpha cuts at which the fuzzy variables are
      to be evaluated. If *objMgr* was specified, the alphacuts are taken from there and this parameter
      is ignored.
      If there are no fuzzy objects in *objMgr* then this paramter must be specified.
    - *consts*: any additional constants to consider in the analysis.

        {$constName$:{desc:string, value:float}, ...}
    - *args*: additional command line arguments to pass to the test script. This is a list with
      the following format:

          ['--param1=value1','--param2=value2', ...]
    - *n_prob*: the number of realizations to use when calculating the effect of the
      probabilistic variables. If there are probabilistic objects in *objMgr*, this parameter
      is obtained from the number of realizations of those objects instead.
    - *n_fuzzy*: the number of realizations to use when calculating the effect of the fuzzy
      variables. If there are fuzzy objects in *objMgr*, this parameter
      overrides the number of realizations of those objects. If there are too few realizations
      for the objects in *objMgr* existing realizations are replicated.
    - *seed*: initialize random seed to an integer.
    - *sweep_cb*: a callback which is called after every puq run. I.e., after an hdf5 file
      is generated, this function is called. It must take two arguments. The first argument
      is the puq sweep object which generated the hdf5. The second is a handle to the hdf5 file.
      The return value must be True or False. If True, the sweep is over. See the puq documentation
      for more information about the callback.

      Note: since the callback is called after every sweep, for fuzzy and mixed runs, it will
      be called multiple times. Ie. after each hdf5 file is generated.
    - *outfiles*: a list of files outut by the testProgram which will be copied to the hdf5 file.
    """
    def __init__(self,testProgScriptFile=None,testProgFunc=None,args=[],testProgDesc=None,workingDir=None,
                 objMgr=None,baseShapesName='shapes.json',probVars={},fuzzyVars={},
                 fuzzyVarACuts=[],consts={},n_prob=None,n_fuzzy=None,seed=None,
                 outfiles='',sweep_cb=None):

        if probVars==None:
            raise Exception('probVars cannot be None')
        if fuzzyVars==None:
            raise Exception('fuzzyVars cannot be None')
        if fuzzyVarACuts==None:
            raise Exception('fuzzyVarACuts cannot be None')
        if consts==None:
            raise Exception('consts cannot be None')

        if testProgScriptFile==None and testProgFunc==None:
            raise Exception('Must specify testProgScriptFile or testProgFunc')
        if testProgScriptFile!=None and testProgFunc!=None:
            raise Exception('Must specify testProgScriptFile or testProgFunc, not both')

        self._hdf5_basename=time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()).replace(':','-')
        if len(self._hdf5_basename)<1:
            raise Exception('Output hdf5 filename must be specified')
        if '@' in self._hdf5_basename or '#' in self._hdf5_basename:
            raise Exception('{} must not contain @ or #'.format(self._hdf5_basename))
        if self._hdf5_basename[0]=='.':
            raise Exception('{} must not start with a period'.format(self._hdf5_basename))

        if len(probVars)==0 and len(fuzzyVars)==0 and objMgr==None:
            raise Exception('No uncertain objects or parameters specified. Nothing to do!')

        self._extra_args=args

        if seed!=None:
            random.seed(seed)
            np.random.seed(seed)

        #see run method
        self._dryrun=False
        self._parallel_jobs=None
        self._sweep_cb=sweep_cb
        self._outfiles=outfiles

        #get the filename of the calling script
        frame,filename,line_number,function_name,lines,index = inspect.stack()[1]
        self._calling_script=os.path.realpath(filename)

        #check working dir
        if workingDir==None:
            workingDir=os.getcwd()
        if not os.path.isdir(workingDir):
            raise Exception('{} does not exist'.format(workingDir))
        self._oldwd=os.getcwd()
        self._workingDir=workingDir

        #build the path to the test program
        if testProgScriptFile!=None:
            if os.path.splitext(testProgScriptFile)[1]!='.py':
                raise Exception('testProgScriptFile must be a python script')
            self._testProgScriptFile=os.path.join(self._workingDir,testProgScriptFile)
        else:
            self._testProgScriptFile=None

        self._testProgFunc=testProgFunc
        self._testProgDesc=testProgDesc

        #disable shapes processing if there are none
        if objMgr==None or len(objMgr.objects)==0:
            self._baseShapesFileName=None
        else:
            self._baseShapesFileName=baseShapesName

        self._objMgr=objMgr

        self._puqparams_prob=[]
        self._n_probval=n_prob
        self._n_prob=n_prob
        self._probVars=probVars

        self._consts=consts

        self._fuzzyVarAcuts=fuzzyVarACuts
        self._n_fuzzyval=n_fuzzy
        self._fuzzyVars=fuzzyVars
        self._puqparams_fuzzy={} #see doc for ObjectManager.fuzzyObjects2PuqParams
        self._n_fuzzy={} #keys=acuts value=float
        self._acuts=[]

    @property
    def basename(self):
        """
        Returns the basename of the folder/files
        """
        return self._hdf5_basename

    def _setup_parameters(self):
        #change to output dir. Affects crispObjects2PuqParams, fuzzyObjects2PuqParams below.
        outputdir=os.path.join(self._workingDir,self._hdf5_basename)
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        os.chdir(outputdir)
        utilities.msg('Current output dir = {}'.format(os.getcwd()),'v')

        #*********************************
        #set up probabilistic variables
        self._setup_parameters_prob()
        #*********************************
        #set up constants
        self._puqparams_const=[]
        for cname,data in self._consts.iteritems():
            self._puqparams_const.append(puq.ConstantParameter(cname,data['desc'],
                                                               attrs=[('uncert_type','const')],
                                                               value=data['value']))

        #*********************************
        #set up fuzzy variables

        if self._objMgr!=None and len(self._objMgr.fuzzyObjects)>0:
            self._acuts=self._objMgr.fuzzyObjects[0].realizations.keys()
            if len(self._fuzzyVarAcuts)>0:
                utilities.msg('ignoring fuzzyVarACuts. Taking the ones from objMgr instead','w')
        else:
            #if there are no fuzzy objects in the object manager, then we need the
            #fuzzyVarAcuts parameter
            if len(self._fuzzyVarAcuts)==0 and len(self._fuzzyVars)>0:
                raise Exception('fuzzyVars, but no fuzzy objects found. Therefore fuzzyVarACuts must be specified!')
            if len(self._fuzzyVarAcuts)>0 and len(self._fuzzyVars)==0:
                raise Exception('FuzzyVarACuts was specified therefore fuzzyVars must also be given')
            if len(self._fuzzyVarAcuts)==0 and len(self._fuzzyVars)==0:
                #there are no fuzzy things of any kind
                self._acuts=[]
            else:
                if self._n_fuzzy==None:
                    raise Exception('n_fuzzy must be specified when fuzzyVars are given but there are no fuzzy objects')
                self._acuts=self._fuzzyVarAcuts

        self._acuts.sort()

        #get the puq parameters for all shapes
        if self._objMgr!=None and len(self._objMgr.fuzzyObjects)>0:
            self._puqparams_fuzzy=self._objMgr.fuzzyObjects2PuqParams(self._baseShapesFileName)
            if len(self._puqparams_fuzzy.keys())!=len(self._acuts):
                raise Exception('This error should not happen. Inconsistent alpha cut arrays length.')
            if not all([v in self._acuts for v in self._puqparams_fuzzy.keys()]):
                print(self._acuts,self._puqparams_fuzzy.keys())
                raise Exception('This error should not happen. Inconsistent alpha cut arrays content.')

            #go through each alpha cut and get the number of realizations. ObjectManager guarantees that
            #for non-constant parameters, all parameters will have the same number of realizations
            if self._n_fuzzyval!=None:
                utilities.msg('n_fuzzy specified. Ignoring n_fuzzy specified for the objects in objMgr','w')
            for acut in self._acuts:
                self._n_fuzzy[acut]=self._puqparams_fuzzy[acut]['num_realizations']

        #add the non-shape fuzzy parameters
        for acut in self._acuts:
            for fuzzyVar,data in self._fuzzyVars.iteritems():
                #build a new fuzzy number
                fn=fuzz.TrapezoidalFuzzyNumber(kernel=(data['kl'],data['ku']),
                                               support=(data['sl'],data['su']))

                cut=fn.alpha(acut)

                #build a puq parameter for this alpha cut
                param=puq.UniformParameter(fuzzyVar,data['desc'],
                                           attrs=[('uncert_type','fuzzy')],
                                           min=cut[0],max=cut[1])

                if not acut in self._puqparams_fuzzy:
                    self._puqparams_fuzzy[acut]={'params':[]}
                self._puqparams_fuzzy[acut]['params'].append(param)

            if self._n_fuzzyval!=None:
                #if n_fuzzy was specified in __init__, overwrite any values already in the
                #n_fuzzy dict.
                self._n_fuzzy[acut]=self._n_fuzzyval
            elif not acut in self._n_fuzzy:
                #TODO: this is not ideal since for each alpha level,
                #we will sample n_fuzzy times. This is in contrast to
                #how FObjects are sampled which is n_fuzzy at the
                #lowest alpha level.  How to fix this: use the same
                #algorithm as FObjects. This would need modification here
                #and wherever the parameters are actually sampled. May need to
                #pre-generate parameter samples instead of letting puq do it.
                self._n_fuzzy[acut]=self._n_fuzzyval


        #*********************************

    def _setup_parameters_prob(self):
        self._puqparams_prob=[]
        if self._objMgr!=None and len(self._objMgr.probabilisticObjects)>0:
            self._puqparams_prob.extend(self._objMgr.crispObjects2PuqParams(self._baseShapesFileName))

            #take the number of realizations from the first custom parameter, if present.
            for p in self._puqparams_prob:
                if type(p) is puq.CustomParameter:
                    self._n_prob=np.size(p.values)
                    if self._n_probval!=None:
                        utilities.msg('overwrote given n_prob value ({}) with one from objMgr ({})'.format(self._n_probval,self._n_prob))
                    break
        else:
            #there are no probabilistic shapes
            if len(self._probVars)>0 and self._n_prob==None:
                raise Exception('probVars specified. n_prob must also be specified')
            if self._n_prob!=None and len(self._probVars)==0:
                raise Exception('n_prob specified but probVars not specified')

        #setup scalar probabilistic vars
        for varname,data in self._probVars.iteritems():
            if data['dist']=='normal':
                p=puq.NormalParameter(varname,
                                      data['desc'],attrs=[('uncert_type','prob')],
                                      **data['kwargs'])
            elif data['dist']=='uniform':
                p=puq.UniformParameter(varname,
                                       data['desc'],attrs=[('uncert_type','prob')],
                                       **data['kwargs'])
            elif data['dist']=='triangular':
                p=puq.TriangParameter(varname,
                                      data['desc'],attrs=[('uncert_type','prob')],
                                      **data['kwargs'])
            else:
                raise Exception("'{}' distribution is not supported for variable {}".format(data['dist'],
                                varname))

            self._puqparams_prob.append(p)

    def run(self,plot=True,dryrun=False,keep_files=False,parallel_jobs=0):
        """
        Executes the UQ run.

        - *plot*: Automatically plots the results when the run is complete.
        - *dryrun*: Conducts a run without actually executing the TestProgram.
        - *keep_files*: Keeps the individual input/output files generated by the run.
        - *parallel_jobs*: The number of jobs which will be run in parallel. Default is
          to set this number equal to the number of cpus on the machine.

        Returns a variable suitable for the *hdf5* parameter of :func:`UQ.plot`.

        Note: depending on your model, setting this number to be higher
        than the number of cpus may yield speed improvements. As a starting
        point, try making this number double the number of cpus and do
        a couple test runs.
        """
        #This function plays the role of the 'control script' from the puq docs

        self._setup_parameters()

        self._dryrun=dryrun
        self._parallel_jobs=parallel_jobs
        self._keep=keep_files

        if self._parallel_jobs==0:
            self._parallel_jobs=multiprocessing.cpu_count()

        try:
            if len(self._puqparams_prob)>0 and len(self._acuts)==0:
                #we only have a probabilistic run
                outfiles=self._run_probabilistic()
            elif len(self._puqparams_prob)==0 and len(self._acuts)>0:
                #we have a fuzzy run only
                outfiles=self._run_fuzzy()
            else:
                #its a combined run
                outfiles=self._run_probfuzzy()

            if plot:
                UQ.plot(outfiles)
                plt.show()

            return outfiles
        except Exception,e:
            traceback.print_exc()
            utilities.msg('error: {}'.format(str(e)),level='e')
        finally:
            os.chdir(self._oldwd)
            print('')

    def _run_probabilistic(self):
        """
        Runs a probabilistic sweep. Returns the name of the hdf5 file that was generated.
        """
        sweep=self._sweep_setup(self._puqparams_prob+self._puqparams_const,
                                self._n_prob,'single probabilistic sweep')

        utilities.msg('Starting a probabilistic run')

        ctypes.windll.kernel32.SetConsoleTitleA('{} jobs in puq run 1 of 1'.format( self._n_prob))

        utilities.msg('Puq run 1 of 1...')
        print('')
        if not sweep.run(self._hdf5_basename +'.hdf5',dryrun=self._dryrun):
            raise Exception('error running sweep')
        print('')
        utilities.msg('Puq run 1 of 1...Done')

        return os.path.join(os.getcwd(),self._hdf5_basename +'.hdf5')

    def _run_fuzzy(self):
        """
        Runs a fuzzy sweep.

        Returns the names of the hdf5 files that were generated in a
        dictionary:

            {$acut$:string, ...}

        The file name has the convention $basename$ @x where x is the alpha cut, expressed as a
        decimal number between and including 0 and 1.
        """
        utilities.msg('Starting a fuzzy run')
        ret={}
        pool=None

        try:
            for i,acut in enumerate(sorted(self._acuts,reverse=True)):
                pool=self._get_processPool(pool)
                sweep=self._sweep_setup(self._puqparams_fuzzy[acut]['params']+self._puqparams_const,
                                        self._n_fuzzy[acut],
                                        'fuzzy run (alpha-cut {} from set {})'.format(acut,self._acuts),
                                        pool)

                ctypes.windll.kernel32.SetConsoleTitleA( \
                    '{} jobs in puq run {} of {} (a-cut {})'.format(self._n_fuzzy[acut],
                                                                    i+1,len(self._acuts),
                                                                    acut))

                utilities.msg('Puq run {} of {} (alpha-cut {})...'.format(i+1,len(self._acuts),acut))
                print('')
                hdf5=self._hdf5_basename +' @{:1.1f}.hdf5'.format(acut)
                if not sweep.run(hdf5,dryrun=self._dryrun):
                    raise Exception('error running sweep')
                print('')

                utilities.msg('Puq run {} of {} (alpha-cut {})...Done'.format(i+1,len(self._acuts),acut))

                ret[acut]=os.path.join(os.getcwd(),hdf5)

        finally:
            #closes the process pool even if there is an exception
            if not pool==None:
                pool.close()
                pool.join()

        return ret

    def _run_probfuzzy(self):
        """
        Runs a mixed sweep.

        Returns the names of the hdf5 files that were generated in a
        dictionary:

            {$acut$:[string], ...}

        Each element in the list [string] is a file name corresponding to the probabilistic
        runs for alpha-cut $acut$. The file name has the convention $basename$ @x#y where
        x is the alpha cut and y is an integer indicating the run number.
        """
        #this flag controls whether the same realizations of the probabilistic variables
        #are used for each iteration of the fuzzy calculation. This should normally be set to
        #False so a different set of realizations are used for each fuzzy loop. Only set it
        #to True for testing purposes.
        #
        #For UA, it's probably better to use a different set each time so as to better
        #explore the probabilistic space
        repeatProbabilisticRealizations=False

        utilities.msg('Starting a probabilistic-fuzzy run')

        #make sure that all parameters at all acut levels have enough samples
        #to be able to run for the requested amount of iterartions. (only if the parameter
        #has had the samples generated already).
        #ConstantParameters are excepted since they always have only 1 sample.
        #This code may be removed once the code is more mature and no
        #problems have shown up with index out of bound errors
#        for acut in self._acuts:
#            for p in self._puqparams_fuzzy[acut]['params']:
#                if not type(p) is puq.parameter.ConstantParameter and \
#                    hasattr(p,'values') and np.size(p.values)<self._n_fuzzy[acut]:
#                    raise Exception('_run_probfuzzy: {} has {} samples, expected {}'.format(p.name,
#                                    np.size(p.values),self._n_fuzzy[acut]))

        totalpuqruns=0
        cumpuqruns=0
        for acut in self._acuts:
            totalpuqruns+=self._n_fuzzy[acut]

        if repeatProbabilisticRealizations:
            py_rndstate_fuzzyloop=random.getstate()
            np_rndstate_fuzzyloop=np.random.get_state()
            py_rndstate_probloop=py_rndstate_fuzzyloop
            np_rndstate_probloop=np_rndstate_fuzzyloop

        ret={}
        pool=None

        try:
            for acut in sorted(self._acuts,reverse=True):
                ret[acut]=[]

                for j in range(self._n_fuzzy[acut]):
                    puq_fuzzy_consts=[]
                    pool=self._get_processPool(pool)

                    #grab a sample of each fuzzy object and variable and make a constant out of them
                    for param in self._puqparams_fuzzy[acut]['params']:
                        #if the parameter already has values, use those. If not get a new random one.
                        if hasattr(param,'values'):
                            if type(param) is puq.parameter.ConstantParameter:
                                #note: can't just append existing ConstantParameter unfortunately.
                                #h5py gives an error if you try.
                                puq_fuzzy_consts.append(
                                    puq.ConstantParameter(param.name,
                                                          '[C@] '+param.description,
                                                          attrs=[('uncert_type','fuzzy-const')],
                                                          value=param.values[0]))
                            else:
                                puq_fuzzy_consts.append(
                                    puq.ConstantParameter(param.name,
                                                          '[C@] '+param.description,
                                                          attrs=[('uncert_type','fuzzy-const')],
                                                          value=param.values[j%len(param.values)]))
                        else:
                            puq_fuzzy_consts.append(
                                puq.ConstantParameter(param.name,
                                                      '[C@] '+param.description,
                                                      attrs=[('uncert_type','fuzzy-const')],
                                                      value=param.pdf.random(1)[0]))
                    #end for param in self._puqparams_fuzzy[acut]['params']:

                    if repeatProbabilisticRealizations:
                        #save the current state of the random number generator
                        py_rndstate_fuzzyloop=random.getstate()
                        np_rndstate_fuzzyloop=np.random.get_state()

                        #inner probabilistic sweep. each sweep uses the random state as it was before
                        #entering the outermost loop. This means that on every time the probabilistic run
                        #is executed, the same sequence of random numbers is generated.
                        random.setstate(py_rndstate_probloop)
                        np.random.set_state(np_rndstate_probloop)

                    sweep=self._sweep_setup(puq_fuzzy_consts+self._puqparams_prob+self._puqparams_const,
                                    self._n_prob,'probabilistic-fuzzy sweep, a-cut:{} run:{}'.format(acut,j+1),
                                    proc_pool=pool)

                    cumpuqruns+=1

                    ctypes.windll.kernel32.SetConsoleTitleA( \
                    '{} jobs in puq run {} of {} (run {} of {} in a-cut {})'.format(self._n_prob,
                                                                    cumpuqruns,totalpuqruns,
                                                                    j+1,self._n_fuzzy[acut],acut))

                    utilities.msg('Puq run {} of {} (alpha-cut {}, run {})...'.format(cumpuqruns,
                                      totalpuqruns,acut,j+1))
                    print('')
                    hdf5=self._hdf5_basename +' @{:1.1f}#{}.hdf5'.format(acut,j)
                    if not sweep.run(hdf5,dryrun=self._dryrun):
                        raise Exception('error running sweep')
                    ret[acut].append(os.path.join(os.getcwd(),hdf5))
                    print('')
                    utilities.msg('Puq run {} of {} (alpha-cut {}, run {})...Done'.format(cumpuqruns,
                                      totalpuqruns,acut,j+1))

                    if repeatProbabilisticRealizations:
                        #restore the true random state, so that the next sample in the
                        #fuzzy loop may be taken properly.
                        random.setstate(py_rndstate_fuzzyloop)
                        np.random.set_state(np_rndstate_fuzzyloop)
                    else:
                        if self._objMgr!=None:
                            for obj in self._objMgr.probabilisticObjects:
                                obj.generateLastUsed(self._n_prob)
                        self._setup_parameters_prob()


                #end for j in range(self._n_fuzzy[acut]):
            #end for acut in sorted(self._acuts,reverse=True)

        finally:
            #closes the process pool even if there is an exception
            if not pool==None:
                pool.close()
                pool.join()

        return ret

    def _sweep_setup(self,puqparams,n,desc='',proc_pool=None):
        """
        Sets up a single sweep for the given parameters.

        - *puqparams* list of puq parameter objects.
        - *n*: The number of runs.
        - *proc_pool*: an instance of multiprocessing.Pool. If not specified, a new Pool will
          be created.

        Returns a Sweep object.
        """
        uq=puq.MonteCarlo(params=puqparams,num=n,response=False,iteration_cb=self._sweep_cb)
        return self._sweep_setup_helper(uq,desc,proc_pool)

    def _sweep_setup_helper(self,sweep,desc='',proc_pool=None):
        utilities.msg('_sweep_setup() Called by module {}'.format(self._calling_script),'d')

        #if we are at a message level of verbose or debug,
        #enable extra output in puq.
        if utilities.MESSAGE_LEVEL>3:
            puq.options['verbose']=2

        #set up the argument to pass to the test program. Use the file method of passing
        #arguments. This means that --paramsFile an --baseShapes are always required.
        #If a particular test program doesn't require shapes, it can ignore the baseShapes
        #argument when it parses args
        args='--paramsFile=puq_input_params.txt --baseShapes=../'+str(self._baseShapesFileName)
        for arg in self._extra_args:
            args+=' ' + arg

        if self._testProgScriptFile!=None:
            prog=puq.TestProgram(exe='"' + sys.executable+ '" "' + self._testProgScriptFile + '" '
                                      + args,
                                 newdir=True,paramsByFile=True,desc=self._testProgDesc,
                                 outfiles=self._outfiles)
            sweep=puq.Sweep(sweep,puq.InteractiveHost(cpus_per_node=self._parallel_jobs),prog,
                            description='Wiggly UQ ' + desc)
        elif self._testProgFunc!=None:
            prog=puq.TestProgram(func=self._testProgFunc,
                                 func_args=args,
                                 newdir=True,paramsByFile=True,desc=self._testProgDesc,
                                 outfiles=self._outfiles)
            sweep=puq.Sweep(sweep,puq.InteractiveHostMP(cpus_per_node=self._parallel_jobs,
                                                     proc_pool=proc_pool),
                            prog,
                            description='Wiggly UQ ' + desc)

        sweep.input_script=self._calling_script

        #keep or don't keep files for individual runs
        puq.options['keep']=self._keep

        return sweep

    def _get_processPool(self,old_pool=None):
        """
        Returns a multiprocessing.Pool object, if self._testProgFunc is set.

        - *old_pool*: a Pool object. This must be given if this function is called
          in a loop.

        If this function has been called 60 times or less, old_pool is returned. Else a new
        Pool is created and returned.

        If *old_pool* is None, a new Pool is always created, if self._testProgFunc is set.
        If self._testProgFunc is not set, None is always returned.
        """
        if not hasattr(self,'_get_processPool_counter'):
            self._get_processPool_counter=0

        self._get_processPool_counter+=1

        utilities.msg('_get_processPool: call count: {}'.format(self._get_processPool_counter),'d')

        #if the test model is a function instead of a script, it means _sweep_setup
        #will create an InteractiveHostMP object. We can pre-initialize the pool
        #for a decent speedup which avoids the overhead of repeatedly creating a
        #new set of worker processes for fuzzy and mixed runs.
        #
        #If required, we can re-create the process pool after a certain number of
        #calls to this function.
        p=old_pool
        if old_pool==None and self._testProgFunc!=None :
            p=multiprocessing.Pool(processes=self._parallel_jobs)
            utilities.msg('_get_processPool: create new pool {}'.format(repr(p)),'d')
        elif self._testProgFunc==None:
            p=None
        elif self._get_processPool_counter>60 and self._testProgFunc!=None:
            old_pool.close()
            old_pool.join()
            p=multiprocessing.Pool(processes=self._parallel_jobs)
            utilities.msg('_get_processPool: create new pool {}'.format(repr(p)),'v')
            self._get_processPool_counter=0
        else:
            p=old_pool

        utilities.msg('_get_processPool: returning {}'.format(repr(p)),'d')
        return p

    @staticmethod
    def get_results(hdf5,baseshapes_file='shapes.json'):
        """
        Returns the results of the run in varying formats, without plotting.

        Parameters are the same as :func:`UQ.plot` (plotting-related parameters are excepted of course).
        """

        utilities.msg('Retrieving results...')

        ret=None

        if type(hdf5) is str:
            ret=UQ._plot_probabilistic(hdf5,baseshapes_file,plot=False)
        elif type(hdf5) is dict:
            if len(hdf5)>0:
                if type(hdf5[hdf5.keys()[0]]) is str:
                    ret=UQ._plot_fuzzy(hdf5,baseshapes_file,plot=False)
                elif type(hdf5[hdf5.keys()[0]]) is list:
                    ret=UQ._plot_probfuzzy(hdf5,plot=False)
                else:
                    utilities.msg('Could not get results. hdf5 parameter dictionary must contain strings or lists of strings','e')
            else:
                utilities.msg('Could not get results. Nothing to get.','w')
        else:
            utilities.msg('hdf5 parameter of type {} isn not supported'.format(str(type(hdf5))),'e')

        utilities.msg('Retrieving results...Done')
        sys.stdout.flush()

        return ret

    @staticmethod
    def plot(hdf5,baseshapes_file='shapes.json',**kwargs):
        """
        Plots results, and also returns them in varying formats. See table below.

        - *hdf5*: The hdf5 files to plot.
        - *baseshapes_file*: The full path of the json file containing the
          baseshapes . If no path is specified, it will be taken from the *hdf5* parameter.
          If that contains no path, then the current working directory is used.

          If this parameter is set to None, any shapes will be ignored. This can be used
          in cases where the model contains no shapes.
        - *kwargs*: Keyword arguments. Depend on plot type, see table below.

        Returns the data used to generate each plot. This data can be used for
        generating custom plots or perform additional analyses.

        ============== =================================================================
        type(hdf5)      plot type
        ============== =================================================================
        string         Probabilistic.
                       CDF and histogram plot for each output.

                       hdf5: is the hdf5 file to process.

                       Valid kwargs:

                           - *hist*: Boolean. If True, plots a
                             histogram. Default is True.
                           - *nbins*: Number of bins to use for
                             histogram. Default is None (auto).
                           - *return_fullshapesdata*: If False, the 'shapesdata'
                             value will contain a
                             list of Shapely shapes. See below. Default is False.

                       Returns: a dictionary. The 'shapesdata' key contains all
                       the shapely shapes generated in this run in the form of a list,
                       [shpdata].

                           {$outputName$:r_[samples (float)], ..., 'shapesdata':[shpdata] }

                       If *return_fullshapesdata* =True then [shpdata] becomes a dictionary:

                           {'name':<string>, 'shp':<Shapely object>, 'desc':<string>,
                           'type':<string>, 'alphacut':None, ...}

        dict of str    Fuzzy.
                       Membership function for each output.

                       hdf5: Each
                       str is a file name corresponding to the
                       results at a particular alpha cut,
                       given by the key of the dict.

                       Valid kwargs:

                           - *alphacuts*: process only the alpha cuts given
                             in this list. If not specified. All alpha cuts
                             are processed.
                           - *color*: Plots membership in color if set to
                             True. If False grayscale is used. If it is float, it
                             must be between 0.0 and 1.0, indicating a
                             specific shade of grey.
                           - *return_fullshapesdata*: If False, the 'shapesdata'
                             value will contain a
                             list of Shapely shapes. See below. Default is False.

                       Returns: a dictionary. The 'shapesdata' key contains all
                       the shapes for all alpha cuts.

                           {$outputName$:{$acut$:r_[samples (float)], ...}, ...,
                           'shapesdata':{$acut:[shpdata], ...}}

                       If *return_fullshapesdata* =True then [shpdata] is a dictionary:

                           {'name':<string>, 'shp':<Shapely object>, 'desc':<string>,
                           'type':<string>, 'alphacut':<float>, ...}

        dict of list   Probabilistic-fuzzy.
                       Horsetail plot where each 'strand' has
                       an associated value of membership.

                       hdf5: Each
                       list contains files corresponding to a
                       set of results at a particular alpha cut
                       given by the key of the dict.

                       Valid kwargs:

                           - *cdf_pdf*: String value.

                               - 'cdf' (default): A plot is
                                 generated at each alpha cut,
                                 containing all cdfs at that
                                 alpha value
                                 (i.e., a horsetail plot).
                               - 'pdf': Same as 'cdf' except
                                 fitted pdfs are plotted.
                               - 'none': No pdfs or cdfs are
                                 generated for each alpha cut.
                                 The composite pdf is still
                                 generated.

                           - *cdfs_fuzzy3d*: Plots all cdfs
                             at all alpha levels in 3D, with
                             alpha being the z-coordinate.
                             Default is True.
                           - *alphacuts*: process only the alpha cuts given
                             in this list. If not specified. All alpha cuts
                             are processed.
                           - *color*: Plots membership in color if set to
                             True. If False grayscale is used. If it is float, it
                             must be between 0.0 and 1.0 indicating a specific
                             shade of grey.
                           - *estimate_at_value*: A dictionary.
                             Estimates the membership function
                             of output 'X' at value yyy. yyy
                             can be a single number or a list
                             of numbers.

                                 {'X':yyy, ...}

                             Default value is None.

                        Returns: dictionary.

                            {$outputName$:{$acut$:[mincdf,maxcdf], ...}, ...}

                        Mincdf and maxcdf are dictionaries containing the
                        minimum and maximum cdfs at each alpha cut:

                            { 'x':r_[x-values], 'y':r_[y-values] }
        ============== =================================================================
        """

        utilities.msg('Plotting results...')

        ret=None

        if kwargs.pop('UASA',None)=='SA':
            import sa #prevent circular import errors
            cls=sa.SA
        else:
            cls=UQ

        if type(hdf5) is str:
            ret=cls._plot_probabilistic(hdf5,baseshapes_file,**kwargs)
        elif type(hdf5) is dict:
            if len(hdf5)>0:
                if type(hdf5[hdf5.keys()[0]]) is str:
                    ret=cls._plot_fuzzy(hdf5,baseshapes_file,**kwargs)
                elif type(hdf5[hdf5.keys()[0]]) is list:
                    ret=cls._plot_probfuzzy(hdf5,**kwargs)
                else:
                    utilities.msg('Could not plot. hdf5 parameter dictionary must contain strings or lists of strings','e')
                    return None
            else:
                utilities.msg('Could not plot results. Nothing to plot.','w')
                return None
        else:
            utilities.msg('hdf5 parameter of type {} isn not supported'.format(str(type(hdf5))),'e')
            return None

        utilities.msg('Plotting results...Done')
        sys.stdout.flush()

        if baseshapes_file!=None:
            plt.show(block=False)

        return ret

    @staticmethod
    def _plot_probfuzzy(hdf5,cdf_pdf='cdf',cdfs_fuzzy3d=True,alphacuts=None,color=True,
                        estimate_at_value=None,estimate_at_prob=None,plot=True):
        """
        Plots a mixed probabilistic-fuzzy run. A plot is generated which shows
        A lower bound and upper bound CDF for each alpha cut.

        - *hdf5*: a dictionary of lists. Key is the alpha cut. Each list contains file names.
        - *cdf_pdf*: If set to 'pdf', plots the fitted pdf of each output at every alpha cut. If
          set to 'cdf', plots the cdf of each output at every alpha cut (a "horsetail" plot).
          If set to 'none', doesn't plot this.
        - *cdfs_fuzzy3d*: Plots the cdfs and associated membership function in 3D, to show
          cdfs which may be overlapping.
        - *color*: Plots membership values in color. Else, grayscale is used.
        - *estimate_at_value*: a dictionary which indicates at which values of the output variables the
          membership function will be calculated. Appears as a vertical line on the cdf plots.

              {<output name>:[float, float, ...], ...}

          <output name> is a name of an ouput in the hdf5 files, the value is a list of numbers
          indicating the values at which to calculate the membership function at.
        - *estimate_at_prob*: a dictionary which indicates at which quantiles of the output variables the
          membership function will be calculated. Appears as a horizontal line on the cdf plots.
        - *plot*: If False only returns data (used by :func:`UQ.get_results`)
        """
        utilities.msg('Type of run: probabilistic-fuzzy run')

        if not cdf_pdf in ['none','cdf','pdf']:
            raise Exception ('Invalid value, {}, of parameter cdf_pdf'.format(cdf_pdf))

        #sort in reverse order so we can take advantage of the
        #nesting property of the alpha cuts.
        sorted_acuts=sorted(hdf5.keys(),reverse=True)
        if alphacuts==None:
            alphacuts=sorted_acuts
        else:
            alphacuts=sorted(alphacuts,reverse=True)
        if not all([x in sorted_acuts for x in alphacuts]):
            raise Exception('A value in the *alphacuts* list was not found in the hdf5 files')

        hdf5_path=os.path.dirname(hdf5[sorted_acuts[0]][0])
        if hdf5_path=='':
            utilities.msg('Current dir: {}'.format(os.getcwd()))
        else:
            utilities.msg('Current dir: {}'.format(hdf5_path))

        utilities.msg('available alpha cuts: {}'.format(sorted_acuts),'v')
        outputs=puq.hdf.get_output_names(hdf5[sorted_acuts[0]][0])
        utilities.msg('loaded {}'.format(hdf5[sorted_acuts[0]][0]),'d')
        utilities.msg('Available output vars: {}'.format(outputs),'v')

        if color:
            cm=plt.get_cmap('jet')
        else:
            cm=plt.get_cmap('binary')

        ret={}
        for output in outputs:
            utilities.msg('processing output variable {}'.format(output))

            #keep track of the total number of samples, across all alpha cuts
            total_numsamp=0

            #keep track of the min/max cdf at each acut
            min_cdf={}
            max_cdf={}

            ret[output]={}

            for i,acut in enumerate(alphacuts):
                if cdf_pdf!='none' and plot:
                    utilities.msg('\t{} alpha cut'.format(acut),'v')
                    plt.figure()
                    plt.xlabel('value')
                    plt.title('{} a-cut:{}'.format(output,acut))
                    utilities.msg('\t   Figure {}'.format(plt.gcf().number),'v')
                    utilities.msg('\t\tname: {}'.format(output),'v')
                    utilities.msg('\t\tdesc: {}'.format(puq.hdf.data_description(hdf5[acut][0],output)),'v')
                    utilities.msg('\t\tsource (1st run): {}'.format(os.path.basename(hdf5[acut][0])),'d')

                min_cdf_x=None
                min_cdf_y=None
                max_cdf_x=None
                max_cdf_y=None

                for run in hdf5[acut]:
                    samples=puq.hdf.get_result(run,output)
                    utilities.msg('\t   loaded {} samples from {}'.format(np.size(samples),
                                  os.path.split(run)[-1]),'d')
                    total_numsamp+=np.size(samples)

                    srt=np.sort(samples)
                    cdf_y=np.arange(len(srt))/float(len(srt))

                    if cdf_pdf=='pdf' and plot:
                        pdf=puq.ExperimentalPDF(samples)
                        pdf.plot()
                        plt.ylabel('probability density')
                    elif cdf_pdf=='cdf' and plot:
                        plt.plot(srt,cdf_y,'k-')
                        plt.ylabel('probability')

                    #find the minimum and maximum cdf at this alpha cut.
                    #Note that the min/max may contain pieces of multiple cdfs.
                    if min_cdf_x==None or max_cdf_x==None:
                        #initialize the min/max cdfs
                        #Note, the min_cdf_y arrays are the same for all cdfs at this
                        #alpha cut due to how cdf_y is calculated above. Same for max_cdf_y
                        min_cdf_x=srt
                        min_cdf_y=cdf_y
                        max_cdf_x=srt
                        max_cdf_y=cdf_y
                    else:
                        min_cdf_x=UQ._plot_probfuzzy_minmax_cdf(min_cdf_x,srt,'min')
                        max_cdf_x=UQ._plot_probfuzzy_minmax_cdf(max_cdf_x,srt,'max')
                    #end if min_cdf_x==None or max_cdf_x==None:
                #end for run in hdf5[acut]

                #plot the min/max cdf for this alpha cut.
                if cdf_pdf=='cdf' and plot:
                    plt.plot(min_cdf_x,min_cdf_y,'-r')
                    plt.plot(max_cdf_x,max_cdf_y,'-r')
                    plt.tight_layout()

                #check the previous alpha cut to see if the min there is less than
                #the min here. If it is, replace this min with that one. This takes
                #advantage of the nesting property of the alpha cuts
                if i>0:
                    min_cdf_x_prev=min_cdf[alphacuts[i-1]]['x']
                    max_cdf_x_prev=max_cdf[alphacuts[i-1]]['x']
                    min_cdf_x=UQ._plot_probfuzzy_minmax_cdf(min_cdf_x,min_cdf_x_prev,'min')
                    max_cdf_x=UQ._plot_probfuzzy_minmax_cdf(max_cdf_x,max_cdf_x_prev,'max')
                min_cdf[acut]={'x':min_cdf_x,'y':min_cdf_y}
                max_cdf[acut]={'x':max_cdf_x,'y':max_cdf_y}

                ret[output][acut]=[min_cdf[acut],max_cdf[acut]]

            #end for acut in alphacuts:

            utilities.msg('\tTotal number of samples: {}'.format(total_numsamp),'v')

            #the final plots
            if plot:
                f1=plt.figure()
                plt.xlabel('value')
                plt.ylabel('probability')
                plt.title(output)
                if cdfs_fuzzy3d:
                    f2=plt.figure()
                    ax=f2.gca(projection='3d')
                    plt.xlabel('value')
                    plt.ylabel('probability')
                    ax.set_zlabel('membership')
                    plt.title(output)
                    plt.tight_layout()
                for acut in sorted_acuts:
                    #the acut value is a string when plot is called directly
                    ac=float(acut)

                    #the colormap value 1 maps to 0 so multiply so that it's never 1
                    #also prevent the 0 ac from disappearing into white by setting it to 0.05
                    ac=0.05 if ac==0 else ac
                    clr=str(color) if color>0 and color<1 else cm(ac*0.999999)

                    plt.figure(f1.number)
                    plt.plot(min_cdf[acut]['x'],min_cdf[acut]['y'],color=clr,label=acut)
                    plt.plot(max_cdf[acut]['x'],max_cdf[acut]['y'],color=clr)

                    if cdfs_fuzzy3d:
                        plt.figure(f2.number)
                        ax.plot(min_cdf[acut]['x'],min_cdf[acut]['y'],float(acut),color=clr)
                        ax.plot(max_cdf[acut]['x'],max_cdf[acut]['y'],float(acut),color=clr)
                #end for acut in sorted_acuts:

                plt.figure(f1.number)
                plt.legend(title='membership',bbox_to_anchor=(1.05, -0.05), loc='lower right',
                           borderaxespad=0.)
                plt.ylim(-0.005,1.005)
                plt.tight_layout()


                #calcualate the memb func at a given value
                if estimate_at_value!=None and len(estimate_at_value)>0:
                    if output in estimate_at_value.keys():
                        if not type(estimate_at_value[output]) is list:
                            values=[estimate_at_value[output]]
                        else:
                            values=estimate_at_value[output]
                        for value in values:
                            memb={}
                            for acut in sorted_acuts:
                                memb[acut]=[np.interp(value,min_cdf[acut]['x'],min_cdf[acut]['y']),
                                            np.interp(value,max_cdf[acut]['x'],max_cdf[acut]['y'])]

                            x,y=UQ._plot_fixMembFcn(memb)
                            plt.figure()
                            plt.plot(x,y,'-k')
                            plt.xlabel('probability')
                            plt.ylabel('membership')
                            plt.title('{} at value={}'.format(output,value))
                            plt.ylim(-0.005,1.1)
                            plt.xlim(min(x)-0.007,)
                            plt.tight_layout()

                            plt.figure(f1.number)
                            plt.plot([value,value],[0,1],'--',color='0.5')
                    else:
                        utilities.msg('Can''t estimate. {} not found in the list of outputs'.format(output))
                #end if estimate_at
                if estimate_at_prob!=None and len(estimate_at_prob)>0:
                    if output in estimate_at_value.keys():
                        if not type(estimate_at_prob[output]) is list:
                            values=[estimate_at_prob[output]]
                        else:
                            values=estimate_at_prob[output]
                        for value in values:
                            memb={}
                            for acut in sorted_acuts:
                                memb[acut]=[np.interp(value,min_cdf[acut]['y'],min_cdf[acut]['x']),
                                            np.interp(value,max_cdf[acut]['y'],max_cdf[acut]['x'])]

                            x,y=UQ._plot_fixMembFcn(memb)
                            plt.figure()
                            plt.plot(x,y,'-k')
                            plt.xlabel('values')
                            plt.ylabel('membership')
                            plt.title('{} at probability={}'.format(output,value))
                            plt.ylim(-0.005,1.1)
                            plt.xlim(min(x)-0.007,)
                            plt.tight_layout()

                            plt.figure(f1.number)
                            plt.plot(plt.gca().get_xlim(),[value,value],'--',color='0.5')
                    else:
                        utilities.msg('Can''t estimate. {} not found in the list of outputs'.format(output))
                #end if estimate_at
            #end if plot
        #end for output in outputs:
        return ret

    @staticmethod
    def _plot_probfuzzy_minmax_cdf(srt_samples1,srt_samples2,minmax):
        """
        Finds the min (or max) of the cdfs of the two given arrays.

        - *srt_samples1, srt_samples2*: arrays of samples of which to find the min (or max)
          cdf. The arrays must be sorted ascending and have the same number of elements.
        - *minmax*: If set to 'min' calculates the minimum cdf. If set to 'max' calculates the
          maximum.

        Returns an array with the min (max) of the cdfs of the two sample arrays. The
        y-values of the cdf can be calculated by

            cdf_y=np.arange(len(srt_samples1))/float(len(srt_samples1))
        """
        #find the entries out of the two cdfs which are the minimum/maximum
        if minmax=='min':
            mins=np.minimum(srt_samples1,srt_samples2)
        elif minmax=='max':
            mins=np.maximum(srt_samples1,srt_samples2)
        else:
            raise Exception('minmax must be either "min" or "max". Got {}'.format(minmax))

        #Find the indices in each array which contain the minimum.
        #If the array does not contain the minimum at a particular location,
        #that index will be false.
        #Also, find the indices in which both arrays are equal. We need this
        #since min1_indices and min2_indices will both be true for elements which are equal
        #in both arrays.
        min1_indices=mins==srt_samples1
        min2_indices=mins==srt_samples2
        eq_indices=srt_samples1==srt_samples2

        #select the entries out of each array which correspond to
        #the indices we found. Then combine them into a single array.
        #First, build an array with entries that exist in both srt_samples1 and srt_samples2.
        #Then, pick out the elements in each array which are NOT equal, then select the minimum
        #out of those.
        return eq_indices*srt_samples1 + \
               ~eq_indices*min1_indices*srt_samples1 + \
               ~eq_indices*min2_indices*srt_samples2

    @staticmethod
    def _plot_fuzzy(hdf5,baseshapes_file,return_fullshapesdata=False,alphacuts=None,color=True,plot=True):
        """
        Plots the membership function of all outputs in a fuzzy run.

        - *hdf5*: a dictionary of filenames. Key is the alpha cut.
        - *alphacuts*: processes only the alpha cuts in this list.
        - *color*: plots in color if True
        - *plot*: If False, only returns data (used by :func:`UQ.get_results`)
        """
        utilities.msg('Type of run: fuzzy only')

        #sort in reverse order so we can take advantage of the
        #nesting property of the alpha cuts.
        sorted_acuts=sorted(hdf5.keys(),reverse=True)

        if alphacuts==None:
            alphacuts=sorted_acuts
        if not all([x in sorted_acuts for x in alphacuts]):
            raise Exception('A value in the *alphacuts* list was not found in the hdf5 files')

        hdf5_path=os.path.dirname(hdf5[sorted_acuts[0]])
        if hdf5_path=='':
            utilities.msg('Current dir: {}'.format(os.getcwd()))
        else:
            utilities.msg('Current dir: {}'.format(hdf5_path))

        utilities.msg('available alpha cuts: {}'.format(sorted_acuts),'v')
        outputs=puq.hdf.get_output_names(hdf5[sorted_acuts[0]])
        utilities.msg('loaded {}'.format(os.path.basename(hdf5[sorted_acuts[0]])),'d')
        utilities.msg('Available output vars: {}'.format(outputs),'v')

        #data to plot
        # {$varName$:{$acut$:[min,max], ...}, ...}
        plot_data={}
        ret={}
        for output in outputs:
            utilities.msg('processing output variable {}'.format(output))
            plot_data[output]={}
            ret[output]={}

            prev_min=np.inf
            prev_max=-np.inf
            total_numsamp=0
            for acut in alphacuts:
                samples=puq.hdf.get_result(hdf5[acut],output)
                utilities.msg('\tloaded {} samples from {}'.format(np.size(samples),
                              os.path.basename(hdf5[acut])),'d')
                total_numsamp+=np.size(samples)

                #Since a-cuts are nested, it means
                #that a lower alpha level will always contain a higher one.
                #This code ensures that lower alpha levels will be at least as
                #wide as the higher levels.
                minval=np.min(samples)
                maxval=np.max(samples)
                if minval<prev_min:
                    prev_min=minval
                else:
                    minval=prev_min
                if maxval>prev_max:
                    prev_max=maxval
                else:
                    maxval=prev_max

                ret[output][acut]=samples
                plot_data[output][acut]=[minval,maxval]

                #UQ._plot_probabilistic(hdf5[acut],outputName=output,title_append=", a-cut:"+acut)

            utilities.msg('\tTotal number of samples: {}'.format(total_numsamp),'v')

        if plot:
            for outvar,data in plot_data.iteritems():
                #note: point_to_poly_runner. comment out all lines tagged with fig5
                plt.figure()                                        #fig5 off from here to plt.ylim()
                plt_x,plt_y=UQ._plot_fixMembFcn(data)
                plt.plot(plt_x,plt_y,'k-')
                plt.xlabel('value')
                plt.ylabel('membership')
                plt.title(outvar)
                plt.ylim(-0.005,1.1)
                plt.tight_layout()

                utilities.msg('Figure {}'.format(plt.gcf().number),'v')
                utilities.msg('\tname: {}'.format(outvar),'v')
                utilities.msg('\tdesc: {}'.format(puq.hdf.data_description(hdf5[sorted_acuts[0]],outvar)),'v')
                utilities.msg('\tsource (1st a-cut): {}'.format(os.path.basename(hdf5[sorted_acuts[0]])),'d')

        #plot the shapes, as obtained from the hdf5 files.
        if baseshapes_file!=None:
            utilities.msg('Getting shapes')

            ret['shapesdata']={}

            if hdf5_path!='' and os.path.dirname(baseshapes_file)=='':
                baseshapes_file=os.path.join(hdf5_path,baseshapes_file)
            if not os.path.isfile(baseshapes_file):
                raise Exception('Base shapes file {} not found'.format(os.path.realpath(baseshapes_file)))

            if plot:
                plt.figure()                                            #fig5 off
                plt.axis('equal')                                       #fig5 off
                if color==True:
                    cm=plt.get_cmap('jet')
                elif color==False:
                    cm=plt.get_cmap('binary')

            for acut in sorted_acuts[::-1]:
                if acut in alphacuts:
                    if return_fullshapesdata:
                        shapesdata=UQ._plot_getshapes(hdf5[acut],baseshapes_file,'full')
                    else:
                        shapesdata=UQ._plot_getshapes(hdf5[acut],baseshapes_file,'shapely')
                    ret['shapesdata'][acut]=shapesdata

                    if plot:
                        #the colormap value 1 maps to 0 so multiply so that it's never 1
                        #also prevent the 0 ac from disappearing into white by setting it to 0.05
                        ac=0.05 if float(acut)==0 else float(acut)
                        clr=str(color) if color>0 and color<1 else cm(ac*0.999999)
                        for shpdt in shapesdata:
                            if return_fullshapesdata:
                                shp=shpdt['shp']
                            else:
                                shp=shpdt
                            shpcoords=utilities.shape2Points(shp)
                            if np.size(shpcoords[:,0])==1:
                                sym='o'
                            else:
                                sym='-'
                            plt.plot(shpcoords[:,0],shpcoords[:,1],sym,markeredgecolor='none',color=clr)
                        #end for
                    #end if plot
                #end if acut in alphacuts
            #end for
        #end if
        plt.tight_layout()
        return ret

    @staticmethod
    def _plot_fixMembFcn(data):
        """
        Makes the data from a membership function defined as a set of intervals
        easier to plot.

        Returns a tuple, (x,y), containing the points to plot.

        - *data*: A dictionary in the format

            {<a-cut>:[lower bound, upper bound]}
        """
        acuts=sorted(data.keys())

        plt_x=np.zeros(2*len(data))
        plt_y=np.zeros(2*len(data))
        i=0
        for acut in acuts:
            lowerupper=data[acut]
            plt_x[i]=lowerupper[0]
            plt_x[2*len(data)-1-i]=lowerupper[1]
            plt_y[i]=acut
            plt_y[2*len(data)-1-i]=acut
            i+=1

        return plt_x,plt_y

    @staticmethod
    def _plot_probabilistic(hdf5,baseshapes_file,return_fullshapesdata=False,hist=True,nbins=None,title_append='',plot=True,msg=True):
        """
        Plots the CDF of each outpupt of a probabilistic run.

        - *hdf5*: a string indicating the name of the hdf5 file to process.
        - *return_fullshapesdata*: returns the complete data to allow for external plotting of shapes.
          If False, returns only shapely shapes in the 'shapesdata' key of the return dictionary.
        - *hist*: If true, also plots a histogram.
        - *nbins*: The number of bins to use when plotting the histogram. If not specified, this
          value is automatically calculated.
        - *title_append*: Appends this text to the title of the plot.
        - *plot*: If false, only returns data (used by :func:`UQ.get_results` and plotting
          functions in SA module.)
        - *msg*: shows the message (used by SA module plotting functions)
        """
        outputs=puq.hdf.get_output_names(hdf5)

        if msg:
            utilities.msg('Type of run: probabilistic only')

        hdf5_path=os.path.dirname(hdf5)
        if msg:
            if hdf5_path=='':
                utilities.msg('Current dir: {}'.format(os.getcwd()))
            else:
                utilities.msg('Current dir: {}'.format(hdf5_path))

            utilities.msg('loaded {}'.format(hdf5),'d')
            utilities.msg('Available output vars {}'.format(outputs),'v')

        n=-1
        ret={} #{$varName$:r_[samples (float)], ...}
        for output in outputs:
            if msg:
                utilities.msg('processing output variable {}'.format(output))
            samples=puq.hdf.get_result(hdf5,output)

            ret[output]=samples
            if plot:
                if nbins==None:
                    nbins_output=UQ._hist_nbins(samples)
                else:
                    nbins_output=nbins

                fig_cdf=plt.figure()
                #better way to do it since it doesn't depend on binning
                #http://stackoverflow.com/a/11692365
                #yvals is an array = [0, 1/N, 2/N, ... (N-1)/N]
                #Each value in samples is assigned the corresponding value in yvals
                #The more a particular value appears in samples, the more yvals will
                #be associated with it, therefore increasing the height of the ECDF at
                #that particular value.
                srt=np.sort(samples)
                n=len(srt)
                yvals=np.arange(n)/float(n)
                plt.plot(srt, yvals)
                plt.xlabel('value')
                plt.ylabel('probability')
                plt.title(output + title_append)
                plt.tight_layout()

                if hist:
                    fig_hist=plt.figure()
                    plt.hist(samples,bins=nbins_output,normed=True)
                    plt.xlabel('value')
                    plt.ylabel('probability density')
                    plt.title(output + title_append)

                txt='Figure {}'.format(fig_cdf.number)
                if hist:
                    txt+=' and {}'.format(fig_hist.number)
                utilities.msg(txt)
                utilities.msg('\tname: {}'.format(output))
                utilities.msg('\tdesc: {}'.format(puq.hdf.data_description(hdf5,output)))
                utilities.msg('\tnum samples: {}'.format(np.size(samples)))
                utilities.msg('\tsource: {}'.format(os.path.basename(hdf5)),'d')
            #end if plot
        #enf for output in outputs

        #plot the shapes, as obtained from the hdf5 files.
        ret['shapesdata']=None
        if baseshapes_file!=None:
            utilities.msg('Getting shapes')

            if hdf5_path!='' and os.path.dirname(baseshapes_file)=='':
                baseshapes_file=os.path.join(hdf5_path,baseshapes_file)
            if not os.path.isfile(baseshapes_file):
                raise Exception('Base shapes file {} not found'.format(os.path.realpath(baseshapes_file)))

            if return_fullshapesdata:
                shapesdata=UQ._plot_getshapes(hdf5,baseshapes_file,'full')
            else:
                shapesdata=UQ._plot_getshapes(hdf5,baseshapes_file,'shapely')
            ret['shapesdata']=shapesdata

            if plot:
                plt.figure()
                for shpdt in shapesdata:
                    if return_fullshapesdata:
                        shp=shpdt['shp']
                    else:
                        shp=shpdt
                    shpcoords=utilities.shape2Points(shp)
                    if np.size(shpcoords[:,0])==1:
                        sym='o'
                    else:
                        sym='-'
                    plt.plot(shpcoords[:,0],shpcoords[:,1],sym,color='0.7')
                plt.axis('equal')
        #end if baseshapes_file

        return ret


    @staticmethod
    def _plot_getshapes(hdf5,baseshapes_file,returntype='coords'):
        """
        Gets all the shapes from the given hdf5 file and returns them in the form of a list.

        - *hdf5*: the full path to the hdf5 file.
        - *baseshapes_file*: the base shapes file to use.
        - *returntype*:

            - 'coords': the returned list contains coordinates in the form of a
              2D array.
            - 'shapely': the returned list contains Shapely shapes.
            - 'full': the returned list contains dictionaries with keys

                {'name':<string>, 'shp':<Shapely object>, 'desc':<string>,
                 'type':<string>, 'alphacut':<float or None>, ...}

              which is similar in format to the return value of :func:`ObjectManager.puq2Shapes`.
        """
        params_hdf5=puq.hdf.get_params(hdf5)

        #use the first input parameter to get the number of realizations
        n=np.size(params_hdf5[0].values)

        retlist=[]

        for i in range(n):
            params=[]
            for param in params_hdf5:
                if objectmanager.ObjectManager.isShapeParam(param.name) or \
                    objectmanager.ObjectManager.isShapeParamLegacy(param.name):
                    params.append(dict(name=param.name, desc=param.description,value=param.values[i]))

            shapes=objectmanager.ObjectManager.puq2Shapes(baseshapes_file,params=params)

            for shpname,shpdata in shapes.iteritems():
                shp=shpdata['shp']
                if returntype=='coords':
                    a=utilities.shape2Points(shp)
                    retlist.append(a)
                elif returntype=='shapely':
                    retlist.append(shp)
                elif returntype=='full':
                    retlist.append({'name':shpname,'shp':shp,'desc':shpdata['desc'],
                                    'type':shpdata['type'],'alphacut':shpdata['alphacut']})
                else:
                    raise Exception('Invalid value for option "returntype": {}'.format(returntype))
        return retlist

    @staticmethod
    def _hist_nbins(data):
        """
        Calculates the number of bins for a histogram given the 1D array data.

        From puq analyzer.py
        """
        nbins=2
        iqr = scipy.stats.scoreatpercentile(data, 75) - scipy.stats.scoreatpercentile(data, 25)
        if iqr == 0.0:
            nbins =11
        else:
            nbins = int((np.max(data) - np.min(data)) / (2*iqr/len(data)**(1.0/3)) + .5)
        if nbins<2:
            nbins=2

        return nbins

    @staticmethod
    def find_all_hdf5(basename,path=None):
        """
        Used for the :func:`UQ.plot` function. Returns a string or dictionary with the
        hdf5 files (without path) found in *path* which begin with *basename*.
        See :func:`UQ.plot` for more details regaring the possible return types.

        - *basename*: the filename of the hdf5 file(s) to process. Basename includes the
          name up to the first @ symbol, if any. File extension is not included. E.g., the
          basename in the file name below is indiccated

            $basename$ @x#y.hdf5

          where x is the alpha cut (0.0 to 1.0) and y is the run number for alpha cut x.
        - *path*: the location to look for the files begining with *basename*. If
          not specified, the current python working directory is used.

        """
        if path!=None:
            if os.path.exists(path):
                os.chdir(path)
            else:
                raise Exception ('path {} not found'.format(path))

        #use glob pacage
        files=glob.glob(basename + '*.hdf5')

        if len(files)==0:
            raise Exception('no files with basename {} found'.format(os.path.realpath(basename)))
        elif len(files)==1:
            #we have a probabilistic run
            return os.path.join(path,files[0])
        else:
            filtered_files=[f for f in files if re.search('@[0,1]\.\d+\.hdf5$',f)]
            if len(filtered_files)>0:
                #matched $basename$ @x.hdf5. we have a fuzzy run.
                filtered_files_dict={}
                for f in filtered_files:
                    #for each file, get the alpha cut value
                    #the 2 in [0,1,2] is for the special case of prob-fuzzy SA
                    r=re.search('@(?P<acut>[0,1,2]\.\d+)\.hdf5$',f)
                    if r:
                        filtered_files_dict[float(r.group('acut'))]=os.path.join(path,f)
                    else:
                        raise Exception('Unexpected format for the a-cut value for {}'.format(f))
                return filtered_files_dict

            filtered_files=[f for f in files if re.search('@[0,1]\.\d+#\d+\.hdf5$',f)]
            if len(filtered_files)>0:
                #matched $basename$ @x#y. We have a mixed run
                filtered_files_dict={}
                for f in filtered_files:
                    #extract the alpha cut
                    r=re.search('@(?P<acut>[0,1]\.\d+)#(?P<run>\d+)\.hdf5$',f)
                    if r:
                        if float(r.group('acut')) not in filtered_files_dict.keys():
                            filtered_files_dict[float(r.group('acut'))]=[]
                        filtered_files_dict[float(r.group('acut'))].append(os.path.join(path,f))
                    else:
                        raise Exception('Unexpected format for the a-cut value for {}'.format(f))
                return filtered_files_dict
