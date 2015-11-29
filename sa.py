import uq,utilities,objectmanager
import puq,fuzz
import multiprocessing,traceback,os,ctypes,inspect
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import scipy.stats
"""
Sensitivity analysis using the Morris method.
"""

class SA(uq.UQ):
    """
    Manages a SA run in a simplified way.

    Parameters are the same as :class:`UQ.uq` except for num_trajectories and num_levels.
    """
    def __init__(self,testProgScriptFile=None,testProgFunc=None,args=[],testProgDesc=None,workingDir=None,
                 objMgr=None,baseShapesName='shapes.json',probVars={},fuzzyVars={},
                 fuzzyVarACuts=[],consts={},num_trajectories=10,num_levels=4,seed=None,
                 outfiles='',sweep_cb=None,pool_recycle_every=60):

        uq.UQ.__init__(self,testProgScriptFile,testProgFunc,args,testProgDesc,workingDir,
                 objMgr,baseShapesName,probVars,fuzzyVars,
                 fuzzyVarACuts,consts,n_prob=None,n_fuzzy=None,seed=seed,
                 outfiles=outfiles,sweep_cb=sweep_cb,pool_recycle_every=pool_recycle_every)

        #get the filename of the calling script
        frame,filename,line_number,function_name,lines,index = inspect.stack()[1]
        self._calling_script=os.path.realpath(filename)

        self._r=num_trajectories
        self._p=num_levels
        if num_levels%2!=0:
            #to follow saltelli
            raise Exception ('num_levels must be even!')
        #to follow the recommended delta = p/2 * 1/(p-1) from Saltelli.
        #SALib calculates delta = grid_jump * 1/(p-2).
        self.grid_jump=self._p/2.

        if not self._objMgr._ignore_certain_vertices:
            utilities.msg('objMgr should have ignore_certain_vertices=True! Else there may be too many parameters to test in the senstivity analysis','w')


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
            if 'desc' not in data:
                raise Exception('{} is missing attribute "desc"'.format(cname))
            if 'value' not in data:
                raise Exception('{} is missing attribute "value"'.format(cname))
            desc=data['desc']
            value=data['value']
            if desc=='' or desc==None:
                desc='N/A' #there will be an error in objectmanager if this is blank
            if value=='' or value==None:
                raise Exception('"value" for {} cannot be None'.format(cname))
            self._puqparams_const.append(puq.ConstantParameter(cname,desc,
                                                               attrs=[('uncert_type','const')],
                                                               value=value))

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
            self._puqparams_fuzzy=self._objMgr.fuzzyObjects2PuqParams(self._baseShapesFileName,use_samples=False)
            if len(self._puqparams_fuzzy.keys())!=len(self._acuts):
                raise Exception('This error should not happen. Inconsistent alpha cut arrays length.')
            if not all([v in self._acuts for v in self._puqparams_fuzzy.keys()]):
                print(self._acuts,self._puqparams_fuzzy.keys())
                raise Exception('This error should not happen. Inconsistent alpha cut arrays content.')

        #add the non-shape fuzzy parameters
        for acut in self._acuts:
            for fuzzyVar,data in self._fuzzyVars.iteritems():
                #build a new fuzzy number
                fn=fuzz.TrapezoidalFuzzyNumber(kernel=(data['kl'],data['ku']),
                                               support=(data['sl'],data['su']))

                cut=fn.alpha(acut)

                desc=data['desc']
                if desc=='' or desc==None:
                    desc='N/A' #there will be an error in objectmanager if this is blank

                #build a puq parameter for this alpha cut
                param=puq.UniformParameter(fuzzyVar,desc,
                                           attrs=[('uncert_type','fuzzy')],
                                           min=cut[0],max=cut[1])

                if not acut in self._puqparams_fuzzy:
                    self._puqparams_fuzzy[acut]={'params':[]}
                self._puqparams_fuzzy[acut]['params'].append(param)


    def _setup_parameters_prob(self):
        self._puqparams_prob=[]
        if self._objMgr!=None and len(self._objMgr.probabilisticObjects)>0:
            self._puqparams_prob.extend(self._objMgr.crispObjects2PuqParams(self._baseShapesFileName,use_samples=False))

        #setup scalar probabilistic vars
        for varname,data in self._probVars.iteritems():
            if 'dist' not in data:
                raise Exception('{} is missing the "dist" attribute'.format(varname))
            if 'desc' not in data:
                raise Exception('{} is missing the "desc" attribute'.format(varname))
            desc=data['desc']
            if desc=='' or desc==None:
                desc='N/A' #there will be an error in objectmanager if this is blank
            if data['dist']=='normal':
                p=puq.NormalParameter(varname,
                                      desc,attrs=[('uncert_type','prob')],
                                      **data['kwargs'])
            elif data['dist']=='lognormal':
                p=puq.LognormalParameter(varname,
                                      desc,attrs=[('uncert_type','prob')],
                                      **data['kwargs'])
            elif data['dist']=='uniform':
                p=puq.UniformParameter(varname,
                                       desc,attrs=[('uncert_type','prob')],
                                       **data['kwargs'])
            elif data['dist']=='triangular':
                p=puq.TriangParameter(varname,
                                      desc,attrs=[('uncert_type','prob')],
                                      **data['kwargs'])
            else:
                raise Exception("'{}' distribution is not supported for variable {}".format(data['dist'],
                                varname))

            self._puqparams_prob.append(p)

    def run(self,plot=True,dryrun=False,keep_files=False,parallel_jobs=0):
        """
        Executes the SA run.

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
                SA.plot(outfiles,baseshapes_file=self._baseShapesFileName)
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
        Runs a probabilistic sweep.

        Returns the name of the hdf5 file that was generated.
        """
        utilities.msg('Starting a probabilistic run')

        params=self._puqparams_prob+self._puqparams_const
        sweep=self._sweep_setup(params,
                                self._r,self._p,self.grid_jump,'single probabilistic sweep -- Morris')

        ctypes.windll.kernel32.SetConsoleTitleA('{} jobs in puq run 1 of 1'.format( self._r*(len(params)+1)))

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
                sweepid='@{:1.1f}'.format(acut)
                params=self._puqparams_fuzzy[acut]['params']+self._puqparams_const
                sweep=self._sweep_setup(params,
                                        self._r,self._p,self.grid_jump,
                                        'fuzzy run (alpha-cut {} from set {})'.format(acut,self._acuts),
                                        pool,sweepid=sweepid)

                ctypes.windll.kernel32.SetConsoleTitleA( \
                    '{} jobs in puq run {} of {} (a-cut {})'.format( self._r*(len(params)+1),
                                                                    i+1,len(self._acuts),
                                                                    acut))

                utilities.msg('Puq run {} of {} (alpha-cut {})...'.format(i+1,len(self._acuts),acut))
                print('')
                hdf5=self._hdf5_basename +' ' +sweepid+'.hdf5'
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

        The file name has the convention $basename$ @x#0 where x is the alpha cut, expressed as a
        decimal number between and including 0 and 1. The #0 is to distinguish it as as mixed run
        (see :func:`UQ._run_probfuzzy`). There is only 1 hdf5 per alpha cut.

        NOTE: the probabilistic variables are re-sampled for every alpha-cut.
        """
        utilities.msg('Starting a probabilistic-fuzzy run')

        ret={}
        pool=None

        try:
            for i,acut in enumerate(sorted(self._acuts,reverse=True)):
                pool=self._get_processPool(pool)
                sweepid='@{:1.1f}#0'.format(acut)
                params=self._puqparams_fuzzy[acut]['params']+self._puqparams_prob+self._puqparams_const
                sweep=self._sweep_setup(params,
                                        self._r,self._p,self.grid_jump,
                                        'probabilistic-fuzzy sweep (alpha-cut {} from set {})'.format(acut,self._acuts),
                                        pool,sweepid=sweepid)

                ctypes.windll.kernel32.SetConsoleTitleA( \
                    '{} jobs in puq run {} of {} (a-cut {})'.format( self._r*(len(params)+1),
                                                                    i+1,len(self._acuts),
                                                                    acut))

                utilities.msg('Puq run {} of {} (alpha-cut {})...'.format(i+1,len(self._acuts),acut))
                print('')
                hdf5=self._hdf5_basename +' ' +sweepid+'.hdf5'
                if not sweep.run(hdf5,dryrun=self._dryrun):
                    raise Exception('error running sweep')
                print('')

                utilities.msg('Puq run {} of {} (alpha-cut {})...Done'.format(i+1,len(self._acuts),acut))

                ret[acut]=[os.path.join(os.getcwd(),hdf5)]
        finally:
            #closes the process pool even if there is an exception
            if not pool==None:
                pool.close()
                pool.join()

        return ret

    def _sweep_setup(self,puqparams,r,p,gj,desc='',proc_pool=None,sweepid=''):
        """
        Sets up a single sweep for the given parameters.

        - *puqparams* list of puq parameter objects.
        - *n*: The number of runs.
        - *proc_pool*: an instance of multiprocessing.Pool. If not specified, a new Pool will
          be created.
        - *sweepid*: extra info to identify this sweep as part of a particular hdf5 file.
          Only fuzzy and mixed analyses will set this flag. Probabilistic runs only have 1
          hdf5 file.

        Returns a Sweep object.
        """
        sa=puq.Morris(params=puqparams,numtrajectories=r,levels=p,gridjump=gj,iteration_cb=self._sweep_cb)
        return self._sweep_setup_helper(sa,desc,proc_pool,sweepid)

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

        ============== =====================================================================
        type(hdf5)      plot type
        ============== =====================================================================
        string         Probabilistic.
                       Single plot of sigma vs mu\*

                       hdf5: is the hdf5 file to process.

                       Valid kwargs:

                           - *title_append*: a string to append to any plots generated.
                           - *labels*: Boolean. If True (default), plots data labels
                             to identify the points.
                           - *labels_top*: only displays the top n data labels, sorted by
                             the sensitivity index. Default is 20.
                           - *labels_bbox_hgt, labels_bbox_wdt*: The bounding box for labels
                             isn't automatically calculated. These numbers are the
                             proportion of the axis limits which to make the bounding boxes.
                             Default values are 0.05, 0.04 respectively.
                           - *plot_top*: Only plots the top x entries.
                             x=None (plot all, default), x>0 (plot top x) x<0 (plots the
                             bottom x).
                             
                       Returns: a dictionary. The 'shapesdata' key contains all
                       the shapely shapes generated in this run in the form of a list,
                       [shpdata].

                           {$outputName$:r_[samples (float)], ..., 'shapesdata':[shpdata] }

        dict of str    Fuzzy.
                       Plot of sigma vs mu\* for each alpha cut. Also a plot of top 16
                       variables, sorted descending according to mu\* of highest alpha
                       cut.

                       hdf5: Each
                       str is a file name corresponding to the
                       results at a particular alpha cut,
                       given by the key of the dict.

                       Valid kwargs:

                           - *alphacuts*: process only the alpha cuts given
                             in this list. If not specified. All alpha cuts
                             are processed.
                           - *alphacut_orderby*: the alpha level to order the final results
                             by. Must be a value contained in the *alphacuts* list. Defaults
                             to the lowest available alpha level.
                           - *print_alpha_rank*: prints the Morris indices at each alpha
                             level to the console.
                           - *morris_plots*: If False (default) does not show the morris
                             plots at each alpha-cut.
                           - *title_append*: a string to append to any plots generated.
                           - *labels*: If True (default), plots data labels
                             to identify the points.
                           - *labels_top*: only displays the top n data labels, sorted by
                             the sensitivity index. Default is 20.
                           - *labels_bbox_hgt, labels_bbox_wdt*: The bounding box for labels
                             isn't automatically calculated. These numbers are the
                             proportion of the axis limits which to make the bounding boxes.
                             Default values are 0.05, 0.04 respectively.
                           - *plot_top*: Only plots the top x entries.
                             x=None (plot all, default), x>0 (plot top x) x<0 (plots the
                             bottom x).
        dict of list   Probabilistic-fuzzy.
                       The results are the same as the fuzzy case. The probabilistic and
                       fuzzy variables are plotted together. Probabilistic are solid
                       symbols, fuzzy are hollow.

                       Parameters are the same as the Fuzzy case.

        ============== =====================================================================

        """
        kwargs['UASA']='SA'
        return uq.UQ.plot(hdf5,baseshapes_file,**kwargs)

    @staticmethod
    def _plot_probabilistic(hdf5,baseshapes_file,plot_top=None,labels=True,labels_top=20,title_append='',plot=True,
                            labels_bbox_hgt=.05,labels_bbox_wdt=.04):
        """
        Plots the morris results and ranking of the Morris run. Ranking is in the form
        of a table with the top 20 entries ranked by u\*.

        - *hdf5*: a string indicating the name of the hdf5 file to process.
        - *baseshapes_file*: The json file with the baseshapes
        - *plot_top*: Only plots the top x entries.
          x=None (plot all, default), x>0 (plot top x) x<0 (plots the bottom x).
        - *labels*: plots data labels.
        - *labels_top*: only displays the top n data labels, sorted by the sensitivity index. Default is 20.
        - *title_append*: Appends this text to the title of the plot.
        - *plot*: If false, only returns data (used by :func:`UQ.get_results`)

        Returns a dictionary containting the sensitivity indices:

            {$output$:{$paramName$:{mu*,sigma,mu*95conf}, ...}, ... }

        There is an extra key named 'shapesdata' which contains the shapes for each realization
        in the form of a list of shapely shapes.
        """
        outputs=puq.hdf.get_output_names(hdf5)

        utilities.msg('Type of run: probabilistic only')

        hdf5_path=os.path.dirname(hdf5)
        if hdf5_path=='':
            utilities.msg('Current dir: {}'.format(os.getcwd()))
        else:
            utilities.msg('Current dir: {}'.format(hdf5_path))

        utilities.msg('loaded {}'.format(hdf5),'d')
        utilities.msg('Available output vars {}'.format(outputs),'v')

        ret={} #{$output$:{$paramName$:{mu*,sigma,mu*95conf}, ...}, ... }

        data=uq.UQ._plot_probabilistic(hdf5,baseshapes_file,return_fullshapesdata=False,hist=False,
                            nbins=None,title_append=title_append,plot=False,msg=False)['shapesdata']
        ret['shapesdata']=data


        for output in outputs:
            utilities.msg('processing output variable {}'.format(output))
            sens=puq.hdf.get_sensitivity(hdf5,output)
            ret[output]=sens

            sens_data=SA._plot_sensdata2array(sens,hdf5)
            if plot:
                SA._plot_morrisplot(sens_data,output + title_append,labels,labels_top,
                                    labels_bbox_hgt,labels_bbox_wdt,plot_top=plot_top)

            print('\tname\t\tmu*\t\tsigma\t\tmustar_conf')
            for row in sens_data:
                print('\t{}\t\t{:.5e}\t{:.5e}\t{:.5e}'.format(row['name'],
                      row['mustar'],row['sigma'],row['mustar_conf']))
            print('total number of parameters: {}'.format(len(sens_data)))

        return ret

    @staticmethod
    def _plot_sensdata2array(sens,hdf5):
        """
        returns an array of records of type sens_data_dtype
        """
        sens_data_dtype=[('name','S20'),('mustar',float),('sigma',float),('mustar_conf',float),
                         ('is_shp',bool),('uncert_type','S15')]

        attrs=dict(puq.hdf.get_params_with_attrs(hdf5)) #keys are parameter names
        values=[]
        for data in sens:
            paramname=data[0]
            paramsens=data[1]
            paramuncert_type=None
            try:
                paramuncert_type=attrs[paramname]['uncert_type']
            except Exception:
                utilities.msg('uncert_type attribute not found for \'{}\' in {}'.format(paramname,
                              os.path.basename(hdf5)),'w')
            if objectmanager.ObjectManager.isShapeParam(paramname):
                s=paramname.split('__')
                name=s[1]+'_'+s[2]+s[3]
                isshp=True
            else:
                name=paramname
                isshp=False
            values.append((name,paramsens['ustar'],paramsens['std'],paramsens['ustar_conf95'],
                           isshp,paramuncert_type))

        return np.array(values,dtype=sens_data_dtype)

    @staticmethod
    def _plot_morrisplot_markerfacecolor(sens_data,sort_by_items=None):
        #fuzzy or fuzzy-const variables will be plotted with no fill
        #given sens_data (see _plot_sensdata2array), returns an nx4 matrix
        #where each row corresponds to an element in sens_data (a parameter).
        #the row is an RGBA tuple.
        #
        #i'th row in the return array corresponds to i'th parameter in
        #the array sens_data['name']. If a different ordering is needed,
        #pass the desired ordering as an array of parameter names in
        #sort_by_items
        if sort_by_items==None:
            sort_by_items=sens_data['name']
        sens_data_fill=np.zeros((np.size(sens_data['name']),4)) #RGBA
        for i,item in enumerate(sort_by_items):
            idx=np.argwhere(sens_data['name']==item)
            if len(idx)!=1:
                #since each var is unique there should only be 1 record from the
                #previous operation. The single record is contained in a nested numpy array.
                #for some reason: [[idx]]
                #use [0][0] to extract the index. Without this, all further operations with idx
                #will be returned in a nested array.
                raise Exception('error plotting morris plot. too many records')
            idx=idx[0][0]
            if 'fuzzy' in sens_data['uncert_type'][idx]:
                sens_data_fill[i]=(1,1,1,1)
            else:
                sens_data_fill[i]=(0,0,0,1)
        return sens_data_fill

    @staticmethod
    def _plot_morrisplot(sens_data,title='',labels=True,labels_top=20,
                         labels_bbox_hgt=.05,labels_bbox_wdt=.04,marker_override=False,
                         plot_top=None):
        """
        Plots the traditional morris plot.

        Parameters which correspond to fuzzy variables are plotted as hollow. This is to be
        able to differentiate fuzzy vs prob parameters at each alpha level.
        *marker_override* changes this behavior.
        """

        if plot_top!=None and plot_top<0:
            sens_data=np.sort(sens_data,order='mustar')[0:-plot_top][::-1]
        elif plot_top!=None and plot_top>0:
            sens_data=np.sort(sens_data,order='mustar')[::-1][0:plot_top]
        else:
            sens_data=np.sort(sens_data,order='mustar')[::-1]

        #put the parameters that are shapes and those that aren't into separate arrays
        #sens_data_nonshapes=sens_data[sens_data['is_shp']==False]
        #sens_data_shapes=sens_data[sens_data['is_shp']==True]

        sens_data_fill=SA._plot_morrisplot_markerfacecolor(sens_data)

        #get the top 20 values to show in the table
        sens_data_top=sens_data[0:20]

        gs=gridspec.GridSpec(1,2,width_ratios=[3.0,1.5])
        plt.figure()
        plt.subplot(gs[0])

        if marker_override:
            plt.plot(sens_data['mustar'],sens_data['sigma'],'ok')
        else:
            plt.scatter(sens_data['mustar'],sens_data['sigma'],marker='o',facecolors=sens_data_fill)

        if labels:
            lbl=[l if i<labels_top else '' for i,l in enumerate(sens_data['name'])]

            sens_data_mustar=sens_data['mustar']
            sens_data_sigma=sens_data['sigma']

            if abs(labels_top)>plot_top and plot_top!=None:
                labels_top=abs(plot_top)

            sens_data_mustar=sens_data_mustar[0:labels_top]
            sens_data_sigma=sens_data_sigma[0:labels_top]

            utilities.plot_datalabels(sens_data_mustar+np.random.random(len(sens_data_mustar))*1e-5,
                                      sens_data_sigma+np.random.random(len(sens_data_sigma))*1e-5,
                                      plt.gca(),lbl,labels_bbox_hgt,labels_bbox_wdt)

        plt.xlabel('$\\mu^*$',fontsize=16)
        plt.ylabel('$\\sigma$',fontsize=16)
        plt.title(title)
        plt.xlim(plt.gca().get_xlim()[0]*.97,plt.gca().get_xlim()[1])

        plt.subplot(gs[1])
        col_labels=['name','mu*','sigma']
        table_data=np.vstack((sens_data_top['name'],
                              ['{:.2e}'.format(x) for x in sens_data_top['mustar']],
                              ['{:.2e}'.format(x) for x in sens_data_top['sigma']])).T
        #http://stackoverflow.com/questions/8524401/how-can-i-place-a-table-on-a-plot-in-matplotlib
        #http://stackoverflow.com/questions/10388462/matplotlib-different-size-subplots
        plt.table(cellText=table_data,
                  colWidths = [0.60,0.47,0.47],
                  colLabels=col_labels,
                  loc='center',
                  bbox=[-0.15, 0, 1.18, 1])
        plt.gca().axis('off')
        plt.tight_layout(pad=0)

    @staticmethod
    def _plot_fuzzy(hdf5,baseshapes_file,alphacuts=None,alphacut_orderby=None,
                    print_alpha_rank=False,labels=True,labels_top=20,
                    labels_bbox_hgt=.05,labels_bbox_wdt=.04,plot=True,title_append='',mixed=False,
                    morris_plots=False,plot_top=None):
        """
        Plots the membership function of all outputs in a fuzzy run.

        - *hdf5*: a dictionary of filenames. Key is the alpha cut.
        - *alphacuts*: processes only the alpha cuts in this list.
        - *alphacut_orderby*: the alpha level to order the final results by. Must be a value
          contained in the *alphacuts* list. Defaults to the lowest available alpha level.
        - *print_alpha_rank*: prints the Morris indices at each alpha level.
        - *labels*: plots data labels.
        - *labels_top*: only displays the top n data labels, sorted by the sensitivity index. Default is 20.
        - *plot*: If False, only returns data
        - *morris_plots*: If False (default) does not show the morris plots at each alpha-cut. *Plot* must be
          True for this option to have an effect)
        - *title_append*: appends this text to any plot title.
        - *mixed*: used by :func:`SA._plotprobfuzzy` ONLY.
        - *plot_top*: Only plots the top x entries.
          x=None (plot all, default), x>0 (plot top x) x<0 (plots the
          bottom x).

        Returns the sensitivity indices in a dictionary of the form

            {$varName$:{$acut$:sens_indices, ...}, ...}

        where sens_indices is a numpy record array with columns equal to the names of the indices.
        There is an extra key named 'shapesdata' which contains the shapes for each alpha cut in
        the form of a list of shapely shapes.

            {'shapesdata':{$acut$:[shapes], $acut$:[shapes], ...}}
        """
        if not mixed:
            utilities.msg('Type of run: fuzzy only')
            marker_override=True
        else:
            marker_override=False

        #sort in reverse order so we can take advantage of the
        #nesting property of the alpha cuts.
        sorted_acuts=sorted(hdf5.keys(),reverse=True)

        if alphacuts==None:
            alphacuts=sorted_acuts
        else:
            alphacuts=sorted(alphacuts,reverse=True)
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
        # {$varName$:{$acut$:sens_indices, ...}, ...}
        plot_data={'shapesdata':{}}
        for output in outputs:
            utilities.msg('processing output variable {}'.format(output))
            plot_data[output]={}

            for acut in alphacuts:
                utilities.msg('a-cut:{}'.format(acut))
                sens=puq.hdf.get_sensitivity(hdf5[acut],output)
                sens_data=SA._plot_sensdata2array(sens,hdf5[acut])

                plot_data[output][acut]=sens_data

                if print_alpha_rank:
                    print('\tname\t\tmu*\t\tsigma\t\tmustar_conf')
                    for row in sens_data:
                        print('\t{}\t\t{:.5e}\t{:.5e}\t{:.5e}'.format(row['name'],
                              row['mustar'],row['sigma'],row['mustar_conf']))

                data=uq.UQ._plot_probabilistic(hdf5[acut],baseshapes_file,return_fullshapesdata=False,hist=False,
                            nbins=None,title_append=title_append,plot=False,msg=False)['shapesdata']
                plot_data['shapesdata'][acut]=data

                if plot and morris_plots:
                    SA._plot_morrisplot(sens_data,'{}, alpha:{}{}'.format(output,acut,title_append),labels,labels_top,
                                        labels_bbox_hgt,labels_bbox_wdt,marker_override,plot_top=plot_top)

            if plot:
                SA._plot_morris_summary(plot_data[output],alphacut_orderby,title=output+title_append,plot_top=plot_top)

                #plot summary graph
                SA._plot_morris_rhoplot(plot_data[output],labels,labels_top,
                                        labels_bbox_hgt,labels_bbox_wdt,marker_override,
                                        title=output+title_append,plot_top=plot_top)
        #end for output in outputs

        #put the parameters that are shapes and those that aren't into separate arrays
        #EDIT: don't need this for now
        #sens_data_nonshapes=sens_data[sens_data['is_shp']==False]
        #sens_data_shapes=sens_data[sens_data['is_shp']==True]

        #18-nov-14: changed the return type from a single dictionary with keys equal to alphacuts
        #to the return type given in the docstring.
        return plot_data

    @staticmethod
    def _plot_probfuzzy(hdf5,baseshapes_file,alphacuts=None,alphacut_orderby=None,
                    print_alpha_rank=False,labels=True,labels_top=20,
                    labels_bbox_hgt=.05,labels_bbox_wdt=.04,title_append='',plot=True,
                    morris_plots=False,plot_top=None):
        """
        Plots the membership function of all outputs in a fuzzy run.

        - *hdf5*: a dictionary of filenames. Key is the alpha cut. There should only be 1
          file name for each alpha cut.
        - *alphacuts*: processes only the alpha cuts in this list.
        - *alphacut_orderby*: the alpha level to order the final results by. Must be a value
          contained in the *alphacuts* list. Defaults to the lowest available alpha level.
        - *print_alpha_rank*: prints the Morris indices at each alpha level.
        - *labels*: plots data labels.
        - *labels_top*: only displays the top n data labels, sorted by the sensitivity index. Default is 20.
        - *plot*: If False, only returns data.
        - *title_append*: appends this text to any plot title.
        - *mixed*: used by :func:`UQ._plotprobfuzzy` ONLY.
        """
        utilities.msg('Type of run: mixed probabilistic-fuzzy')
        hdf5_2={}
        for k in hdf5.iterkeys():
            hdf5_2[k]=hdf5[k][0] #extract the filename from the list (there should only be 1)
        return SA._plot_fuzzy(hdf5_2,None,alphacuts=alphacuts,alphacut_orderby=alphacut_orderby,
                             print_alpha_rank=print_alpha_rank,labels=labels,labels_top=labels_top,
                             plot=plot,mixed=True, title_append=title_append,
                             labels_bbox_hgt=labels_bbox_hgt,labels_bbox_wdt=labels_bbox_wdt,
                             morris_plots=morris_plots,plot_top=plot_top)

    @staticmethod
    def _plot_morris_summary(plot_data,alphacut_orderby,plot_grid_sz=4,title='',plot_top=None):
        """
        Plots the morris indices for the first *plot_grid_sz* x *plot_grid_sz*
        parameters as a function of membership level.

        - *plot_data*: a dictionary with keys equal to alpha levels. Value is the
          output of the indices at that level as output by :func:`_plot_sensdata2array`.
        - *alphacut_orderby*: the alpha level to order the final results by. Must be a value
          contained in the keys of *plot_data*. Defaults to the lowest available alpha level.
        - *plot_grid_sz*: the number of rows and columns the plot will have.
        - *plot_top*: plots only this number of variables. If >0 plots the top variables, <0
          plots the bottom ones. The number of variables is limited to plot_grid_sz**2.
          Default is None which sets plot_top=plot_grid_sz**2.
        """
        alphacuts=plot_data.keys()

        #sort input params according to mustar of indicated acut
        #alphacuts[0] is the highest available acut
        if alphacut_orderby==None:
            plot_mustar_acut_sort=alphacuts[-1]
        else:
            if not alphacut_orderby in alphacuts:
                raise Exception('alpha level {} not in the alphacuts list'.format(alphacut_orderby))
            plot_mustar_acut_sort=alphacut_orderby

        #plot only the top or bottom n variables. n is given by plot_top
        if plot_top!=None and plot_top<0:
            #gets the bottom variables
            plot_top*=-1
            sens_data_top=np.sort(plot_data[plot_mustar_acut_sort],order='mustar')[0:plot_top]
            plot_top=min(plot_top,plot_grid_sz*plot_grid_sz)
            #gets the top ones out of those
            sens_data_top=sens_data_top[::-1][0:plot_top]
        elif plot_top!=None and plot_top>0:
            plot_top=min(plot_top,plot_grid_sz*plot_grid_sz)
            sens_data_top=np.sort(plot_data[plot_mustar_acut_sort],order='mustar')[::-1][0:plot_top]
        else:
            plot_top=plot_grid_sz*plot_grid_sz
            sens_data_top=np.sort(plot_data[plot_mustar_acut_sort],order='mustar')[::-1][0:plot_top]

        plt.figure()
        gs=gridspec.GridSpec(plot_grid_sz,plot_grid_sz)
        axs1=[None]*len(sens_data_top)
        axs2=[None]*len(sens_data_top)
        for i in range(len(sens_data_top)):
            ax1=plt.subplot(gs[i])
            ax2=ax1.twinx()
            axs1[i]=ax1
            axs2[i]=ax2

            #share both y axes across all subplots
            #http://stackoverflow.com/questions/12919230/how-to-share-secondary-y-axis-between-subplots-in-matplotlib
            axs1[0].get_shared_y_axes().join(axs1[i],axs1[0])
            axs2[0].get_shared_y_axes().join(axs2[i],axs2[0])

            #for the current variable, get all values from all available acuts
            #and plot them in the subplot
            curr_var=sens_data_top[i]['name']
            mustar=np.zeros(len(alphacuts))
            sigma=np.zeros(len(alphacuts))
            markerfill=np.zeros((len(alphacuts),4))
            for j,acut in enumerate(alphacuts):
                sens_params_alpha=plot_data[acut][plot_data[acut]['name']==curr_var]
                if len(sens_params_alpha)!=1:
                    #since each var is unique there should only be 1 record from the
                    #previous operation. The single record is contained in a numpy array.
                    #use [0] on each column to extract the value from the array.
                    raise Exception('error plotting morris summary. too many records')
                mustar[j]=sens_params_alpha['mustar'][0]
                sigma[j]=sens_params_alpha['sigma'][0]
                if 'fuzzy' in sens_params_alpha['uncert_type'][0]:
                    markerfill[j]=(1,1,1,1)
                else:
                    markerfill[j]=(0,0,0,1)

            ax2.scatter(alphacuts,mustar,marker='*',facecolors=markerfill,s=55)
            ax1.scatter(alphacuts,sigma,marker='o',facecolors=markerfill)

            #plot the sort-by value in red
            ax2.plot(plot_mustar_acut_sort,
                     plot_data[plot_mustar_acut_sort][plot_data[plot_mustar_acut_sort]['name']==curr_var]['mustar'],
                     '*r',markeredgecolor='red',zorder=1000)

            if ax1.get_ylim()[0]<=0:
                ax1.set_ylim(ax1.get_ylim()[0]*.2,ax1.get_ylim()[1])
            if ax2.get_ylim()[0]<=0:
                ax2.set_ylim(ax2.get_ylim()[0]*.2,ax2.get_ylim()[1])

            plt.title(curr_var)

        #adjust the labels. Need to do this after all axes have been joined
        #or else there will be extra labels.
        for i in range(len(axs1)):
            ax1=axs1[i]
            ax2=axs2[i]
            ax1.set_xlim(-0.05,1.05)
            if i>=plot_grid_sz*plot_grid_sz-(plot_grid_sz*plot_grid_sz-len(axs1))-plot_grid_sz:
                utilities.tickLabels_keepEvery(ax1,'x',2)
                ax1.set_xlabel(r'$\alpha$')
            else:
                ax1.set_xticklabels([])

            if i%plot_grid_sz==0:
                ax1.set_ylabel(r'$\sigma$ ($\bullet$)')
                utilities.tickLabels_keepEvery(ax1,'y',3)
            else:
                ax1.set_yticklabels([])
            if (i+1)%plot_grid_sz==0:
                ax2.set_ylabel(r'$\mu^*$ ($\star$)')
                utilities.tickLabels_keepEvery(ax2,'y',3)
            else:
                ax2.set_yticklabels([])
        plt.suptitle(title,fontsize=14)
        plt.tight_layout(pad=0,h_pad=0.2)
        plt.subplots_adjust(top=0.89)

        #so we can retrieve the title later
        plt.gcf().set_label(title)

    @staticmethod
    def _plot_morris_rhoplot(plot_data,labels=True,labels_top=20,labels_bbox_hgt=.05,
                             labels_bbox_wdt=.04,marker_override=False,title='',plot_top=None):
        """
        Plots the std dev of mu* across all memb levels vs E(mu*) for each parameter.

        - *plot_data*: a dictionary with the data to plot.

            {$varName$:{$acut$:sens_indices, ...}, ...}

        - *marker_override*: by default, fuzzy variables are plotted with hollow points
          and probabilistic with solid. Setting this to True plots everything as solid.
        """
        #'mustar': rank by avg(mustar) over all a-cuts. plot std(mustar) v avg(mustar)
        #'mustar_norm': rank by avg(mustar/max(mustar)) over all a-cuts. if max(mustar)==0
        #   at any a-cut, that parameter is ignored when calculating avg.
        #'rank': rank by avg(rank) where rank is the ranking position at an a-cut.
        rank_metric='mustar'

        #first, for each alpha cut, get the ranking of each parameter
        alphacuts=plot_data.keys()

        sens_data_0=plot_data[alphacuts[0]]

        #compute the rankings for each a-cut
        rankings_by_acut={}
        for acut in alphacuts:
            sens_data=plot_data[acut]
            #ties are assigned the min rank
            ranked=scipy.stats.rankdata(sens_data['mustar'],method='min')
            rankings_by_acut[acut]=[sens_data['name'],(np.size(ranked)+1)-ranked]

        #re-organize rankings_by_acut to group by parameter.
        #each key in rankings in the parameter name. The value is a list of rankings for that
        #value across all a-cuts
        rankings={}
        rankings_rankorder_for_print={}
        for acut in alphacuts:
            r=rankings_by_acut[acut]
            if rank_metric=='mustar':
                m=1
            if rank_metric=='mustar_norm':
                m=np.max(plot_data[acut]['mustar'])
                if m==0: m=np.nan #avoid a warning later on

            for i in range(len(r[0])):
                if not r[0][i] in rankings.keys():
                    rankings[r[0][i]]=[]
                    rankings_rankorder_for_print[r[0][i]]=[]
                rankings_rankorder_for_print[r[0][i]].append(r[1][i])
                if rank_metric=='mustar' or rank_metric=='mustar_norm':
                    rankings[r[0][i]].append(plot_data[acut][plot_data[acut]['name']==r[0][i]]['mustar'][0]/m)

        print('ranking by parameter and acut (unsorted). rows are parameters, columns are ranks\nacuts: {}'.format(alphacuts))
        for paramname,ranks in rankings_rankorder_for_print.iteritems():
            print('{}: {} (m{} s{})'.format(paramname,ranks,
                  np.round(np.mean(ranks),1),np.round(np.std(ranks),1)))

        if rank_metric=='rank':
            rankings=rankings_rankorder_for_print

        rankednames=rankings.keys()

        x=np.zeros(len(rankings))
        y=np.zeros(len(rankings))
        for i,ranks in enumerate(rankings.itervalues()):
            ranks=np.asarray(ranks)
            x[i]=np.mean(ranks[~np.isnan(ranks)])
            y[i]=np.std(ranks[~np.isnan(ranks)]) #this is the population std

            #alternative rank methods
#                maxrank=np.min(ranks)
#                x[i]=maxrank
#                y[i]=np.count_nonzero(ranks==maxrank)

#                m,c=scipy.stats.mode(ranks)
#                x[i]=m[0]
#                y[i]=c[0]

        #calculate ranking coefficients - dont use for mean/std method
        #final_rank[i] is a weighted sum where the mode is scaled to be between 0 and 1 and
        #the count is scaled between -1 and 1. This (tries) to give highest importance to
        #points falling in the upper right of the plot, medium to the ones in the lower and middle,
        #and lowest to the ones in the upper left.
#        final_rank=np.zeros(len(rankings))
#            for i in range(len(modes)):
#                final_rank[i]=(1-1/(len(rankings)-1.)*(modes[i]-1))*modes[i] + (1-2/(len(rankings)-1.)*(modes[i]-1))*counts[i]
#        x=final_rank

        #sort_idx=np.argsort(final_rank)[::-1] #use this one when using the other ranking methods
        sort_idx=np.argsort(x)[::-1] #use this one when ranking using mean and std
        if rank_metric=='rank':
            sort_idx=np.argsort(x)
        sorted_ranks_x=[]
        sorted_ranks_y=[]
        sorted_vars=[]
        print('name\tranking coeff.')
        for i in sort_idx:
            print('{}\t{}'.format(rankednames[i],x[i]))
            sorted_ranks_x.append(x[i])
            sorted_ranks_y.append(y[i])
            sorted_vars.append(rankednames[i])
        print('total number of parameters: {}'.format(len(sort_idx)))

        if plot_top!=None and plot_top<0:
            sorted_ranks_x=sorted_ranks_x[::-1][0:-plot_top]
            sorted_ranks_y=sorted_ranks_y[::-1][0:-plot_top]
            sorted_vars=sorted_vars[::-1][0:-plot_top]
        elif plot_top!=None and plot_top>0:
            sorted_ranks_x=sorted_ranks_x[0:plot_top]
            sorted_ranks_y=sorted_ranks_y[0:plot_top]
            sorted_vars=sorted_vars[0:plot_top]


        #for marker fills, get the uncert_type from the first available acut.
        #(the uncert_type wont change across memb levels.)
        #e.g., a parameter may change from fuzzy to fuzzy-const but never to prob.)
        markerfill=SA._plot_morrisplot_markerfacecolor(sens_data_0,sorted_vars)

        gs=gridspec.GridSpec(1,2,width_ratios=[3.0,1.5])

        plt.figure()
        plt.subplot(gs[0])
        if marker_override:
            plt.plot(sorted_ranks_x,sorted_ranks_y,'ok')
        else:
            plt.scatter(sorted_ranks_x,sorted_ranks_y,marker='o',facecolor=markerfill)

        if rank_metric=='mustar':
            plt.xlabel('$E(\\mu^*)$',fontsize=16)
            plt.ylabel('$\\sigma(\\mu^*)$',fontsize=16)
            col_labels=['name','E(u*)']
        if rank_metric=='mustar_norm':
            plt.xlabel('$E(\\mu^*/\\mu^*_{max})$',fontsize=16)
            plt.ylabel('$\\sigma(\\mu^*/\\mu^*_{max})$',fontsize=16)
            col_labels=['name','E(u*/max(u*))']
        if rank_metric=='rank':
            plt.xlabel('$E(\\rho)$',fontsize=16)
            plt.ylabel('$\\sigma(\\rho)$',fontsize=16)
            col_labels=['name','E(rho)']
            plt.gca().invert_xaxis()
            plt.xlim(len(sorted_vars)*1.05,-0.03)



        #add small random offsets to avoid overlapping labels
        if labels:
            if plot_top!=None and plot_top<0:
                sorted_ranks_x=sorted_ranks_x[::-1]
                sorted_ranks_y=sorted_ranks_y[::-1]
                sorted_vars=sorted_vars[::-1]

            lbl=[l if (i-1)<=labels_top else '' for i,l in enumerate(sorted_vars)]

            if plot_top!=None and labels_top>abs(plot_top):
                labels_top=abs(plot_top)

            x=sorted_ranks_x[0:labels_top]
            y=sorted_ranks_y[0:labels_top]

            utilities.plot_datalabels(x+np.random.random(len(x))*1e-5,
                                      y+np.random.random(len(y))*1e-5,plt.gca(),
                                      lbl,labels_bbox_hgt,labels_bbox_wdt)


        plt.subplot(gs[1])

        #get only top 20 for the table
        table_data=np.vstack((sorted_vars[0:20],
                              ['{:.2e}'.format(r) for r in sorted_ranks_x[0:20]])).T

        #http://stackoverflow.com/questions/8524401/how-can-i-place-a-table-on-a-plot-in-matplotlib
        #http://stackoverflow.com/questions/10388462/matplotlib-different-size-subplots
        plt.table(cellText=table_data,
                  colWidths = [0.50,0.47],
                  colLabels=col_labels,
                  loc='center',
                  bbox=[-0.22, 0, 1.35, 1])
        plt.gca().axis('off')

        plt.gcf().axes[0].title.set_text(title)
        plt.tight_layout()