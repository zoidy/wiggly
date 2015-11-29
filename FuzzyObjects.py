"""
This module contains all classes related to non-crisp objects.
"""

from FObjects import FObject
import numpy as np
import matplotlib.pyplot as plt
import utilities
import _distributions as distributions
import fuzz as fzy
import shapely.geometry

class FuzzyObject(FObject):
    """
    Base class for all fuzzy objects. Non instantiable.

    - *x,y*: numpy arrays of coordinates of the vertices of the object, each of size N.
    - *isClosed*: whether to consider the object as having an open or closed boundary.
    - *fuzzyVariables*: a list of TrapezoidalFuzzyNumbers corresponding to vertex coordinate
      offsets or edge offsets. See derived classes.
    """

    def __init__(self,x,y,fuzzyVariables,isClosed):
        if type(self) is FuzzyObject:
            raise NotImplementedError('FObject.__init__(): abstract class')

        FObject.__init__(self,x,y,isClosed)

        if np.size(x)!=np.size(y):
            raise Exception("the size of x and y must be the same!")
        if np.size(x)<1:
            raise Exception("Object needs at least 1 vertex.")
        if np.size(x)==1 and isClosed:
            raise Exception("A point cannot be closed.")

        self._fuzzyVariables=fuzzyVariables

        #the generated realizations, stored in a dictionary see the realzations property
        self._realizations=None

        #dictionary of shapely polygons
        self._realizationsPolygons={}

        #when it comes time to generate samples, stores the last
        #distribution and last method ('random','reducedtransformation') used
        self._lastDistr=None
        self._lastMethod=None

    @property
    def uncertainVertices(self):
        """
        Returns the array of uncertain vertices (vertex defined object) or edges (edge defined obj).

        The array contains a 1 at indices corresponding to vertices/edges which are uncertain.
        """
        unc_pts=np.ones(len(self._fuzzyVariables))
        for i,fnum in enumerate(self._fuzzyVariables):
            #if the support has width zero, its a constant
            if fnum.support[0]==fnum.support[1]:
                unc_pts[i]=0

        return unc_pts

    @property
    def realizations(self):
        """
        Returns all realizations.

        .. _fuzzyobjects.realizations:

        Returns the realizations of the *fuzzyVariables* (defined in the class constructor)
        at all alpha cuts given in
        in :ref:`generateRealizations <fuzzyobjects.generaterealizations>`  in the form of
        a dictionary with the keys set to alpha cuts. Each key has an associated value
        which is a list of numpy arrays.

        The lists each have length N where N is the number of fuzzy numbers in *fuzzyVariables*.
        The order of arrays in the list corresponds to the same order specified in
        *fuzzyVariables*.

        Each array contains samples of an individual *fuzzyVariable* which fall within the
        given alpha cut. The length of each array varies depending on how many realizations of the
        *fuzzyVariable* a particular alpha cut contains. The length of each array is determined
        by *n* from :ref:`generateRealizations <fuzzyobjects.generaterealizations>`

        For example, consider an EdgeDefinedObject with
        three fuzzy uncertain edges, and the fuzzy numbers corresponding to each edge are discretized
        into two alpha cuts (0.1 and 0.9).

        In this case the return value of this properity is a dictionary with two entries.
        Taking the first entry of the dictionary (alpha-cut=0.1), it consists of a list (of
        length 3, one for each fuzzy edge) of numpy arrays. The first entry in the list corresponds
        to the samples of the first fuzzy edge which are contained in the 0.1 alpha-cut.

        In table form, this looks like shown below:
        For each table, the sequence of column headers is a list. Each list element (ie. the
        columns in the table) is a numpy array

        DICTIONARY:

        Key={alphacut=0.1}
            ========== ========== ==========
            FuzzyEdge1 FuzzyEdge2 FuzzyEdge3
            ========== ========== ==========
            1.2        3.4        1
            -0.2       2.0        0.4
            4.1        0.6        1.7
            5.0        4.0        1.2
            ========== ========== ==========
        Key={alphacut=0.9}
            ========== ========== ==========
            FuzzyEdge1 FuzzyEdge2 FuzzyEdge3
            ========== ========== ==========
            4.1        3.4        0.4
            --          2.0        --
            ========== ========== ==========

        Note that for alphacut=0.9, some edges have fewer samples than others. This can happen
        due to different shapes of fuzzy numbers (ie. ones that are narrower at the top).
        """
        if self._realizations==None:
            raise Exception("no realizations have been generated yet!")

        return self._realizations

    @property
    def realizationsPolygons(self):
        """
        Abstract property: Returns the shapes of all realizations at all alpha cuts.

        .. _fuzzyobjects.realizationsPolygons:

        Returns a dictionary with keys equal to alpha cuts and values consisting all the
        realizations of the object belonging to that alpha cut in the form
        of Shapely shapes.

        Each dictionary entry has a key equal to one of the alpha cuts given
        in :ref:`generateRealizations <fuzzyobjects.generaterealizations>`. The value associated
        with each key is a list of Shapely shapes (Polygons, Points, LineStrings )
        which are the realizations corresponding to that particular alpha cut level.

        The generation of shapes from the samples of the individual fuzzy numbers, as given in
        *fuzzyVariables* in the constructor, proceeds as follows:

        1.  pick an alpha-cut, *ac*
        2.  From all fuzzy numbers at level *ac*, find the one with the largest number of samples within
            *ac*. Call this number *nmax_ac*.
        3.  Build a realization of the object at level *ac* by first selecting a sample from each fuzzy
            number at level *ac*. Then with the chosen set of samples, construct a Shapely shape.
        4.  Repeat step 3 *nmax_ac*  times.
        5.  Store the generated shapes (there are *nmax_ac*  of them) in a list.
        6.  Add the list from step 5 to a dictionary *d* and set the key equal to *ac*
        7.  Return to step 1, choosing a different alpha-cut
        8.  When all alpha-cuts have been picked, return *d*

        Note: since the algorithm generates *nmax_ac*  shapes for each alpha-cut,
        the total number of shapes generated will be greater than the value of *n* specified
        in :ref:`generateRealizations <fuzzyobjects.generaterealizations>`. The exact number
        depends on *n*, the number of alpha-cut levels, and the shape of the fuzzy numbers.`
        For each fuzzy edge. The upper bound for the number of shapes generated is
        *n x number-of-alphaCuts* although in reality, the number will be much less than that.

        Note 2: for the smallest alpha-cut value in *alphaCuts* specified in
        :ref:`generateRealizations <fuzzyobjects.generaterealizations>`, the number of shapes
        generated is always exactly equal to *n*, since in this case, *nmax_ac* = *n*
        """
        raise NotImplementedError("FuzzyObject.realizationsPolygons: abstract property")

    def getRealizations4Sim(self,acut):
        """
        Returns all realizations at alpha level *acut*.

        Returns a 2D numpy array where the rows are realizations and columns are the
        fuzzy variables. The order of the columns are in the same order as the *fuzzyVariables*
        parameter in the :class:`FuzzyObject` constructor. The number of rows is equal to the
        fuzzy number with the most samples at level *acut* and it depends not only on *acut*
        but also on the shape of the fuzzy numbers.

        Unlike
        :attr:`FuzzyObject.realizations` which returns all
        realizations in a raw form (each fuzzy variable may have a different number of
        realizations at a particular alpha-cut), this function returns realizations for a particular
        alpha cut in a form that is directly usable for generating polygons (all fuzzy variables
        are made to have the same number of realizations as the fuzzy variable with the most
        realizations).

        - *acut*: the alpha cut at which to return the realization. The value values of acut
          can be determined by realizations.keys().
        """

        #get all the realizations of the fuzzy variables at this alpha cut
        #fuzzyvariables is a list of numpy arrays
        fuzzyvariables=self.realizations[acut]

        #at this alpha cut, find the variable with the most realizations.
        #from generateRealizations, each edge is guaranteeed at least 1 realization
        #at each alpha cut.
        #This number is the number of shapes that will be generated for this alpha cut
        nmax_acut=0
        for i in range(len(fuzzyvariables)):
            if np.size(fuzzyvariables[i])>nmax_acut:
                nmax_acut=np.size(fuzzyvariables[i])

        #for each fuzzy var. in the alpha cut, pick a realization of the variable using
        #matchRandom or matchByRealization. Repeat nmax_acut times.
        returnArray=np.zeros((nmax_acut,len(fuzzyvariables)))
        for i in range(nmax_acut):

            #pick one or the other
            #realization=_realizations_matchRandom(fuzzyvariables)
            realization=self._realizations_matchByRealization(fuzzyvariables,i)

            returnArray[i,:]=realization

        return returnArray


    def generateRealizations(self,n,alphaCuts,method,shuffle=True):
        """
        Generates random samples of the fuzzy numbers given in *fuzzyVariables* (defined in
        the constructor) at a given membership level.

        .. _fuzzyobjects.generaterealizations:

        It is guaranteed that at least one sample will be generated for each
        *fuzzyVariable* at each alpha level.  Use the realizations property to get the results.

        - *n*: the total number of samples to generate at the LOWEST alpha
          level of each fuzzy number and must be >=2. These same samples will be used for the higher alpha
          levels in the following fashion:

          For a particular alpha level higher than the lowest, select samples from
          the lowest level which fall in the alpha cut interval of the particular level. This ensures
          that as the alpha level increases, the number of realizations in the associated interval
          decrease while maintaining randomness.

          This parameter is only used when *method* is 'random', 'linspace' or
          'reducedtransformation-subset'.
        - *alphaCuts*: a numpy array containing the alpha cuts at which samples will be genrated.
        - *method*: one of various methods to use. Can be 'random', 'linspace', 'reducedtransformation'.

          - random: randomly selects values at the lowest alpha level. Interval enpoints are included.
            A realization
            is generated by randomly matching values of all fuzzy variables at a particular alpha cut.
          - linspace: selects values based on equal spacing. Interval endpoints are included. A realization
            is generated by randomly matching values of all fuzzy variables at a particular alpha cut.
          - reducedtransformation: uses the reduced transformation method. Only interval endpoints
            are selected. Realizations are generated by a full factorial design (all possible
            combinations are considered) and the value of *n* is ignored
          - 'reducedtransformation-subset': same as 'reducedtransformation' except n is used to
            limit the total number of realizations to a random subset.

        - *shuffle*: Boolean which indicates whether to shuffle the generated realizations
          ('linspace' method only). Without shuffling, when building a shape out of the realizations
          all low values of the sampled fuzzy numbers will be matched to all low values and all
          high values are matched with all high values. This results in a poor coverage of the
          sample space. If not specified, it defaults to True.
          Setting it to false may be desirable
          for EdgeDefinedObjects in order to generate object whose fuzzy region is evenly spaced.
          For VertexDefinedObjects, this should
        """
        #After this function runs, _realizations will contain a dictionary with
        #keys of alphacut values and values of numpy arrays
        alphaCuts=np.asarray(alphaCuts)
        if np.size(alphaCuts)<1:
            raise Exception("need at least 1 alpha cut")
        if np.any(alphaCuts>1) or np.any(alphaCuts<0):
            raise Exception("{}. Alpha cuts must be in the interval [0,1]".format(alphaCuts))
        if n<2 and method!='reducedtransformation':
            raise Exception('need at least 2 realizations for method {}!'.format(method))
        if method=='reducedtransformation' and n!=None:
            raise Exception('Parameter n and method=reducedtransformation cannot both be specified in generateRealizations')

        self._lastDistr=None
        self._lastMethod=None

        #important
        alphaCuts.sort()
        self._realizationsPolygons.clear()

        self._realizations=self._realizations_generateRealizations(n,alphaCuts,
                                                               self._fuzzyVariables,
                                                               method,shuffle)

        self._lastDistr=distributions.DIST_UNIF
        self._lastMethod=method

    def plotFuzzyNumbers(self,n=-1,fuzzyNumbersLabels=[],showAlphaCuts=True,
                         showAcutRealizations=[],rlzColors=False,showMembFcn=True):
        """
        Plots each fuzzy number in *fuzzyVariables*. *fuzzyVarialbes* is defined in the constructor.

        .. _fuzzyobjects.plotfuzzynumbers:

        - *n*: Plots the first *n* fuzzy numbers into new figures. If n<1,
          all fuzzy numbers are plotted.
        - *labels*: list of labels to show in the plot, one for each fuzzy number.
        - *showAlphaCuts*: Shows the alphaCut levels as specified in
          :ref:`generateRealizations <fuzzyobjects.generaterealizations>`.
        - *showAcutRealizations*: A list containing alphaCuts to shows the realizations of.

          -If set to None, no realizations are shown.
          -If not specified, the realizations of all alphaCuts are shown.

        - *showMembFcn*: plots the membership function of each fuzzy number.

        Note: if *showAlphaCuts* is True and *showAcutRealizations*!=None, then
        :ref:`generateRealizations <fuzzyobjects.generaterealizations>` must have been
        previously called.
        """
        if self._realizations==None and (showAcutRealizations!=None or showAlphaCuts):
            raise Exception("plotFuzzyNumbers: No realizations have been generated yet!")

        if len(fuzzyNumbersLabels)==0:
            fuzzyNumbersLabels=[]
            for i in range(len(self._fuzzyVariables)):
                fuzzyNumbersLabels.append("Fuzzy number '{}'".format(i))

        if n<1:
            n=len(self._fuzzyVariables)

        plt.rc('font', serif=['Arial'])
        for i in range(n):
            fn=self._fuzzyVariables[i]

            plt.figure()

            acuts=np.linspace(0,1,11)
            nacuts=np.size(acuts)
            pts_x=np.zeros(nacuts*2)
            pts_y=np.zeros(nacuts*2)
            for l in range(nacuts):
                cut=fn.alpha(acuts[l])
                pts_x[l]=cut[0]
                pts_x[-l-1]=cut[1]
                pts_y[l]=acuts[l]
                pts_y[-l-1]=acuts[l]

            if showMembFcn:
                plt.plot(pts_x,pts_y,'k')

            s=fn.support
            k=fn.kernel
    #        plt.plot([s[0],k[0],k[1],s[1]],[0,1,1,0],'k')

            if showMembFcn:
                title="Supp:{} Kern:{}".format(np.round(s,4),np.round(k,4))
            else:
                title="Supp:N/A Kern:N/A"
            if showAlphaCuts:
                for acut in self._realizations:
                    ac=fn.alpha(acut)
                    plt.plot(ac,[acut,acut],'--',color='0.5')
            if showAcutRealizations!=None:
                if len(showAcutRealizations)==0:
                    showAcutRealizations=self._realizations.keys()
                for acut_rlz in showAcutRealizations:
                    edgeRealizations=self._realizations[acut_rlz][i]
                    if rlzColors:
                        #http://wiki.scipy.org/Cookbook/Matplotlib/Show_colormaps
                        cm=plt.get_cmap('Paired')
                        ax=plt.gca()
                        ax.set_color_cycle([cm(1.*cc/15) for cc in range(15)])
                        for rlz in edgeRealizations:
                            plt.plot(rlz,acut_rlz,'o')

                    else:
                        plt.plot(edgeRealizations,[acut_rlz]*len(edgeRealizations),'ko')

                min_acut=min(showAcutRealizations)
                title=title+". at a-cut {}, n={}".format(min_acut,np.size(self._realizations[min_acut][i]))

            print("Plot in figure {}: Fuzzy number '{}': Supp={} kern={}".format(plt.gcf().number,i,
                  np.round(s,4),np.round(k,4)))

            if showAcutRealizations!=None:
                if len(showAcutRealizations)==0:
                    showAcutRealizations=self._realizations.keys()
                showAcutRealizations.sort()
                total=0
                for j in range(len(showAcutRealizations)):
                    num=np.size(self._realizations[showAcutRealizations[j]][i])
                    print("\tfor a-cut={}, n={}".format(showAcutRealizations[j],num))
                    total+=num
                print("\tTotal n: {}".format(total))

            plt.ylim([-0.005,1.2])
            plt.title(title)
            plt.xlabel(fuzzyNumbersLabels[i])
            plt.ylabel("u")

    def plot(self,fignum=-1,alphaCuts=[],meanLine=True,color=True):
        """
        Plots the generated shapes.

        - *fignum*: the figure number to plot into. If fignum<1, plots into a new figure
        - *alphaCuts*: a list of alpha cuts to plot. All the values in this list must have a
          match in the *alphaCuts* parameter from
          :ref:`generateRealizations <fuzzyobjects.generaterealizations>`.
          If this parameter isn't specified, plots all alpha-cuts
        - *meanLine*: plots the undistorted shape.
        - *color*: If True, plots in color, if False, greyscale is used. If set to 'fixed'
          a single shade of grey is used for all membership values.
        """
        if len(alphaCuts)==0:
            alphaCuts=self.realizations.keys()

        self._plot_shapes(fignum,alphaCuts,self._x,self._y,self.realizationsPolygons,
                     self._isClosed,meanLine,color)

    def _realizations_generateRealizations(self,n,alphaCuts,fuzzyNumbers,method,shuffle=None):
        """
        Generates n realizations of each fuzzy number.
        - *n*: the number of realizations to generate (see *method* below)
        - *alphaCuts*: a list of numbers each in the range [0,1] indicating the alpha cuts
          to generate realizations for
        - *fuzzyNumbers*: a list of TrapezoidalFuzzyNumbers which will be sampled
        - *method*: If method='random' or 'linspace':

            - Realizations are generated at the lowest level alphacut.
              For higher level alpha cuts, the realizations of the loweset cut are filtered
              so that only the ones in the higher alpha cut are included. This means that
              a higher alpha cut's realizations are a subset of the lowest level alpha cut's realizations.
            - 'random': realizations are generated at the lowest alphacut randomly
            - 'linspace': realizations are generated at the lowest alphacut linearly spaced

          If method='reducedtransformation' or 'reducedtransformation-subset'

            - Realizations are generated at each alpha cut by generating all possible combinations
              of the interval endpoints for each fuzzy varable (2-level full factorial design).
              The value of n is ignored..
            - 'reducedtransformation-subset' same as 'reducedtransformation' except the value of n
              is used to limit the number of returned realizations. If n>the full factorial design,
              n=all possible combinations.

        - *shuffle*: Boolean which indicates whether to shuffle the generated realizations
          ('linspace' method only)
        """
        if method=='linspace':
            if shuffle==None:
                raise Exception("'linspace' method selected but shuffle not specified (must be true or false)")


        returnDictionary={}

        #for testing purposes only. When False and method==random or linspace
        #the endpoints of each alpha cut are not explicitly included
        #in the realizations. Normally this value should be True.
        #In shape plots, you may notice skewing of figures from lower left to upper right
        #this is due to including the end points since they are matched lowest to
        #lowest and highest to higest value.
        #setting this False works best with  method 'random'
        #set this            v     to False when needed.
        useEndptsForRandom= True if method=='random' else True
        if useEndptsForRandom and method=='random':
            utilities.msg("Note: 'random' method selected. Two of the sample points will always be on the" +
                            " upper and lower bound of each fuzzy number and therefore are not random.",
                            'v')

        ###generate the requested number of realizations
        #use the range of the lowest alpha cut
        lbounds=np.zeros(np.size(fuzzyNumbers))
        ubounds=np.zeros(np.size(fuzzyNumbers))
        minacut=min(alphaCuts)
        for i in range(np.size(fuzzyNumbers)):
            lbounds[i]=fuzzyNumbers[i].alpha(minacut)[0]
            ubounds[i]=fuzzyNumbers[i].alpha(minacut)[1]

        #generate uncorrelated realizations
        #realizations is an n-by-N numpy array
        if utilities.isEdgeDefinedObject(self):
            objtype="EdgeDefinedObj"
        if utilities.isVertexDefinedObject(self):
            objtype="VertexDefinedObj"
        utilities.msg("{} Fuzzy number realizations: method '{}'".format(objtype,method),'v')
        if method=='random':
            cor=np.eye(np.size(fuzzyNumbers))
            realizations=distributions.generateUniform(lbounds,ubounds,cor,n)
            utilities.msg("\tgenerated {} samples for {} fuzzy variables".format(n,np.size(lbounds)),'v')
        elif method=='linspace':
            realizations=np.zeros((n,np.size(fuzzyNumbers)))
            for i in range(np.size(fuzzyNumbers)):
                if lbounds[i]==ubounds[i]:
                    samples=np.r_[lbounds[i]]
                    step=0
                else:
                    samples,step=np.linspace(lbounds[i],ubounds[i],n,retstep=True)

                    #grab all the elements except the endpoints.
                    middle=samples[1:-1]

                    #Very important. without randomizing the evenly spaced values,
                    #when generating shapes using _realizations_matchByRealization, all the low values
                    #of all fuzzy number realizations will be matched with each other (same with high values).
                    #this results in wonky results since the fuzzy shape won't be sampled at combinations
                    #of low and high values. Eg., compare the result of _testVertexDefined with
                    #this line commented out. Note the narrow location of the top right edge. This
                    #phenomenon only appears for vertexdefinedobjects since for edge defined, there
                    #is only 1 fuzzy number per edge while for vertex defined, there are two.
                    #Shuffling shouldn't affect the results when using _realizations_matchRandom
                    #in realizationsPolygons
                    if shuffle:
                        np.random.shuffle(middle)

                    #add the interval endpoints as the first and second elements. This is so
                    #that when pairing samples of different variables. the interval lbound
                    #will always be paired with the lbound of all other variables. Same for ubound
                    samples=np.hstack((np.r_[samples[0],samples[-1]],middle))
                #end if

                utilities.msg("\tfuzzy variable {}: generated {} samples with step {}".format(i,n,step),'v')
                realizations[:,i]=samples
            #end for
        elif method=='reducedtransformation':
            return self._realizations_generateRealizations_reducedTrans(alphaCuts,fuzzyNumbers,False,None)
        elif method=='reducedtransformation-subset':
            return self._realizations_generateRealizations_reducedTrans(alphaCuts,fuzzyNumbers,True,n)
        else:
            raise Exception("method {} not recognized".format(method))

        ###


        #classify the realizations into the alpha cuts they fall into
        utilities.msg("\tFuzzy number realizations: post processing...",'v')

        #delete the first two rows since we need them for the end points.
        #realizations is guaranteed to have at least 2 rows
        if useEndptsForRandom:
            realizations=realizations[2:,:]
        for i in range(np.size(alphaCuts)):
            ac=alphaCuts[i]

            #each element of the list below will hold a numpy array of varying lengths where
            #each array contains the realizations that fall within alpha cut ac
            ac_realizations=[None]*np.size(fuzzyNumbers)

            for fn in range(np.size(fuzzyNumbers)):
                #this a fuzzypy RealRange
                cut_lbound=fuzzyNumbers[fn].alpha(ac)[0]
                cut_ubound=fuzzyNumbers[fn].alpha(ac)[1]

                #add the interval bounds to the realizations if useEndPtForRandom is set.
                #for both 'random' and 'linspace', this code ensures that the first and second
                #realizations in the array are the interval end points
                #1. this ensures that each interval will have at least 2 realizations,
                #   except for the degenerate (crisp) interval
                #2. If whatever function we are trying to find the membership value of is
                #   monotonic, including the intervals has a better chance to find the true memb. func.
                #   NOTE: this is not the reduced transformation method since it doesn't include all
                #   combinations of all the interval endpoints for all the fuzzy variables
                #   but only if we use _realizations_matchbyrealization
                #3. Only bother to add the remaining realizations if the interval isn't degenerate,
                #   This meeans for a crisp interval, there will only be 1 realization.

                #add the lower bound of the acut
                rlz=np.r_[cut_lbound]
                if cut_lbound!=cut_ubound:
                    #we don't have a degenerate (crisp) interval, add the upper bound as well.
                    rlz=np.r_[rlz,cut_ubound]

                    #extract the fn'th column and get only the rows of
                    #that column that  meet the criteria required to be included in the alpha cut
                    rlz=np.r_[rlz,realizations[(realizations[:,fn]>=cut_lbound) *
                                               (realizations[:,fn]<=cut_ubound),fn]]
                    if not useEndptsForRandom:
                        #reset rlz to not include the end points
                        rlz=realizations[(realizations[:,fn]>=cut_lbound)*(realizations[:,fn]<=cut_ubound),fn]
                        if np.size(rlz)<1:
                            #if there are no realizations at this alpha cut, add one
                            rlz=np.r_[(cut_lbound+cut_ubound)/2]

                ac_realizations[fn]=rlz
                #print("fuzzy number {} alpha cut {}".format(fn,ac))
                #print(rlz)
            #end for

            #append the results to the dictionary
            returnDictionary[ac]=ac_realizations
         #end for

        utilities.msg("\tFuzzy number realizations: post processing...Done",'v')


        return returnDictionary

    def _realizations_generateRealizations_reducedTrans(self,alphaCuts,fuzzyNumbers,subset,n):

        #generates all possible combinations of interval enpoints at each alpha cut
        #over all variables.
        returnDictionary={}

        if n!=None and n<2 and not subset:
            n=2
            print("reducedTransformation: the number of requested realizations was less than 2. It was reset to 2")

        for i in range(np.size(alphaCuts)):
            ac=alphaCuts[i]
            cuts_lbound=np.zeros(np.size(fuzzyNumbers))
            cuts_ubound=np.zeros(np.size(fuzzyNumbers))
            for fn in range(np.size(fuzzyNumbers)):
                #this a fuzzypy RealRange
                cuts_lbound[fn]=fuzzyNumbers[fn].alpha(ac)[0]
                cuts_ubound[fn]=fuzzyNumbers[fn].alpha(ac)[1]

            #generate all possible combinations of lower and upper bounds over all the
            #fuzzy variables for this alpha cut.
            a=utilities.DOE_ff2l(cuts_lbound,cuts_ubound)
            #remove duplicate rows. can happen when the lower and upper levels are teh same
            #for alpha cut 1
            a=utilities.unique_rows(a)

            #a now contains unique combinations of the inputs. E.g., for an edge defined
            #object with 5 sides, 1 of which is certain at a particular a-cut, there will be
            #5-1=4 variables therefore a contains 2^4=16 rows
            utilities.msg("reducedTransformation: generated 2^{}={} samples at a-cut {}".format(np.size(a,1),
                  np.size(a,0),ac),'v')

            if subset:
                #take only a random (without replacement) subset of all the possibilities
                n_subset=min(np.size(a,0),n)
                rand=np.random.choice(np.size(a,0),n_subset,replace=False)
                a=a[rand,:]
                utilities.msg("reducedTransformation: subset of FF DOE, requested:{} actual:{}".format(n,n_subset),
                              'v')

            #convert to list of 1d arrays since that is what other functions expect
            ac_realizations=[None]*np.size(fuzzyNumbers)
            for fn in range(np.size(fuzzyNumbers)):
                ac_realizations[fn]=a[:,fn]
            returnDictionary[ac]=ac_realizations

        return returnDictionary

    def _plot_shapes(self,fignum,alphaCuts,x,y,realizationsPolygons,isClosed,meanLine,color):
        #important so plotting with color works properly
        alphaCuts.sort()

        #get the shapes to plot -- dictionary
        all_shapes=realizationsPolygons
        numshapes=0

        if color:
            cm=plt.get_cmap('jet')
        else:
            cm=plt.get_cmap('binary')


        #get the shape type. assume all shapes are of the same type as the first one
        shape_the_first=all_shapes[alphaCuts[0]][0]
        if utilities.isShapelyPolygon(shape_the_first):
            shape_type=2
        elif utilities.isShapelyLineString(shape_the_first):
            shape_type=1
        elif utilities.isShapelyPoint(shape_the_first):
            shape_type=0
        else:
            raise Exception("Unsupported shape {}".format(str(type(shape_the_first))))

        for ac in alphaCuts:
            if fignum<0:
                f=plt.figure()
                fignum=f.number
            else:
                plt.figure(fignum)

            plt.axis('equal')

            #get the shapes for the given alpha-cut -- list
            ac_shapes=all_shapes[ac]
            numshapes+=len(ac_shapes)

            for shape in ac_shapes:
                if shape_type==2:
                    bdy=np.array(shape.exterior.coords)
                elif shape_type==0 or shape_type==1:
                    bdy=np.array(shape.coords)

                #plot it
                if color=='fixed':
                    clr='0.8'
                else:
                    #the colormap value 1 maps to 0 so multiply so that it's never 1
                    #also prevent the 0 ac from disappearing into white by setting it to 0.05
                    ac=0.05 if ac==0 else ac
                    clr=cm(ac*0.999999)
                if shape_type!=0:
                    plt.plot(bdy[:,0],bdy[:,1],'-',color=clr)
                else:
                    plt.plot(bdy[:,0],bdy[:,1],'.',color=clr)

            if meanLine:
                if isClosed:
                    plt.plot(np.concatenate((x,np.r_[x[0]])),
                             np.concatenate((y,np.r_[y[0]])),'--',color='0.6')
                else:
                    plt.plot(x,y,'--',color='0.6')

        utilities.msg("Plot in figure {}: Alpha cuts shown:{}, Total # of shapes:{}".format(plt.gcf().number,
                      alphaCuts,numshapes),'v')

    def _realizations_matchByRealization(self,rlzOfFuzzyVars,realizToMatch):
        """
        Given a list of numpy arrays, *rlzOfFuzzyVars*, this function returns a numpy
        array of samples from each
        array such that the samples were taken from the same row across all arrays. The return
        array is 1D has length equal to the length of *rlzOfFuzzyVars*.
        If a particular array has fewer rows than the rest, the sample is taken in
        a round robin fashion.

        - *rlzOfFuzzyVars*: a list of numpy arrays. Each array, i, has dimension
          (n_i,1) where n_i is the number of realizations corresponding to the i'th
          fuzzy variable.
        - *realizToMatch*: an integer specifying the row from which the sample will be taken.
          If n_i > realizToMatch, the sample is wrapped around to the beginning of array i.

        For example, consider 3 fuzzy variables, A, B, C. The first two have 3 realizations (i.e,
        their arrays have dimension (3,1)) and the third has 2 realizations. Let realiztToMatch=3.
        The array returned is
        [A_2,B_2,C_0] where the subscripts indicate the row in the arrays corresponding to the
        particular fuzzy variable.

        Subsequent calls to this function will yield the same result if the same values are
        passed in.
        """
        edgeOffsets=np.empty(len(rlzOfFuzzyVars))
        edgeOffsets.fill(np.nan)
        for j in range(len(rlzOfFuzzyVars)):
            edgeOffsets[j]=rlzOfFuzzyVars[j][realizToMatch%rlzOfFuzzyVars[j].size]
        #print(edgeOffsets)
        return edgeOffsets

    def _realizations_matchRandom(self,rlzOfFuzzyVars):
        """
        Given a list of numpy arrays, *rlzOfFuzzyVars*, this function returns a numpy
        array containing 1 sample from each array such that the samples selected randomly
        (with replacement). The return array is 1D has length equal to the length of
        *rlzOfFuzzyVars*.

        - *rlzOfFuzzyVars*: a list of numpy arrays. Each array, i, has dimension
          (n_i,1) where n_i is the number of realizations corresponding to the i'th
          fuzzy variable.

        For example, consider 3 fuzzy variables, A, B, C. The first two have 3 realizations (i.e,
        their arrays have dimension (3,1)) and the third has 2 realizations. The return array is
        a random sample of the realizations of A, B and C., eg. [A_0,B_0,C_1]. A subsequent call
        to this function may yield [A_2,B_0,C_0].
        """
        edgeOffsets=np.empty(len(rlzOfFuzzyVars))
        edgeOffsets.fill(np.nan)
        for j in range(len(rlzOfFuzzyVars)):
            edgeOffsets[j]=np.random.choice(rlzOfFuzzyVars[j])
        #print(edgeOffsets)
        return edgeOffsets

class VertexDefinedObject(FuzzyObject):
    """
    Represents a vertex-defined fuzzy object. The x and y coordinates of each vertex are
    treated as an individual fuzzy variable.

    - *x,y*: numpy arrays of coordinates of the vertices of the object, each of size N
    - *fuzzyVertices_x:* a list of fuzzpy TrapezoidalFuzzyNumbers corresponding to the fuzzy membership
      function of the x coordinate of each vertex.  The i'th fuzzy number in this list represents
      the i'th number in *x*.

      The membership function is in the form of an offset with respect to the i'th element in *x*.
      The fuzzy number should be centered at zero to avoid confusion but this is not required.
    - *fuzzyVertices_y:* same as *fuzzyVertices_x*.
    - *isClosed*: whether to consider the object as having an open or closed boundary
    """
    def __init__(self,x,y,fuzzyVertices_x,fuzzyVertices_y,isClosed=True):
        if len(fuzzyVertices_x)!=len(fuzzyVertices_y):
            raise Exception("the length of fuzzyVertices_x and fuzzyVertices_y must be the same!")
        if len(fuzzyVertices_x)!=np.size(x):
            raise Exception("The size of x and y must be the same as fuzzyVertices_x and fuzzyVertices_y")
        if np.size(x)<1:
            raise Exception("A vertex-defined object needs at least 1 vertex.")

        #convert the relative fuzzy numbers to absolute
        #could also do it in realizationsPolygons but its probably faster to do it here
        membFcn_x=utilities.fuzzyOffset2Absolute(fuzzyVertices_x,x)
        membFcn_y=utilities.fuzzyOffset2Absolute(fuzzyVertices_y,y)

        FuzzyObject.__init__(self,x,y,membFcn_x+membFcn_y,isClosed)

        for i in range(len(fuzzyVertices_x)):
            k=membFcn_x[i].kernel
            middleValue=k[0]+(k[1]-k[0])/2
            if x[i]!=middleValue:
                print("Warning: vertex {}, x-coord {} is not equal to the kernel midpoint {}".format(i,
                      x[i],k))
        for i in range(len(fuzzyVertices_y)):
            k=membFcn_y[i].kernel
            middleValue=k[0]+(k[1]-k[0])/2
            if y[i]!=middleValue:
                print("Warning: vertex {}, y-coord {} is not equal to the kernel midpoint {}".format(i,
                      y[i],k))

    @property
    def realizationsPolygons(self):
        """
        Returns a dictionary with keys equal to alpha cuts and values equal to list of shapes.

        See :ref:`realizationsPolygons <fuzzyobjects.realizationsPolygons>` in the parent class.
        """
        if self._realizations==None:
            raise Exception("No realizations have been generated yet!")
        if len(self._realizationsPolygons)>0:
            return self._realizationsPolygons

        utilities.msg("VertexDefinedObj: generating shapes",'d')

        #initialize the dictionary to the alpha-cuts which were given by the user
        shapes={key:None for key in self.realizations.keys()}

        counter=0
        for acut in self.realizations:
            #rows are realizations
            vertexOffsetsAll=self.getRealizations4Sim(acut)

            shapes_acut=[]
            numShapes=np.size(vertexOffsetsAll,0)
            for i in range(numShapes):
                #vertexOffsets contains a numpy array of length 2N, N=#of vertices
                vertexOffsets=vertexOffsetsAll[i,:]

                numvert=np.size(self._x)
                pts_x=vertexOffsets[0:numvert]
                pts_y=vertexOffsets[numvert:2*numvert]

                shape=utilities.pointslist2Shape(pts_x,pts_y,self._isClosed)

                shapes_acut.append(shape)

            #add the list of generated shapes at this alpha cut to the global list
            shapes[acut]=shapes_acut

            utilities.msg("\tAlpha-cut:{} -- {} shapes were  generated".format(acut,numShapes),'d')
            counter+=numShapes

        utilities.msg("\t{} shapes were generated over all alpha-cuts".format(counter),'d')

        self._realizationsPolygons=shapes
        return shapes

    def generateRealizations(self,n,alphaCuts,method):
        """
        Generates random samples of each coordinate of each fuzzy vertex. See
        :ref:`generateRealizations <fuzzyobjects.generaterealizations>`.
        """
        FuzzyObject.generateRealizations(self,n,alphaCuts,method,shuffle=True)

    def plotFuzzyNumbers(self,n=-1,showAlphaCuts=True,showAcutRealizations=[],rlzColors=False,showMembFcn=True):
        """
        Plots each fuzzy number corresponding to a vertex coordinate. See
        :func:`FuzzyObject.plotFuzzyNumbers`.
        """
        labelsx=[]
        labelsy=[]
        for i in range(len(self._fuzzyVariables)/2):
            labelsx.append("Vertex {}, x-coordinate".format(i))
            labelsy.append("Vertex {}, y-coordinate".format(i))

        FuzzyObject.plotFuzzyNumbers(self,n,labelsx+labelsy,showAlphaCuts,showAcutRealizations,
                                     rlzColors,showMembFcn)

class EdgeDefinedObject(FuzzyObject):
    """
    Represents an Edge-defined fuzzy object. Each fuzzy edge is treated as a separate
    fuzzy variable.

    - *x,y*: numpy arrays of coordinates of the vertices of the object, each of size N
    - *fuzzyEdges:* a list of fuzzpy TrapezoidalFuzzyNumbers corresponding to the fuzzy membership
      function of each edge. A fuzzy number represents the offset of the edge, e, formed
      by the i'th and (i+1)th vertices of the arrays *x,y*. The length of the list must be N if
      *isClosed* is true or N-1 if it is false.
    - *isClosed*: whether to consider the object as having an open or closed boundary
    """

    def __init__(self,x,y,fuzzyEdges,isClosed=True):
        if np.size(x)<2:
            raise Exception("An Edge Defined object needs at least 2 vertices.")
        if len(fuzzyEdges)<np.size(x) and isClosed:
            raise Exception("fuzzyEdges must be the same length as the number of edges for closed objects")
        if len(fuzzyEdges)<np.size(x)-1 and not isClosed:
            raise Exception("fuzzyEdges must be the same length as the number of edges in the object less 1 for open objects")

        FuzzyObject.__init__(self,x,y,fuzzyEdges,isClosed)

        #check to ensure max membership is centered at zero. If it's not, can still calculate
        #but results might cause confusion since the max membership won't be centered at the
        #mean line when using plot.
        for i in range(len(fuzzyEdges)):
            k=fuzzyEdges[i].kernel
            if not np.allclose((k[0]+k[1])/2,0):
                print("Warning: edge {} offset not equal to kernel midpoint.".format(i))

    @property
    def realizationsPolygons(self):
        """
        Returns a dictionary with keys equal to alpha cuts and values equal to list of shapes.

        See :ref:`realizationsPolygons <fuzzyobjects.realizationsPolygons>` in the parent class.
        """
        if self._realizations==None:
            raise Exception("No realizations have been generated yet!")
        if len(self._realizationsPolygons)>0:
            return self._realizationsPolygons

        utilities.msg("EdgeDefinedObj: generating shapes",'d')

        #initialize the dictionary to the alpha-cuts which were given by the user
        shapes={key:None for key in self.realizations.keys()}

        #build our base shape from which we will do all the offsetting
        #use shapely to fix the orientation, of the polygon to clockwise since
        #the offsetting function expects a clockwise orientation
        if len(self._x)>2:
            l=shapely.geometry.asLineString(np.vstack((self._x,self._y)).transpose())
            shape=shapely.geometry.Polygon(l)
            bdy=np.array(shape.exterior.coords)
        else:
            bdy=np.vstack((self._x,self._y)).transpose()

        counter=0
        for acut in self.realizations:
            #rows are realizations
            edgeOffsetsAll=self.getRealizations4Sim(acut)

            shapes_acut=[]
            numShapes=np.size(edgeOffsetsAll,0)
            for i in range(numShapes):
                #edgeOffsets contains a numpy array of length N, N=# of edges
                edgeOffsets=edgeOffsetsAll[i,:]

                #edgeOffsets now contains a numpy array of length N, N=#of edges
                #Each element corresponds to an one offset for an edge.
                #we can now build a shape
                buff=self._offset(bdy,edgeOffsets)
                if self._isClosed and len(self._x)>2:
                    shape=shapely.geometry.Polygon(buff)
                else:
                    shape=shapely.geometry.LineString(buff)

                shapes_acut.append(shape)

            #add the list of generated shapes at this alpha cut to the global list
            shapes[acut]=shapes_acut

            utilities.msg("\tAlpha-cut:{} -- {} shapes were  generated".format(acut,numShapes),'d')
            counter+=numShapes

        utilities.msg("\t{} shapes were generated over all alpha-cuts".format(counter),'d')

        self._realizationsPolygons=shapes
        return shapes

    def generateRealizations(self,n,alphaCuts,method):
        """
        Generates random samples of each coordinate of each fuzzy vertex. See
        :ref:`generateRealizations <fuzzyobjects.generaterealizations>`.
        """
        #set shuffle to true. If you want evenly spaced contours in
        #the boundaries, set to false.
        FuzzyObject.generateRealizations(self,n,alphaCuts,method,shuffle=True)

    def _testOffsetting(self,edgeOffsets):
        """
        Plots a single offset of the EdgeDefinedObject

        - *edgeOffsets*: a numpy array of offsets, one for each edge
        """
        #use shapely to fix the orientation, of the polygon to clockwise.
        if len(self._x)>2:
            l=shapely.geometry.asLineString(np.vstack((self._x,self._y)).transpose())
            shape=shapely.geometry.Polygon(l)
            bdy=np.array(shape.exterior.coords)
        else:
            bdy=np.vstack((self._x,self._y)).transpose()

        buff=self._offset(bdy,edgeOffsets)

        #plot it
        print("exterior offsetted:\n{}".format(buff))
        if self._isClosed and len(self._x)>2:
            shape=shapely.geometry.Polygon(buff)
            bdy=np.array(shape.exterior.coords)
        else:
            shape=shapely.geometry.LineString(buff)
            bdy=np.array(shape.coords)

        plt.plot(bdy[:,0],bdy[:,1],'-')
        if self._isClosed:
            plt.plot(np.concatenate((self._x,np.r_[self._x[0]])),
                     np.concatenate((self._y,np.r_[self._y[0]])),'--k')
        else:
            plt.plot(self._x,self._y,'--k')

        plt.axis('equal')

    def _getoffsetintercept(self,pt1, pt2, m, offset):
        """
        getoffsetintercept gets the b in y = mx + b needed to calculate the new point:

        From points pt1 and pt2 defining a line
        in the Cartesian plane, the slope of the
        line m, and an offset distance,
        calculates the y intercept of
        the new line offset from the original.
        """
        xy=self._getoffsetpoint(pt1,pt2,offset)

        #return the intercept
        return xy[1] - m * xy[0]

    def _getoffsetpoint(self,pt1, pt2,offset):
        """
        From points pt1 and pt2 defining a line
        in the Cartesian plane, and an offset distance,
        offsets, pt1 perpendicular to the line.
        Returns a numpy array with one tuple [(x,y)]
        """
        #find angle between line and the axis.
        theta = np.arctan2(pt2[1] - pt1[1],
                           pt2[0] - pt1[0])
        #rotate by 90
        theta += np.pi/2.0

        #calculate the amount of the offset corresponding to x and y and subtract it from the
        #original x y, thereby offseting the point
        x=pt1[0] - np.cos(theta) * offset
        y=pt1[1] - np.sin(theta) * offset

        #return the intercept
        return np.r_[(x,y)]

    def _getpt(self,pt1, pt2, pt3,offset12,offset23):
        """
        Gets intersection point of the two
        lines defined by pt1, pt2, and pt3;

        - pt1, pt2, pt3: Three points defining two lines (clockwise)
        - offset12, offset12: The offset distance of the line segment defined by pt1 and pt2 or pt2 and pt3.
        """

        # get first offset intercept
        m = (pt2[1] - pt1[1])/(pt2[0] - pt1[0])
        boffset = self._getoffsetintercept(pt1, pt2, m, offset12)
        # get second offset intercept
        mprime = (pt3[1] - pt2[1])/(pt3[0] - pt2[0])
        boffsetprime =self._getoffsetintercept(pt2, pt3, mprime, offset23)

        # get intersection of two offset lines, handling the case of vertical lines
        if m==float('inf') or m==float('-inf'):
            newx=pt1[0]+offset12*np.sign(m)
            newy=mprime*newx+boffsetprime
        elif mprime==float('inf') or mprime==float('-inf'):
            newx=pt2[0]+offset23*np.sign(mprime)
            newy=m*newx+boffset
        else:
            newx = (boffsetprime - boffset)/(m - mprime)
            newy = m * newx + boffset
        return newx, newy

    def _offset(self,shape, edgeOffsets):
        """
        Offsets the boundaries of a clockwise or counterclockwise shape.

        - *shape:* a N-by-2 numpy array repesenting the x,y coords of N vertices defined clockwise.
          For closed polygons, it is not necessary to make the first and last points equal.
        - *edgeOffsets:* a length N (or N-1 if the EdgeDefinedObject was created with *isClosed* set
          to False) numpy array of offset distances for each edge.
          A positive value moves the edge "to the left". For clockwise defined objects,
          this will result in an expansion of the object.

        Returns an N-by-2 numpy array representing the vertices of the offset polygon.

        Note in the current implementation, no extra processing
        is done to ensure that offset edges do not intersect other edges or go beyond the
        point where they disappear. This means this function does not behave like a true
        buffering operation.

        Also, no checking is done to make sure that there are no coincident points. If three
        consecutive vertices are colinear, they are ignored.
        """

        #based on http://pyright.blogspot.com/2011/02/simple-polygon-offset.html

        if shape.ndim!=2:
            raise Exception("poly must have dimension N-by-2")
        if np.size(shape,1)!=2:
            raise Exception("poly must have two columns")
        if edgeOffsets.ndim!=1:
            raise Exception("edgeOffsets must have dimension 1")
        if len(shape)<2:
            raise Exception("the object must have at least 2 vertices")

        #+number is shrink, -number is grow, so flip the sign of the given input
        edgeOffsets=-edgeOffsets

        #count the number of edges we have. If the last point equals the first,
        #ignore the last point
        nVerts=len(shape)
        if shape[0,0]==shape[-1,0] and shape[0,1]==shape[-1,1]:
            nVerts=len(shape)-1

        if nVerts==2:
            nEdges=1
        else:
            nEdges=nVerts

        #WARNING: this is a hack to make it work when the user specifies an
        #open shape and omits the entry for the "missing" edge in the edgeOffsets
        #the dummy value is not used in the calculation but it is needed so the loops work
        if not self._isClosed and len(edgeOffsets)==nEdges-1:
            edgeOffsets=np.hstack((edgeOffsets,0))

        if self._isClosed and len(edgeOffsets)<nEdges:
            raise Exception("The length of the uncertain edges array (={}) must be >= the number \
                            of edges (={}) for objects with closed boundaries".format(len(edgeOffsets),nEdges))
        if not self._isClosed and len(edgeOffsets)<(nEdges-1):
            raise Exception("The length of the uncertain edges array (={}) must be >= the number \
                            of edges (={}) for objects with open boundaries".format(len(edgeOffsets),nEdges-1))
        polyOffset = []
        if nEdges==1:
            pt=self._getoffsetpoint(shape[0],shape[1],edgeOffsets[0])
            polyOffset.append(pt)
            pt=self._getoffsetpoint(shape[1],shape[0],-edgeOffsets[0])
            polyOffset.append(pt)
        else:
            for i in range(nEdges):
                #for each edge, calculate the location of the point which corresponds
                #to the offset line
                pt_iminus1=shape[(i-1)%nVerts]
                pt_i=shape[i%nVerts]
                pt_iplus1=shape[(i+1)%nVerts]

                edgeOffsets_iminus1=edgeOffsets[(i-1)%nEdges]
                edgeOffsets_i=edgeOffsets[i%nEdges]

                #check for co-linear points
                if (pt_i[1]-pt_iminus1[1])*(pt_iplus1[0]-pt_i[0])!=(pt_iplus1[1]-pt_i[1])*(pt_i[0]-pt_iminus1[0]):
                    #for an open shape, make the end lines go all the way to the end of the original one
                    if i==0 and not self._isClosed:
                        #pt_iminus1 can be any point perpendicular to the line defined by shape[0]
                        #and shape[1], at location of the first point (shape[0])
                        #EXCEPTION: pt_iminus1 can't be equal to shape[0], else you will get
                        #a runtime warning when yo try to calculate the slope in _getpt
                        #Therefore can set the offset in _getoffsetpoint to any number except 0
                        pt_iminus1=self._getoffsetpoint(shape[0],shape[1],1)
                        edgeOffsets_iminus1=0
                    if i==nEdges-1 and not self._isClosed:
                        pt_iplus1=self._getoffsetpoint(shape[-1],shape[-2],1)
                        edgeOffsets_i=0

                    pt = self._getpt(pt_iminus1, pt_i, pt_iplus1,edgeOffsets_iminus1,edgeOffsets_i)
                    polyOffset.append(pt)
                else:
                    print("Warning [offsetting]: the edge with index {} was ignored because some of its \
                          constituent vertices are colinear with the constituent vertices of its \
                          neighboring edges".format(i))

            #calculate the last point of the last edge
            if self._isClosed:
                pt_i=shape[i%nVerts]
                pt_iplus1=shape[(i+1)%nVerts]
                pt_iplus2=shape[(i+2)%nVerts]
                edgeOffsets_i=edgeOffsets[i%nEdges]
                edgeOffsets_iplus1=edgeOffsets[(i+1)%nEdges]
                if (pt_iplus1[1]-pt_i[1])*(pt_iplus2[0]-pt_iplus1[0])!=(pt_iplus2[1]-pt_iplus1[1])*(pt_iplus1[0]-pt_i[0]):
                    pt = self._getpt(pt_i, pt_iplus1, pt_iplus2,edgeOffsets_i,edgeOffsets_iplus1)
                    polyOffset.append(pt)

        return np.array(polyOffset)

class VertexDefinedObjectFromValues(VertexDefinedObject):
    """
    A vertex-defined object, constructed from previously generated values.

    - *x*: A numpy array of x coordinates, length N
    - *y*: A numpy array of y coordinates, length N
    - *isClosed*: whether the object will be treated as having a closed boundary.
    - *samples_vertices_dict*: a dictionary holding samples of the fuzzy vertices.
      The first half of the array are the x vertex values and the other half the y.

        {<alpha-cut>:numpy array of length 2N, ...}

      Only 1 sample of each vertex is allowed per alpha cut.

    """
    def __init__(self,x,y,isClosed,samples_vertices_dict):
        #build a set of fake fuzzy vertices
        dummyvert=[]
        for i in range(np.size(x)):
            dummyvert.append(fzy.TrapezoidalFuzzyNumber())

        VertexDefinedObject.__init__(self,x,y,dummyvert,dummyvert,isClosed)

        #initialize the dictionary to the alpha-cuts which were given by the user.
        #and build the realizations dictionary. See the realizations property of
        #FuzzyObject
        self._realizations={key:None for key in samples_vertices_dict}
        for k in self._realizations:
            #build a list of 1 element arrays containing each sample value
            self._realizations[k]=[np.r_[val] for val in samples_vertices_dict[k]]


    def plotFuzzyNumbers(self,n=-1,showAlphaCuts=True,
                         showAcutRealizations=[],rlzColors=False):

        VertexDefinedObject.plotFuzzyNumbers(self,n=n,
                              showAlphaCuts=showAlphaCuts,rlzColors=rlzColors,
                              showAcutRealizations=[],showMembFcn=False)

    def generateRealizations(self,n,alphaCuts,method):
        raise NotImplementedError("generateRealizations isn't supported for this type of object")

class EdgeDefinedObjectFromValues(EdgeDefinedObject):
    """
    An Edge-defined object, constructed from previously generated values.

    - *x*: A numpy array of x coordinates, length N
    - *y*: A numpy array of y coordinates, length N
    - *isClosed*: whether the object will be treated as having a closed boundary.
    - *samples_edges_dict*: a dictionary with samples of the fuzzy edges

        {<alpha-cut>:numpy array of length N (closed object) or N-1 (open object), ...}

       Only 1 sample of each edge is allowed per alpha cut.

    """
    def __init__(self,x,y,isClosed,samples_edges_dict):
        #build a set of fake fuzzy edges
        if isClosed:
            numedges=np.size(x)
        else:
            numedges=np.size(x) - 1

        dummyedges=[]
        for i in range(numedges):
            dummyedges.append(fzy.TrapezoidalFuzzyNumber())

        EdgeDefinedObject.__init__(self,x,y,dummyedges,isClosed)

        #initialize the dictionary to the alpha-cuts which were given by the user.
        #and build the realizations dictionary. See the realizations property of
        #FuzzyObject
        self._realizations={key:None for key in samples_edges_dict}
        for k in self._realizations:
            self._realizations[k]=[np.r_[val] for val in samples_edges_dict[k]]


    def plotFuzzyNumbers(self,n=-1,fuzzyNumbersLabels=[],showAlphaCuts=True,
                         showAcutRealizations=[],rlzColors=False):

        EdgeDefinedObject.plotFuzzyNumbers(self,n=n,fuzzyNumbersLabels=fuzzyNumbersLabels,
                              showAlphaCuts=showAlphaCuts,rlzColors=rlzColors,
                              showAcutRealizations=[],showMembFcn=False)

    def generateRealizations(self,n,alphaCuts,method):
        raise NotImplementedError("generateRealizations isn't supported for this type of object")


##############################################################################
# testing
##############################################################################
def _testEdgeDefined1():
    #tests the offseting of a polygon's/polyline's edges

    #define a clockwise polygon. negative offsets shrink the bdy
    #positive grow it. If isClosed=False, you should get a polyline
#    pt_x=np.r_[480.,520,510,520,485]
#    pt_y=np.r_[123.,117,110,105,100]
#    edgeOffsets=np.r_[-3,0,0,-5,5]

    #define the same polygon but counterclockwise polygon and with an enpoint
    #equal to the start point. If isClosed is True, you should get exactly the
    #same result as the clockwise polygon
    #the EdgeDefined object should handle cw and ccw orientations and
    #the duplicated end point with no problem.
#    pt_x=np.r_[480.,485,520,510,520,480]
#    pt_y=np.r_[123.,100,105,110,117,123]
#    edgeOffsets=np.r_[-5,5,0,0,3]

    #define a more complex linestring with horizontal and vertical lines,
    #and a colinear point. counter clockwise
    pt_x=np.r_[480.,520,510,500,500,515,530,530,520,518]
    pt_y=np.r_[123.,117,110,110,100,100,110,115,123,123]
    edgeOffsets=np.r_[-2,-2,-2,-2,-2,2,0,-2,-2,-2]

    #define a line segment
#    pt_x=np.r_[490.,495]
#    pt_y=np.r_[110.,100]
#    edgeOffsets=np.r_[-3]

    #define a triangle. ccw
#    pt_x=np.r_[480.,485,520]
#    pt_y=np.r_[123.,100,105]
#    edgeOffsets=np.r_[-4,0,1]

    #define an open triangle
#    pt_x=np.r_[480.,520,510]
#    pt_y=np.r_[123.,117,110]
#    edgeOffsets=np.r_[-4,2]

    edo=EdgeDefinedObject(pt_x,pt_y,[fzy.TrapezoidalFuzzyNumber((0,0),(0,0))]*len(pt_x),False)
    edo._testOffsetting(edgeOffsets)

def _testEdgeDefined2():
    np.random.seed(969316594)

    #tests generating realizations of fuzzy numbers at any alpha level

    fn1=fzy.TrapezoidalFuzzyNumber((0.2, 0.2), (-3, 3))
    fn2=fzy.TrapezoidalFuzzyNumber((-4, -2), (-5, 3))

    alphas=np.r_[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    s=fn1.support
    k=fn1.kernel
    plt.subplot(211)
    plt.plot([s[0],k[0],k[1],s[1]],[0,1,1,0],'k')
    for i in range(np.size(alphas)):
        ac=fn1.alpha(alphas[i])
        plt.plot(ac,[alphas[i],alphas[i]],'--',color='0.5')
    plt.ylim([0,1.2])
    plt.xlim([-5.5,3.5])

    s=fn2.support
    k=fn2.kernel
    plt.subplot(212)
    plt.plot([s[0],k[0],k[1],s[1]],[0,1,1,0],'k')
    for i in range(np.size(alphas)):
        ac=fn2.alpha(alphas[i])
        plt.plot(ac,[alphas[i],alphas[i]],'--',color='0.5')
    plt.ylim([0,1.2])
    plt.xlim([-5.5,3.5])

    fuzzyEdges=[fn1,fn2]
    pt_x=np.r_[480.,520,510]
    pt_y=np.r_[123.,117,110]

    edo=EdgeDefinedObject(pt_x,pt_y,fuzzyEdges,isClosed=False)

    edo.generateRealizations(11,alphas,'random')
    rlz=edo.realizations

    #plot reazliations across the support of each fuzzy number as vertical lines
    acut=0.6
    acut_plot=acut if acut>=0.1 else 0.05
    plotarray=rlz[acut][0]
    plt.subplot(211)
    plt.title("acut:{} range:[{}] n:{}".format(acut,fuzzyEdges[0].alpha(acut),plotarray.size))
    for i in range(plotarray.size):
        plt.plot([plotarray[i],plotarray[i]],[0,acut_plot],'-',color='0.80')

    plotarray=rlz[acut][1]
    plt.subplot(212)
    plt.title("acut:{} range:[{}] n:{}".format(acut,fuzzyEdges[1].alpha(acut),plotarray.size))
    for i in range(plotarray.size):
        plt.plot([plotarray[i],plotarray[i]],[0,acut_plot],'-',color='0.80')

    #p=edo.realizationsPolygons

    edo.plot()


def _testEdgeDefined3():
    #tests generating realizations of shapes, given a set of fuzzy numbers
    #and alpha cuts

    np.random.seed(96931694)

    #define a clockwise polygon. negative offsets shrink the bdy
    #positive grow it.
    pt_x=np.r_[480.,520,510,520,485]
    pt_y=np.r_[123.,117,110,105,100]
    isclosed=True

    #define fuzzy numbers for all the edges.
    #trapezoidal fuzzy numbers are in the form
    #   (kernel_lower,kernel_upper), (support_lower,support_upper)
    edgeMembFcn=[fzy.TrapezoidalFuzzyNumber((0, 0), (0, 0)),
                 fzy.TrapezoidalFuzzyNumber((1, 1), (-3, 3)),
                 fzy.TrapezoidalFuzzyNumber((-2, 1), (-5, 7)),
                 fzy.TrapezoidalFuzzyNumber((-1, 1), (-3, 3)),
                 fzy.TrapezoidalFuzzyNumber((-0.5, 1), (-1, 1))]

    #a different shape
#    pt_x=np.r_[480.,520]
#    pt_y=np.r_[123.,117]
#    edgeMembFcn=[fzy.TrapezoidalFuzzyNumber((-1, 1), (-3, 3))]
#    isclosed=False


    edo=EdgeDefinedObject(pt_x,pt_y,edgeMembFcn,isclosed)
    edo.generateRealizations(500,np.r_[0,0.5,0.8,1],'linspace')

    plt.close('all')
    #edo.plotFuzzyNumbers()
    edo.plot(alphaCuts=[],color=True)

def _testVertexDefined():
    #tests generating realizations of shapes, given a set of fuzzy numbers
    #and alpha cuts

    np.random.seed(96931694)
    plt.close('all')

    #define a clockwise polygon.
    pt_x=np.r_[480.,520,510,520,485]
    pt_y=np.r_[123.,117,110,105,100]

    #define fuzzy numbers for all the vertices.
    #trapezoidal fuzzy numbers are in the form
    #   (kernel_lower,kernel_upper), (support_lower,support_upper)
    membFcn_x=[  fzy.TrapezoidalFuzzyNumber((0, 0), (0, 0)),
                 fzy.TrapezoidalFuzzyNumber((0, 0), (-2, 2)),
                 fzy.TrapezoidalFuzzyNumber((-2, 2), (-2, 2)),
                 fzy.TrapezoidalFuzzyNumber((-1, 1), (-1.5, 3)),
                 fzy.TrapezoidalFuzzyNumber((-0.5, 0.5), (-2, 1))]
    membFcn_y=membFcn_x

    vdo=VertexDefinedObject(pt_x,pt_y,membFcn_x,membFcn_y,True)
    vdo.generateRealizations(500,np.r_[0,0.5,0.8,1],method='linspace')
    shapes=vdo.realizationsPolygons

    #vdo.plotFuzzyNumbers(showAlphaCuts=False,showAcutRealizations=None)
    vdo.plotFuzzyNumbers()
    vdo.plot(alphaCuts=[])

def _testVertexDefined2():
    #tests generating realizations of shapes, given a set of fuzzy numbers
    #and alpha cuts

    np.random.seed(96931694)
    plt.close('all')

    #define a point
    pt_x=np.r_[520]
    pt_y=np.r_[117]

    #define fuzzy numbers for all the vertices.
    #trapezoidal fuzzy numbers are in the form
    #   (kernel_lower,kernel_upper), (support_lower,support_upper)
    membFcn_x=[fzy.TrapezoidalFuzzyNumber((0, 0), (-2, 2))]
    membFcn_y=[fzy.TrapezoidalFuzzyNumber((0, 0), (-3, 5))]

    vdo=VertexDefinedObject(pt_x,pt_y,membFcn_x,membFcn_y,False)
    vdo.generateRealizations(100,np.r_[0,0.5,0.8,1],method='reducedtransformation')
    shapes=vdo.realizationsPolygons

    vdo.plotFuzzyNumbers()
    vdo.plot(alphaCuts=[])
    plt.title('Fuzzy point. points are evaluation location, Color=membership.')
    plt.xlabel('X')
    plt.ylabel('Y')

if __name__=='__main__':
    #utilities.clearall()

    #_testVertexDefined()
    #_testVertexDefined2()

    #_testEdgeDefined1()
    #_testEdgeDefined2()
    _testEdgeDefined3()
    pass