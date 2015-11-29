"""
This module contains all classes related to crisp objects.
"""
import FObjects
import numpy as np
import matplotlib.pyplot as plt
import utilities
import _distributions as distributions

class CrispObject(FObjects.FObject):
    """
    Base class for all crisp objects. Not instantiable

    - *x,y*: Numpy arrays of x and y coordinates, each of length N
    - *cov*: A numpy 2n-by-2n matrix containing the variances and co-variances within and
      between points where n is the number of points. The rows and columns correspond
      to the x and y covariances in sequence. E.g., for 2 points x0, y0, x1, y1, the
      covariance  matrix's rows and columns are

      == == == == ==
      \  x0 x1 y0 y1
      == == == == ==
      x0
      x1
      y0
      y1
      == == == == ==

    - *unc_pts*: numpy array of length >= N. if the i'th position is 0, it indicates that the
      i'th in the x,y arrays has no uncertainty.
    - *isClosed*: Boolean indicating whether the object will be treated as having a closed boundary.
    """

    def __init__(self,x,y,cov,unc_pts,isClosed):
        if type(self) is CrispObject:
            raise NotImplementedError('FObject.__init__(): abstract class')

        FObjects.FObject.__init__(self,x,y,isClosed)

        if np.ndim(cov)!=2:
            raise Exception("The covariance matrix must have dimension 2")
        if np.size(cov,0)!=np.size(cov,1):
            raise Exception("The covariance matrix must be square")

        #http://stackoverflow.com/questions/5320324/testing-if-a-numpy-array-is-symmetric
        if not np.allclose(np.transpose(cov), cov):
            raise Exception("Covariance matrix must be symmetric")
        if np.any(np.diagonal(cov)<=0):
            raise Exception("Invalid covariance matrix. One of the diagonal entries is <=0")

        if np.size(unc_pts)>np.size(x):
            print("Warning: {} elements in unc_pts, expected {}. Ignoring extra ones".format(
                np.size(unc_pts),np.size(x)))
        if  np.size(unc_pts)<np.size(x):
            raise Exception("{} elements in unc_pts. Need at least {}".format(
                np.size(unc_pts),np.size(x)))

        self._unc_pts=unc_pts
        self._cov=cov

        #stores the genreated realizations in a 2D numpy array. The rows are realizations, columns
        #are: x0,...,xn,y0,...,yn of the distorted points
        self._realizations=None

        #list of shapely polygons
        self._realizationsPolygons=[]

        self._lastDistr=None

    @property
    def realizations(self):
        """
        Returns the vertices of the distorted object in two 2D numpy arrays x,y.
        The columns are the x or y coordinates for all vertices and the rows are the realizations.
        """
        if self._realizations==None:
            raise Exception("No realizations have been generated yet!")

        numpts=np.size(self._x)
        pts_x=self._realizations[:,0:numpts]
        pts_y=self._realizations[:,numpts:2*numpts]

        return pts_x, pts_y

    @property
    def realizationsPolygons(self):
        """
        Returns a list of Shapely shapes.
        (LineStrings if the shape is open, Point if it's a point).
        """
        objtype='RigidObj' if utilities.isRigidObject(self) else 'DeformableObj'
        utilities.msg('{}: generating shapes'.format(objtype),'v')
        if self._realizations==None:
            raise Exception("No realizations have been generated yet!")
        if len(self._realizationsPolygons)>0:
            return self._realizationsPolygons

        pts_x,pts_y=self.realizations

        n=np.size(pts_x,0)

        for i in range(n):
            shape=utilities.pointslist2Shape(pts_x[i],pts_y[i],self._isClosed)
            self._realizationsPolygons.append(shape)

        utilities.msg('\t{} shapes were generated'.format(n),'v')

        return self._realizationsPolygons

    @property
    def uncertainVertices(self):
        """
        Returns the array of uncertain vertices.

        The array contains a 1 at indices corresponding to vertices which are uncertain.
        """
        return self._unc_pts

    def generateNormal(self,n):
        """
        Abstract method which generates a set of realizations for a CrispObject using
        a multivariate normal. Derived classes may have addtional parameters.

        - *n*: the number of realizations to generate
        """
        raise NotImplementedError("CrispObject.generateNormal(): abstract method")
    def generateUniform(self,n):
        """
        Abstract method which generates a set of realizations for a CrispObject using
        a multivariate uniform distribution. Derived classes may have addtional parameters.

        - *n*: the number of realizations to generate
        """
        raise NotImplementedError("CrispObject.generateUniform(): abstract method")
    def generateLastUsed(self,n):
        """
        Abstract method which generates a set of realizations for a CrispObject using
        the last used distribution. Derived classes may have addtional parameters.

        - *n*: the number of realizations to generate
        """
        if self._lastDistr==None:
            raise Exception('must call generateNormal or generateUniform at least once before callling this!')

        if self._lastDistr==distributions.DIST_NORM:
            self.generateNormal(n)
        elif self._lastDistr==distributions.DIST_UNIF:
            self.generateUniform(n)
        else:
            raise Exception('{} distribution not known'.format(self._lastDistr))

    def plot(self,fignum=-1,meanLine=True,lines=True,points=False,n=-1):
        """
        Plots this object.

        - *fignum*: the figure to plot into
        - *meanLine*: plots the mean line
        - *lines*: If true connects each point with lines
        - *points*: If drue draws the generated points
        - *n*: the number of realizations (starting from the first one) to plot. If <1 plots all

        Plots into the given figure but doesn't show it until until matplotlib show() is called.
        """
        if self._realizations==None:
            raise Exception("No realizations have been generated yet! Nothing to plot.")

        if not lines and  not points:
            raise Exception("Nothing to plot. Lines and Points parameters are both false")

        if fignum<0:
            f=plt.figure()
            fignum=f.number
        else:
            plt.figure(fignum)
        plt.axis('equal')

        numvars=np.size(self._realizations,1)
        numrealizations=np.size(self._realizations,0)

        if n>=np.size(self._realizations,0):
            n=np.size(self._realizations,0)

        #make the loop easier to read
        if n<1:
            n=numrealizations+1

        #plot each row. Split the array in half (down the columns) since X and Y are
        #arranged like that in the data.
        if lines:
            for i in range(min(numrealizations,n)):
                plt.plot(self._realizations[i,0:numvars/2],self._realizations[i,-numvars/2:],
                         '-',color='0.80')
                if self._isClosed:
                    #need a bunch of np.newaxis b/c slicing numpy arrays takes away dimensions.
                    plt.plot(np.concatenate((self._realizations[i,numvars/2-1,np.newaxis],self._realizations[i,0,np.newaxis])),
                             np.concatenate((self._realizations[i,-1,np.newaxis],self._realizations[i,numvars/2,np.newaxis])),
                             '-',color='0.80')

        if points:
            for i in range(min(numrealizations,n)):
                plt.plot(self._realizations[i,0:numvars/2],self._realizations[i,-numvars/2:],
                    'x',color='0.2')

        if meanLine:
            if self._isClosed:
                plt.plot(np.concatenate((self._x,np.r_[self._x[0]])),
                         np.concatenate((self._y,np.r_[self._y[0]])),'--k')
            else:
                plt.plot(self._x,self._y,'--k')

    def plotStats(self,fignum=None,labels=[]):
        """
        Outputs  statistics for the first 3 variables. The plot is not shown
        until matplotlib show() is called

        - fignum: the figure in which to plot the stats. If <0 plots into a new figure
        - labels: a list of length <=3 with labels to use for the statsOfPoints plot
        """
        if self._realizations==None:
            raise Exception("No realizations have been generated yet! Nothing to plot.")

        if fignum is None:
            f=plt.figure()
            fignum=f.number
        else:
            plt.figure(fignum)

        plt.axis('equal')

        numvars=np.size(self._realizations,1)
        numpointstoplot=np.min([3,numvars/2])
        var_x=self._realizations[:,0:numpointstoplot]
        var_y=self._realizations[:,numvars/2:(numvars/2+numpointstoplot)]
        data=np.hstack((var_x,var_y))

        #build the labels
        for i in range(numpointstoplot):
            labels.append("x"+str(i))
        for i in range(numpointstoplot):
            labels.append("y"+str(i))

        print("\n"+str(type(self)).replace("<class '","").replace("'>",""))
        print("-----------------")
        print("Statistics for the first {} points".format(numpointstoplot))
        print("means {}:".format(labels))
        print(np.mean(data,axis=0)) #calculate the mean along the rows (ie down the columns)
        cor_err=False
        try:
            print("corr & cov (columns/rows are {}".format(labels))
            print(np.corrcoef(data,rowvar=0))
        except:
            print('Warning: Couldn\'t compute correlation coefficients. Probably not all vertices are uncertain')
            cor_err=True
        print("covariance matrix")
        print(np.cov(data,rowvar=0))

        #plot
        if not cor_err:
            utilities.pairs(data=data,figurenum=fignum,labels=labels,rankcorr=True)
        else:
            print('Error: can\'t make pairs plot since at least one vertex has covariance of zero')

class RigidObject(CrispObject):
    """
    Represents a crisp, rigid object.

    - *x, y*: numpy arrays of coordinate values
    - *origin_x, origin_y*: the x,y coords of the rotation/translation origin
    - *theta*: the mean of the rotation angle in degrees
    - *cov*: A numpy 3-by-3 matrix containing the variances and co-variances of the center of
      rotation and translation

      =====   == == =====
      \       x0 y0 theta
      =====   == == =====
      x0
      y0
      theta
      =====   == == =====

    - *unc_pts*: numpy array of 0 or 1 of length N indicating whether point i is to be considered
      uncertain or not. 1=uncertain
    - *isClosed*: whther the object has a closed boundary
    """

    def __init__(self,x,y,origin_x, origin_y,theta,cov,unc_pts,isClosed=True):
        CrispObject.__init__(self,x,y,cov,unc_pts,isClosed)

        if cov.shape!=(3,3):
            raise Exception("The covariance matrix for a rigid object must be 3x3")

        self._origin_x=origin_x
        self._origin_y=origin_y
        self.__theta=theta

        #stores the realizations of the translation an rotation parameters. Columns
        #are: origin_x, origin_y, theta
        self._realizationsParams=0


    def _distortObject(self):
        """
        From the realizations of the parameters, generate the realizations of the
        distorted object.
        """
        n=np.size(self._realizationsParams,0)

        #perform an affine transformation (rotation and translation)
        self._realizations=np.zeros((n,self._x.size+self._y.size))
        for i in range(n):
            x=self._realizationsParams[i,0]
            y=self._realizationsParams[i,1]
            theta=self._realizationsParams[i,2]

            output=utilities.transformPoints(self._x,self._y,
                                      x,y,
                                      theta,
                                      self._origin_x,self._origin_y)

            #add the result to the array of points
            self._realizations[i,:]=output.reshape((1,-1))

        #reset any points not considered uncertain to the mean values
        for i in range(np.size(self._unc_pts)):
            if self._unc_pts[i]==0:
                self._realizations[:,i]=self._x[i]
                self._realizations[:,self._x.size+i]=self._y[i]

    @property
    def origin(self):
        """
        Returns the origin of this rigid object as a tuple (origin_x, origin_y).
        """
        return(self._origin_x,self._origin_y)

    @property
    def realizationsParams(self):
        """
        Returns the realizations of the rigid object's parameters.

        Unlike :func:`CrispObject.realizations` which returns the realizations
        of the x and y coordinates of each vertex, *realizationsParams* returns
        the rigid object's x, y, and theta realizations which apply to all vertiecs
        in the object.

        The return value are 3 numpy arrays, x, y, and theta. The values in
        each array are the realizations for that variable.
        """
        x=self._realizationsParams[:,0]
        y=self._realizationsParams[:,1]
        t=self._realizationsParams[:,2]
        return x,y,t

    def generateNormal(self,n,translate=True,rotate=True):
        """
        Generates a set of realizations for a rigid object using a multivariate normal.

        .. _crispobjects.rigidobject.generatenormal:

        - n: the number of realizations to generate
        - translate: enable translational uncertainty. If false, the x and y entries in the covariance matrix will be ignored.
        - rotate: enable rotational uncertainty. If false, the theta entries in the covariance matrix will be ignored

        Note: If there has been a previous call to any of the generate\* functions, if translate or
        rotate is false, the previously generated values will be used. This means that
        upon subsequent calls, the genrated values won't be correlated with the values from the
        previous run.
        If there have been no previously generated values, the mean values will be used.
        E.g.,

        - 1st run: translate=True, rotate=False
            The x and y values of translation are correlated according to the cov matrix.
            All realizations of the rotation angle are equal to the mean value
        - 2nd run: translate=False, rotate=True
            The x and y values from run 1 are used and randomly generated values of theta
            are used according to the variance of theta specified in cov. x,y are correlated
            but are uncorrelated with theta, regardless of what is in the cov matrix.

        The advantage to doing things this way is that we can have different distributions for
        translation and rotation.
        """
        utilities.msg("CrispObj realizations for x,y,theta: dist 'normal'",'v')

        if not translate and not rotate:
            raise Exception("You must specify at least one of translational or rotational uncertainty")
        if n<1:
            raise Exception("At least 1 realization is needed.")

        self._lastDistr=None

        #clear the list
        self._realizationsPolygons=[]

        #initialize the results to the mean values
        self._realizations=np.zeros((n,2*np.size(self._x)),dtype='f8')
        self._realizationsParams=np.zeros((n,3),dtype='f8')

        for i in range(n):
            self._realizations[i,:]=np.concatenate((self._x,self._y))
            self._realizationsParams[i,:]=np.r_[self._origin_x,self._origin_y,self.__theta]

        realizations=distributions.generateNormal(np.r_[self._origin_x,self._origin_y,self.__theta],
                                                 self._cov,n)

        #only copy over the values for variables that are uncertain. Since we initialized
        #the realization arrays to the mean in __init__, the arrays will retain the
        #mean values unless specifically set.
        if rotate==True:
            self._realizationsParams[:,2]=realizations[:,2]
        if translate==True:
            self._realizationsParams[:,0]=realizations[:,0]
            self._realizationsParams[:,1]=realizations[:,1]

        self._distortObject()
        self._lastDistr=distributions.DIST_NORM

        #used by objectmanager
        self._variances=np.diag(self._cov)
        self._means=np.r_[self._origin_x,self._origin_y,self.__theta]

        utilities.msg('\tgenerated {} samples for {} vertices'.format(n,np.size(self._x)),'v')

    def generateUniform(self,n,translate=True,rotate=True):
        """
        Generates a set of realizations using a multivariate uniform.

        - n: the number of realizations to generate
        - translate: enable translational uncertainty. If false, the x and y entries in the covariance matrix will be ignored.
        - rotate: enable rotational uncertainty. If false, the theta entries in the covariance matrix will be ignored.

        Note on the covariance matrix:
            This matrix is specified in the same way
            as for normally distributed variables.
            The covariances are used to calculate the correlation coefficient, which
            here, is interpreted as Spearman's (rank) correlation. Therefore, the outputs
            will be rank correlated.

        See :ref:`generateNormal <crispobjects.rigidobject.generatenormal>` for a note about
        subsequent calls.
        """
        utilities.msg("CrispObj realizations for x,y,theta: dist 'uniform'",'v')

        if not translate and not rotate:
            raise Exception("You must specify at least one of translational or rotational uncertainty")
        if n<1:
            raise Exception("At least 1 realization is needed.")

        self._lastDistr=None
        self._realizationsPolygons=[]

        #initialize the results to the mean values
        self._realizations=np.zeros((n,2*np.size(self._x)),dtype='f8')
        self._realizationsParams=np.zeros((n,3),dtype='f8')

        for i in range(n):
            self._realizations[i,:]=np.concatenate((self._x,self._y))
            self._realizationsParams[i,:]=np.r_[self._origin_x,self._origin_y,self.__theta]

        #the range of variation of the uniform distr is calculated from the
        #definition of the variance for the uniform distr
        var=np.diag(self._cov)
        mean=np.r_[self._origin_x,self._origin_y,self.__theta]
        bound_lower=mean-np.sqrt(12*var)/2.
        bound_upper=mean+np.sqrt(12*var)/2.
        cor= utilities.cov2cor(self._cov)
        realizations=distributions.generateUniform(bound_lower,bound_upper,cor,n)


        #only copy over the values for variables that are uncertain. Since we initialized
        #the realization arrays to the mean in __init__, the arrays will retain the
        #mean values unless specifically set.
        if rotate==True:
            self._realizationsParams[:,2]=realizations[:,2]
        if translate==True:
            self._realizationsParams[:,0]=realizations[:,0]
            self._realizationsParams[:,1]=realizations[:,1]

        self._distortObject()
        self._lastDistr=distributions.DIST_UNIF

        #used by objectmanager
        self._variances=var
        self._means=mean
        self._bound_lower=bound_lower
        self._bound_upper=bound_upper

        utilities.msg('\tgenerated {} samples for {} vertices'.format(n,np.size(self._x)),'v')

    def plotStats(self,fignum=None,statsOfPoints=False,labels=[]):
        """
        Outputs some statistics for the 3 generated parameters. The plot is not shown
        until matplotlib show() is called

        - *fignum*: the figure in which to plot the stats. If <0 plots into a new figure
        - *statsOfPoints*: Generates a plot with the statisistics of the first 3 points
          instead of origin_x, origin_y, and theta.
        - *labels*: a list of length <=3 with labels to use for the statsOfPoints plot
        """

        if statsOfPoints==False:
            numpointstoplot=3
            data=self._realizationsParams
            if len(labels)==0:
                labels=["origin_x","origin_y","theta (Deg)"]

            print("\n"+str(type(self)).replace("<class '","").replace("'>",""))
            print("-----------------")
            print("Statistics for the first {} points".format(numpointstoplot))
            print("means {}:".format(labels))
            print(np.mean(data,axis=0)) #calculate the mean along the rows (ie down the columns)
            print("correlation (columns/rows are {}".format(labels))
            print(np.corrcoef(data,rowvar=0))
            print("covariance matrix")
            print(np.cov(data,rowvar=0))
            utilities.pairs(data=data,figurenum=fignum,labels=labels)
        else:
            CrispObject.plotStats(self,fignum,labels)

class RigidObjectFromValues(RigidObject):
    """
    A rigid object constructed out of a previously generated set of realizations of
    x,y, and theta.

    - *x, y*: numpy arrays of coordinate values of the original shape.
    - *origin_x, origin_y*: the x,y coords of the rotation/translation origin.
    - *theta*: the mean of the rotation angle in degrees.
    - *unc_pts*: numpy array of 0 or 1 of length N indicating whether point i is to be considered
      uncertain or not. 1=uncertain.
    - *isClosed*: whther the object has a closed boundary
    - *samples_x,samples_y,samples_t*: numpy arrays containing samples of the x,y coords and
      rotation angle theta, of the origin.

    This class is used by :class:`ObjectManager`.
    """
    def __init__(self,x,y, origin_x, origin_y,unc_pts,isClosed,
                 samples_x,samples_y,samples_t):

        #build a fake covariance matrix. It doesn't matter for this object
        #for rigid objects, its always a 3x3
        cov=np.eye(3)

        RigidObject.__init__(self, x,y, origin_x,origin_y,
                             np.nan,cov,unc_pts,isClosed)

        #stores the realizations of the translation an rotation parameters. Columns
        #are: origin_x, origin_y, theta
        self._realizationsParams=np.vstack((samples_x,samples_y,samples_t)).T

        self._distortObject()

    def generateNormal(self,n,translate=True,rotate=True):
        raise NotImplementedError("CrispObject.RigidObjectFromValues(): this method is not supported for this object type")

    def generateUniform(self,n,translate=True,rotate=True):
        raise NotImplementedError("CrispObject.RigidObjectFromValues(): this method is not supported for this object type")

class DeformableObject(CrispObject):
    """
    Represents a crisp, deformable object.

    - *x*: A numpy array of x coordinates, length N
    - *y*: A numpy array of y coordinates, length N
    - *cov*: A numpy 2n-by-2n matrix containing the variances and co-variances within and
      between points where n is the number of points. The rows and columns correspond
      to the x and y covariances in sequence. E.g., for 2 points x0, y0, x1, y1, the
      covariance  matrix's rows and columns are

      == == == == ==
      \  x0 x1 y0 y1
      == == == == ==
      x0
      x1
      y0
      y1
      == == == == ==

    - *unc_pts*: numpy array of length >= N. if the i'th position is 0, it indicates that point in the x,y arrays should not be considered uncertain.
    - *isClosed*: whether the object will be treated as having a closed boundary
    """

    def __init__(self,x,y,cov,unc_pts,isClosed=True):
        if np.mod(np.size(cov,1),2)!=0:
            raise Exception("the number of columns in the covariance matrix must be even")

        CrispObject.__init__(self,x,y,cov,unc_pts,isClosed)


    def generateNormal(self,n):
        """
        Generates a set of realizations for a deformable object using a multivariate normal.

        - *n*: the number of realizations to generate
        """
        utilities.msg("DeformableObj realizations for x,y: dist 'normal'",'v')

        #the column i in the output are the random points that corresponding
        #to point i in the arrays x and y

        self._lastDistr=None
        self._realizationsPolygons=[]

        self._realizations=distributions.generateNormal(np.concatenate((self._x,self._y)),
                                                         self._cov,n)

        #adjust the output for points which are not uncertain: For those points, set all
        #the realizations to be equal to the specified x and y values.
        numvars=np.size(self._realizations,1)
        for i in range(len(self._unc_pts)):
            if self._unc_pts[i]==0:
                self._realizations[:,i]=self._x[i]
                self._realizations[:,numvars/2+i]=self._y[i]

        self._lastDistr=distributions.DIST_NORM

        #used by objectmanager
        self._variances=np.diag(self._cov)
        self._means=np.concatenate((self._x,self._y))


        utilities.msg('\tgenerated {} samples for {} vertices'.format(n,np.size(self._x)),'v')

    def generateUniform(self,n):
        """
        Generates a set of realizations using a multivariate uniform.

        - *n*: the number of realizations to generate

        Note on the covariance matrix.
            This matrix is specified in the same way
            as for normally distributed variables.
            The covariances are used to calculate the correlation coefficient, which
            here, is interpreted as Spearman's (rank) correlation. Therefore, the outputs
            will be rank correlated.
        """
        utilities.msg("DeformableObj realizations for x,y: dist 'uniform'",'v')

        self._lastDistr=None

        #the x vars are var[0:var.size/2]
        #the y vars are var[var.size/2:var.size]
        var=np.diag(self._cov)
        mean=np.concatenate((self._x,self._y))
        bound_lower=mean-np.sqrt(12*var)/2
        bound_upper=mean+np.sqrt(12*var)/2
        cor= utilities.cov2cor(self._cov)

        self._realizations=distributions.generateUniform(bound_lower,bound_upper,cor,n)
        self._realizationsPolygons=[]

        #adjust the output for points which are not uncertain: For those points, set all
        #the realizations to be equal to the specified x and y values.
        numvars=np.size(self._realizations,1)
        for i in range(len(self._unc_pts)):
            if self._unc_pts[i]==0:
                self._realizations[:,i]=self._x[i]
                self._realizations[:,numvars/2+i]=self._y[i]

        self._lastDistr=distributions.DIST_UNIF

        #used by objectmanager
        self._variances=var
        self._means=mean
        self._bound_lower=bound_lower
        self._bound_upper=bound_upper

        utilities.msg('\tgenerated {} samples for {} vertices'.format(n,np.size(self._x)),'v')


class DeformableObjectFromValues(DeformableObject):
    """
    Represents a crisp, deformable object constructed from previously generated values.

    - *x*: A numpy array of x coordinates, length N
    - *y*: A numpy array of y coordinates, length N
    - *unc_pts*: numpy array of length >= N. if the i'th position is 0, it indicates that point in the x,y arrays should not be considered uncertain.
    - *isClosed*: whether the object will be treated as having a closed boundary.
    - *samples_x,samples_y*: numpy arrays containing samples of the x,y coords.
    """

    def __init__(self,x,y,unc_pts,isClosed,samples_x,samples_y):
        #build a fake covariance matrix. It doesn't matter for this object
        cov=np.eye(np.size(x)*2)

        DeformableObject.__init__(self,x,y,cov,unc_pts,isClosed)

        #voodoo.
        #this creates a matrix no matter if samples_x and samples_y are single values
        #or arrays of values or ndarrays. This matrix is then converted back to
        #an ndarray so as expected by the parent class
        self._realizations=np.asarray(np.r_['r',np.hstack((samples_x,samples_y))])

    def generateNormal(self,n):
        raise NotImplementedError("CrispObject.DeformableObjectFromValues(): this method is not supported for this object type")
    def generateUniform(self,n):
        raise NotImplementedError("CrispObject.DeformableObjectFromValues(): this method is not supported for this object type")
####################################################################################################
# testing functions
####################################################################################################
def _testRigid():
    pt_x=np.r_[480.,485,520,510,520]
    pt_y=np.r_[123.,100,105,110,117]
    centroid_x=np.mean(pt_x)
    centroid_y=np.mean(pt_y)
    print("Centroid: {} {}".format(centroid_x,centroid_y))
    centroid_xv=2.
    centroid_yv=2.

    #in degrees
    theta_mn=0.
    theta_v=10

    variances=np.r_[centroid_xv,centroid_yv,theta_v]
    cor=np.r_[[
                [1,     0,      0.8],
                [0,     1,      0.8],
                [0.8,   0.8,      1]
            ]]
    cov=utilities.cor2cov(cor,variances)

    uncertain_pts=np.r_[1,0,1,1,1]

    ro=RigidObject(pt_x,pt_y,centroid_x,centroid_y,theta_mn,cov,uncertain_pts,isClosed=False)
    #test normal
    ro.generateNormal(n=100,translate=True,rotate=False)
    ro.plot(fignum=1)
    plt.figure(1)                   #plot the original object
    plt.plot(np.concatenate((pt_x,np.r_[pt_x[0]])),np.concatenate((pt_y,np.r_[pt_y[0]])),'-k')
    ro.plotStats(fignum=2)

    #test uniform rotation. The translation uncertainty from above is preserved
    ro.generateUniform(n=100,translate=False,rotate=True)
    ro.plot(fignum=3)
    plt.figure(3)                   #plot the original object
    plt.plot(np.concatenate((pt_x,np.r_[pt_x[0]])),np.concatenate((pt_y,np.r_[pt_y[0]])),'-k')
    ro.plotStats(fignum=4,statsOfPoints=False)

def _testDeformable():
    pt_x=np.r_[480.,485,520,510,520]
    pt_y=np.r_[123.,100,105,110,117]

    pt_xv=np.r_[10.,20,10,10,10]
    pt_yv=np.r_[10.,20,10,10,10]

    uncertain_pts=[1,0,1,1,1]

    cov=np.r_[[
        #x0         x1          x2          x3          x4          y0          y1          y2          y3          y4
        [pt_xv[0],  0,          0,          0,          0,          -9,          0,          0,          0,          0],         #x0
        [0,         pt_xv[1],   0,          0,          0,          0,          0,          0,          0,          0],         #x1
        [0,         0,          pt_xv[2],   0,          0,          0,          0,          0,          0,          0],         #x2
        [0,         0,          0,          pt_xv[3],   0,          0,          0,          0,          0,          0],         #x3
        [0,         0,          0,          0,          pt_xv[4],   0,          0,          0,          0,          0],         #x4
        [-9,         0,          0,          0,          0,          pt_yv[0],   0,          0,          0,          0],         #y0
        [0,         0,          0,          0,          0,          0,          pt_yv[1],   0,          0,          0],         #y1
        [0,         0,          0,          0,          0,          0,          0,          pt_yv[2],   0,          0],         #y2
        [0,         0,          0,          0,          0,          0,          0,          0,          pt_yv[3],   0],         #y3
        [0,         0,          0,          0,          0,          0,          0,          0,          0,          pt_yv[4]]   #y4
    ]]

    do=DeformableObject(x=pt_x,y=pt_y,cov=cov,unc_pts=uncertain_pts,isClosed=False)

    do.generateNormal(n=5)     #first generate normally distributed uncertainty
    do.plot(fignum=1,n=200)         #plot realizations
    do.plotStats(fignum=2)
    plt.figure(1)                   #plot the original object
    plt.plot(np.concatenate((pt_x,np.r_[pt_x[0]])),np.concatenate((pt_y,np.r_[pt_y[0]])),'-k')
    #do.simpleBuffer()

    do.generateUniform(n=1000)    #now try uniformly distributed unceratinty.
    do.plot(fignum=3,n=100)         #plot realization
    do.plotStats(fignum=4)
    plt.figure(3)                   #plot the original object
    plt.plot(np.concatenate((pt_x,np.r_[pt_x[0]])),np.concatenate((pt_y,np.r_[pt_y[0]])),'-k')

    plt.show()

if __name__ == '__main__':
    #utilities.clearall()

    #_testRigid()
    _testDeformable()
