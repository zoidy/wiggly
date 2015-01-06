"""
Serves as a collection of general purpose utilites
"""
import numpy as np
import scipy
import matplotlib.pyplot as plt
import fuzz as fzy
import shapely.geometry
import os,time,re

#: Sets the level at which messages are printed to the console.
#: Level 5 (debug level) prints the most messages while 1 (error level)
#: prints only critical errors.
#:
#: 5=error, 4=warning, 3=information (default), 2=verbose, 1=debug
MESSAGE_LEVEL=3
def msg(string,level='i',indent=0,timelevel=True):
    """
    Prints a message.

    - *indent*: set a particular indent level.
    - *level*: the level of the message. can be
    - *timelevel*: include a timestamp and log level indicator in the message.

        === =============
        e   error 1
        w   warning 2
        i   information 3
        v   verbose 4
        d   debug 5
        === =============

      From top to bottom means lowest number to highest number of message output.
    """
    if level=='e':
        levelnum=1
    elif level=='w':
        levelnum=2
    elif level=='i':
        levelnum=3
    elif level=='v':
        levelnum=4
    elif level=='d':
        levelnum=5

    timelevelstr=''
    if timelevel:
        timelevelstr=time.strftime("%H:%M:%S",time.localtime()) + "["+level+"]: "

    if levelnum<=MESSAGE_LEVEL:
        #print(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()) + "["+level+"]: " + "\t"*indent + string)
        print(timelevelstr + "\t"*indent + string)

def clearall():
    """
	clear all globals.

	IMPORTANT: If you put this function in another script, comment it out until you need it.
	To use it, uncomment, run once, and comment out again. If you don't comment out,
	it will clear out libraries that have just been loaded.

	Call this any time you need to clear all globals from the python environment.
	Use whenever testing so that old definitions are cleared out for sure.
	From http://blog.datasingularity.com/?p=134

	I recently took a class on scripting ArcGIS with python. We were doing exercises in
	PythonWin and were constantly running and rerunning scripts into the same interpreter.
	So any variables from the last script would still be set in the interpreter's globals().
	Most would get overwritten, but some actually caused the programs to act weird. To remedy this,
	they had us use 'del' statements on our variables at the end of each script. This seemed like
	a waste of time and got really old fast. I looked around for a function that clears all global
	variables but was met with posts telling me that python garbage collection takes care of it,
	well, not in this situation. This was the remedy I came up with on my down time.
	if you put this in a script and run it once, or directly into the interpreter, you can call the
	clearall() function from anywhere. It also makes sure not to delete itself. It's not perfect
	but works well in most situations.
	"""
    for uniquevar in [var for var in globals().copy() if var[0] != "_" and var != 'clearall']:
        del globals()[uniquevar]

def varianceToIntervalUniform(means,variances):
    """
    Given a variance, returns the lower and upper bound of the associated interval
    for a uniform distribution. Returns the lower and upper bound as a tuple of
    numpy arrays

    - *means*: a numpy array of means
    - *var*: a numpy array of variances.
    """
    if type(means)!=np.ndarray:
        raise Exception("The means must be a numpy array")
    if type(variances)!=np.ndarray:
        raise Exception("The variances must be a numpy array")
    lbounds=means-np.sqrt(12*variances)/2
    ubounds=means+np.sqrt(12*variances)/2
    return lbounds,ubounds

def cov2cor(covariance):
    """
    Computes the correlation matrix from the covariance matrix.

    - *covariance*: a numpy matrix/array containing the variances/covariances.

    From the package OTB
    https://bitbucket.org/forieux/otb/src/73e68868eaacd7a644ed59ab8499c331a3b7015b/otb/utils.py?at=default
    """
    if np.ndim(covariance)!=2:
        raise Exception("The covariance matrix must have dimension 2")
    if np.size(covariance,0)!=np.size(covariance,1):
        raise Exception("The covariance matrix must be N-by-N")

    std = np.sqrt(np.diag(covariance))[:, np.newaxis]
    return (covariance / std) / std.T

def cor2cov(correlation,variances):
    """
    Computes the covariance matrix from a correlation matrix and the variances.

    - *correlation*: an N-by-N correlation matrix (numpy)
    - *variances*: a numpy matrix/array of length N containing the variances. The i'th element of this
      array corresponds to the i'th column in the correlation matrix
    """
    N=np.size(variances)
    if np.ndim(correlation)!=2:
        raise Exception("The covariance matrix must have dimension 2")
    if np.shape(correlation)!=(N,N):
        raise Exception("Size mismatch between the correlation matrix and the variance array")
    if np.size(correlation,0)!=np.size(correlation,1):
        raise Exception("The covariance matrix must be N-by-N")

    cov=np.zeros((N,N), dtype='f8')

    for i in range(N):
        for j in range(N):
            cov[i,j] = correlation[i,j]*np.sqrt(variances[i])*np.sqrt(variances[j])

    return cov
def fuzzyOffset2Absolute(fuzzyNumbers,referenceNumbers):
    """
    Converts a list of TrapezoidalFuzzyNumbers from offsets relative to some number to absolute fuzzy  numbers.

    - *fuzzyNumbers*: a list of TrapezoidalFuzzyNumbers which represent offsets with
      respect to the numbers in *referenceNumbers*
    - *referenceNumbers*: a list  or numpy array of numbers which serve as the reference
      point for each element in *fuzzyNumbers*

    returns a list of TrapezoidalFuzzyNumbers which now are expressed in absolute terms
    with respect to *referenceNUmbers*
    """

    if np.size(fuzzyNumbers)!=np.size(referenceNumbers):
        raise Exception("fuzzyNumbers must have the same length as referenceNumbers")

    transformedFn=[]
    if isinstance(fuzzyNumbers[0],fzy.TrapezoidalFuzzyNumber):
        for i in range(len(fuzzyNumbers)):
            k=fuzzyNumbers[i].kernel
            s=fuzzyNumbers[i].support
            refnum=referenceNumbers[i]

            newfn=fzy.TrapezoidalFuzzyNumber((k[0]+refnum, k[1]+refnum),
                                             (s[0]+refnum, s[1]+refnum))
            transformedFn.append(newfn)
    elif isinstance(fuzzyNumbers[0],fzy.GaussianFuzzyNumber):
        raise Exception("not yet done")
    else:
        raise Exception ("Fuzzy numbers must be gaussian or trapezoidal")

    return transformedFn

def makeSymmetric(m,lower2upper=True):
    """
    Makes the square matrix m symmetric along the diagonal.

    - *m*: the matrix to make symmetrical
    - *lower2upper*: uses the elements in the lower triangular part of the matrix. If false
      usees the elements in the upper triangular part

    Returns the symmetric matrix
    """
    if np.ndim(m)!=2:
        raise Exception("the matrix must be 2D")
    if np.size(m,0)!=np.size(m,1):
        raise Exception("the matrix must be square")

    ret=0
    if lower2upper:
        ret=np.tril(m,-1)+np.tril(m).T
    else:
        ret=np.triu(m,1)+np.triu(m).T

    return ret

def puqParams_fromFile(filename):
    """
    Converts the file written by puq when paramsByFile is True for TestProgram.

    This function is useful in the puq TestProgram script to parse the
    parameters file.

    - *filename*: The file to convert. The file must have three columns with data
      types string, float, string. These columns correspond to parameter names, values
      and description respectively.

    Returns: a dictionary

        {$paramName$:{'value':float, 'desc':string}, ... }

    'desc', 'value' are the parameter name description and value respectively.
    """
    #paramvalues is a list of tuples.
    paramValues=np.loadtxt(filename,
                   dtype={"names":("p_name","p_val","p_desc"),
                          "formats":(np.object,np.float,np.object)})
    params={}
    for tupleParam in paramValues:
        params[tupleParam[0]]={'value':tupleParam[1],'desc':tupleParam[2]}

    return params

def transformPoints(x,y,new_x,new_y,theta,origin_x=0,origin_y=0):
    """
    Performs an affine transformation (translation & rotation only) of x,y.

    - *x, y*: points to transform. *x* and *y* may be arrays where each x[i],y[i] pair
      is a point. They may also be 2D matrices produced by the meshgrid command.
    - *new_x, new_y, theta*: Move all *x,y* to a new location given by scalars *new_x, new_y*.
    - *origin_x, origin_y*: scalar values indicating the origin of rotation and translation.

    Returns:
        - if *x,y* were arrays, a matrix containing the displaced points. The first row is the x coords
          and the second is the y.
        - If *x,y* were matrices, returns, *xx,yy*, which are two matrices with the same shape as
          *x,y*.

    In essence, this function

        - First, rotates all points *x,y* by an angle *theta* with respect to *origin_x,origin_y*.
        - Transforms the origin of all points from *origin_x, origin_y* to *new_x, new_y*.
    """
    if np.size(origin_x)>1 or np.size(origin_y)>1:
        raise Exception('origin_x and origin_y must be scalars')
    if np.size(new_x)>1 or np.size(new_y)>1:
        raise Exception('new_x and new_y must be scalars')
    if np.ndim(x)!=np.ndim(y):
        raise Exception ('x and y must have the same dimension')
    if np.ndim(x)>2:
        raise Exception('x and y must be arrays or 2D matrices')

    shape=None
    if np.ndim(x)==2:
        shape=np.shape(x)
        x=x.ravel()
        y=y.ravel()

    #first make a matrix out of the points, moving the object to the origin.
    #Need to move it to the origin so the rotation will be around the object's centroid
    allpoints=np.matrix(np.vstack((x-origin_x,y-origin_y)))

    #'r' indicates T is a matrix instead of an ndarray
    T=np.r_["r",[  [new_x],
                   [new_y]
                ]]

    theta=theta*np.pi/180
    R=np.r_["r",[   [np.cos(theta),-np.sin(theta)],
                    [np.sin(theta),np.cos(theta)]
                ]]

    #apply the affine transform
    output=  R*allpoints+T

    if shape!=None:
        return np.asarray(np.reshape(output[0,:],shape)),np.asarray(np.reshape(output[1,:],shape))

    return output


#from Handbook of Mathematical Functions. http://http://stackoverflow.com/questions/457408/
#This way, we don't need SciPy or to import the Python 2.7 math lib  which has this function
#Vectorized by me. Performance is similar to scipy.
#UPDATE: puq uses scipy so this is probably not necessary anymore. Leave it here in case
#wiggly is used without puq.
def erf(z):
    """
    Calculates the error function.

    - *z*: The argument to the error function. If it is a numpy array, it calculates the error
      function for all its elements
    """
    # save the sign of z
    z_ge_0=np.greater_equal(z,0)
    sign=z_ge_0*1 + np.logical_not(z_ge_0)*-1   #is element z>=0? Yes: sign=1 No sign=-1

    z = np.abs(z)

    # constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911

    # A&S formula 7.1.26
    t = 1.0/(1.0 + p*z)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*np.exp(-z*z)
    return sign*y # erf(-x) = -erf(x)

def pairs(data, labels=None, figurenum=None,rankcorr=False):
    """
    Generate something similar to R `pairs`.

    - *data*: the data to plot. Variables are arranged column-wise. This is a numpy array
    - *labels*: a vector of labels
    - *fignum*: The figure is plotted into figurenum but is not shown. If <0, it plots into a new fig.
    - *rankcorr*: computes Spearman's rank correlation instead of Pearson's rho.

    The numbers in each plot give the following information:

        =========== ===================
        Location    Description
        =========== ===================
        Left-upper  Variance/covariance
        Right-upper Correlation coef.
        Left-lower  Mean
        =========== ===================
    """
    #http://stackoverflow.com/questions/2682144/matplotlib-analog-of-rs-pairs
    if figurenum==None:
        fig=plt.figure()
        figurenum=fig.number
    else:
        plt.close(figurenum)
        fig = plt.figure(figurenum)

    nVariables = data.shape[1]
    if labels is None:
        labels = ['var%d'%i for i in range(nVariables)]

    for i in range(nVariables):
        for j in range(nVariables):
            nSub = i * nVariables + j + 1
            ax = fig.add_subplot(nVariables, nVariables, nSub)
            ax.locator_params(tight=True)#, nbins=4)

            if i == j:
                ax.hist(data[:,i])
                ax.set_xticks([])
                ax.text(.1,.57,np.round(np.mean(data[:,i]),2),
                        size=9,transform=ax.transAxes,horizontalalignment='left',
                        bbox=dict(facecolor='white', alpha=0.75))
                ax.text(0.5, 0.02,labels[i], horizontalalignment='center',
                     verticalalignment='bottom',
                     transform=ax.transAxes,color='white',bbox=dict(facecolor='red', alpha=0.6))
            else:
                ax.plot(data[:,j], data[:,i], '.k',markersize=2)
                ax.set_xticks([])

            if rankcorr:
                c,p=scipy.stats.stats.spearmanr(data[:,i],data[:,j])
            else:
                c=np.corrcoef(data[:,i],data[:,j],rowvar=False)[0,1]
            ax.text(.95,.77,np.round(c,2),
                    size=9,transform=ax.transAxes,horizontalalignment='right',
                    bbox=dict(facecolor='white', alpha=0.75))

            ax.text(.1,.77,np.round(np.cov(data[:,i],data[:,j],rowvar=False)[0,1],2),
                    size=9,transform=ax.transAxes,horizontalalignment='left',
                    bbox=dict(facecolor='white', alpha=0.75))

            #ax.set_title(np.round(np.cov(data[:,i],data[:,j],rowvar=False)[0,1],2),size=11)
            ax.set_yticklabels([])

    #tight layout for the whole plot
    fig.tight_layout()
    return fig

def isCrispObject(obj):
    """
    Returns true if *obj* is a CrispObject
    """
    #import this here due to the dreaded python circular import problem
    #If this is placed at the top, a circular import with CrispObjects will
    #be triggered
    import CrispObjects

    return True if isinstance(obj,CrispObjects.CrispObject) else False
def isFuzzyObject(obj):
    """
    Returns true if *obj* is a FuzzyObject
    """
    import FuzzyObjects

    return True if isinstance(obj,FuzzyObjects.FuzzyObject) else False
def isRigidObject(obj):
    import CrispObjects
    return True if isinstance(obj,CrispObjects.RigidObject) else False
def isDeformableObject(obj):
    import CrispObjects
    return True if isinstance(obj,CrispObjects.DeformableObject) else False
def isVertexDefinedObject(obj):
    import FuzzyObjects
    return True if isinstance(obj,FuzzyObjects.VertexDefinedObject) else False
def isEdgeDefinedObject(obj):
    import FuzzyObjects
    return True if isinstance(obj,FuzzyObjects.EdgeDefinedObject) else False

def isShapelyPolygon(obj):
    """
    Determines whether *obj* is a Polygon object of the Shapely library.
    Returns True if it is.
    """

    #this is the safest, most compatible way to do it.
    #In the type comparisons below, compare using the string representation
    #of the type because for some reason, when calling this module by
    #using the wiggly package from another script, the traditional type comparison
    #type(object) is class doesn't work
    if 'shapely.geometry.polygon.Polygon' in str(type(obj)):
        return True
    else:
        return False

def isShapelyMultiPolygon(obj):
    """
    Determines whether *obj* is a MultiPolygon object of the Shapely library.
    Returns True if it is.
    """

    if 'shapely.geometry.multipolygon.MultiPolygon' in str(type(obj)):
        return True
    else:
        return False
def isShapelyGeometryCollection(obj):
    """
    Determines whether *obj* is a GeometryCollection object of the Shapely library.
    Returns True if it is.
    """

    if 'shapely.geometry.collection.GeometryCollection' in str(type(obj)):
        return True
    else:
        return False

def isShapelyLineString(obj):
    """
    Determines whether *obj* is a LineString object of the Shapely library.
    Returns True if it is.
    """

    if 'shapely.geometry.linestring.LineString' in str(type(obj)):
        return True
    else:
        return False

def isShapelyPoint(obj):
    """
    Determines whether *obj* is a Point object of the Shapely library.
    Returns True if it is.
    """

    if 'shapely.geometry.point.Point' in str(type(obj)):
        return True
    else:
        return False

def fuzzyCalcResultToXY(fuzzyResultString):
    """
    Converts a string representation of a discrete fuzzy number to numpy arrays.

    The string represetation must be in the format output by
    the fuzzy calculator from the Fuzzy Webtools located at
    https://ffem.mech.kuleuven.be/fuzzy/html/calculator.html.

    - *fuzzyResultString*: a string representation of a discrete fuzzy number. Copy and
      paste it from the fuzzy calculator website.

    Returns x, y which are numpy arrays containing the x and y coordinates of the
    membership function of the *fuzzyResultString* which can be passed directly to
    the matplotlib plot function.
    """
    membfcn_acuts=fuzzyResultString.split(";")
    membfcn_numAcuts=np.size(membfcn_acuts)
    membfcn_x=np.zeros(2*membfcn_numAcuts)
    membfcn_y=np.zeros(2*membfcn_numAcuts)
    for i in range(membfcn_numAcuts):
        acut_plus_membval=membfcn_acuts[i].split(":")
        acut_membval=acut_plus_membval[1]
        acut=acut_plus_membval[0].split(",")
        membfcn_x[i]=acut[0]
        membfcn_x[2*membfcn_numAcuts-i-1]=acut[1]
        membfcn_y[i]=acut_membval
        membfcn_y[2*membfcn_numAcuts-i-1]=acut_membval

    return membfcn_x,membfcn_y

def DOE_ff2l(levels_lower,levels_upper):
    """
    Generates a 2-level full factorial design with k variables.

    - *levels_lower*: A numpy array specifying the lower level for each variable
    - *levels_upper*: A numpy array specifying the upper level for each variable

    Returns a numpy array of dimension 2^k-by-k where each row is a unique combination
    of the input variables. In the full factorial design, all possible combinations are
    generated (there are 2^k of them).
    """
    if np.size(levels_lower)!=np.size(levels_upper):
        raise Exception("levels_lower and levels_upper must be the same size!")
    if type(levels_lower)!=np.ndarray or type(levels_upper)!=np.ndarray:
        raise Exception("levels_lower and levels_upper must be numpy arrays!")


    k=np.size(levels_lower)
    n=2**k

    a=np.ones((n,k))

    step=1
    for j in range(k):
        step=2**j
        switch=False
        counter=0
        for i in range(n):
            if counter==step:
                switch=not switch
                counter=0
            if switch:
                a[i,j]=levels_upper[j]
                #a[i,j]=-1
            else:
                #a[i,j]=1
                a[i,j]=levels_lower[j]
            counter+=1

    return a

def unique_rows(a):
    """
    Returns a numpy array containing the unique rows of *a*.
    """
    #http://http://stackoverflow.com/questions/8560440/
    # first, sort the 2D NumPy array row-wise so dups will be contiguous
    # and rows are preserved. a.T creates the keys for to pass to lexsort
    #note: this method is much faster than the second answer on that page
    order = np.lexsort(a.T)
    a = a[order]
    # now diff by row
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)
    return a[ui]

def shape2Points(shp):
    """
    Given a shapely shape, converts it to numpy arrays.

    Returns two numpy arrays x, y containing the coordinates of the vertices. For polygons, only the
    exterior vertices are returned
    """
    if isShapelyPolygon(shp):
        a=np.asarray(shp.exterior)
    else:
        a=np.asarray(shp.coords)
    return a

def pointslist2Shape(pts_x,pts_y,isClosed):
    """
    Given point arrays, converts them to an appropriate Shapely shape.

    - *pts_x, pts_y*: Numpy arrays of x and y coordinates
    - *isClosed*: Boolean indicating whether the geometry is to be considered closed. If True,
      then the length of pts_x and pts_y must be 3 or greater.
    """
    shape=None
    points=np.vstack((pts_x,pts_y)).transpose()
    if np.size(pts_x)==1:
        p=shapely.geometry.asPoint(points.flatten())
    else:
        l=shapely.geometry.asLineString(points)

    if isClosed:
        shape=shapely.geometry.Polygon(l)
    else:
        #create a new linestring or point instead of referecing the
        #linestring from l which references numpy array.
        #without this, any operation on the linestring will be
        #too slow.
        if np.size(pts_x)==1:
            shape=shapely.geometry.Point(p)
        else:
            shape=shapely.geometry.LineString(l)
    return shape


def plot_datalabels(x_data,y_data,axis,labels=None,bbox_height=0.03,bbox_width=0.02,size=None,labelsbg=None):
    """
    Adds data labels to the specified axis.

    - *x_data, y_data*: The data that was previously plotted (arrays).
    - *axis*: The axis into which the data was plotted
    - *labels*: The label to set for each data point (list). If None, the value of *y* is used.
      If a particular label is an empty string, the label and arrow won't be plotted.
    - *labelsbg*: a color specifying the background color of the labels. Can
      be any matplotlib color e.g., 'white' or and RGBA tuple (1,1,1,0.5) etc.
    - *size*: font size in points to use. If None use the default.
    - *bbox_height,bbox_width*: bounding boxes for text are not automatically computed. These
      numbers multiply the axis xrange (width) and yrange (height) which are used to define
      the fixed text bboxes. If the plot labels are still colliding, tweak these numbers.

    Note: inverted x-axis is supported but it must axes.invert_xaxis must be called
    before plotting the data.

    Note2: if two points are exactly coincident, then their labels will overlap.

    Based on http://stackoverflow.com/a/10739207
    """
    def _get_text_positions(x_data, y_data, txt_width, txt_height,labels=None):
        if labels==None:
            labels=['.']*len(x_data)
        a = zip(y_data, x_data)
        text_positions = y_data.copy()
        for index, (y, x) in enumerate(a):
            local_text_positions = [i for i in a if i[0] >= (y - txt_height)
                                and (abs(i[1] - x) <= txt_width * 2)  and i != (y,x)]
            if local_text_positions:
                sorted_ltp = sorted(local_text_positions)
                if labels[index]!='': #only consider labels that are to be plotted
                    if abs(sorted_ltp[0][0] - y) <= txt_height: #True == collision
                        differ = np.diff(sorted_ltp, axis=0)
                        a[index] = (sorted_ltp[-1][0] + txt_height, a[index][1])
                        text_positions[index] = sorted_ltp[-1][0] + txt_height
                        for k, (j, m) in enumerate(differ):
                            #j is the vertical distance between words
                            if j > txt_height * 2: #if True then room to fit a word in
                                a[index] = (sorted_ltp[k][0] + txt_height, a[index][1])
                                text_positions[index] = sorted_ltp[k][0] + txt_height
                                break
        return text_positions

    def _text_plotter(x_data, y_data,text_positions, axis,labels=None,labelsbg=None,lblsize='medium'):
        inverted=1
        if axis.xaxis_inverted():
            inverted=-1
        a_height = 0.03*(axis.get_ylim()[1] - axis.get_ylim()[0])
        a_width = 0.02*(inverted*axis.get_xlim()[1] - inverted*axis.get_xlim()[0])

        lbltxtargs=dict(color='blue',size=lblsize)
        if labelsbg!=None:
            lbltxtargs['backgroundcolor']=labelsbg

        if labels==None:
            for x,y,t in zip(x_data, y_data, text_positions):
                axis.annotate('%d'%int(y),xy=(x,t),xytext=(0,8),textcoords='offset points',
                              annotation_clip=False,ha='center',va='center',**lbltxtargs)
                if y != t:
                    axis.arrow(x, t,0,y-t, color='red',alpha=0.3, width=a_width*0.1,
                               head_width=a_width/2., head_length=a_height/2.,
                               zorder=0,length_includes_head=True)
        else:
            for x,y,t,l in zip(x_data, y_data, text_positions,labels):
                if l!='':
                    axis.annotate(l,xy=(x,t),xytext=(0,8),textcoords='offset points',
                                  annotation_clip=False,ha='center',va='center',**lbltxtargs)
                    if y != t:
                        axis.arrow(x, t,0, y-t, color='red',alpha=0.3, width=a_width*0.1,
                                   head_width=a_width/2., head_length=a_height/2.,
                                   zorder=0,length_includes_head=True)

    #set the bbox for the text. Increase txt_width for wider text.
    inverted=1
    if axis.xaxis_inverted():
        inverted=-1
    txt_height = bbox_height*(axis.get_ylim()[1] - axis.get_ylim()[0])
    txt_width = bbox_width*(inverted*axis.get_xlim()[1] - inverted*axis.get_xlim()[0])

    #Get the corrected text positions, then write the text.
    text_positions = _get_text_positions(x_data, y_data, txt_width, txt_height,labels)
    _text_plotter(x_data, y_data,text_positions, axis,labels=labels,labelsbg=labelsbg,
                  lblsize='medium' if size==None else size)

    #adjust axis ranges to make the labels visible
    max_textpos_y=np.max(text_positions)
    max_textpos_x=np.max(x_data)
    min_textpos_x=np.min(x_data)
    if max_textpos_y+txt_height>axis.get_ylim()[1]:
        axis.set_ylim(axis.get_ylim()[0],txt_height+max_textpos_y)
    if not axis.xaxis_inverted():
        if max_textpos_x+txt_width>axis.get_xlim()[1]:
            axis.set_xlim(axis.get_xlim()[0],max_textpos_x+txt_width)
        if min_textpos_x-txt_width<axis.get_xlim()[0]:
            axis.set_xlim(min_textpos_x-txt_width,axis.get_xlim()[1])
    else:
        if max_textpos_x+txt_width>axis.get_xlim()[0]:
            axis.set_xlim(axis.get_xlim()[0],axis.get_xlim()[0]+1.3*txt_width)

def tickLabels_keepEvery(axis,xy,n,keep_endpoints=False):
    """
    For the given axis, keeps every nth tick label.

    - *axis*: a matplotlib axis object
    - *xy*: 'x'=Operate on the x axis. 'y'=operate on the y axis.
    - *n*: keep every nth
    - *keep_endpoints*: always include the endpoints of the axis range
    """
    #http://stackoverflow.com/questions/20337664/cleanest-way-to-hide-every-nth-tick-label-in-matplotlib-colorbar
    if xy=='x':
        curr_labels=axis.get_xticklabels()
    elif xy=='y':
        curr_labels=axis.get_yticklabels()
    else:
        raise Exception('only x axis supported at this time')

    for lbl in curr_labels:
        lbl.set_visible(False)
    for lbl in curr_labels[::n]:
        lbl.set_visible(True)
    if keep_endpoints:
        curr_labels[0].set_visible(True)
        curr_labels[-1].set_visible(True)

def plot_setFont(ax,size=20,items=['title','xlabel','ylabel','xtick','ytick']):
    """
    Sets the font size of the given items to the given value for the given axis
    """
    axis_items=[]
    if 'title' in items:
        axis_items.append(ax.title)
    if 'xlabel' in items:
        axis_items.append(ax.xaxis.label)
    if 'ylabel' in items:
        axis_items.append(ax.yaxis.label)
    if 'xtick' in items:
        axis_items+=ax.get_xticklabels()
    if 'ytick' in items:
        axis_items+=ax.get_yticklabels()
    #increase font size
    #http://stackoverflow.com/a/14971193
    for item in axis_items:
        item.set_fontsize(size)

def saveAllFigures(fmt='png',appendtxt='',fignum=None,dpi=125,outdir=None):
    """
    Saves all currently opened figures to the specified format.

    The figures are named fig <n>.<fmt> where n is the figure number. If the figure
    has a label (e.g., set by the figure('some label') command), the label is used instead
    of the number.

    - *fmt*: The file extension of the format to use. Any format supported by Matplotlib works.
    - *appendtxt*: Appends this text to the name of the figure.
    - *fignum*: if specified, saves only the figure given by the number or label (can be a list).
    - *dpi*: The output dpi (non-vector figures only).
    - *outdir*: The directory in which the files will be saved. If not specified, the
      current working directory is used
    """
    if outdir!=None and not os.path.isdir(outdir):
        raise Exception('{} not found'.format(outdir))

    if fignum!=None:
        if type(fignum) is list:
            fignum=[str(a) for a in fignum]
        else:
            fignum=[str(fignum)]

    oldwd=os.getcwd()

    if outdir!=None:
        os.chdir(outdir)

    msg('Saving figures to {}'.format(os.getcwd()))
    try:
        for i in plt.get_fignums():
            lbl=plt.figure(i).get_label()
            if lbl=='':
                lbl=str(i)

            s='fig {}{}.{}'.format(lbl,appendtxt,fmt)
            if fignum!=None:
                if lbl in fignum:
                    msg("saving '{}'".format(s))
                    plt.savefig(s,dpi=dpi)
            else:
                msg("saving '{}'".format(s))
                plt.savefig(s,dpi=dpi)
    except Exception,e:
        msg('Could not save figures. {}.'.format(str(e)))
    finally:
        os.chdir(oldwd)

def natural_keys(text):
    def atoi(text):
        return int(text) if text.isdigit() else text
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)

    If list elements are tuples, sorts on the first element of the tuple.
    '''
    if type(text) is tuple:
        text=text[0]
    return [ atoi(c) for c in re.split('(\d+)', text) ]

#check if we're executing this script as the main script
if __name__ == '__main__':
    lbounds=np.r_[-2,1,0,6,-5.5]
    ubounds=np.r_[3,-1,2.2,7,5.5]
    a=DOE_ff2l(lbounds,ubounds)
    print(a)
    print(np.shape(a))
    au=unique_rows(a)
    print(au)
    print(np.shape(au))
