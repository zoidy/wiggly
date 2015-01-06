"""
Implements the Gband error model using the GPC
polygon clipping library.

Use this in case shapely is not available or for
testing purposes

http://www.cs.man.ac.uk/~toby/gpc/#Ports
"""

#in pyscripter, need to add these to special packages for autocompletion
import numpy as np
from Polygon import *
import matplotlib.pyplot as plt
   
def __test2():
    if withNumPy==False:
        raise Exception("gpc lib doesn't have numpy!")
    
    setDataStyle(STYLE_NUMPY)       

    plotEllipses=False      #enable or disable showing the ellipses
    isClosed=False
    ellipseSpacing=0.25
    numPtsPerEllipse=20
    
    #define a polygon
    pt_x=np.r_[480,485,520,510,520]
    pt_y=np.r_[123,100,105,110,117]
    
    var_x=np.r_[10,10,10,10,10]
    var_y=np.r_[10,10,10,10,10]
    cor=np.r_[[
        #x0      x1       x2      x3       x4      y0       y1       y2      y3      y4
		[1,		0,		0,		0,		0,		0,		0,		0,		0,		0],    #x0
		[0,		1,		0,		0,		0,		0,		0,		0,		0,		0],    #x1
		[0,		0,		1,		0,		0,		0,		0,		0,		0,		0],    #x2
		[0,		0,		0,		1,		0,		0,		0,		0,		0,		0],    #x3
		[0,		0,		0,		0,		1,		0,		0,		0,		0,		0],    #x4
		[0,		0,		0,		0,		0,		1,		0,		0,		0,		0],    #y0
		[0,		0,		0,		0,		0,		0,		1,		0,		0,		0],    #y1
		[0,		0,		0,		0,		0,		0,		0,		1,		0,		0],    #y2
		[0,		0,		0,		0,		0,		0,		0,		0,		1,		0],    #y3
		[0,		0,		0,		0,		0,		0,		0,		0,		0,		1]    #y4
		]]
    cov=__cor2cov(cor,np.concatenate((var_x,var_y)))       

    band=GBand(pt_x,pt_y,cov,isClosed,ellipseSpacing,numPtsPerEllipse)
    band.generateGBand()
    band.plot(plotEllipses=plotEllipses)

    plt.xlim(470,530)
    plt.ylim(95,125)

    plt.show()
    try:
        #only works when not using embedded plots in IPython
        wm = plt.get_current_fig_manager()
        #wm.window.showMaximized()
        wm.window.raise_()
        wm.window.attributes('-topmost', 1)
        wm.window.attributes('-topmost', 0)
    except:
        pass
    
def __test():
   
    if withNumPy==False:
        raise Exception("gpc lib doesn't have numpy!")
    
    setDataStyle(STYLE_NUMPY)
    
    plotEllipses=False      #enable or disable showing the ellipses
    ellipseSpacing=0.5
    numPtsPerEllipse=20
    testPoint=[515,115]

    #define a bunch of point pairs (not a polygon)
    pt0_x=np.r_[480,480,500,520,520,520,500,480,500]
    pt0_y=np.r_[110,100,100,100,100,120,120,120,115]
    pt1_x=np.r_[520,520,500,480,480,480,500,485,510]
    pt1_y=np.r_[110,120,120,120,100,100,100,100,110]
    
    #define the variances and covariances of the points
    #note: can't use correlations/variance of exactly zero. Will cause
    #errors
    varx0=np.r_[10,0.01]
    vary0=np.r_[10,0.01]
    cor=np.r_[[
        #x0         x1       y0       y1
        [1,         .999,   -.8,      0],     #x0
        [.999,      1,       0,       0],     #x1
        [-.8,       0,       1,       0],     #y0
        [0,         0,       0,       1]      #y1
    ]]
    cov=__cor2cov(cor,np.concatenate((varx0,vary0)))
       
    #Plot the line from the defined point
    #remember range doesn't include the end point
    for i in range(0,pt0_x.size):
        band=GBand(np.r_[pt0_x[i],pt1_x[i],],
           np.r_[pt0_y[i],pt1_y[i]],
            cov,False,ellipseSpacing,numPtsPerEllipse)   
        band.generateGBand()

        #plot into a subplot. add 1 to i make it conform to the 1-based subplot indexing
        plt.subplot(3,3,i+1,aspect=1)
        plt.plot([pt0_x[i],pt1_x[i]],[pt0_y[i],pt1_y[i]],color='0.85')
        plt.plot(pt0_x[i],pt0_y[i],'ks',fillstyle='none',markersize=3)
        plt.plot(pt1_x[i],pt1_y[i],'k+')
        plt.xlim(470,530)
        plt.ylim(95,125)
        
        #plot gband and optoinal ellipses.
        points=band.polygonPoints
        for i in range(len(points[:])):
            band_x=points[i][0]
            band_y=points[i][1]
            plt.fill(band_x,band_y,'-',fill=False)
        if plotEllipses:
            for e in band.ellipses:
                plt.plot(e.x,e.y,'-k')
    
        plt.plot(testPoint[0],testPoint[1],'o')                
        plt.title('({},{}) Inside:{}'.format(testPoint[0],testPoint[1],
                        band.polygon.isInside(testPoint[0],testPoint[1])))
        plt.xlabel('x')
        plt.ylabel('y')
    
    plt.show() 
    try:
        #only works when not using embedded plots in IPython
        wm = plt.get_current_fig_manager()
        #wm.window.showMaximized()
        wm.window.raise_()
        wm.window.attributes('-topmost', 1)
        wm.window.attributes('-topmost', 0)
    except:
        pass
       
    
    
def __cor2cov(correlation,variances):
    """
    The covariance matrix from a correlation matrix.
    
    correlation: an N-by-N correlation matrix
    variances: an array of length N containing the variances. The i'th element of this
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

########################################################################################

        
class GBand:
    """
    Represents a G-Band object.
    """
    
    class Ellipse:
        """
        An ellipse
        
        - *pointsx*: x values of the points of the ellipse wrt to X-Y axes
        - *pointsy*:  y values of the points of the ellipse wrt to X-Y axes
        - *angle*: angle of the semi major axis wrt to X-Y (in rad)
        """
        def __init__(self,pointsx,pointsy,angle):
            self.__x=pointsx
            self.__y=pointsy
            self.__angle=angle
        @property
        def angleDeg(self):
            return self.__angle*(180/np.pi)
        @property
        def x(self):
            return self.__x
        @property
        def y(self):
            return self.__y
        
    __ellipses=0    #The error ellipses associated with the gband
    __polygon=0     #an object of type Polygon
       
    def __init__(self,pts_x,pts_y,cov,isClosed=True,ellipseSpacing=1,numPtsPerEllipse=20):
        """
        - *pts_x, pts_y*: numpy arrays with thelocation of uncertain points. Each array has length N
        - *cov*: the covariance matrix between the points. It is of size 2N-by-2N    
        - *isClosed*: if True, considers the points to represent a closed boundary
        - *EllipseSpacing*: maximum spacing between error ellipses along a line segment. This
          value is automatically adjusted if the spacing is too large
        - *numPtsPerEllipse*: the number of points to evaluate each ellipse on.
            
        Note: It is up to the caller to make sure that the variance and covariance 
        arrays are appropriately constructed taking holes into account
        """
        if withNumPy==False:
            raise Exception("gpc lib doesn't have numpy!")    
        if pts_x.size!=pts_y.size:
            raise Exception("There must be the same number of elements in pts_x and pts_y")
        if isClosed and pts_x.size<3:
            raise Exception("Need at least 3 points for a closed polygon")
        if 2*np.size(pts_x)!=np.size(cov,0) or 2*np.size(pts_x)!=np.size(cov,1):
            raise Exception("Covariance matrix must be 2N-by-2N")
        if np.any(np.diagonal(cov)<=0):
            raise Exception("Invalid covariance matrix. One of the diagonal entries is <=0")
            
        setDataStyle(STYLE_NUMPY)
        
        #http://stackoverflow.com/questions/5320324/testing-if-a-numpy-array-is-symmetric
        if not np.allclose(np.transpose(cov), cov):
            raise Exception("Covariance matrix must be symmetric")
        
        self.__pts_x=pts_x
        self.__pts_y=pts_y
        self.__cov=cov
        self.__ellipseSpacing=ellipseSpacing
        self.__numPtsPerEllipse=numPtsPerEllipse
        self.__isClosed=isClosed    
            
    
    @property
    def polygon(self):
        """Returns a Polygon object generated after a call to generateGBand"""
        return self.__polygon
    @property
    def ellipses(self):
        """The collection of ellipses that comprise the G-band after a call to generateGBand"""
        return self.__ellipses
    
    @property
    def polygonPoints(self):
        """
        returns a list of numpy arrays of x and of y points that make up the gband
        the first index is the contour. The second index selects the array of
        x and y points for that contour. E.g., ret[0][1] is the first contour, y points.
        
        Plot with fill instead of plot to ensure the boundary is closed off
        """
        ret=[]
        for c in range(len(self.__polygon)):
            points=self.__polygon[c]
            band_x=np.zeros(len(points))
            band_y=np.zeros(len(points))
            for i in range(len(points)):
                band_x[i]=points[i][0]
                band_y[i]=points[i][1]
            ret.append([band_x,band_y])
        return ret
        
    def plot(self,fignum=-1,plotEllipses=False,plotLine=True):
        """
        Plots the gband using matplotlib but doesn't show the plot. To show it, call matplotlib's
        show() function.
        
        - *fignum*: The figure number to which the band will be plotted. If <0 will plot into a new figure
        - *plotEllipses*: Whether to show the associated error ellipses or not.
        """
        if fignum<0:
            plt.figure()
        else:
            plt.figure(fignum)
        plt.axis('equal')
                
        #get the points comprising the gband. Plot each
        #contour (part of the polygon) in sequence
        points=self.polygonPoints
        
        for c in range(len(points)):
            band_x=points[c][0]
            band_y=points[c][1]
               
            #plot gband
            plt.fill(band_x,band_y,'-',fill=False,zorder=99)

        if plotEllipses:
            for e in self.ellipses:
                plt.plot(e.x,e.y,'-k')
                
        if plotLine:
            if self.__isClosed:
                plt.plot(np.r_[self.__pts_x,self.__pts_x[0]],
                         np.r_[self.__pts_y,self.__pts_y[0]],color='0.85')
            else:
                plt.plot(self.__pts_x,self.__pts_y,color='0.85')
            

            
                
    def generateGBand(self, confidenceLevel=0.39):
        """
        Generates the Gband
        
        - *ConfidenceLevel* is a float and can be one of {0.39, 0.6, 0.9, 0.95}
        
        Returns nothing. To access the results, use the polygon, polygonpoints, or ellipses properties
        """
        def varx_t(t):
            return (1-t)**2*vx0   + 2*t*(1-t)*vx0x1         + t**2*vx1
        def vary_t(t):
            return (1-t)**2*vy0   + 2*t*(1-t)*vy0y1         + t**2*vy1
        def varyx_t(t):
            return (1-t)**2*vx0y0 + t*(1-t)*(vx1y0+vx0y1) + t**2*vx1y1
        def varxy_t(t):
            return (1-t)**2*vy0x0 + t*(1-t)*(vy1x0+vy0x1) + t**2*vy1x1
        
      
        if confidenceLevel!=0.39 and confidenceLevel!=0.60 and confidenceLevel!=0.90 and confidenceLevel!=0.95:
            raise Exception("Unsupported confidence level")
        
        ellipses=[]
        gband=0
        
        if self.__isClosed:
            numpts=self.__pts_x.size
        else:
            if self.__pts_x.size<3:
                numpts=self.__pts_x.size
            else:
                numpts=self.__pts_x.size-1
        
        for j in range(numpts):
            x0=self.__pts_x[j]
            y0=self.__pts_y[j]
            vx0=self.__cov[j,j]#varx0[j]
            vy0=self.__cov[numpts+j,numpts+j]#vary0[j]
            vx0y0=self.__cov[j,numpts+j]#varx0y0[j]
            vy0x0=vx0y0
            
            if j<self.__pts_x.size-1:
                x1=self.__pts_x[j+1]
                y1=self.__pts_y[j+1]
                vx0x1=self.__cov[j,j+1]#varx0x1[j]
                vx0y1=self.__cov[j,numpts+j+1]#varx0y1[j]
                vy0x1=self.__cov[j+1,numpts+j]#vary0x1[j]
                vy0y1=self.__cov[numpts+j,numpts+j+1]#vary0y1[j]
                vx1=self.__cov[j+1,j+1]#varx0[j+1]
                vy1=self.__cov[numpts+j+1,numpts+j+1]#vary0[j+1]
                vx1y1=self.__cov[j+1,numpts+j+1]#varx0y0[j+1]
            else:
                x1=self.__pts_x[0]
                y1=self.__pts_y[0]
                vx0x1=self.__cov[j,0]#varx0x1[j]
                vx0y1=self.__cov[j,numpts]#varx0y1[j]
                vy0x1=self.__cov[0,numpts+j]#vary0x1[j]
                vy0y1=self.__cov[numpts+j,numpts]#vary0y1[j]
                vx1=self.__cov[0,0]#varx0[j+1]
                vy1=self.__cov[numpts,numpts]#vary0[j+1]
                vx1y1=self.__cov[0,numpts]#varx0y0[j+1]
                
            vx1x0=vx0x1
            vx1y0=vy0x1
            vy1x1=vx1y1
            vy1x0=vx0y1
            vy1y0=vy0y1
        
            #find the angle of the line segment counterclockwise from the x-axis
            #op=y1-y0
            #ad=x1-x0
            #angle_line=np.arctan2(op,ad)
            #angle_line_deg=angle_line*(180/np.pi)
    
            #calculate the number of ellipses
            linelen=np.sqrt((x1-x0)**2+(y1-y0)**2)
            numEllipses=linelen/self.__ellipseSpacing
    
            t=0
            dt=1/numEllipses
            while t<1:
                #eq (14) from Shi & Liu
                x_t=(1-t)*x0+t*x1
                y_t=(1-t)*y0+t*y1
                
                #create the (rows,cols) cov matrix
                cov=np.zeros((2,2))
                cov[0,0]=varx_t(t)
                cov[1,0]=varxy_t(t)
                cov[0,1]=varyx_t(t)
                cov[1,1]=vary_t(t)
                
                w=np.linalg.eigvals(cov)
        
                #assign the largest eigenvalue to the semi-major axis
                sql1=np.sqrt(w[0])
                sql2=np.sqrt(w[1])
                A_t=max(sql1,sql2)
                B_t=min(sql1,sql2)
                
                #for any confidence level other than 39% (the orignal gband)
                #sccale the axes according to the associated Mahalanobis distance
                if confidenceLevel==0.6:
                    A_t=A_t*1.3537
                    B_t=B_t*1.3537
                elif confidenceLevel==0.9:
                    A_t=A_t*2.1459
                    B_t=B_t*2.1459
                elif confidenceLevel==0.95:                    
                    A_t=A_t*2.4477
                    B_t=B_t*2.4477
        
                #3rd part of eq (16). This is the rotation angle of the ellipse (angle
                #of the semi-major axis).
                #note that for the special case when the numerator & denominator are zero,
                #it means that the variances for both axes are equal and the covariance is zero.
                #This occurs for a circular error ellipse,
                #In this case, set angle to 0.
                den=varx_t(t)-vary_t(t)
                num=2*varxy_t(t)   
                if den==0:
                    angle_t=np.pi/2
                if num==0 and den==0:
                    angle_t=0
                else:
                    angle_t=np.arctan(num/den)/2
        
                #since A_t is the semi-major axis and is the largest of the eigenvalues
                #if the variance of Y is greater than var X, we need to assign the
                #axis of max variance to Y instead of X. This is done by rotation of pi/2
                if cov[0,0]<cov[1,1]:
                  angle_t=angle_t+np.pi/2
                
                #calculate the ellipse
                theta=np.linspace(start=0, stop=2 * np.pi, num=self.__numPtsPerEllipse)
                x_ell=x_t + A_t * np.cos(theta) * np.cos(angle_t) - B_t * np.sin(theta) * np.sin(angle_t)
                y_ell=y_t + A_t * np.cos(theta) * np.sin(angle_t) + B_t * np.sin(theta) * np.cos(angle_t)
                                
                #create a polygon and union it
                points=np.concatenate((x_ell,y_ell)).reshape(2,-1).transpose()
                poly=Polygon(points)
                if gband==0:
                    gband=poly
                else:
                    gband=gband+poly

                #adaptively change the ellipse spacing to make sure there are no gaps.                    
                dl=dt*linelen
                #print("t: {} dt: {} B_t: {} leng:{}".format(t,dt,B_t,dl))
                if t>0:                    
                    if dl>B_t:
                        dt=np.minimum(dt/2,1/numEllipses)
       
                t=t+dt
                #print(dt,t)
                
                #add the ellipse to the array
                ellipses.append(GBand.Ellipse(x_ell,y_ell,angle_t))
            #END for t in ellipselocations_t:
        #END for j in range(pts_x.size-1):
        
        self.__ellipses=ellipses
        self.__polygon=gband
        
#check if we're executing this script as the main script
if __name__ == '__main__':
    __test()    
    #__test2()
