"""
This module implements the Gband error model using the Shapely
library.

This version is preferred over gband2 which uses
the GPC libary instead of Shapely
"""

#in pyscripter, need to add these to special packages for autocompletion
import numpy as np
import matplotlib.pyplot as plt
import shapely.geometry
import shapely.ops
import utilities
import sys

def __test2():

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
    cov=utilities.cor2cov(cor,np.concatenate((var_x,var_y)))

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
    cov=utilities.cor2cov(cor,np.concatenate((varx0,vary0)))

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


        if plotEllipses:
            for e in band.ellipses:
                plt.plot(e.x,e.y,'-k')

        for i in range(len(points)):
            band_x=points[i][0]
            band_y=points[i][1]
            plt.fill(band_x,band_y,'-',fill=False)

        plt.plot(testPoint[0],testPoint[1],'o')
        plt.title('({},{}) Inside:{}'.format(testPoint[0],testPoint[1],
                        band.polygon.intersection(shapely.geometry.Point(testPoint[0],testPoint[1]))))
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


########################################################################################


class GBand:
    """
    Represents a G-Band object.

    - *pts_x, pts_y*: numpy arrays with thelocation of uncertain points. Each array has length N
    - *cov*: the covariance matrix between the points. It is of size 2N-by-2N
    - *isClosed*: if True, considers the points to represent a closed boundary
    - *EllipseSpacing*: maximum spacing between error ellipses along a line segment. This value is automatically adjusted if the spacing is too large
    - *numPtsPerEllipse*: the number of points to evaluate each ellipse on.

    Note: It is up to the caller to make sure that the variance and covariance
    arrays are appropriately constructed taking holes into account
    """

    class Ellipse:
        """
        An error ellipse used by GBand.

        .. _gband.ellipse:

        - *ellipsePoly*: shapely Polygon object representing the ellipse
        - *angle*: angle of the semi major axis wrt to X-Y (in rad)
        """
        def __init__(self,ellipsePoly,angle):
            self.__ell=ellipsePoly
            self.__angle=angle
        @property
        def angleDeg(self):
            """
            The orientation angle of the ellipse. *ReadOnly*
            """
            return self.__angle*(180/np.pi)
        @property
        def x(self):
            """
            The x coordinates of the ellipse as a numpy array. *ReadOnly*
            """
            return np.array(self.__ell.exterior.coords)[:,0]
        @property
        def y(self):
            """
            The y coordinates of the ellipse as a numpy array. *ReadOnly*
            """
            return np.array(self.__ell.exterior.coords)[:,1]

    def __init__(self,pts_x,pts_y,cov,isClosed=True,ellipseSpacing=1,numPtsPerEllipse=20):

        if pts_x.size!=pts_y.size:
            raise Exception("There must be the same number of elements in pts_x and pts_y")
        if isClosed and pts_x.size<3:
            raise Exception("Need at least 3 points for a closed polygon")
        if 2*np.size(pts_x)!=np.size(cov,0) or 2*np.size(pts_x)!=np.size(cov,1):
            raise Exception("Covariance matrix must be 2N-by-2N")
        if np.any(np.diagonal(cov)<=0):
            raise Exception("Invalid covariance matrix. One of the diagonal entries is <=0")

        #http://stackoverflow.com/questions/5320324/testing-if-a-numpy-array-is-symmetric
        if not np.allclose(np.transpose(cov), cov):
            raise Exception("Covariance matrix must be symmetric")

        self.__pts_x=pts_x
        self.__pts_y=pts_y
        self.__cov=cov
        self.__ellipseSpacing=ellipseSpacing
        self.__numPtsPerEllipse=numPtsPerEllipse
        self.__isClosed=isClosed
        self.__ellipses=[]    #The array of Ellipse objects associated with the gband.
        self.__polygon=None     #an object of type Polygon (shapely)

    @property
    def polygon(self):
        """Returns a shapely Polygon object generated after a call to generateGBand. *read only*"""
        return self.__polygon
    @property
    def ellipses(self):
        """The collection of :ref:`Ellipse <gband.ellipse>` objects that comprise the G-band
        after a call to generateGBand *read only*"""
        return self.__ellipses

    @property
    def polygonPoints(self):
        """
        returns a list of numpy arrays of x and of y points that make up the gband.
        The first index is the contour. The second index selects the array of
        x and y points for that contour. E.g., ret[0][1] is the first contour, y points. *Read only*.

        Plot with fill instead of plot to ensure the boundary is closed off
        """
        ret=[]

        if utilities.isShapelyPolygon(self.__polygon):
            #if the gband is a single polygon, get the x and y arrays
            #of the exterior LinearRing
            pts_x=np.array(self.__polygon.exterior)[:,0]
            pts_y=np.array(self.__polygon.exterior)[:,1]
            ret.append([pts_x,pts_y])
            for interior in self.__polygon.interiors:
                pts_x=np.array(interior)[:,0]
                pts_y=np.array(interior)[:,1]
                ret.append([pts_x,pts_y])
        elif utilities.isShapelyMultiPolygon(self.__polygon):
            #if the gband is a multi polygon need to do the same as
            #above for all the component polygons
            for c in range(len(self.__polygon.geoms)):
                aPoly=self.__polygon.geoms[c]
                pts_x=np.array(aPoly.exterior)[:,0]
                pts_y=np.array(aPoly.exterior)[:,1]
                ret.append([pts_x,pts_y])
                for interior in aPoly.interiors:
                    pts_x=np.array(interior)[:,0]
                    pts_y=np.array(interior)[:,1]
                    ret.append([pts_x,pts_y])
        else:
            raise Exception("The gband is invalid")

        return ret

    def plot(self,fignum=-1,plotEllipses=False,plotLine=True):
        """
        Plots the gband using matplotlib but doesn't show the plot. To show it, call matplotlib's
        show() function.

        - *fignum*: The figure number to which the band will be plotted. If fignum<0, it will plot into a new figure
        - *plotEllipses*: Whether to show the associated error ellipses or not.
        - *plotLine*: plots the mean line in a light gray color.
        """
        if fignum<0:
            plt.figure()
        else:
            plt.figure(fignum)
        plt.axis('equal')

        #get the points comprising the gband. Plot each
        #contour (part of the polygon) in sequence
        contours=self.polygonPoints

        for c in range(len(contours)):
            band_x=contours[c][0]
            band_y=contours[c][1]

            #plot gband
            plt.fill(band_x,band_y,'-',fill=False,zorder=99)

        if plotEllipses:
            for e in self.ellipses:
                plt.plot(e.x,e.y,'-',color='0.5')

        if plotLine:
            if self.__isClosed:
                plt.plot(np.r_[self.__pts_x,self.__pts_x[0]],
                         np.r_[self.__pts_y,self.__pts_y[0]],color='0.85')
            else:
                plt.plot(self.__pts_x,self.__pts_y,color='0.85')




    def generateGBand(self, confidenceLevel=0.39):
        """
        Generates the Gband.

        - *ConfidenceLevel*: is a float and can be one of {0.39, 0.6, 0.9, 0.95}

        Returns nothing. To access the results, use the polygon, polygonpoints, or ellipses properties
        of the GBand object
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

        ells=[]

        if self.__isClosed:
            numpts=self.__pts_x.size
        else:
            if self.__pts_x.size<3:
                numpts=self.__pts_x.size
            else:
                numpts=self.__pts_x.size-1

        numpts=self.__pts_x.size
        print('Calculating g-band. num vertices {}'.format(numpts))
        for j in range(numpts):
            x0=self.__pts_x[j]
            y0=self.__pts_y[j]
            vx0=self.__cov[j,j]
            vy0=self.__cov[numpts+j,numpts+j]
            vx0y0=self.__cov[j,numpts+j]
            vy0x0=vx0y0

            print("process vertex pairs")
            print("\tidx:{} -- vtx0={}".format(j,(x0,y0)))
            if j>=numpts-1:
                if self.__isClosed:
                    #if its a closed polygon, and we're at the last vertex wrap around
                    #to the beginning
                    x1=self.__pts_x[0]
                    y1=self.__pts_y[0]
                    vx0x1=self.__cov[j,0]
                    vx0y1=self.__cov[j,numpts]
                    vy0x1=self.__cov[0,numpts+j]
                    vy0y1=self.__cov[numpts+j,numpts]
                    vx1=self.__cov[0,0]
                    vy1=self.__cov[numpts,numpts]
                    vx1y1=self.__cov[0,numpts]
                    print("\tObject has closed boundary. Wrapping the vertex index")
                    print("\tidx:{} -- vtx1={}".format(0,(x1,y1)))
                elif numpts==1:
                    #for the special case of a single point
                    x1=x0
                    y1=y0
                    vx0x1=0
                    vx0y1=0
                    vy0x1=0
                    vy0y1=0
                    vx1=vx0
                    vy1=vy0
                    vx1y1=0
                    print("\tObject is a single point. Processing done")
                else:
                    print("\tLast vertex and boundary is not closed. Processing done")
                    sys.stdout.flush()
                    break
            else:
                x1=self.__pts_x[j+1]
                y1=self.__pts_y[j+1]
                vx0x1=self.__cov[j,j+1]
                vx0y1=self.__cov[j,numpts+j+1]
                vy0x1=self.__cov[j+1,numpts+j]
                vy0y1=self.__cov[numpts+j,numpts+j+1]
                vx1=self.__cov[j+1,j+1]
                vy1=self.__cov[numpts+j+1,numpts+j+1]
                vx1y1=self.__cov[j+1,numpts+j+1]
                print("\tidx:{} -- vtx1={}".format(j+1,(x1,y1)))

            print("\tvx0:{} vy0:{} vx0y0:{}".format(vx0,vy0,vx0y0))
            print("\tvx1:{} vy1:{} vx1y1:{}".format(vx1,vy1,vx1y1))
            print("\tvx0x1:{} vx0y1:{} vy0x1:{} vy0y1:{}".format(vx0x1,vx0y1,vy0x1,vy0y1))
            sys.stdout.flush()
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
                #print('t:{}  A:{}\tB:{}\tangle:{}'.format(t,A_t,B_t,angle_t*180./np.pi))
                theta=np.linspace(start=0, stop=2 * np.pi, num=self.__numPtsPerEllipse)
                x_ell=x_t + A_t * np.cos(theta) * np.cos(angle_t) - B_t * np.sin(theta) * np.sin(angle_t)
                y_ell=y_t + A_t * np.cos(theta) * np.sin(angle_t) + B_t * np.sin(theta) * np.cos(angle_t)

                #prepare the points for polygon creation as an N x 2 array
                #points=np.concatenate((x_ell,y_ell)).reshape(2,-1).transpose()
                points=np.vstack((x_ell,y_ell)).transpose()

                #construct a linestring from the points and create an ellipse polygon
                lr=shapely.geometry.asLinearRing(points)
                ell=shapely.geometry.Polygon(lr)
                ells.append(ell)
                self.__ellipses.append(GBand.Ellipse(ell,angle_t))

                #Gradually decrease the ellipse spacing if it's too big
                #to reduce gaps between ellipses, while at the same time,
                #trying to resepect the user specified ellipse spacing
                dl=dt*linelen
                #print("t: {} dt: {} B_t: {} leng:{}".format(t,dt,B_t,dl))
                if t>0:
                    if dl>B_t:
                        dt=np.minimum(dt/2,1/numEllipses)

                t=t+dt
                #print(dt,t)

            #END while t<1
        #END for j in range(numpts):

        #union all the ellipses to form the gband
        self.__polygon=shapely.ops.unary_union(ells)

#check if we're executing this script as the main script
if __name__ == '__main__':
    __test()
    #__test2()
