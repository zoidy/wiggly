#################################################################
# Example which shows how to use Wiggly in a mixed uncertainty
# analysis.
#
# The model consists of a mixture of shapes and non-shape
# parameters, both probabilistic and fuzzy.
#################################################################

#REQUIRED to protect code with __name__=='__main__'.
#Since in this example, we will pass the test model as a function handle,
#Wiggly will make use of multiprocessing. See the Python documentation
#on multiprocessing for more details.
if __name__=='__main__':

    import random
    import numpy as np
    import matplotlib.pyplot as plt
    import sys
    sys.path.append("P:\\dissertation\\4-research\\1.my work\\software")
    import wiggly as w
    import test_model


    plt.close('all')

    #the number of probabilistic and fuzzy samples to use.
    #Keep in mid the nested nature of a mixed evaluation (Fuzzy outer loop, probabilistic inner)
    n_fuzzy=10
    n_prob=100

    seed=12345
    random.seed(seed)
    np.random.seed(seed)

    #the object manager handles all shapes and shape conversions
    puqOm=w.objectmanager.ObjectManager()

    #*************************************
    # probabilistic shapes and parameters
    #*************************************

    #####
    #define the OPD polygon coordinates(ccw)
    pt_x=np.r_[480.,485,520,510,520]
    pt_y=np.r_[123.,100,105,110,117]

    #variances of the coordinates of each vertex
    pt_xv=np.r_[10.,20,10,10,10]
    pt_yv=np.r_[10.,20,10,10,10]

    #all vertices of this polygon are uncertain
    uncertain_pts=[1,1,1,1,1]

    #correlation matrix. Vertex is uncorrelated with neighbors, but correlated between x, y
    cor=np.r_[[
        #x0     x1      x2      x3      x4      y0      y1      y2      y3      y4
        [1,     0,      0,      0,      0,      0,      0,      0,      0,      0],  #x0
        [0,     1,      0,      0,      0,      0,      0,      0,      0,      0],  #x1
        [0,     0,      1,      0,      0,      0,      0,      0,      0,      0],  #x2
        [0,     0,      0,      1,      0,      0,      0,      0,      0,      0],  #x3
        [0,     0,      0,      0,      1,      0,      0,      0,      0,      0],  #x4
        [.8,    0,      0,      0,      0,      1,      0,      0,      0,      0],  #y0
        [0,     .8,     0,      0,      0,      0,      1,      0,      0,      0],  #y1
        [0,     0,      .8,     0,      0,      0,      0,      1,      0,      0],  #y2
        [0,     0,      0,      .8,     0,      0,      0,      0,      1,      0],  #y3
        [0,     0,      0,      0,      .8,     0,      0,      0,      0,      1],  #y4
    ]]

    cor=w.utilities.makeSymmetric(cor)
    cov=w.utilities.cor2cov(cor,np.concatenate((pt_xv,pt_yv)))

    do=w.CrispObjects.DeformableObject(pt_x,pt_y,cov,uncertain_pts)

    #####
    #define the OPR polyline
    pt_x=np.r_[470,487,502,523]
    pt_y=np.r_[140,145,142,143]+20

    #the last vertex is certain
    uncertain_pts=[1,1,1,0]

    #set the point of rotation to be the centroid
    centroid_x=np.mean(pt_x)
    centroid_y=np.mean(pt_y)

    #variance of the coordinates of the centroid
    centroid_xv=2.
    centroid_yv=2.

    #mean and variance of the rotation angle in degrees
    theta_mn=0.
    theta_v=10

    variances=np.r_[centroid_xv,centroid_yv,theta_v]

    #no correlation
    cor=np.r_[[
                #x      y       theta
                [1,     0,      0], #x
                [0,     1,      0], #y
                [0,     0,      1]  #theta
            ]]
    cor=w.utilities.makeSymmetric(cor)
    cov=w.utilities.cor2cov(cor,variances)

    ro=w.CrispObjects.RigidObject(pt_x,pt_y,centroid_x,centroid_y,theta_mn,cov,uncertain_pts,isClosed=False)

    #####
    #add the shapes to the object manager
    do.generateNormal(n=n_prob)
    ro.generateNormal(n=n_prob)
    puqOm.addObject(do,name='OPD')
    puqOm.addObject(ro,name='OPR')

    #uncomment below to plot the shapes if desired
    do.plot(1)
    ro.plot(1)
    print('DO perim:{}  area:{}'.format(do.polygon.length,do.polygon.area))
    print('RO perim:{}  area:{}'.format(ro.polygon.length,ro.polygon.area))

    #####
    #add the non-shape parameters
    #see the Wiggly documentation for the UQ constructor for details on the format of these
    #dictionaries.
    probVars={}
    consts={}
        
    probVars['x']={'dist':'normal', 'kwargs':{'mean':0, 'dev':6}, 'desc':'normal, mu=0 sd=6'}
    probVars['y']={'dist':'uniform', 'kwargs':{'min':-3,'max':7.5}, 'desc':'uniform, min=-3, max=7.5'}
    consts['c']={'value':1.103, 'desc':'constant c'}

    #*************************************
    # fuzzy objects and params.
    #*************************************

    #define the alpha-cuts to evaluate. Maximum is 11 alpha cuts.
    acuts=np.linspace(0,1,num=11)

    #####
    #define the OFE polygon (ccw)
    pt_x=np.r_[480.,485,520,510,520]+60
    pt_y=np.r_[123.,100,105,110,117]

    #define fuzzy numbers for all the edges.
    #trapezoidal fuzzy numbers are in the form
    #   (kernel_lower,kernel_upper), (support_lower,support_upper)
    edgeMembFcn=[w.fuzz.TrapezoidalFuzzyNumber((0, 0), (0, 0)),
                 w.fuzz.TrapezoidalFuzzyNumber((0, 0), (-3, 3)),
                 w.fuzz.TrapezoidalFuzzyNumber((-1.5, 1.5), (-5, 7)),
                 w.fuzz.TrapezoidalFuzzyNumber((-1, 1), (-3, 3)),
                 w.fuzz.TrapezoidalFuzzyNumber((-.75, .75), (-1, 1))]

    edo=w.FuzzyObjects.EdgeDefinedObject(pt_x,pt_y,edgeMembFcn,isClosed=True)

    #####
    #define the OFV line (vertex defined)
    pt_x=np.r_[470,487,502,523]+60
    pt_y=np.r_[140,145,142,143]+20
    membFcn_x=[ w.fuzz.TrapezoidalFuzzyNumber((0, 0), (-2, 2)),
                w.fuzz.TrapezoidalFuzzyNumber((-1.5, 1.5), (-2, 2)),
                w.fuzz.TrapezoidalFuzzyNumber((-1, 1), (-1.5, 3)),
                w.fuzz.TrapezoidalFuzzyNumber((0, 0), (0, 0))]
    membFcn_y=membFcn_x

    vdo=w.FuzzyObjects.VertexDefinedObject(pt_x,pt_y,membFcn_x,membFcn_y,isClosed=False)

    #####
    #add the shapes to the object manager
    edo.generateRealizations(n_fuzzy,acuts,method='random')
    vdo.generateRealizations(n_fuzzy,acuts,method='random')
    puqOm.addObject(edo,name='OFE')
    puqOm.addObject(vdo,name='OFV')

    #uncomment below to plot the shapes if desired
    vdo.plot(1)
    edo.plot(1)
    print('EDO perim:{}  area:{}'.format(edo.polygon.length,edo.polygon.area))
    print('VDO perim:{}  area:{}'.format(vdo.polygon.length,vdo.polygon.area))

    #####
    #add the non-shape parameters
    #see the Wiggly documentation for the UQ constructor for details on the format of these
    #dictionaries.
    fuzzyVars={}
    fuzzyVars['f']={'desc':'approximation to N(0,6) truncated at 1st and 99th percentiles',
                    'sl':-13.95,'su':13.95,'kl':-2.1,'ku':2.1}
    fuzzyVars['g']={'desc':'resembles a uniform dist min:-3 max:7.5',
                    'sl':-3,'su':7.5,'kl':-3,'ku':7.5}


    ##############

    #set the working directory if desired. If None, it will use the current
    #Python working directory.
    wdir=None

    #run the analysis. The output files will be saved in a subdirectory
    #of wdir, named using the current date and time.
    uq=w.uq.UQ(testProgFunc=test_model.run,workingDir=wdir,
               seed=seed,objMgr=puqOm,
               probVars=probVars,n_prob=n_prob,
               fuzzyVars=fuzzyVars,n_fuzzy=n_fuzzy,fuzzyVarACuts=acuts,
               consts=consts)

    uq.run(dryrun=False,parallel_jobs=12)

    #when we're finished, show the plots
    plt.show()