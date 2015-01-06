"""
Statistical distributions.
Used by crispobjects and fuzzyobjects to generate samples from statistical distributions.
"""
import numpy as np
import utilities

#constants denoting distributions
#:Normal distribution identifier
DIST_NORM="normal"
#:Truncated Normal distribution identifier
DIST_TNORM="truncated normal"
#:Uniform distribution identifier
DIST_UNIF="uniform"

def generateNormal(mean,cov,n):
    """
    Generates random numbers from the multivariate normal.
    
    - *mean*: numpy array of means, length N.
    - *cov*: Covariance matrix. Numpy array of size NxN
    - *n*: the number of samples to generate
    
    Returns and n-by-N array where the columns are the sequence of random
    numbers corresponding to each of the specified distributions
    """
    
    if n<1:
        raise Exception("The number of realizations must be >=1")
        
    if type(cov)!=np.ndarray:
        raise Exception("The covariance matrix must be a numpy array")
    
    if np.ndim(mean)!=1:
        raise Exception("The means array must be 1D")
        
    if np.ndim(cov)!=2:
        raise Exception("The covariance matrix must have 2 dimensions")
    
    if np.size(cov,0)!=np.size(mean) or np.size(cov,1)!=np.size(mean):
        raise Exception("The size of the correlation matrix must be N x N")

    #check for positive definetness
    #https://github.com/numpy/numpy/pull/3938
    try:
        np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        print("Warning: The covariance matrix is not positive definite."
        "The covariance of the output will differ from what is expected")
    
    out=np.random.multivariate_normal(mean,cov,n)
    return out
    
def generateTruncatedNormal(mean, var,a,b,n):
    """
    Generates a univariate truncated normal between a and b.
    
    - *mean*, var: mean and variance
    - *a, b*: lower and upper bounds
    - *n*: the number of numbers to generate
    
    Note: the standard deviation sqrt(var) must be less than (b-a)/2
    """
    
    if np.size(mean)!=1 or np.size(var)!=1:
        raise Exception("The means and variance must be scalars")
        
    if np.size(a)!=1 or np.size(b)!=1:
        raise Exception("The values of a and b must be scalar")
        
    if np.sqrt(var)>=(b-a)/2:
        raise Exception("The variance must be < (b-a)/2")
    
    sigma=np.sqrt(var
    )
    U=np.random.rand(n)
    alpha=(a-mean)/sigma
    beta=(b-mean)/sigma

    #use the formula from wikipedia
    return __stdnormCDF_inv(  __stdnormCDF(alpha) + U*(__stdnormCDF(beta)-__stdnormCDF(alpha))  )*sigma + mean
    
def generateUniform(lbound,ubound,cor,n):
    """
    Generates random numbers from a multivariate uniform distribution.
    
    - *lbound, ubound*: numpy arrays of length N containing the lower and upper bounds of the 
      uniformly distributed variables.
    - *cor*: a numpy N-by-N array containing the correlation coefficients
    - *n*: the number of random numbers to generate
    
    Returns an n-by-N array where the columns correspond to variates and rows are the samples
    """
    if n<1:
        raise Exception("The number of realizations must be >=1")
    
    if type(cor)!=np.ndarray:
        raise Exception("The correlation matrix must be a numpy array")
        
    if np.ndim(lbound)!=1 or np.ndim(ubound)!=1:
        raise Exception("The lbound and ubound array must be 1D")

    if np.size(lbound)!=np.size(ubound):
        raise Exception("the size of lbound and ubound must be the same!")        
        
    if np.ndim(cor)!=2:
        raise Exception("The correlation matrix must have 2 dimensions")
        
    if np.size(cor,0)!=np.size(lbound) or np.size(cor,1)!=np.size(lbound):
        raise Exception("The size of the correlation matrix must be N x N")
        
    #check for positive definetness
    #https://github.com/numpy/numpy/pull/3938
    try:
        np.linalg.cholesky(cor)
    except np.linalg.LinAlgError:
        print("Warning: The correlation matrix is not positive definite."+ 
        "The covariance of the output will differ from what is expected")        

    #replace the correlation by the modified correlation coefficient.
    #because transforming the normally generated values to uniform
    #may change the correlation, we transform the correlation matrix 
    #with the relationship below. After transpfrmation, the uniformly
    #distributed variables will have the original correlation
    cor_pearson=2*np.sin(cor*np.pi/6)
    
    #generate N correlated standard normal variables
    std_norm=np.random.multivariate_normal(np.zeros(np.size(lbound)),cor_pearson,n)
    
    #map the generated numbers through the CDF of the standard normal
    U=__stdnormCDF(std_norm)
        
    #rescale the standard uniform distributed values to the original bounds
    for i in range(np.size(lbound)):
        U[:,i]=lbound[i]+(ubound[i]-lbound[i])*U[:,i]
        
    return U
    
def __stdnormCDF(x):
    """Standard normal CDF. returns the probability of X<x (area under the PDF curve)"""
    return (1+utilities.erf( x/np.sqrt(2) ))/2
def __stdnormPDF(x):
    """Standard normal PDF. returns the probability density at x"""
    return np.exp((-x**2)/2)/np.sqrt(2*np.pi)

def __stdnormCDF_inv(p):
    """
    Inverse of standard normal CDF.
    
    From http://www.johndcook.com/python_phi_inverse.html. Vectorized by me.
    """
    def rational_approximation(t):

        # Abramowitz and Stegun formula 26.2.23.
        # The absolute value of the error should be less than 4.5 e-4.
        c = [2.515517, 0.802853, 0.010328]
        d = [1.432788, 0.189269, 0.001308]
        numerator = (c[2]*t + c[1])*t + c[0]
        denominator = ((d[2]*t + d[1])*t + d[0])*t + 1.0
        return t - numerator / denominator
    
    assert np.all([p>0])  and np.all([p<1])

    #See article above for explanation of this section.
    P_lt_half=-rational_approximation( np.sqrt(-2.0*np.log(p)) )
    P_ge_half=rational_approximation( np.sqrt(-2.0*np.log(1.0-p)) )
    
    #if p<0.5 assign the value P_lf_half to that matrix cell
    #else assign P_ge_half to the cell
    out= [p<0.5]*P_lt_half + [p>=0.5]*P_ge_half
    return out[0,:]

    