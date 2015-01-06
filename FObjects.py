"""
Base class for all F-Objects. Non instantiable.
"""
import utilities
import numpy as np


#Elements outside the __init__ method are static elements, it means, they belong to the class.
#Elements inside the __init__ method are elements of the object (self), they don't belong to the class.

class FObject(object):
    """
    Base class for all F-Objects.

    - *x, y*: Numpy arrays of x and y coordinates, each of length N.
    - *isClosed*: Boolean indicating whether the object will be treated as having a closed boundary.
    """

    def __init__(self,x,y,isClosed):
        #using the ABC metaclass to define an abstract base class
        #doesn't work if you want the base class to do something
        #Instead, just prevent instantiation manually.
        if type(self) is FObject:
            raise NotImplementedError('FObject.__init__(): abstract class')

        if np.size(x)<1:
            raise Exception("The number of points for a crisp object must be >= 1")
        if isClosed and np.size(x)==1:
            raise Exception("A single point cannot be closed")
        if np.size(x)!=np.size(y):
            raise Exception("the size of x and y must be the same size!")

        #use single underscore. Derived classes have no special access to the attributes defined
        #in its parent; there is no equivalent of C++'s "protected" variables.
        #With double underscore, child classes can't access these without "unmangling"
        #the name.
        self._x=x
        self._y=y
        self._isClosed=isClosed

        #the type of distribution for the most recently called method
        #that generates realizations. Is one of the constant values in _distributions,
        #or is None if no realizations have yet been generated
        self._lastDistr=None

    @property
    def coords(self):
        """
        Returns 2 arrays x,y containing the x and y coordinates of the vertices
        """
        return self._x,self._y

    @property
    def polygon(self):
        """
        Returns a Shapely polygon representing the original shape.
        """
        return utilities.pointslist2Shape(self._x,self._y,self._isClosed)

    @property
    def realizations(self):
        """
        Abstract property which returns the realizations of an object. See
        implementation-specific details in child classes.
        """
        raise NotImplementedError("FObject.realizations: abstract property")
    @property
    def isClosed(self):
        """
        Does the object have a closed or open boundary?
        """
        return self._isClosed
