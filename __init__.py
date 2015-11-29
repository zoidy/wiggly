import sys, os   

#important so sub packages can be used from wiggly folder
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not parent_dir+'\wiggly' in sys.path:
    sys.path.insert(1, parent_dir+'\wiggly')

#allow accessing these things when you import wiggly
import utilities
import shapely
import shapely.geometry
import fuzz

import FObjects
import CrispObjects
import FuzzyObjects
import objectmanager
import uq,sa

# http://mikegrouchy.com/blog/2012/05/be-pythonic-__init__py.html
# name1 and name2 will be available in calling module's namespace 
# when using "from package import *" syntax
__all__ = ["utilities","CrispObjects","FuzzyObjects","FObjects","objectmanager","uq","sa"]