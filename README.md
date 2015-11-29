wiggly
======
### Documentation
See (http://zoidy.github.io/wiggly/)

### Dependencies
* wiggly
  * puq-wiggly
    * jsonpickle 0.7.1
    * h5py 2.3.0
    * sympy 0.7.5
    * puqutil (part of puq-wiggly)
    * SALib (https://github.com/zoidy/SALib)
  * shapely 1.5.1
  * fuzz 0.4.2
  * numpy 1.7.1
  * scipy 0.13.0
    
QGIS 2.6:
* need to update QGIS Python numpy version to at least 1.7.1 and scipy 0.13
        (can do it manually, just delete old numpy and scipy dir and egg-info files
        and replace with new ones). Also need all of their dependencies.
* need to update shapely to at least 1.5.1
        (can do it manually, delete old shapely dir and replace with new one and corresponding
        egg-info)
* tkinter  (can extract from python 32 or 64bit installer) 
  * lib-tk  in Lib (not site packages)
  * _tkinter in libs (copy entire folder)
  * _tkiniter, tcl85.dll, tclpip85.dll, tk85.dll in DLLs
        copy entire tcl directory
* create a bootstrap script to be able to launch a QGIS python shell (see my scripts)
* edit processing to reduce bad file descriptor errors (fixed after 2.4?)
     (https://github.com/qgis/QGIS/commit/322cd0d03bfa7b1971c04a0adff973619289f583)

### License
Copyright (C) 2015 Fernando Rios.  Licensed under the LGPL v3
     
### License for PUQ-Wiggly
License for PUQ: MIT License. The modified version of PUQ used by Wiggly is also MIT licensed.

### License for SALib
Licensed under the GNU Lesser General Public License.

The Sensitivity Analysis Library is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

The Sensitivity Analysis Library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the Sensitivity Analysis Library.  If not, see <http://www.gnu.org/licenses/>.
