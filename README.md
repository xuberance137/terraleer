# terraleer
Geospatial Image access and processing 

On max OS X:

Install GDAL and landsat util to run python code. Preferrably in a virtual environment.

Installing GDAL : 
http://gis.stackexchange.com/questions/198425/install-gdal-python-package-in-virtualenv-on-mac

Installing pycurl :  
env ARCHFLAGS="-arch x86_64" pip install pycurl

Install landsat-util :  
pip install landsat-util

TODO:

Create Docker container with preinstalled GDAL and landsat dependencies.
