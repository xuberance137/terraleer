#!/Users/gopal/projects/harvesting/code/venv/bin/python

"""

Finding relevant satellite scenes based on a collection of shape coordinates


Info:

All Landsat scene identifiers are based on the following naming convention: LXSPPPRRRYYYYDDDGSIVV
where:
L = Landsat
X = Sensor
S = Satellite
PPP = WRS path
RRR = WRS row
YYYY = Year
DDD = Julian day of year
GSI = Ground station identifier
VV = Archive version number
Examples:
LC80390222013076EDC00 (Landsat 8 OLI and TIRS)
"""


import shapefile
import matplotlib
from get_wrs import ConvertToWRS
import matplotlib.pyplot as plt
import numpy as np
from landsat.search import Search
from landsat.downloader import Downloader, RemoteFileDoesntExist, IncorrectSceneId
from landsat.settings import GOOGLE_STORAGE, S3_LANDSAT
from landsat.image import Simple, PanSharpen
from landsat.ndvi import NDVI, NDVIWithManualColorMap


TEST_RANGE = 2
DEBUG_PRINT = True
POST_DOWNLOAD = False

# Order preserving unique sequence generation
def f(seq): 
  seen = set()
  return [x for x in seq if x not in seen and not seen.add(x)]

#plot county map based on rectangular coordinate data from shape file
def plotCountyMap(countyCoord):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = ['red', 'yellow', 'green', 'blue']
    x = np.array(countyCoord, np.float)
    maxCoord = x.max(axis=0)
    minCoord = x.min(axis=0)

    for index in range(len(countyCoord)):   
        rect1 = matplotlib.patches.Rectangle((countyCoord[index][0], countyCoord[index][1]), countyCoord[index][2]-countyCoord[index][0], countyCoord[index][3]-countyCoord[index][1], color=colors[3], alpha=0.4)
        ax.add_patch(rect1)
        plt.xlim([minCoord[0]-1, maxCoord[2]+1])
        plt.ylim([minCoord[1]-1, maxCoord[3]+1])

    plt.ylabel('Latitude (in deg)')
    plt.xlabel('Longitude (in deg)')
    plt.title('County Coodinates Map')
    plt.show()

# transform (lat, lon) coordinate data to unique set of {path, row} data for landsat WRS access
def findPathRow(countyCoord):
    conv = ConvertToWRS()
    pathrowWRS = []

    for index in range(len(countyCoord)):
        pathrow = conv.get_wrs(countyCoord[index][1], countyCoord[index][0])  #converting (lat, lon) to WRS {path, row}
        for item in pathrow: #handling multiple path rows for a given (lat, lon)
            pathrowWRS.append((item['path'], item['row']))
    
    #find unique (path, row) tuples
    pathrowWRS = list(set(pathrowWRS))

    return pathrowWRS

def searchLandsat(pathrowWRS):
    s = Search()
    start_date = '2015-05-01'
    end_date = '2015-08-01'
    cloud_max = 25.0 #maximum allowed cloud is 5%

    result =[] 
    for index in range(len(pathrowWRS)):
        paths_rows = str(pathrowWRS[index][0]) + ',' + str(pathrowWRS[index][1])
        result.append(s.search(paths_rows=paths_rows, start_date=start_date, end_date=end_date, cloud_max=cloud_max))

    return result    

def downloadLandsat(landsatSceneData):
    for item in landsatSceneData:
        if item['results']:
            sceneIDList.append(str(item['results'][0]['sceneID']))
        else:
            print 'Missing scene bundle identified'
    
    d = Downloader(download_dir='../data/sceneData')
    files = d.download(sceneIDList)

    return files

def processLandsat(filePaths):
    processPath = []
    for filePath in filePaths:
        p = NDVI(path=filePath, dst_path='../data/NDVI')
        path = p.run()
        processPath.append(path)

    return processPath

if __name__ == '__main__':

    sf = shapefile.Reader("../data/Shapefile/IOWA_Counties/IOWA_Counties.shp")

    shapes = sf.shapes()
    countyCoord = [item.bbox for item in shapes]
    pathrowWRS = findPathRow(countyCoord)
    landsatData = searchLandsat(pathrowWRS)
    filePaths = downloadLandsat(landsatData)
    processPaths = processLandsat(filePaths)      

    if DEBUG_PRINT:
        if POST_DOWNLOAD == False:        
            print
            for index in range(len(countyCoord)):
                print countyCoord[index]
            x = np.array(countyCoord, np.float)
            delta_x = x[:,2] - x[:,0]
            delta_y = x[:,3] - x[:,1]       
            print
            print "Min longitudinal difference (in deg) : ", delta_x.min()
            print "Max longitudinal difference (in deg) : ", delta_x.max()
            print "Min latitudinal difference (in deg)  : ", delta_y.min()
            print "Max latitudinal difference (in deg)  : ", delta_y.max()
            print
            for item in pathrowWRS:
                print item
            print 
            print "Number of satellite scenes : ", len(landsatData)
            print
            print landsatData
        else:
            print
            print filePaths
            print
            print processPaths



