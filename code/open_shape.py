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
import cPickle
import json
from get_wrs import ConvertToWRS
import matplotlib.pyplot as plt
import numpy as np
from landsat.search import Search
from landsat.downloader import Downloader, RemoteFileDoesntExist, IncorrectSceneId
from landsat.settings import GOOGLE_STORAGE, S3_LANDSAT
from landsat.image import Simple, PanSharpen
from landsat.ndvi import NDVI, NDVIWithManualColorMap
from landsat.image import Simple, PanSharpen


START_DATE = '2015-05-01'
END_DATE = '2015-08-01'
CLOUD_MAX = 25.0 

TEST_RANGE = 2
DEBUG_PRINT = True
POST_DOWNLOAD = False


sceneIndexRefList = [[0, 12], [4, 14], [4], [9], [4], [4], [4], [0, 12], [0, 12], [1], [1], [5, 3], [5, 3], [5, 3, 1], [13, 8, 6], [13, 8], [13, 1], [13], [1], [0], [0], [0, 1, 12], [13], [13, 1], [0, 6], [0], [0, 6], [0, 6], [13, 6], [7, 1], [13], [13, 1], [13, 8], [13], [1], [13], [13, 1], [13, 1], [13, 1], [1], [13, 3], [3], [3], [3], [3], [3], [11, 3], [11, 13, 3], [11, 3], [11], [13, 8], [11, 10, 8], [9], [11], [13], [11], [11, 10], [11], [11], [6], [6], [4], [9], [4, 8], [8], [4, 9, 8], [8], [9, 6], [9, 8, 6], [10], [11, 10], [9, 6], [9], [8], [4, 8], [4, 8], [4, 14], [4], [4, 14], [2, 10], [4], [6], [8], [8], [8], [6], [13, 8], [2, 10], [10], [10], [4, 8], [8], [4], [8], [11, 10], [11, 10], [2, 10], [2, 10], [10]]

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

    pathrowIndex = [] # collection of index sets for each county
    for index in range(len(countyCoord)):
        pathrowIndices = []  #multiple scene indicies for each county
        pathrow = conv.get_wrs(countyCoord[index][1], countyCoord[index][0])  #converting (lat, lon) to WRS {path, row}
        for item in pathrow: #handling multiple path rows for a given (lat, lon)
            pathrowVal = (item['path'], item['row']) 
            for index1 in range(len(pathrowWRS)):
                if pathrowVal == pathrowWRS[index1]:
                    pathrowIndices.append(index1)     
        pathrowIndex.append(pathrowIndices)  
    return pathrowWRS, pathrowIndex

def searchLandsat(pathrowWRS, start_date, end_date, cloud_max):
    s = Search()
    # start_date = '2015-05-01'
    # end_date = '2015-08-01'
    # cloud_max = 30.0 #maximum allowed cloud is 5%

    result =[] 
    for index in range(len(pathrowWRS)):
        paths_rows = str(pathrowWRS[index][0]) + ',' + str(pathrowWRS[index][1])
        searchResult = s.search(paths_rows=paths_rows, start_date=start_date, end_date=end_date, cloud_max=cloud_max)
        if searchResult['status'] == 'SUCCESS':
            result.append(searchResult)
        else: # TODO: Make iterative adjustment till you get some scene data. Right now, one time update by 40.0 for this data set
            print "Increasing Cloud Percentage by 40.0 to get relevant scene data"
            cloud_max = cloud_max + 40.0 
            searchResult = s.search(paths_rows=paths_rows, start_date=start_date, end_date=end_date, cloud_max=cloud_max)
            result.append(searchResult)

    return result    

def downloadLandsat(landsatSceneData):
    sceneIDList = []
    print landsatData

    for item in landsatSceneData:
        print item
        if item['status'] == 'SUCCESS':
            sceneIDList.append(str(item['results'][0]['sceneID']))
        else:
            print "Missing scene bundle identified."
    
    d = Downloader(download_dir='../data/sceneData')
    files = d.download(sceneIDList)

    return files

def downloadClip(scenePaths, countyCoord, pathrowIndex):
    clipPaths = []

    for index in range(0, len(countyCoord)):
        print index
        clipPath = []
        bounds = countyCoord[index]
        for item in pathrowIndex[index]:
            scenePathIndex = item
            p = Simple(path=scenePaths[item], dst_path='../data/clipData', bounds=bounds)
            path = p.run()
            clipPath.append(path)
            p = Simple(path=scenePaths[item], dst_path='../data/clipData', bands=[3,2,1], bounds=bounds)
            path = p.run()
            clipPath.append(path)
            p = NDVI(path=scenePaths[item], dst_path='../data/clipData', bounds=bounds)
            path = p.run()    
            clipPath.append(path)

        clipPaths.append(clipPath)

    return clipPaths


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
    pathrowWRS, pathrowIndex = findPathRow(countyCoord)
    landsatData = searchLandsat(pathrowWRS, START_DATE, END_DATE, CLOUD_MAX)
    filePaths = downloadLandsat(landsatData)
    clipPaths = downloadClip(filePaths, countyCoord, pathrowIndex)
    
    print
    print clipPaths
    print
    print len(countyCoord)
    print len(pathrowIndex)
    print len(clipPaths) 
    print
    for item in clipPaths:
        for item1 in item:
            print item1
        print
    print

    print "Writing files"
    print
    cPickle.dump(clipPaths, open('../data/ImageFilePaths.p', 'wb')) 
    
    jsonfile = open('../data/ImageFilePaths.json', 'w')
    json.dumps(clipPaths, jsonfile, indent=4)
    jsonfile.close()



    # countyCoord = [item.bbox for item in shapes]
    # pathrowWRS = findPathRow(countyCoord)
    # landsatData = searchLandsat(pathrowWRS)
    # filePaths = downloadLandsat(landsatData)
    # processPaths = processLandsat(filePaths)      

    # if DEBUG_PRINT:  
    #         print
    #         for index in range(len(countyCoord)):
    #             print countyCoord[index]
    #         x = np.array(countyCoord, np.float)
    #         delta_x = x[:,2] - x[:,0]
    #         delta_y = x[:,3] - x[:,1]       
    #         print
    #         print "Min longitudinal difference (in deg) : ", delta_x.min()
    #         print "Max longitudinal difference (in deg) : ", delta_x.max()
    #         print "Min latitudinal difference (in deg)  : ", delta_y.min()
    #         print "Max latitudinal difference (in deg)  : ", delta_y.max()
    #         print
    #         for item in pathrowWRS:
    #             print item
    #         print 
    #         print "Number of satellite scenes : ", len(landsatData)
    #         print
    #         print landsatData
    #         print
    #         print filePaths
    #         print
    #         print processPaths



