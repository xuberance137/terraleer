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

REF:

GDAL python
https://pcjericks.github.io/py-gdalogr-cookbook/raster_layers.html
Obtain Latitude and Longitude from a GeoTIFF File
http://stackoverflow.com/questions/2922532/obtain-latitude-and-longitude-from-a-geotiff-file
http://gis.stackexchange.com/questions/6669/converting-projected-geotiff-to-wgs84-with-gdal-and-python

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
from locateShapeData import *
import gdal
import osr
import os
from subprocess import call
import json

TEST_RANGE = 2
DEBUG_PRINT = False
POST_DOWNLOAD = False

START_DATE = '2015-05-01'
END_DATE = '2015-08-01'
CLOUD_MAX = 5.0 #maximum allowed cloud is 5%


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

def searchLandsat(pathrowWRS, START_DATE, END_DATE, CLOUD_MAX):
    s = Search()
    start_date = START_DATE
    end_date = END_DATE
    cloud_max = CLOUD_MAX #maximum allowed cloud is 5%

    result =[] 
    for index in range(len(pathrowWRS)):
        paths_rows = str(pathrowWRS[index][0]) + ',' + str(pathrowWRS[index][1])
        result.append(s.search(paths_rows=paths_rows, start_date=start_date, end_date=end_date, cloud_max=cloud_max))

    return result    

def downloadLandsat(landsatSceneData):
    sceneIDList = []
    #print landsatData

    for item in landsatSceneData:
        #print item
        if item['status'] == 'SUCCESS':
            sceneIDList.append(str(item['results'][0]['sceneID']))
        else:
            print "Missing scene bundle identified."
    
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

def sampleImage(pts, coord, processPath):

    # check for sample point locations 
    for item in pts:
        if item[0] < coord[0] or item[0] > coord[2] or item[1] < coord[1] or item[1] > coord[3]:
            print "[ISSUE ALERT] Sample point out of county boundary ..."

    # Converting from projected (PROJ4) to latlon (WGS84)
    infile_name = processPath[0]
    outfile_name = infile_name[:-4] + '_latlom' + infile_name[-4:]
    #Reference command line
    #gdalwarp NDVI_PROJ4.TIF NDVI_WGS84.TIF -t_srs "+proj=longlat +ellps=WGS84"
    call(["gdalwarp", '-t_srs', '+proj=longlat +ellps=WGS84', infile_name, outfile_name])
    
    src_ds = gdal.Open(outfile_name)
    rb = src_ds.GetRasterBand(1)
    gt = src_ds.GetGeoTransform()

    # adfGeoTransform[0] /* top left x */
    # adfGeoTransform[1] /* w-e pixel resolution */
    # adfGeoTransform[2] /* 0 */
    # adfGeoTransform[3] /* top left y */
    # adfGeoTransform[4] /* 0 */
    # adfGeoTransform[5] /* n-s pixel resolution (negative value) */
    pixels = []
    for index in range(len(pts)):   
        mx,my = pts[index][0], pts[index][1] #coord in map units
        px = int((mx - gt[0]) / gt[1]) #x pixel
        py = int((my - gt[3]) / gt[5]) #y pixel
        #get value from array as numpy uint8 and covert to int16 value
        pixels.append(np.int16(rb.ReadAsArray(px,py,1,1)[0][0]).item())  
        #print pts[index], px, py, pixels[index]
    return pixels

                 
if __name__ == '__main__':

    #sf0 = shapefile.Reader("../data/Shapefile/IOWA_Counties/IOWA_Counties.shp")
    #sf1 = shapefile.Reader("../data/Shapefile/IOWA_Counties/IOWA_Counties.shp") #counties in IA
    sf2 = shapefile.Reader("../data/cb_2015_us_county_500k/cb_2015_us_county_500k.shp") #counties in US
    #sf3 = shapefile.Reader("../data/cb_2015_us_state_500k/cb_2015_us_state_500k.shp") # states

    #prototype CDL GEOTIFF file names in data folder
    tifnames = ['../data/croppoly/CDL_'+str(y)+'.tif' for y in range(2000,2016)]

    # output JSON
    jsomFileName = '../data/pixelValues_2015.json'

    #reads the shapefile into shapes and records objects
    shps = sf2.shapes()
    recs = sf2.records()
    #narrows down the shapefile to hold only IOWA shapes and records
    iowafips = '19'
    iowashapes = []
    iowarecs = []
    for i in range(len(shps)):
        if recs[i][0]==iowafips:
            iowashapes.append(shps[i])
            iowarecs.append(recs[i])
    countyCoord = [item.bbox for item in iowashapes]

    objects = { 'pixelData': [] }

    #identify 1000 points (N) within county shape that have corn data (testval) in each county for a particular year
    for i in range(len(iowarecs)):
        rec = iowarecs[i]
        cname = rec[5].upper().replace("'"," ")
        yieldVals = countyyield(cname)

        for year in range(2015, 2016): 
            print "Data access for: ", cname, " county ID : ", i, " Year : ", year 
            START_DATE = str(year) + '-05-01'
            END_DATE = str(year) + '-08-01'
            CLOUD_MAX = 5.0 #maximum allowed cloud is 5%
            countyCoordLimited = [countyCoord[i]]
            yieldValue = yieldVals[year]              
            print "Generating random sampling points in county ..."
            rpts = genimagesamplepoints(iowashapes[i], tifnames, testfunc=True, testval=1, N=128, bbox=False, nyear = year)
            print "Computing Path/Row for county ..."
            pathrowWRS = findPathRow(countyCoordLimited)
            print "Searching for Landsat Images ..." 
            landsatData = searchLandsat(pathrowWRS, START_DATE, END_DATE, CLOUD_MAX)
            print "Downloading Landsat Images ..."
            filePaths = downloadLandsat(landsatData)
            print "Processing Landsat Images ..."
            processPaths = processLandsat(filePaths)   
            print "Sampling Image points ..."
            pixelVals = sampleImage(rpts, countyCoord[0], processPaths)
            print "Creating Dictionary ..."
            objects['pixelData'].append({
                'year': year,
                'countyName': cname,
                'countyId': i,
                'yield': yieldValue,
                'samplePoints': rpts,
                'pixelVals': pixelVals
            })

    print "Writing to JSON ..."
    with open(jsomFileName, 'w') as outfile:
        json.dump(objects, outfile, sort_keys = True, indent = 4)
    outfile.close()





