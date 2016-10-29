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

def searchLandsat(pathrowWRS):
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

def processLandsat(filePaths):
    processPath = []
    for filePath in filePaths:
        p = NDVI(path=filePath, dst_path='../data/NDVI')
        path = p.run()
        processPath.append(path)

    return processPath

def sampleImage(pts, coord, processPath):

    print "NDVI Image : ", processPath
 
    for item in pts:
        if item[0] < coord[0] or item[0] > coord[2] or item[1] < coord[1] or item[1] > coord[3]:
            print "[ISSUE ALERT] Sample point out of county boundary ..."

    infile_name = '../data/NDVI/LC80280302015160LGN00/LC80280302015160LGN00_NDVI.TIF'
    outfile_name = '../data/NDVI/LC80280302015160LGN00/LC80280302015160LGN00_NDVI_latlon.TIF'

    #gdalwarp ../data/NDVI/LC80280302015160LGN00/LC80280302015160LGN00_NDVI.TIF ../data/NDVI/LC80280302015160LGN00/LC80280302015160LGN00_NDVI_latlon.TIF -t_srs "+proj=longlat +ellps=WGS84"

    call(["gdalwarp", '-t_srs', '+proj=longlat +ellps=WGS84', infile_name, outfile_name])
    
    src_ds = gdal.Open(outfile_name)
    rb = src_ds.GetRasterBand(1)
    gt = src_ds.GetGeoTransform()
    
    print rb
    print gt

    #width = src_ds.RasterXSize
    #height = src_ds.RasterYSize

    # # get old coordinate system
    # old_cs= osr.SpatialReference()
    # old_cs.ImportFromWkt(src_ds.GetProjectionRef())
    # # create the new coordinate system
    # wgs84_wkt = """
    # GEOGCS["WGS 84",
    #     DATUM["WGS_1984",
    #         SPHEROID["WGS 84",6378137,298.257223563,
    #             AUTHORITY["EPSG","7030"]],
    #         AUTHORITY["EPSG","6326"]],
    #     PRIMEM["Greenwich",0,
    #         AUTHORITY["EPSG","8901"]],
    #     UNIT["degree",0.01745329251994328,
    #         AUTHORITY["EPSG","9122"]],
    #     AUTHORITY["EPSG","4326"]]"""
    # new_cs = osr.SpatialReference()
    # new_cs .ImportFromWkt(wgs84_wkt)
    # # create a transform object to convert between coordinate systems
    # transform = osr.CoordinateTransformation(new_cs, old_cs) 

    #get the point to transform, pixel (0,0) in this case
    # width = src_ds.RasterXSize
    # height = src_ds.RasterYSize
    # gt = src_ds.GetGeoTransform()
    # minx = gt[0]
    # miny = gt[3] + width*gt[4] + height*gt[5] 

    #get the coordinates in lat long
    # latlong = transform.TransformPoint(x,y) 

    pixels = []


    # adfGeoTransform[0] /* top left x */
    # adfGeoTransform[1] /* w-e pixel resolution */
    # adfGeoTransform[2] /* 0 */
    # adfGeoTransform[3] /* top left y */
    # adfGeoTransform[4] /* 0 */
    # adfGeoTransform[5] /* n-s pixel resolution (negative value) */

    for index in range(len(pts)):   
        mx,my = pts[index][0], pts[index][1] #transform.TransformPoint(pts[index][0], pts[index][1])  #coord in map units
        px = int((mx - gt[0]) / gt[1]) #x pixel
        py = int((my - gt[3]) / gt[5]) #y pixel
        pixels.append(rb.ReadAsArray(px,py,1,1)[0][0])
        print pts[index], px, py, pixels[index]

                 
if __name__ == '__main__':

    sf0 = shapefile.Reader("../data/Shapefile/IOWA_Counties/IOWA_Counties.shp")
    sf1 = shapefile.Reader("../data/Shapefile/IOWA_Counties/IOWA_Counties.shp") #counties in IA
    sf2 = shapefile.Reader("../data/cb_2015_us_county_500k/cb_2015_us_county_500k.shp") #counties in US
    sf3 = shapefile.Reader("../data/cb_2015_us_state_500k/cb_2015_us_state_500k.shp") # states

    #prototype CDL GEOTIFF file names in data folder
    tifnames = ['../data/croppoly/CDL_'+str(y)+'.tif' for y in range(2000,2016)]

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

    #yield data for all years for corn in all counties
    for i in range(1): #len(iowarecs)):
        rec = iowarecs[i]
        cname = rec[5].upper().replace("'"," ")
        #print i, cname
        yielddata = countyyield(cname)
        yieldval = zip(yielddata['Year'], yielddata['Value'])

    #identify 1000 points (N) within county shape that have corn data (testval) in each county for a particular year
    for i in range(1): #range(len(iowarecs)):
        year = 2015
        rpts = genimagesamplepoints(iowashapes[i], tifnames, testfunc=True, testval=1, N=100, bbox=False, nyear = year)
        print
        print rpts
        print "Resulting sample points : ", len(rpts)


    countyCoordLimited = []
    countyCoordLimited.append(countyCoord[0])
    
    # print "Computing Path/Row for county ..."
    # pathrowWRS = findPathRow(countyCoordLimited)
    # print "Searching for Landsat Images ..." 
    # landsatData = searchLandsat(pathrowWRS)
    # print "Downloading Landsat Images ..."
    # filePaths = downloadLandsat(landsatData)
    # print "Processing Landsat Images ..."
    # processPaths = processLandsat(filePaths)   
    
    # processPaths = ['../data/NDVI/LC80280302015160LGN00/LC80280302015160LGN00_NDVI.TIF']  
    processPaths = ['../data/sceneData/LC80280302015160LGN00/LC80280302015160LGN00_B1.TIF']
    print "Sampling Image points ..."
    sampleImage(rpts, countyCoord[0], processPaths)




