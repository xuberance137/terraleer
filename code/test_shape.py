import shapefile
import gdal
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

DEBUG_PRINT = False

def inshape(ptlat,ptlon,shape):
    """
    given a shape object from a shapefile, tests if the point given by ptlat,ptlon is in the shape by drawing a line west and checking how many times it intersects with the shape.
    
    Note that this function does not work for shapes near the poles or which cross -180/180 degrees longitude  
    """
    # does not behave well near poles!!!
    pts = shape.points
    lat=[]
    lon=[]
    for pt in pts:
        lat.append(pt[1])
        lon.append(pt[0])
    #finds a bounding box
    lon_max=max(lon)
    lon_min=min(lon)
    lat_max=max(lat)
    lat_min=min(lat)
    
    if lon_min < -170 and lon_max > 170:
        for i in range(len(pts)):
            if pts[i][0]<0:
                pts[i][0] = 360+pts[i]
    #checks if point is outside of bounding box
    if (ptlat > lat_max or ptlat < lat_min or
        ptlon > lon_max or ptlon < lon_min):
        return 0
    else:
        #otherwise for each line, see if the line passes directly west
        #of the point, if so adds to count
        count = 0
        for i in range(len(pts)-1):
            line=[pts[i][0],pts[i][1],
                  pts[i+1][0],pts[i+1][1]]
                  
            if ((line[1] < ptlat and line[3] > ptlat) or 
                (line[1] > ptlat and line[3] < ptlat)):
               frac = (line[1]-ptlat)/(line[1]-line[3])
               if line[0]+frac*(line[2]-line[0]) > ptlon:
                    count+=1
        
        line=[pts[-1][0],pts[0][1],
              pts[-1][0],pts[0][1]]                  
        if line[1] < ptlat and line[3] > ptlat:
           frac = (line[1]-ptlat)/(line[1]-line[3])
           if line[0]+frac*(line[2]-line[0]) > ptlon:
                count+=1
    #if count is odd the point is inside the shape, if even, outside
    return count%2

def boundingbox(shp,buff=0):
    """
    returns coordinates of the smallest possible rectangular boundary about a shape
    """
    pts = shp.points
    reved = zip(*pts[::-1])
    return [min(reved[0]) - buff, max(reved[0]) + buff, min(reved[1]) - buff, max(reved[1]) + buff]

def pt2pixel(lat, lon, tifnames,y):
    """
    returns the landcover at a point given a year y. 1 is corn, 5 is soybeans
    """
    pixels = [-1 for i in tifnames]
    for i in range(len(tifnames)):
        src_ds = gdal.Open(tifnames[i])
        rb = src_ds.GetRasterBand(1)
        gt = src_ds.GetGeoTransform()
        
        mx,my=lon, lat  #coord in map units
        px = int((mx - gt[0]) / gt[1]) #x pixel
        py = int((my - gt[3]) / gt[5]) #y pixel
        
        pixels[i] = rb.ReadAsArray(px,py,1,1)[0][0]
    return pixels[y]

def genpointsinshape(shp, tifnames, testfunc=False, testval=False, N=100, bbox=False, nyear=2000):
    """
    generates N random points which are inside the given shape.
    If testfunc is true, ensures that the given point has a value of test val in the nyear'th year
    testval is value of corp type
    """
    ct = 0
    pts=[]
    if bbox == False:
        bbox = boundingbox(shp,buff=0)
    okpt = True
    while ct < N:
        #random.seed(5000)
        x,y = (random.uniform(bbox[0],bbox[1]), random.uniform(bbox[2],bbox[3]))
        if testfunc!=False:
            okpt = pt2pixel(y,x,tifnames,nyear-2000)==testval
        if inshape(y,x,shp) and okpt:
            pts.append([x,y])
            ct += 1
    return pts


def genimagesamplepoints(shp, tifnames, testfunc=False, testval=False, N=100, bbox=False, nyear=2000):
    """
    For each randomly picked point, finding CDL vlaue for that point for each year. Keep iterating till CDL[point] = testval for nyear
    generates N random points which are inside the given shape.
    If testfunc is true, ensures that the given point has a value of test val in the nyear'th year
    test val is the value of the interested crop
    """
    ct = 0
    pts=[]
    if bbox == False:
        bbox = boundingbox(shp,buff=0)
    okpt = True
    random.seed(5000)

    while ct < N:

        x,y = (random.uniform(bbox[0],bbox[1]), random.uniform(bbox[2],bbox[3]))
        
        #compare with CDL data
        if testfunc==True:
            lat = y
            lon = x
            year = nyear-2000
            pixels = [-1 for i in tifnames]
            for i in range(len(tifnames)):
                src_ds = gdal.Open(tifnames[i])
                rb = src_ds.GetRasterBand(1)
                # print rb.GetRasterCount() #binding missing
                gt = src_ds.GetGeoTransform()
                
                mx,my=lon, lat  #coord in map units
                px = int((mx - gt[0]) / gt[1]) #x pixel
                py = int((my - gt[3]) / gt[5]) #y pixel
                
                pixels[i] = rb.ReadAsArray(px,py,1,1)[0][0]
                #print pixels[i]
            
            okpt = (pixels[year] == testval)
            #print ct, pixels, okpt

        # keep generatng random points till there are N points with test_val(1 for corn) in concerned year and point is in the shape object
        if inshape(y,x,shp) and okpt:
            pts.append([x,y])
            ct += 1

    return pts    


def countyyield(NAME):
    """
    reads the yield values from a given county
    note that NAME must be all caps and correspond to the names found 
    in the file
    """
    cyield = pd.read_csv('../data/countyyield.csv')
    cplant = pd.read_csv('../data/cornplanted.csv')
    pmean = np.mean(cplant.Value[np.logical_and(cplant.County == NAME, cplant.Year>=1990)])
    countyield = cyield[cyield.County==NAME]
    countyplant = cplant[np.logical_and(cplant.County == NAME, cplant.Year >= 2000)].Value/pmean
    return pd.concat([countyield, pd.DataFrame(np.array(countyplant), index=countyield[countyield.Year>=2000].index, columns=['Plant'])], axis=1)
    #return pd.concat([countyield], axis=1)
 

if __name__ == '__main__':

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

    sf = sf3
    shapes = sf.shapes()
    recs = sf.records()
    countyCoord = [item.bbox for item in shapes]   
    # print
    # for index in range(len(countyCoord)):
    #     print index, countyCoord[index]
    # print
    # for index in range(len(recs)):
    #     print index, recs[index][0], recs[index][1], str(recs[index][2]), recs[index][3], recs[index][4], str(recs[index][5]).upper(), recs[index][6], recs[index][7], recs[index][8]
    # fields = sf.fields
    # print fields
    # print 
    # df = pd.read_csv('../data/countyyield-2016-2010-monthly.csv')
    # print df['Year']
    # print len(df['Year'])

    plt.figure()
    labels = []
    #yield data for all years for corn in all counties
    for i in range(20): #len(iowarecs)):
        rec = iowarecs[i]
        cname = rec[5].upper().replace("'"," ")
        #print i, cname
        yielddata = countyyield(cname)

        yieldval = zip(yielddata['Year'], yielddata['Value'])

        countyYieldData =yielddata['Value'].values.tolist()[::-1]
        years = [y for y in range(1982, 2016)]
        if len(countyYieldData) == len(years):
            plt.plot(years,yielddata['Value'].values.tolist()[::-1])
            labels.append(cname)

        if DEBUG_PRINT:
            print
            print i, cname
            print yieldval
            if len(yielddata)<16:
                print cname + ': no data found in ' + yieldfile

    plt.legend(labels, loc='upper left', ncol=4)
    plt.show()


    #identify 1000 points (N) within county shape that have corn data (testval) in each county for a particular year
    for i in range(1): #range(len(iowarecs)):
        # for year in range(2000, 2016):
        year = 2015
        #rpts = genpointsinshape(iowashapes[i], tifnames, testfunc=True, testval=1, N=100, bbox=False, nyear = year)
        rpts = genimagesamplepoints(iowashapes[i], tifnames, testfunc=True, testval=1, N=1000, bbox=False, nyear = year)
        print
        print rpts
        print "Resulting sample points : ", len(rpts)




   