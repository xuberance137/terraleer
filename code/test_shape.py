import shapefile
import gdal

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

def genpointsinshape(shp, testfunc=False, testval=False, N=100, bbox=False, nyear=2000):
    """
    generates N random points which are inside the given shape.
    If testfunc is true, ensures that the given point has a value of test val in the nyear'th year
    """
    ct = 0
    pts=[]
    if bbox == False:
        bbox = boundingbox(shp,buff=0)
    okpt = True
    while ct < N:
        x,y = (random.uniform(bbox[0],bbox[1]), 
               random.uniform(bbox[2],bbox[3]))
        if testfunc!=False:
            okpt = pt2pixel(y,x,tifnames,nyear-2000)==testval
        if inshape(y,x,shp) and okpt:
            pts.append([x,y])
            ct += 1
    return pts

def builddatasets():
    """
    takes all of the data for all of the counties per each year,
    samples weather and reflectivity data at 100 points where there 
    is corn, averages it and puts it into a file by county by year
    
    This should be run once as it takes up to a day to run!
    All data is saved in folder: DataParamsB
    """
    for i in range(len(iowarecs)):
        rec = iowarecs[i]
        cname = rec[5].upper().replace("'"," ")
        yielddata = countyyield(cname, yieldfile)
        if len(yielddata)<16:
            print cname + ': no data found in ' + yieldfile
        for y in range(16):
            year=2000+y
            rpts = genpointsinshape(iowashapes[i], testfunc=False, testval=1, N=100, bbox=False, nyear = year)
            plotshapes([iowashapes[i]])
            plotpoints(rpts)
            pcounts = {}
            mdat=[]
            pixels = getMODpix(year,doy,rpts)
            for pt in rpts:
                latag = str(int(np.floor(pt[1])))
                lonag = str(int(np.floor(pt[0])))
                pkey=latag+"_"+lonag
                if pkey in pcounts:
                    pcounts[pkey][2] = pcounts[pkey][2]+1
                else:
                    pcounts[pkey] = [pt[0],pt[1],1]
            N = sum([p[2] for p in pcounts.values()])
            dfs = []
            for k in pcounts.keys():
                ptds = read_agro_point(agfolder, pcounts[k][1], pcounts[k][0])
                fields = ['INSgd','RAD','T','TMAX','TMIN','RH','RAIN', 'WIND']
                dfs.append(pd.concat([deres(doy, ptds, fields, np.nanmax, year, 'max', width=8)*1.0*pcounts[k][2]/N, 
                                     deres(doy, ptds, fields, np.nanmin, year, 'min', width=8)*1.0*pcounts[k][2]/N, 
                                    deres(doy, ptds, fields, np.nanmean, year, 'mean', width=8)*1.0*pcounts[k][2]/N], axis=1))
            sdfs= sum(dfs)
            sdfs = pd.concat([sdfs,pd.DataFrame(pixels.T,columns = ['NDVI', 'EVI', 'RED', 'NIR', 
                  'BLUE', 'MIR', 'pixrel'])],axis=1)
            sdfs.to_csv('DataParamsB/'+str(year)+'_'+cname)

if __name__ == '__main__':

    sf1 = shapefile.Reader("../data/Shapefile/IOWA_Counties/IOWA_Counties.shp") #counties in IA
    sf2 = shapefile.Reader("../data/cb_2015_us_county_500k/cb_2015_us_county_500k.shp") #counties in US
    sf3 = shapefile.Reader("../data/cb_2015_us_state_500k/cb_2015_us_state_500k.shp") # states

    #reads the shapefile into shapes and records objects
    shps = sf2.shapes()
    recs = sf2.records()

    #narrows down the shapefile to hold only iowa shapes and records
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


    for i in range(len(iowarecs)):
        rec = iowarecs[i]
        cname = rec[5].upper().replace("'"," ")
        yielddata = countyyield(cname, yieldfile)
        if len(yielddata)<16:
            print cname + ': no data found in ' + yieldfile
        for y in range(16):
            year=2000+y
            rpts = genpointsinshape(iowashapes[i], testfunc=False, testval=1, N=100, bbox=False, nyear = year)





   