import os
import shapefile
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import mpld3
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import numpy as np
import scipy.integrate as integrate
import calendar
#from osgeo import gdal,ogr
import osgeo.gdal
import struct
import sys
import utm
import urllib
import random
import requests
import sklearn
from sklearn import linear_model
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, median_absolute_error, mean_squared_error
from scipy.stats.stats import pearsonr
from pyproj import Proj
import pickle
import json
import tqdm

RUN_INDIVIDUAL_PRED = False
PREDICT_ONE = True
LIABILITY_PLOT = False
LOGGING_MODEL_PERF = False

def dmy2doy(day, mon, year):
    """
    Converts dates of day, month, year to day since the first of the year, accounting for leap years [1:365]
    """
    monlens = [31,28,31,30,31,30,31,31,30,31,30,31]
    if calendar.isleap(year):
        monlens[1]=29
    return sum(monlens[0:(mon-1)])+day

def doy2dm(doy, year):
    """
    the inverse of dmy2doy; converts, the day of year (given the year) to month and date
    """
    monlens = [31,28,31,30,31,30,31,31,30,31,30,31]
    if calendar.isleap(year):
        monlens[1]=29
    moncum = [sum(monlens[:(i)]) for i in range(12)]
    for i in range(12):
        if doy < moncum[i]:
            month=i-1
            break
    return [doy-moncum[month],month+1]


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
    return [min(reved[0]) - buff, max(reved[0]) + buff, 
            min(reved[1]) - buff, max(reved[1]) + buff]

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

def plotshape(shape,show=False):
    """
    plots a shape object from a shapefile
    """
    x=[]
    y=[]
    for pt in shape.points:
        x.append(pt[0])
        y.append(pt[1])
    plt.plot(x,y)
    if show:
        plt.show()


def plotshapes(shapes):
    """
    plots all shapes in a list
    """
    for shape in shapes:
        plotshape(shape)

def plotpoints(pts):
    """
    plots all points in a list of form [[x0,y0],[x1,y1],...]
    """
    x=[]
    y=[]
    for xy in pts:
        x.append(xy[0])
        y.append(xy[1])
    plt.scatter(x,y)

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


def splitagrofile(agfile, folder):
    """
    takes worldwide agroweather data file and splits it into degree/degree files in 'folder'
    """
    fin = open(agfile,'r')
    indeg=False
    indata=False
    line = 'kjdsgjds'
    while True and len(line)>1:
        line = fin.readline()
        if line[0:10]=='Location: ':
             fname = line.split()[2]+"_"+line.split()[4]
             fout = open(folder+'/'+fname,'w')
        if not indata:
            if line=='-END HEADER-\n':
                indata=True
                continue
        if indata:
            if line[0:4]=='-BEG':
                indata=False
                fout.close()
            else:
                fout.write(line)
    fin.close()

#already done in repository
#splitagrofile(agfile, 'POWERdata')

def read_agro_point(agfolder, lat, lon):
    """
    reads the agroweather data from the corresponding point
    """
    latag = str(int(np.floor(lat)))
    lonag = str(int(np.floor(lon)))
    olb = pd.read_table(agfolder+'/'+latag+'_'+lonag, 
                  header=None, sep =r"\s*",
                  names=['YEAR', 'DOY', 'INS', 'INSgd', 
                       'RAD', 'T', 'TMIN', 'TMAX', 
                        'RH', 'DP', 'RAIN', 'WIND'],na_values=['-'])
    return olb


def countyyield(NAME, yieldfile):
    """
    reads the yield values from a given county
    note that NAME must be all caps and correspond to the names found 
    in the file
    """
    cyield = pd.read_csv(yieldfile)
    cplant = pd.read_csv('../data/cornplanted.csv')
    pmean = np.mean(cplant.Value[np.logical_and(cplant.County == NAME, 
                    cplant.Year>=1990)])
    countyield = cyield[cyield.County==NAME]
    countyplant = cplant[np.logical_and(cplant.County == NAME,
                                   cplant.Year >= 2000)].Value/pmean
    return pd.concat([countyield,
               pd.DataFrame(np.array(countyplant),
              index=countyield[countyield.Year>=2000].index, 
              columns=['Plant'])], axis=1)

#reads MODIS data 
#Note that the MODIS data is about 50 GB and is not in the bitbucket archive. Let me know at tjivancic@gmail.com if you want the record from 2000-2015 for iowa or download it from http://e4ftl01.cr.usgs.gov/MOLT/MOD13Q1.006/ for each day of interest, put it in the folder MODIS and shorten the name to be of the format MOD13A1.A<YYYY><DDD>.h<HH>v04.hdf  where <DDD> is the day of the year (1:365) <HH> is the horizontal swatch (either 10 or 11) (v04 is the vertical swatch, but that doesn't matter since it always stays the same)
def buildstrings(year, doy, folder='MODIS',left=False):
    """
    builds strings to query modis data gdal objects
    """
    if left:
        stbase = ('HDF4_EOS:EOS_GRID:"MODIS/MOD13A1.A' +
                    str(year) + '%03.0f' % doy +
                 '.h11v04.hdf":MODIS_Grid_16DAY_500m_VI:500m 16 days ')
    else:
        stbase = ('HDF4_EOS:EOS_GRID:"MODIS/MOD13A1.A' +
                   str(year) + '%03.0f' % doy +
                 '.h10v04.hdf":MODIS_Grid_16DAY_500m_VI:500m 16 days ')
    types = ['NDVI', 'EVI', 'red reflectance', 'NIR reflectance', 
              'blue reflectance', 'MIR reflectance', 
              'pixel reliability']
    return [stbase + tp for tp in types]


def getMODpix(year, doys, pts):
    """
    get the relevent modis data timeserieses at different points
    """
    p_modis_grid = Proj('+proj=sinu +R=6371007.181 +nadgrids=@null +wktext')
    pixels = np.zeros([len(pts),7,len(doys)])
    for d in range(len(doys)):
        doy=doys[d]
        strsl = buildstrings(year, doy, folder='MODIS',left=True)
        strs = buildstrings(year, doy, folder='MODIS',left=False)
        for i in range(len(strs)):
            st = strs[i]
            stl= strsl[i]
            VIdata = gdal.Open(st)
            VIdatal = gdal.Open(stl)
            rb = VIdata.GetRasterBand(1)
            gt = VIdata.GetGeoTransform()
            rbl = VIdatal.GetRasterBand(1)
            gtl = VIdatal.GetGeoTransform()
            rar = rb.ReadAsArray()
            rarl = rbl.ReadAsArray()
            for p in range(len(pts)):
                x, y = p_modis_grid(pts[p][0], pts[p][1])
                px = int((x - gt[0]) / gt[1]) #x pixel
                py = int((y - gt[3]) / gt[5]) #y pixel
                pxl = int((x - gtl[0]) / gtl[1]) #x pixel
                pyl = int((y - gtl[3]) / gtl[5]) #y pixel
                try:
                    pixels[p,i,d] = rar[px,py]
                except:
                    pixels[p,i,d] = rarl[pxl,pyl]
    pixlow = sum(pixels)*1.0/len(pts)
    return pixlow

def deres(doy, df, fields, method, year, label, width=8):
    """
    given a data frame with daily data decreaes the resolution using 
    'method' to cover only days within <width> of the given day of 
    years in <doy>
    """
    dfdict = {(field+label):[] for field in fields}
    dfdict['YEAR']=[]
    dfdict['DOY']=[]
    inyear = df.YEAR==year
    for i in doy:
        indrange = np.logical_and(df.DOY > (i-width),df.DOY<(i+width))
        localdata= np.logical_and(inyear,indrange)
        [dfdict[field+label].append(method(np.array(df[field][localdata]).astype(float))) for field in fields]
        dfdict['YEAR'].append(year)
        dfdict['DOY'].append(i)
    return pd.DataFrame(dfdict)

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


def read_all_data(fname,folder = 'DataParamsB'):
    """
    Reads all of the County by year data from DataParamsB as well as 
    the yield data and returns a list of [yields, features, planted
    areas] 
    """
    yieldset = countyyield(fname[5:], yieldfile)
    yieldval =yieldset.Value[yieldset.Year==int(fname[0:4])]
    plantval = yieldset.Plant[yieldset.Year==int(fname[0:4])]
    df = pd.read_csv(folder+'/'+fname)
    teleds = pd.DataFrame(teledats[int(fname[0:4])-2000],columns = tnames)
    return [float(yieldval), pd.concat([df,teleds],axis=1),float(plantval)]





def insuredcornperacre(Yield, APH, Yvar, guarantee, liability, cost):
    """
    This is used as an insurance liability estimation tool
    
    Uses a yield guarantee insurance model to return an estimated
    value for liability if the entire county were insured under the 
    same policy
    
    Yield: is the actual yield
    
    APH: is the expected yield from the county
    
    Yvar: is the variance of yield in that county
    
    guarantee: is the fraction of expected production below which the 
    insurance policy starts paying out for
    
    liability: is the fraction of shortfall below the guarantee
    covered by the policy
    
    cost: cost of a bushel of corn (typically aroung $3.8)
    
    This uses the assumption that there are a large number of farms 
    in the county and that the farm variance can be expressed as 
    Vfarm = Vcounty + Vrandom
    and that Vrandom = .3*Vcounty
    
    these assumption are based on Scott, Thompson, Miller 2014 
    'Exploiting the relationship between Farm-level Yields and County 
    Level Yields for applied analysis'
    """
    Fvar = Yvar*.3
    a = 1/np.sqrt(Fvar)/np.sqrt(2*np.pi)
    b = Yield
    c = np.sqrt(Fvar)
    insum = 0.0
    countsum=0
    payoff=APH*guarantee
    def atleast(dx):
        if dx<0:
            return 0.0
        else:
            return dx
    
    def integrand(x,payoff,liability,a,b,c):
        return (payoff-atleast(x))*liability*a*np.exp(-(x-b)**2/2/c**2)
        #return a*np.exp(-(x-b)**2/2/c**2)
    I = integrate.quad(integrand, -100, payoff, args=(payoff,liability,a,b,c))
    return I[0]*cost


def TTestArray(y,x):
    """
    find the correlation and T-test singificance for each feature in x 
    against y
    """
    N, M = x.shape
    corray=np.zeros(M)
    prray=np.zeros(M)
    for i in range(M):
        pearvals = pearsonr(y,x[:,i])
        corray[i], prray[i] = pearvals
    return corray,prray

def Tcorr(y,x,cnty):
    """
    finds the correlations between y and each feature in x by county
    then adds them using the fischer Z transform.
    The reason for this is to measure only the year-to-year 
    variability rather than the county to county variability
    """
    uctys = np.unique(cnty)
    countval = 0
    zlist = np.zeros(x.shape[1])
    for i in range(len(uctys)):
        countval+=1
        ucty = uctys[i]
        corlist = TTestArray(y,x)[0]
        zlist = zlist + np.arctanh(corlist)
    avcor = np.tanh(zlist/countval)
    return avcor


def Ndayalldata(N,folder,yieldhist=True, plantval=True):
    """
    Builds a dataframe including data from the start of the growing season until the N th satellite report which occurs every 16 days
    """
    files = os.listdir(folder)
    dfl = read_all_data(files[0],folder = '../data/DataParamsB')
    dfl[1].drop(['RAINmin','DOY','YEAR','DOY.1','Unnamed: 0','YEAR.1','DOY.2','DOYtc','pixrel','YEAR.2'], 1,inplace=True)
    ndrop,M = dfl[1].shape
    labelroots = dfl[1].columns
    MN = N*M
    #flattrack = [[v[0],v[1][i]] for v in vartrack for i in range(len(v[1]))]
    #MN = len(vartrack)
    x=np.zeros([len(files), MN + int(yieldhist)+int(plantval)])
    y=np.zeros(len(files))
    yr=np.zeros(len(files))
    cnty = [0 for i in range(len(files))]
    labels=[]
    for j in range(M):
        for i in range(N):
            labels.append(labelroots[j]+str(i))
    labels = labels+['YHIST']+['Plant']
    print "Total Files : ", len(files)
    for i in range(len(files)):
        print N, "File #", i
        dfl = read_all_data(files[i],folder = '../data/DataParamsB')
        df = dfl[1]
        cnty[i] = files[i][5:]
        if yieldhist:
            yielddata = countyyield(cnty[i], yieldfile)
            yieldsub = yielddata.Value[np.logical_and(yielddata.Year < 2000, yielddata.Year < 1990)]
            x[i,-(int(plantval)+1)] = np.nanmean(yieldsub[yieldsub>1])
        if plantval:
            try:
                x[i,-1] = float(yielddata.Plant[yielddata.Year == int(files[i][0:4])])
            except:
                x[i,-1] = 1.0
        y[i] = dfl[0]
        yr[i]= int(files[i][0:4])
        for t in range(M):
            for l in range(N):
                x[i,t*N+l] = df[labelroots[t]][l]
    return [x,y,yr,cnty,labels]


def dono(x):
    """
    returns the input
    """
    return x

def Derivemodfunc(testdat,func=dono, invfunc=dono, alp=.01):
    """
    first applies func to the yield values then
    uses the year-to-year (from Tcorr) variability to select the    
    'best' features using forward stepwise regression and validating 
    the features using one-year-out validation. 
    
    returns a bool array indicating the best variables 
    """
    y0 = testdat[1][testdat[1]>0]
    x0 = testdat[0][testdat[1]>0,:]
    yr = np.array(testdat[2])[testdat[1]>0]
    cty = np.array(testdat[3])[testdat[1]>0]
    years = range(2000,2016)
    testcors = np.abs(Tcorr(y0,x0,cty))
    #xin = np.array(testcors==max(testcors))
    ybar = np.mean(y)
    lastR=-1000
    xin = np.array([False for i in range(x0.shape[1])])
    xtry = np.array([False for i in range(x0.shape[1])])
    xleft = np.array([True for i in range(x0.shape[1])])
    xtry = np.array(testcors==max(testcors))
    xnew=np.array(testcors==max(testcors))
    cycle = 0
    varscount = 0
    while True:
        SSres=0
        SStot=0
        ct= 0
        resid = np.zeros(len(y0))
        for valyear in years:
            valvec = np.array([currentyear == valyear for currentyear in yr])
            #ybar = np.mean(y0[valvec])
            trvec = np.logical_not(valvec)
            xtr=x0[trvec,:][:,xtry]
            ytr=y0[trvec]
            #print x0.shape
            #print valvec.shape
            #print xtry.shape
            xvl = x0[valvec,:][:,xtry]
            yvl = y0[valvec]
            #regr = linear_model.LinearRegression()
            #regr = RandomForestRegressor(random_state=0, n_estimators=100)
            regr = linear_model.Ridge(alpha=alp,normalize=True)
            if len(xtr.shape)==1:
                xtr = xtr.reshape(-1, 1)
                xvl = xvl.reshape(-1,1)
            regr.fit(xtr,func(ytr))
            SSres+=sum((invfunc(regr.predict(xvl)) - yvl) ** 2)
            ct+=len(ytr)
            SStot+=sum((yvl - ybar)**2)
            resid[valvec] = regr.predict(xvl) - func(yvl)
        thisR = 1-SSres/SStot
        #print SSres, SStot, thisR
        if thisR >= lastR:
            #print thisR
            cycle = 0
            varscount+=1
            xin[xnew]=True
            lastR=thisR
        if cycle > 40 or varscount>40:
            return xin
        if sum(xleft)==0:
            return xin
        #print str(sum(xin)) +' var(s) accepted / '+str(sum(xleft))+ ' var(s) left'
        testcors=np.zeros(x0.shape[1])
        testcors[xleft] = np.abs(Tcorr(resid,x0[:,xleft],cty))
        xnew=np.array([False for i in range(x0.shape[1])])
        xnew[np.argmax(testcors)]=True
        #print np.argmax(testcors)
        xleft[np.argmax(testcors)]=False
        xtry[xin]=True
        xtry[xnew]=True


def val2dollazz(value,ctys):
    """
    converts the yields into dollar liability estimate for a policy
    of 
    guarantee = .75
    liability = .75
    cost = 3.8
    """
    cmean=[]
    cvar=[]
    uct = list(np.unique(ctys))
    for ct in uct:
        cmean.append(np.mean(value[ctys==ct]))
        cvar.append(np.var(value[ctys==ct]))
    outvec=np.zeros(len(value))
    for i in range(len(value)):
        j = uct.index(ctys[i])
        outvec[i] = insuredcornperacre(value[i], cmean[j], cvar[j], .75, .75, 3.8)
    return outvec


def oneyearoutbylabfunc(testdat, labs, plotfile='validationplot.pdf', func=dono, invfunc=dono,alp=.00001,ptz=30):
    """
    Validate the model using year by year vailidation, prints the 
    root mean square liability error in $ and returns Rsquare value
    
    also plots and saves a figure to plotfile 
    """
    x=testdat[0][testdat[1]>0,:][:,np.in1d(np.array(testdat[4]),labs)]
    y=testdat[1][testdat[1]>0]
    yr = testdat[2][testdat[1]>0]
    cty = np.array(testdat[3])[testdat[1]>0]
    years = range(2000,2016)
    ybar = np.mean(y)
    SSres=0
    SStot=0
    ct= 0
    if plotfile:
        pp = PdfPages(plotfile)
        plotbits=np.zeros(len(y))
    for valyear in years:
        valvec = np.array([currentyear == valyear for currentyear in yr])
        trvec = np.logical_not(valvec)
        xtr=x[trvec,:]
        ytr=y[trvec]
        xvl = x[valvec,:]
        yvl = y[valvec]
        #regr = linear_model.LinearRegression()
        #regr = RandomForestRegressor(random_state=0, n_estimators=100)
        regr = linear_model.Ridge(alpha=alp,normalize=True)
        regr.fit(xtr,func(ytr))
        SSres+=sum((invfunc(regr.predict(xvl)) - yvl) ** 2)
        ct+=len(ytr)
        SStot+=sum((yvl - ybar)**2)
        if plotfile:
            plotbits[valvec]=invfunc(regr.predict(xvl))
    maxdol = max(val2dollazz(y,cty))
    ptsz = np.array(val2dollazz(y,cty)/maxdol*ptz)
    ptsz[ptsz<1.0]=1
    colors = np.array(['red' for i in ptsz])
    colors[ptsz==1]='b'
    #plt.scatter(y,plotbits,c=colors,s=ptsz)
    plt.scatter(y,plotbits)
    #plt.scatter(yvl,regr.predict(xvl),color='red')    
    plt.ylim([0,250])
    plt.xlim([0,250])
    plt.plot([0,250],[0,250])
    plt.xticks([0,100,200])
    plt.yticks([0,100,200])
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    #plt.title(str(valyear))
    pp.savefig()
    plt.close()    
    #print('Coefficients: ', regr.coef_)
    # The mean square error
    #print("Residual sum of squares: %.2f"
    #      % np.mean((regr.predict(xvl) - yvl) ** 2))
    # Explained variance score: 1 is perfect prediction
    #print('Variance score: '+str(1-SSres/SStot))
    pp.close()
    #print np.sqrt(np.mean((val2dollazz(plotbits,cty)-val2dollazz(y,cty))**2))
    return 1-SSres/SStot

def Add_fancy_terms(ndata):
    """
    Adds square and interaction terms to ndata and returns it.
    
    Note, this generates up to 100,000 features,
    takes forever to run anything, and seriously overfits the model
    """
    N,M = ndata[0].shape
    labels = ndata[4]
    newlabels = [l for l in labels]
    for i in range(M):
        for j in range(i):
            newlabels.append(labels[i]+labels[j])
    
    newdata0 = np.zeros([N,len(newlabels)])
    index=M
    for i in range(M):
        newdata0[:,i]=ndata[0][:,i]
    
    for i in range(M):
        for j in range(i):
            newdata0[:,index]=ndata[0][:,j]*ndata[0][:,i]
            index+=1
    return [newdata0,ndata[1],ndata[2],ndata[3],newlabels]     

# a more standard normalization function
def func1a(y):
    return np.array(y)**2 

def invfunc1a(z):
    return np.sqrt(z)

#normalization function for yield which ends up favoring fits to the low yields
def func1b(y):
    return np.log(300.0-np.array(y)) 

def invfunc1b(z):
    return 300.0-np.exp(z)

#Ndata is a list of lists which hold all the interesting info #about the features/yields 
#
#   Ndata[0] is a np.array of all of the feature values with shape = [number of yields, number of features]
#   Ndata[1] is all of the yield values
#   Ndata[2] is the year corresponding to each yield value
#   Ndata[3] is the county "            "  "    "       " 
#   Ndata[4] is the list of label names kept
def keep_Ndata_features(Ndata, labels_logic):
    # this one returns Ndata with only the labels you care about
    # given by labels_logic (a true/false array corresponding to 
    # the features to keep)
    return [Ndata[0][:,labels_logic], Ndata[1], Ndata[2], Ndata[3], list(np.array(Ndata[4])[labels_logic])]


def split_Ndata_by_year(Ndata,year):
    #this helps split the Ndata object into (Ndata_a, Ndata_b) 
    # where the 'a' is all data not in the given year and 
    # 'b' is all the data in the given year 
    keep = Ndata[2]==float(year)
    dont = np.logical_not(keep)
    return ( [Ndata[0][dont], Ndata[1][dont], Ndata[2][dont],
             list(np.array(Ndata[3])[dont]), Ndata[4]],
            [Ndata[0][keep], Ndata[1][keep], Ndata[2][keep],
             list(np.array(Ndata[3])[keep]), Ndata[4]] )


def trainmodel(Ndata, func=dono, invfunc=dono, alp = .00001):
    #trains the model given Ndata (make sure to use the output of 
    #keep_Ndata_features, or this will fit to all the features)
    #recall that func tries to normalize the yield while invfunc 
    #brings them back to the orginal distribution 
    regr = linear_model.Ridge(alpha=alp,normalize=True)
    regr.fit(Ndata[0],func(Ndata[1]))
    return regr

def predict_model(Ndata, regr, func=dono, invfunc=dono):
    # for a regression model, fit the xvalues and return 
    # y_predictions
    return invfunc(regr.predict(Ndata[0]))

def parse_preformance_data(jsomFileName):

    print "Loading Model Performance data ..."
    with open(jsomFileName) as jsonFile:    
        RegModelPerformance = json.load(jsonFile)

    numSat = []
    inYear = []
    outYear = []
    inSampleVal = []
    outSampleVal = []

    for index in range(len(RegModelPerformance['reg_model'])):
        item = RegModelPerformance['reg_model'][index]
        numSat.append(item['NumberOfSatelliteRuns'])
        inYear.append(item['YearIn'])
        outYear.append(item['YearOut'])
        inSampleVal.append(item['InSampleR2'])
        outSampleVal.append(item['OutSampleR2'])

    invalMaxIndex = np.argmax(inSampleVal)
    outvalMaxIndex = np.argmax(outSampleVal)

    # print len(RegModelPerformance['reg_model'])
    # print invalMaxIndex
    # print numSat[invalMaxIndex], inYear[invalMaxIndex], outYear[invalMaxIndex], inSampleVal[invalMaxIndex]

    # print
    # print outSampleVal[outvalMaxIndex]

    # plt.figure()
    # plt.plot(inSampleVal)
    # plt.figure()
    # plt.plot(outSamepleVal)
    # plt.show()
    return [numSat[invalMaxIndex], inYear[invalMaxIndex], outYear[invalMaxIndex], inSampleVal[invalMaxIndex]]

def measure_single_model_performance(optimal_data_set):
    #pickle.dump(Ndata_one_created, open('../data/MultivariateModelData-One.p', 'wb'))
    NUM_SAT_RUNS = optimal_data_set[0]
    print "Loading data from serialized python object ..."
    #Ndata_one = pickle.load(open('../data/MultivariateModelData-One.p', 'rb'))
    Ndata_all1 = pickle.load(open('../data/MultivariateModelData-Multiple.p', 'rb'))
    Ndata_one = Ndata_all1[NUM_SAT_RUNS]
    # run parameter optimization on single sat pass data returning true on optimal input features
    print "Running Ridge/SBS Regression Model with normalized yield data (Type A) ..."
    labbool= Derivemodfunc(Ndata_one, func1a, invfunc1a, alp=.00001)
    # get label names of optimal input features
    label_names = np.array(Ndata_one[4])[labbool]
    #keep only the features in labbool
    Ndata_one_short = keep_Ndata_features(Ndata_one, labbool)
    #separate into 2000-2014 and 2015
    Ndata_most, Ndata_out = split_Ndata_by_year(Ndata_one, optimal_data_set[2])
    Ndata_most1, Ndata_in = split_Ndata_by_year(Ndata_one, optimal_data_set[1])
    # get the model
    regmod=trainmodel(Ndata_most, func=func1a, invfunc=invfunc1a, alp= .00001)
    #predict using the model
    pred_val = predict_model(Ndata_in, regmod, func=func1a, invfunc=invfunc1a)
    print label_names, len(label_names)
    print 
    print "In sample, predicting year :", optimal_data_set[1]
    print "R2  : ", r2_score(Ndata_in[1], pred_val)
    print "MSE : ", mean_squared_error(Ndata_in[1], pred_val)
    print "MAE : ", mean_absolute_error(Ndata_in[1], pred_val)
    print 

    sorted_index = [i[0] for i in sorted(enumerate(Ndata_in[1]), key=lambda x:x[1])]

    plt.figure()
    #plt.bar(np.arange(len(pred_val))*2, pred_val, color = 'red')
    #plt.bar(np.arange(len(Ndata_2015[1]))*2+1, Ndata_2015[1], color = 'blue')        
    plt.plot(pred_val, color = 'red', label = 'Predicted Yield')
    plt.plot(Ndata_in[1], color = 'blue', label = 'Actual Yield')         
    plt.title('In Sample Prediction for ' + str(optimal_data_set[1]) ) 
    plt.ylabel('Predicted Yield Values')  
    plt.xlabel('County IDs')
    plt.legend(loc='upper left')  

    plt.figure()       
    plt.plot([pred_val[sorted_index[n]] for n in range(len(sorted_index))], color = 'red', label = 'Predicted Yield')
    plt.plot([Ndata_in[1][sorted_index[n]] for n in range(len(sorted_index))], color = 'blue', label = 'Actual Yield')              
    plt.title('Sorted In Sample Prediction for ' + str(optimal_data_set[1]) ) 
    plt.ylabel('Predicted Yield Values')  
    plt.xlabel('Post Sort County IDs')
    plt.legend(loc='upper left') 

    plt.show()  





if __name__ == '__main__':


    #days of year used in MODIS data, they are the same every year
    doy = [113, 129, 145, 161, 177, 193, 209, 225, 241, 257, 273, 289, 305]

    #name of the US-wide county shapefile
    shapefilename = '../data/cb_2015_us_county_500k/cb_2015_us_county_500k.shp'

    #reads the shapefile into shapes and records objects
    print "Reading ShapeFiles ..."
    sf = shapefile.Reader(shapefilename)
    shps = sf.shapes()
    recs = sf.records()

    #narrows down the shapefile to hold only iowa shapes and records
    print "Limiting ShapeFiles to IOWA and related records ..."
    iowafips = '19'
    iowashapes = []
    iowarecs = []
    for i in range(len(shps)):
        if recs[i][0]==iowafips:
            iowashapes.append(shps[i])
            iowarecs.append(recs[i])

    #creates the names of landcover tiffs
    tifnames = ['croppoly/CDL_'+str(y)+'.tif' for y in range(2000,2016)]

    #filenames of agroweather data
    agfile = 'NASAagro.cgi'
    agfolder = 'POWERdata'

    #this is alreads in the repository
    #os.system('mkdir '+agfolder)
    yieldfile= '../data/countyyield.csv'

    print "Reading Teleconnection, MODIS and yield data ..."
    #reads teleconnection data and saves it as a pandas object
    #teleconnections are things like ENSO, which measures el nino, and NAO 
    #(North atlantic Osclillation) which is the temperature difference 
    #between two points that meteorologists like to blame storms and stuff on
    tdf = pd.read_fwf('../data/tele_index.nh', skiprows=range(17)+[18], na_values='-99.90', widths=[4,4,6,6,6,6,6,6,6,6,6,6,7])
    ensodf = pd.read_table('../data/ENSO.txt', sep=r"\s*", engine='python')
    teledats = [np.zeros([len(doy),6]) for i in range(16)]
    tnames = ['YEAR','DOYtc','NAO','EA','WP','ENSO']

    for t in range(16):
    
        okdat = tdf.yyyy==(t+2000)
        okenso = ensodf.YEAR == 2000+t
        ensovec = [ensodf.get_value(np.where(okenso)[0][0],mstr) for mstr
                        in ['DECJAN','JANFEB','FEBMAR','MARAPR',
                            'APRMAY','MAYJUN','JUNJUL','JULAUG',
                            'AUGSEP','SEPOCT','OCTNOV','NOVDEC']]
    
        for d in range(len(doy)):
            index = doy2dm(doy[d], t+2000)[1]-1
            teledats[t][d,0] = 2000+t
            teledats[t][d,1] = doy[d]
            teledats[t][d,2] = np.array(tdf.NAO[okdat])[index]
            teledats[t][d,3] = np.array(tdf.EA[okdat])[index]
            teledats[t][d,4] = np.array(tdf.WP[okdat])[index]
            teledats[t][d,5] = ensovec[index]

    teledf = pd.DataFrame(teledats[0],columns = tnames)

    #generates 13 data sets including, for each county and year the amount of data availible after N weeks for N in range(13)+1
    folder = '../data/DataParamsB'
    JSON_FILE_NAME = '../data/RegModelPerformance.json'

    optimal_data_set = parse_preformance_data(JSON_FILE_NAME)

    #setting to 1 for fast compute, else will find optimal based on performance JSON
    optimal_data_set[0] = 1
    measure_single_model_performance(optimal_data_set)


    if LOGGING_MODEL_PERF:

        objects = { 'reg_model': [], }

        NUM_SAT_RUNS = [n for n in range(1, 13)] 
        YEAR_IN = [n for n in range (2000, 2016)]
        YEAR_OUT = [n for n in range (2000, 2016)]
        
        print "Loading data from serialized python object ..."
        #Ndata_one = pickle.load(open('../data/MultivariateModelData-One.p', 'rb'))
        Ndata_all1 = pickle.load(open('../data/MultivariateModelData-Multiple.p', 'rb'))

        for num_sat_runs in NUM_SAT_RUNS:
            
            Ndata_one = Ndata_all1[num_sat_runs]
            # run parameter optimization on single sat pass data returning true on optimal input features
            print "Running Ridge/SBS Regression Model with normalized yield data (Type A) ..."
            labbool= Derivemodfunc(Ndata_one, func1a, invfunc1a, alp=.00001)
            # get label names of optimal input features
            label_names = np.array(Ndata_one[4])[labbool]
            #keep only the features in labbool
            Ndata_one_short = keep_Ndata_features(Ndata_one, labbool)

            for year_in in YEAR_IN:
                for year_out in YEAR_OUT:
                    if year_out != year_in:
                        print num_sat_runs, year_in, year_out
                        #separate into in sample and out sample data sets
                        Ndata_most, Ndata_out = split_Ndata_by_year(Ndata_one, year_out)
                        Ndata_most1, Ndata_in = split_Ndata_by_year(Ndata_one, year_in)
                        # get the model
                        regmod=trainmodel(Ndata_most, func=func1a, invfunc=invfunc1a, alp= .00001)
                        #predict using the model
                        pred_val = predict_model(Ndata_out, regmod, func=func1a, invfunc=invfunc1a)
                        pred_val1 = predict_model(Ndata_in, regmod, func=func1a, invfunc=invfunc1a)

                        #handling NaN, infiinity and large values beyond dtype(float64)
                        try:
                            InSampleR2 = r2_score(Ndata_in[1], pred_val1)
                            InSampleMSE = mean_squared_error(Ndata_in[1], pred_val1),
                            InSampleMAE = mean_absolute_error(Ndata_in[1], pred_val1)
                        except ValueError:
                            print "[In sample Metrics] Value Error Found. Setting to Zero ..."
                            InSampleR2 = 0
                            InSampleMSE = 0
                            InSampleMAE = 0
                        try:
                            OutSampleR2 = r2_score(Ndata_out[1], pred_val)
                            OutSampleMSE = mean_squared_error(Ndata_out[1], pred_val)
                            OutSampleMAE = mean_absolute_error(Ndata_out[1], pred_val)
                        except ValueError:
                            print "[Out sample Metrics] Value Error Found. Setting to Zero ..."
                            OutSampleR2 = 0
                            OutSampleMSE = 0
                            OutSampleMAE = 0

                        objects['reg_model'].append({
                            'NumberOfSatelliteRuns': num_sat_runs,
                            'YearIn': year_in,
                            'YearOut': year_out,
                            'FeatureLaabels': list(label_names),
                            'InSampleR2': InSampleR2,
                            'InSampleMSE': InSampleMSE,
                            'InSampleMAE': InSampleMAE,
                            'OutSampleR2': OutSampleR2,
                            'OutSampleMSE': OutSampleMSE,
                            'OutSampleMAE': OutSampleMAE                                                 
                        })
        
        print "Writing performance data to JSON File ..."
        with open(JSON_FILE_NAME, 'w') as outfile:
            json.dump(objects, outfile, sort_keys = True, indent = 4)
        outfile.close()

    if RUN_INDIVIDUAL_PRED:

        if PREDICT_ONE:
            #A quick run through of how these work:
            #First generate Ndata and find out which features matter
            # single satellite pass data
            #print "Creating and loading plausible (X, y) training dataset ..."
            #Ndata_one_created = Ndayalldata(1,folder)
            #saving data
            #pickle.dump(Ndata_one_created, open('../data/MultivariateModelData-One.p', 'wb'))
            NUM_SAT_RUNS = 4
            print "Loading data from serialized python object ..."
            #Ndata_one = pickle.load(open('../data/MultivariateModelData-One.p', 'rb'))
            Ndata_all1 = pickle.load(open('../data/MultivariateModelData-Multiple.p', 'rb'))
            Ndata_one = Ndata_all1[NUM_SAT_RUNS]
            print Ndata_one[2]
            # run parameter optimization on single sat pass data returning true on optimal input features
            print "Running Ridge/SBS Regression Model with normalized yield data (Type A) ..."
            labbool= Derivemodfunc(Ndata_one, func1a, invfunc1a, alp=.00001)
            # get label names of optimal input features
            label_names = np.array(Ndata_one[4])[labbool]
            #keep only the features in labbool
            Ndata_one_short = keep_Ndata_features(Ndata_one, labbool)
            #separate into 2000-2014 and 2015
            Ndata_most, Ndata_2015 = split_Ndata_by_year(Ndata_one, 2011)
            Ndata_most1, Ndata_2014 = split_Ndata_by_year(Ndata_one, 2014)
            # get the model
            regmod=trainmodel(Ndata_most, func=func1a, invfunc=invfunc1a, alp= .00001)
            #predict using the model
            pred_val = predict_model(Ndata_2015, regmod, func=func1a, invfunc=invfunc1a)
            print label_names, len(label_names)
            # print pred_val, len(pred_val)
            # print Ndata_2015[1], len(Ndata_2015[1])
            print 
            print "Out of sample, predicting year 2015 "
            print "R2  : ", r2_score(Ndata_2015[1], pred_val)
            print "MSE : ", mean_squared_error(Ndata_2015[1], pred_val)
            print "MAE : ", mean_absolute_error(Ndata_2015[1], pred_val)
            print 
            plt.figure()
            #plt.bar(np.arange(len(pred_val))*2, pred_val, color = 'red')
            #plt.bar(np.arange(len(Ndata_2015[1]))*2+1, Ndata_2015[1], color = 'blue')        
            plt.plot(pred_val, color = 'red', label = 'Predicted Yield')
            plt.plot(Ndata_2015[1], color = 'blue', label = 'Actual Yield')              
            plt.legend()
            pred_val = predict_model(Ndata_2014, regmod, func=func1a, invfunc=invfunc1a)
            print label_names, len(label_names)
            # print pred_val, len(pred_val)
            # print Ndata_2014[1], len(Ndata_2014[1])
            print 
            print "In sample, predicting year 2014 "
            print "R2  : ", r2_score(Ndata_2014[1], pred_val)
            print "MSE : ", mean_squared_error(Ndata_2014[1], pred_val)
            print "MAE : ", mean_absolute_error(Ndata_2014[1], pred_val)
            print 
            plt.figure()
            #plt.bar(np.arange(len(pred_val))*2, pred_val, color = 'red')
            #plt.bar(np.arange(len(Ndata_2015[1]))*2+1, Ndata_2015[1], color = 'blue')        
            plt.plot(pred_val, color = 'red', label = 'Predicted Yield')
            plt.plot(Ndata_2014[1], color = 'blue', label = 'Actual Yield')              
            plt.legend()
            plt.show()        

        else:

            # Ndata_all = [Ndayalldata(i+1,folder) for i in range(13)]

            # pickle.dump(Ndata_all, open('../data/MultivariateModelData-Multiple.p', 'wb'))
            print "Loading data from serialized python object ..."
            Ndata_all1 = pickle.load(open('../data/MultivariateModelData-Multiple.p', 'rb'))


            #derives models for all N , prints the liablity error and Rsq and returns the R sq values as Rsq_all_b for normalization function b
            labels_all_b = []
            Rsq_all_b =[]
            for i in range(len(Ndata_all1)):
                Ndata=Ndata_all1[i]
                labels_all_b.append(Derivemodfunc(Ndata, func1b, invfunc1b, alp=.00001))
                label_names = np.array(Ndata[4])[labels_all_b[i]]
                Rsq_all_b.append(oneyearoutbylabfunc(Ndata, label_names, 'Predictionsb'+str(i)+'.pdf', func1b, invfunc1b, alp=.00001))
                print "R2 : ", Rsq_all_b[i]
                print "Number of Input Params : ", len(labels_all_b[i])
                print "Input Params : ", labels_all_b[i]
                print 

            #derives models for all N , prints the liablity error and Rsq and returns the R sq values as Rsq_all_a for normalization function a
            print "Running Ridge/SBS Regression Model with normalized yield data (Type A) ..."    
            labels_all_a = []
            Rsq_all_a =[]
            for i in range(len(Ndata_all1)):
                Ndata=Ndata_all1[i]
                labels_all_a.append(Derivemodfunc(Ndata, func1a, invfunc1a, alp=.00001))
                label_names = np.array(Ndata[4])[labels_all_a[i]]
                Rsq_all_a.append(oneyearoutbylabfunc(Ndata, label_names, 'Predictionsa'+str(i)+'.pdf', func1a, invfunc1a, alp=.00001))
                print "R2 : ", Rsq_all_a[i]
                print "Number of Input Params : ", len(labels_all_a[i])
                print "Input Params : ", labels_all_a[i]
                print 


    if LIABILITY_PLOT:
        #makes a plot of the liability to a county yield based on a yield 
        #insurance, may or may not be useful to harvesting
        liab = [insuredcornperacre(i, 150, 30, .75, .75, 3.8) for i in range(250)]

        matplotlib.rc('xtick', labelsize=20) 
        matplotlib.rc('ytick', labelsize=20) 
        print "Creating Liability Plot ..."
        pp = PdfPages('LiabilityPlot.pdf')
        plt.plot(range(250),liab)
        #plt.plot([150,150],[0,max(liab)])
        plt.xticks([0,100,200])
        plt.yticks([0,100,200,300])
        plt.xlabel('Yield Bushels/Acre')
        plt.ylabel('Liability $/Acre')
        pp.savefig()
        pp.close()







