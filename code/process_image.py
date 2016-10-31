#!/Users/gopal/projects/harvesting/code/venv/bin/python

#from wand.image import Image
import shapefile
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import cPickle
from tqdm import tqdm
import json
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.cross_validation import train_test_split
import os
from subprocess import call

RESIZE = 128
PLOT_ROWS = 10
PLOT_COLS = 10
NUM_BINS = 256
HIST_SCALE_FAC = 5000000


def plotHistograms(fileList, sf, fig):

	recs = sf.records()

	for index in tqdm(range(len(fileList))):
		im = Image.open(fileList[index][2])
		#print im.size
		#img1 = im.resize((RESIZE, RESIZE), Image.ANTIALIAS)
			
		ima = np.array(im)
		ima = np.reshape(ima, (1, np.product(ima.shape)))   #ima.ravel()
		# print index, ima.shape
		hista = np.histogram(ima, bins=NUM_BINS)
		imMean = np.mean(ima)
		imStd = np.std(ima)
		# print imMean, imStd
		# print hista
		# ax = fig.add_subplot(PLOT_ROWS, PLOT_COLS, index+1)
		# plt.imshow(img1, cmap='Greys_r')
		# need to use true_divide, because divide returns ints and all zero values
		max_hista = np.amax([row[0] for row in hista])
		hista = np.true_divide(hista, max_hista)
		plt.subplot(PLOT_ROWS, PLOT_COLS, index+1)
		plt.plot(hista[0], 'bo')
		plt.xlim(0, NUM_BINS+1)
		plt.ylim(0, 1.0)
		plt.title(str(recs[index][2]))

	fig.suptitle('Distribution Plots for County')


def getYieldValues(yieldValueFileName):

	with open(yieldValueFileName) as f:
		lines = f.readlines()

	yieldVals = [(str(x.split(',')[0]), float(x.split(',')[1].replace('\n', ''))) for x in lines]

	return yieldVals

def createDataFrame(jsomFileName, yieldValueFileName):
	
	with open(yieldValueFileName) as f:
		lines = f.readlines()
	# generate yeild tuples (countyName<str>, yieldValues<float>)
	yieldVals = [(str(x.split(',')[0]), float(x.split(',')[1].replace('\n', ''))) for x in lines]

	with open(jsomFileName) as jsonFile:    
		imageStatistics = json.load(jsonFile)

	countyName = []
	meanVal = []
	stdVal = []
	maxVal = []
	yieldVal = []

	for index in tqdm(range(len(imageStatistics['images']))):
		item = imageStatistics['images'][index]
		for val in yieldVals:
			if item['countyName'] == val[0]:
				yieldVal.append(val[1])
				meanVal.append(item['mean'])
				stdVal.append(item['stdDev'])
				maxVal.append(item['maxHistogramVal'])
				countyName.append(item['countyName'])

	df = pd.DataFrame({
		'countyName' : countyName,
		'mean' : meanVal,
		'std deviation' : stdVal,
		'max histogram' : maxVal,
		'yield' : yieldVal 
		})

	return df

def createDataFrameFromMultipleJSON(jsomFileNames):
	
	countyName = []
	countyId = []
	year = []
	yieldVal = []
	meanVal = []
	stdVal = []
	maxVal = []

	for file in jsomFileNames:
		with open(file) as jsonFile:    
			imagePixels = json.load(jsonFile)

		for index in tqdm(range(len(imagePixels['pixelData']))):
			item = imagePixels['pixelData'][index]
			if item['yield'] != 0 and item['pixelVals'] != []:
				countyName.append(item['countyName'])
				countyId.append(item['countyId'])
				year.append(item['year'])
				yieldVal.append(item['yield'])
				pixelArray = np.array(item['pixelVals'])
				meanVal.append(np.mean(pixelArray))
				stdVal.append(np.std(pixelArray))
				maxVal.append(np.amax(pixelArray))

	df = pd.DataFrame({
		'countyName' : countyName,
		'countyId' : countyId,
		'year' : year,
		'mean' : meanVal,
		'std' : stdVal,
		'max' : maxVal,
		'yield' : yieldVal 
		})

	print "Total Number of sample points : ", len(yieldVal)

	return df


def createAnalysisPlots(df):
	
	sns.set(color_codes = True)
	cols = ['mean', 'std', 'max', 'yield']
	sns.pairplot(df[cols])

	plt.figure()
	cm = np.corrcoef(df[cols].values.T)
	sns.set(font_scale = 1.5)
	hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size':15}, yticklabels=cols, xticklabels=cols)

def createResidualPlots(y_train, y_test, y_train_pred, y_test_pred):

	plt.scatter(y_train_pred, y_train_pred-y_train, c='blue', marker='o', label='Training Data')
	plt.scatter(y_test_pred, y_test_pred-y_test, c='green', marker='s', label='Test Data')

	plt.xlabel('Predicted Values')
	plt.ylabel('Residuals')
	plt.legend(loc='upper left')
	plt.show()

def createRegressionModel(df):

	#X = df[['mean', 'std deviation']].values
	X1 = df['mean'].values
	X2 = df['std'].values
	X3 = df['max'].values
	X = np.vstack((X1, X2, X3)).T
	y = df['yield'].values 
	sc_x = StandardScaler()
	sc_y = StandardScaler()
	X_std = sc_x.fit_transform(X)
	y_std = sc_y.fit_transform(y)
	#print X1

	slr = LinearRegression()
	
	slr.fit(X_std, y_std)
	
	y_pred = slr.predict(X_std)
	
	X_train, X_test, y_train, y_test = train_test_split(X_std, y_std, test_size=0.3, random_state=0)

	slr.fit(X_train, y_train)
	
	y_train_pred = slr.predict(X_train)
	y_test_pred = slr.predict(X_test)

	print "Model Performance"
	print "R2            : ", metrics.r2_score(y_std, y_pred)
	print "R2 In Sample  : ", metrics.r2_score(y_train, y_train_pred)
	print "R2 Out Sample : ", metrics.r2_score(y_test, y_test_pred)
	print "Variance : ", metrics.explained_variance_score(y_std, y_pred)
	print "MSE : ", metrics.mean_squared_error(y_std, y_pred)
	print "MSE Train : ", metrics.mean_squared_error(y_train, y_train_pred)
	print "MSE Test : ", metrics.mean_squared_error(y_test, y_test_pred)
	print "Model Parameters"
	print "Slope : ", slr.coef_
	print "Intercept : ", slr.intercept_

	# plt.figure()
	# createResidualPlots(y_train, y_test, y_train_pred, y_test_pred)

	return slr



def computeImageStatistics(fileList, sf, jsomFileName):

	shapes = sf.shapes()
	recs = sf.records()
	countyCoord = [item.bbox for item in shapes]

	objects = {
		'images': [],	
	}

	# filelist and recs have the same number of items ie number of counties
	for index in tqdm(range(len(fileList))):  
		im = Image.open(fileList[index][2])
		# getting integer pixel values into a numpy array	
		ima = np.array(im)
		# flatten array
		ima = np.reshape(ima, (1, np.product(ima.shape)))  
		numPixels = ima.shape[1]
		# removing zero values
		ima = np.asarray([x for x in ima[0] if x > 0])
		# get histogram
		hista = np.histogram(ima, bins=NUM_BINS)
		imMean = np.mean(ima)
		imStd = np.std(ima)
		# need to use true_divide, because divide returns ints and all zero values
		maxHista = np.amax([row[0] for row in hista])
		# normalizing histogram with max value
		histaNorm = np.true_divide(hista, maxHista)

		objects['images'].append({
			'countyName': recs[index][2],
			'countyCoordinates': list(countyCoord[index]),
			'sourceImageFile': fileList[index][2],
			'histogram': hista[0].tolist(),
			'normalizedHistorgram': histaNorm[0].tolist(),
			'binEedges': hista[1].tolist(),
			'numPixels': numPixels,
			'numNonZeroPixels': ima.shape[0],
			'mean': imMean,
			'stdDev': imStd,
			'maxHistogramVal': maxHista
		})

	print
	print "Number of Statistics Captured : ", len(objects['images'])
	print
	print "Writing  JSON file with Image Statistics"
	print
	with open(jsomFileName, 'w') as outfile:
		json.dump(objects, outfile, sort_keys = True, indent = 4)
	outfile.close()

	return

def processSceneBundleToVisual(sceneDataPath):

	# f = [x[0] for x in os.walk(sceneDataPath)]
	# subdirectories = os.listdir(sceneDataPath)
	
	d = sceneDataPath
	subdirectories = filter(lambda x: os.path.isdir(os.path.join(d, x)), os.listdir(d))

	for scene in subdirectories:
		sceneSource  = '../data/sceneData/' + scene
		processFolder = '../data/processData/'
		for band in range(2, 5):
			argList = '-t_srs EPSG:3857 ' + sceneSource + '/' + scene + '_B' + str(band) + '.TIF ' + processFolder + '/' + scene + '_' +str(band) + '_projected.TIF'   
			#gdalwarp -t_srs EPSG:3857 ../data/sceneData/LC80250302015139LGN00/LC80250302015139LGN00_B$BAND.TIF $BAND-projected.tif
			call(["gdalwarp", '-t_srs', 'EPSG:3857', sceneSource + '/' + scene + '_B' + str(band) + '.TIF', processFolder + scene + '_' +str(band) + '_projected.TIF'])
		#convert -combine {4,3,2}-projected.tif RGB.tif
		call(['convert', '-combine', processFolder + scene + '_4_projected.TIF', processFolder + scene + '_3_projected.TIF', processFolder + scene + '_2_projected.TIF', processFolder + scene + '_RGB.TIF'])
		#convert -channel B -gamma 0.925 -channel R -gamma 1.03 -channel RGB -sigmoidal-contrast 50x16% RGB.tif RGB-corrected.tif
		call(['convert' , '-channel', 'B', '-gamma', '0.925', '-channel', 'R', '-gamma', '1.03', '-channel', 'RGB', '-sigmoidal-contrast', '50x16%', processFolder + scene + '_RGB.TIF', processFolder + scene + '_RGB-corrected.TIF'])
		#convert -depth 8 RGB-corrected.tif RGB-corrected-8bit.tif
		call(['convert', '-depth', '8', processFolder + scene + '_RGB-corrected.TIF', processFolder + scene + '_RGB-corrected-8bit.TIF'])

	return subdirectories


def plotColorScenes(sceneIDs):

	processFolder = '../data/processData/'

	for index in range(len(sceneIDs)):
		if index < 16:
			sceneID = sceneIDs[index]
			processImage = processFolder + sceneID + '_RGB-corrected-8bit.TIF'
			im = Image.open(processImage)
			img = im.resize((RESIZE, RESIZE), Image.ANTIALIAS)
			plt.subplot(4, 4, index+1)
			plt.imshow(img)
			plt.axis("off")
			plt.title('Scene #' + str(index+1) + ":" + sceneID)

	plt.show()


if __name__ == '__main__':

	# sf = shapefile.Reader('../data/Shapefile/IOWA_Counties/IOWA_Counties.shp')
	# fileList = cPickle.load(open('../data/ImageFilePaths.p', 'rb'))
	# jsomFileName = '../data/ImageFileStatistics.json'
	# yieldValueFileName = '../data/YieldByCounty2015.txt'

	# # fig = plt.figure()
	# # plotHistograms(fileList, sf, fig)
	# # plt.show()

	# computeImageStatistics(fileList, sf, jsomFileName)
	# df = createDataFrame(jsomFileName, yieldValueFileName)
	# createAnalysisPlots(df)
	# model = createRegressionModel(df)
	# plt.show()

	year = [2013, 2014, 2015]

	jsomFileNames = []
	for y in year:
		jsomFileName = '../data/pixelValues_' + str(y) + '.json'
		jsomFileNames.append(jsomFileName)

	df = createDataFrameFromMultipleJSON(jsomFileNames)	
	createAnalysisPlots(df)
	model = createRegressionModel(df)
	plt.show()










