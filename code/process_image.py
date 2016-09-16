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

RESIZE = 128
PLOT_ROWS = 10
PLOT_COLS = 10
NUM_BINS = 256
HIST_SCALE_FAC = 5000000

def plotHistograms(fileList, recs, fig):

	recs = sf.records()

	for index in tqdm(range(len(fileList))):
		im = Image.open(fileList[index][2])
		#print im.size
		#img1 = im.resize((RESIZE, RESIZE), Image.ANTIALIAS)
			
		ima = np.array(im)
		ima = np.reshape(ima, (1, np.product(ima.shape)))   #ima.ravel()
		print index, ima.shape
		hista = np.histogram(ima, bins=NUM_BINS)
		imMean = np.mean(ima)
		imStd = np.std(ima)
		print imMean, imStd
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

def createPlots(df):
	
	plt.figure()
	sns.set(color_codes = True)
	cols = ['mean', 'std deviation', 'max histogram', 'yield']
	sns.pairplot(df[cols])

	plt.figure()
	cm = np.corrcoef(df[cols].values.T)
	sns.set(font_scale = 1.5)
	hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size':15}, yticklabels=cols, xticklabels=cols)

def createRegressionModel(df):

	X = df[['mean', 'std deviation']].values
	y = df['yield'].values 
	sc_x = StandardScaler()
	sc_y = StandardScaler()
	X_std = sc_x.fit_transform(X)
	y_std = sc_y.fit_transform(y)
	#print X1

	slr = LinearRegression()
	slr.fit(X, y)
	print "Slope : ", slr.coef_
	print "Intercept : ", slr.intercept_

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


if __name__ == '__main__':

	sf = shapefile.Reader('../data/Shapefile/IOWA_Counties/IOWA_Counties.shp')
	fileList = cPickle.load(open('../data/ImageFilePaths.p', 'rb'))
	jsomFileName = '../data/ImageFileStatistics.json'
	yieldValueFileName = '../data/YieldByCounty2015.txt'

	# fig = plt.figure()
	# plotHistograms(fileList, recs, fig)
	# plt.show()

	#computeImageStatistics(fileList, sf, jsomFileName)
	df = createDataFrame(jsomFileName, yieldValueFileName)
	createPlots(df)

	model = createRegressionModel(df)


	plt.show()













