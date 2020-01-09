'''
    IMAGE SEGMENTATION USING K-MEANS (UNSUPERVISED LEARNING)
    AUTHOR Paul Asselin

    command line arguments:
		python imageSegmentation.py K inputImageFilename outputImageFilename
	where K is greater than 2
'''
import matplotlib.pyplot as plt
import numpy as np
import sys
from PIL import Image
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances

iterations = 5

K = 3

img1 = 'bird.jpg'
img2 = 'plane.jpg'
out1 = 'clustered_bird.jpg'
out2 = 'clustered_plane.jpg'

#	Open input image
image = Image.open(img1)
row = image.size[0]
column = image.size[1]

#feature vector for each pixel use a 5-dimensional feature vector
featureVector = np.ndarray(shape=(row * column, 5), dtype=float)
#Array to store the clustered image
clustered_img = np.ndarray(shape=(row * column), dtype=int)

#Initialize the feature vector (r, g, b, x, y)
for y in range(column):
      for x in range(row):
      	pixel = (x, y)
      	rgb = image.getpixel(pixel)
      	featureVector[x + y * row, 0] = rgb[0]
      	featureVector[x + y * row, 1] = rgb[1]
      	featureVector[x + y * row, 2] = rgb[2]
      	featureVector[x + y * row, 3] = x
      	featureVector[x + y * row, 4] = y

#Get the normalized feature vector
normVector = preprocessing.normalize(featureVector)

# min max for getting distances
minValue = np.amin(normVector)
maxValue = np.amax(normVector)

# initialize locations for post clustering
location = np.ndarray(shape=(K,5))
for index, center in enumerate(location):
	location[index] = np.random.uniform(minValue, maxValue, 5)

for iteration in range(iterations):
	#Get euclidean distances
	for position, data in enumerate(normVector):
		distanceTolocation = np.ndarray(shape=(K))
		for index, center in enumerate(location):
			distanceTolocation[index] = euclidean_distances(data.reshape(1, -1), center.reshape(1, -1))
		clustered_img[position] = np.argmin(distanceTolocation)

	#Cluster towards centroids
	for i in range(K):
		dataInCenter = []

		for index, item in enumerate(clustered_img):
			if item == i:
				dataInCenter.append(normVector[index])
		dataInCenter = np.array(dataInCenter)
		location[i] = np.mean(dataInCenter, axis=0)

#Assign cluster colors to that of centroids
for index, item in enumerate(clustered_img):
	featureVector[index][0] = int(round(location[item][0] * 255))
	featureVector[index][1] = int(round(location[item][1] * 255))
	featureVector[index][2] = int(round(location[item][2] * 255))

#	Save image
image = Image.new("RGB", (row, column))

for y in range(column):
	for x in range(row):
	 	image.putpixel((x, y), (int(featureVector[y * row + x][0]), 
	 							int(featureVector[y * row + x][1]),
	 							int(featureVector[y * row + x][2])))
image.save(out1)