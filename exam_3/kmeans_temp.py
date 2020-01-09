#Exam 3 Question 1 Ben Gowaski
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from PIL import Image

# img1 = 'bird.jpg'
# img2 = 'plane.jpg'
# out1 = 'clustered_bird.jpg'
# out2 = 'clustered_plane.jpg'

def kmeans(n,k):
	K = k
	iterations = 6
	#	Open input image
	imgIn = Image.open(f'{n}.jpg')
	row = imgIn.size[0]
	column = imgIn.size[1]

	#feature vector for each pixel use a 5-dimensional feature vector
	featureVector = np.ndarray(shape=(row * column, 5), dtype=float)
	#Array to store the clustered image
	clustered_img = np.ndarray(shape=(row * column), dtype=int)

	#Initialize the feature vector (r, g, b, x, y)
	for y in range(column):
	      for x in range(row):
	      	pixel = (x, y)
	      	rgb = imgIn.getpixel(pixel)
	      	featureVector[x + y * row, 0] = rgb[0]
	      	featureVector[x + y * row, 1] = rgb[1]
	      	featureVector[x + y * row, 2] = rgb[2]
	      	featureVector[x + y * row, 3] = x
	      	featureVector[x + y * row, 4] = y

	#Get the normalized feature vector
	normVector = preprocessing.normalize(featureVector)

	# initialize locations for post clustering
	location = np.ndarray(shape=(K,5))
	for index, center in enumerate(location):
		location[index] = np.random.uniform(np.amin(normVector), np.amax(normVector), 5)

	for iteration in range(iterations):
		#Get euclidean distances
		for position, data in enumerate(normVector):
			eucDistArray = np.ndarray(shape=(K))
			for index, center in enumerate(location):
				eucDistArray[index] = euclidean_distances(data.reshape(1, -1), center.reshape(1, -1))
			clustered_img[position] = np.argmin(eucDistArray)
		##################################################################################################
		#	Check if a cluster is ever empty, if so append a random datapoint to it
		clusterToCheck = np.arange(K)		#contains an array with all clusters
											#e.g for K=10, array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
		clustersEmpty = np.in1d(clusterToCheck, clustered_img)
											#^ [True True False True * n of clusters] False means empty
		for index, item in enumerate(clustersEmpty):
			if item == False:
				clustered_img[np.random.randint(len(clustered_img))] = index
				# ^ sets a random pixel to that cluster as mentioned in the homework writeup
		##################################################################################################
		#Cluster towards centroids
		for i in range(K):
			centroidLocations = []
			for index, item in enumerate(clustered_img):
				if item == i:
					centroidLocations.append(normVector[index])
			centroidLocations = np.array(centroidLocations)
			location[i] = np.mean(centroidLocations, axis=0)
	#Assign cluster colors to that of centroids
	for index, item in enumerate(clustered_img):
		# mask the r, g, b positions of feature vector to get
		featureVector[index][0] = int(round(location[item][0] * 255))
		featureVector[index][1] = int(round(location[item][1] * 255))
		featureVector[index][2] = int(round(location[item][2] * 255))

	# create new image of output
	imgOut = Image.new("RGB", (row, column))	
	for y in range(column):
		for x in range(row):
		 	imgOut.putpixel((x, y), (int(featureVector[y * row + x][0]), 
		 							int(featureVector[y * row + x][1]),
		 							int(featureVector[y * row + x][2])))
	imgOut.save(f'k{k}.jpg')
	return imgOut, imgIn

if __name__ == '__main__':
    img1 = Image.open('1.jpg')
    img2, _ = kmeans('1', 2)
    img3, _ = kmeans('1', 3)
    img4, _ = kmeans('1', 4)
    img5, _ = kmeans('1', 5)
    img6 = Image.open('2.jpg')
    img7, _ = kmeans('2', 2)
    img8, _ = kmeans('2', 3)
    img9, _ = kmeans('2', 4)
    img10, _ = kmeans('2', 5)
    fig, (ax1 ,ax2) = plt.subplots(2, 5)
    for index, i in enumerate([img1, img2, img3, img4, img5]):
        ax1[index].imshow(i)
    for index1, j in enumerate([img6, img7, img8, img9, img10]):
        ax2[index1].imshow(j)
    plt.tight_layout()
    plt.show()