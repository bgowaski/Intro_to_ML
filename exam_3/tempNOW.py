#Exam 3 Question 1 Ben Gowaski
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from PIL import Image
import cv2
import numpy as np
from sklearn import mixture
import os
from matplotlib import pyplot as plt
from pylab import *

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
#GMM Section of code:
#code adapted from https://github.com/GordonCai/Project-Image-Segmentation-With-Maxflow-and-GMM/blob/master/Code/Final%20Python%20code/EXT1_3_segmentations.py
# Removed F-F algorithm
'''
Note:
1. Use three clusters to get 3 segmentaions
2. Use Numba to speed up F-F algorithm
'''

####################
# Helper Functions #
####################
def DS(img): # Downsampling
    
    img_small = cv2.resize(img,(dimen_limit,dimen_limit), interpolation = cv2.INTER_CUBIC)
    dimen_1, dimen_2, _ = img_small.shape
    img_flat = np.concatenate((img_small[:,:,0].flatten().reshape(-1,1),
                               img_small[:,:,1].flatten().reshape(-1,1),
                               img_small[:,:,2].flatten().reshape(-1,1)),axis=1)
    
    return img_flat, dimen_1, dimen_2

def GMM(img_flat, dimen_1, dimen_2, k): # GMM clustering
    gmm_tmp = mixture.GaussianMixture(k, covariance_type='full', max_iter=500, n_init=5)
    gmm_tmp.fit(img_flat)
    tmp = np.argmax(gmm_tmp.predict_proba(img_flat),axis=1).reshape(-1,1)
    tmp0 = tmp.reshape(dimen_1, dimen_2).astype('float16')
    
    return tmp0

def pplot1(i,j,tmp0): # Merge two clusters
    
    tmp1=np.zeros((dimen_1, dimen_2))
    tmp1[tmp0==i]=1; tmp1[tmp0==j]=1
    
    return tmp1.astype('int16')

global shrinkradio
shrinkradio = 1
def show_clu(img,tmp,ax): # Apply mask to image

    # Form mask to segment image
    img_third = cv2.resize(img, (round(width/shrinkradio), round(height/shrinkradio)), interpolation=cv2.INTER_CUBIC)
    resized_image = cv2.resize(tmp.astype('float32'), 
                               (round(width/shrinkradio), round(height/shrinkradio)), interpolation=cv2.INTER_CUBIC).round()
    mask1 = np.zeros((round(height/shrinkradio), round(width/shrinkradio),3))

    mask1[:,:,0] = mask1[:,:,2] = mask1[:,:,1] = 1-resized_image

    seg_img1 = cv2.resize(np.multiply(mask1.astype('bool'),img_third),
                          (round(width), round(height)), interpolation=cv2.INTER_CUBIC) 
    
    ax.imshow(seg_img1)
    
    return ax

if __name__ == '__main__':
    # img1, _ = kmeans('2', 2)
    # img2, _ = kmeans('2', 3)
    # img3, _ = kmeans('2', 4)
    # img4, _ = kmeans('2', 5)
    # fig, axs = plt.subplots(1, 4)
    # for index, i in enumerate([img1, img2, img3, img4]):
    #     axs[index].imshow(i)
    # plt.tight_layout()
    # plt.show()
    ########
	# Main #
	########

	# Read image
	img1 = cv2.imread('1.jpg')
	img2 = cv2.imread('2.jpg')    
	global height, width, dimen_limit
	dimen_limit = 65

	# Find downsampling dimension for GMM
	height, width, _ = img1.shape
	large = max(height,width)
	for i in range(1,100):
	    if large/i < dimen_limit:
	        break

	# Downsampling for GMM
	img_flat, dimen_0, dimen_1 = DS(img1)
	img_flat2, dimen_2, dimen_3 = DS(img2)
	# Five cluster GMM
	tmp0 = GMM(img_flat, dimen_0, dimen_1, 1)
	tmp1 = GMM(img_flat, dimen_0, dimen_1, 2)
	tmp2 = GMM(img_flat, dimen_0, dimen_1, 3)
	tmp3 = GMM(img_flat, dimen_0, dimen_1, 4)
	tmp4 = GMM(img_flat, dimen_0, dimen_1, 5)
	tmp5 = GMM(img_flat2, dimen_2, dimen_3, 1)
	tmp6 = GMM(img_flat2, dimen_2, dimen_3, 2)
	tmp7 = GMM(img_flat2, dimen_2, dimen_3, 3)
	tmp8 = GMM(img_flat2, dimen_2, dimen_3, 4)
	tmp9 = GMM(img_flat2, dimen_2, dimen_3, 5)
	#a1 =pplot1(0,1,tmp0); a2 =pplot1(0,2,tmp0);
	a0 =pplot1(1,2,tmp0)
	a1 =pplot1(1,2,tmp1)
	a2 =pplot1(1,2,tmp2)
	a3 =pplot1(1,2,tmp3)
	a4 =pplot1(1,2,tmp4)
	a5 =pplot1(1,2,tmp5)
	a6 =pplot1(1,2,tmp6)
	a7 =pplot1(1,2,tmp7)
	a8 =pplot1(1,2,tmp8)
	a9 =pplot1(1,2,tmp9)
	#kmeans section
	kimg0, _ = [img1,img1]
	kimg1, _ = kmeans('1', 2)
	# kimg2, _ = kmeans('1', 3)
	# kimg3, _ = kmeans('1', 4)
	# kimg4, _ = kmeans('1', 5)
	# kimg5, _ = [img2,img2]
	# kimg6, _ = kmeans('2', 2)
	# kimg7, _ = kmeans('2', 3)
	# kimg8, _ = kmeans('2', 4)
	# kimg9, _ = kmeans('2', 5)

	im1og = Image.open('1.jpg')
	im2og = Image.open('2.jpg')
	# fig,(ax1,ax2,ax3,ax4) = plt.subplots(4,5,figsize=(15,8))
	fig,(ax1,ax2,ax3,ax4) = plt.subplots(4,5,figsize=(15,8))
	ax1[0] = show_clu(img1,a0,ax1[0])
	ax1[1] = show_clu(img1,a1,ax1[1])
	ax1[2] = show_clu(img1,a2,ax1[2])
	ax1[3] = show_clu(img1,a3,ax1[3])
	ax1[4] = show_clu(img1,a4,ax1[4])
	ax2[0] = show_clu(img2,a5,ax2[0])
	ax2[1] = show_clu(img2,a6,ax2[1])
	ax2[2] = show_clu(img2,a7,ax2[2])
	ax2[3] = show_clu(img2,a8,ax2[3])
	ax2[4] = show_clu(img2,a9,ax2[4])
	ax3[0] = kimg1
	ax3[1] = imshow(kimg0)
	ax3[2] = imshow(kimg0)
	ax3[3] = imshow(kimg0)
	ax3[4] = imshow(kimg0)
	ax4[0] = imshow(kimg0)
	ax4[1] = imshow(kimg0)
	ax4[2] = imshow(kimg0)
	ax4[3] = imshow(kimg0)
	ax4[4] = imshow(kimg0)
	plt.show()