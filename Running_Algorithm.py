# -*- coding: utf-8 -*-
"""
Created on Sat May  2 18:59:41 2020

@author: Matthijs Schrage en Sietse Schröder
"""

#Our own modules
import Clustering
import Preprocessing
import Plotting
import kNN_DBSCAN

#Imported modules
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.image as mpimg
from matplotlib import style
from sklearn.neighbors import NearestNeighbors, BallTree
import time
from scipy import spatial
import statistics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import DBSCAN
import operator
import copy
from numpy.random import choice
import heapq

import cProfile, pstats, io

def get_labels(filtered_data, eps, min_samples):
	return Clustering.DBSCAN(eps, min_samples).fit(filtered_data)

image_name = "mondriaan_mini.jpeg"
# image_name = "mondriaan_mini1.jpeg"
# image_name = "mondriaan_mini2.jpeg"
# image_name = "mondriaan_mini3.jpeg"
# image_name = "mondriaan_mini4.jpeg"
# image_name = "mondriaan_mini5.jpeg"
# image_name = "mondriaan_mini6.jpeg"
# img = mpimg.imread(image_name)

#Specify some stuff for the plot. 
currentAxis = plt.gca() #Get the plot axis, so we can reference it later when we want to add something to the plot
fig1 = plt.figure(1, axis=currentAxis) #Create a figure object
fig_width = fig1.set_figwidth(20) #State the figure width and height to make the resolution better
fig_height = fig1.set_figheight(fig1.get_figwidth())

# Load and label image
img = mpimg.imread(image_name)
height = len(img[:,0])
width = len(img[0,:])
filtered_data, filtered_image = Preprocessing.ImagePreprocessing(img, show_difference=False).saturate(satu=1.7)
pixel_values = np.array(copy.deepcopy(filtered_data))[:,2:5]

labels = get_labels(filtered_data, eps=45, min_samples=2)
Plotting.labeled_image_plot(10, 10, labels)
plt.show()

#Create some grid data and normalize it
filtered_data = Preprocessing.CreateData.grid_data(gridsize=10, spacing=1)
filtered_data = Preprocessing.DataPreprocessing.scale(filtered_data, mode="normalize")
labels = get_labels(filtered_data, eps=20, min_samples=2)
Plotting.labeled_scatterplot(filtered_data, labels, eps=20)
plt.show()

#Create some random data and normalize it
filtered_data = Preprocessing.CreateData.random_data(data_size=100, spread=100, dimensions=2)
filtered_data = Preprocessing.DataPreprocessing.scale(filtered_data, mode="normalize")
labels = get_labels(filtered_data, eps=10, min_samples=2)
Plotting.labeled_scatterplot(filtered_data, labels, eps=10)
plt.show()
