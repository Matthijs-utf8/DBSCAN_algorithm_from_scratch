# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 13:13:22 2020

@author: Sietse Schröder and Matthijs Schrage
"""
import numpy as np
import random
import math
import Plotting
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import statistics
from sklearn.cluster import KMeans
import time

# random.seed(0)

#A function for copying data so we can safely process it without altering the original data.
def copy(data):
	data2 = data.copy()
	return data2

##### Class for creating random data #####

class CreateData:
	
	def __init__(self):
		pass
	
	#Spiral data is created with a certain number of points per class and a certain number of classes (represented as sprial arms on a scatter plot)
	def spiral_data(points_per_class=100, classes=3, spread=10):
		
		print("Started creating spiral data")
		
		X = np.zeros((points_per_class*classes, 2))  #Data matrix (each row = single example)
		
		for j in range(classes):
			ix = range(points_per_class*j, points_per_class*(j+1))
			r = np.linspace(0.0, 1, points_per_class)
			t = np.linspace(j*4, (j+1)*4, points_per_class) + np.random.randn(points_per_class)*0.2  
			X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)] * spread
		
		print("Finished creating spiral data")
		return X.tolist() #Returns the datapoints
	
	#Create randomly scattered data of size data_size. Spread specifies the amount of spreading in the data.
	def random_data(data_size=150, spread=50, dimensions=2):
		print("Started creating random data")
		random_points = []
		for _ in range(data_size):
			point_tested = False
			while point_tested == False:
				new_datapoint = [random.randint(-0.5*spread, 0.5*spread) for _ in range(dimensions)]
				if new_datapoint not in random_points:
					random_points.append(new_datapoint)
					point_tested = True
		print("Finished creating random data")
		return random_points
	
	#Create a grid of datapoints with spacing >= 1 in a grid the size of -gridsize x gridsize
	def grid_data(gridsize=25, spacing=1):
		print("Started creating grid data")
		X = []
		for x in range(0, gridsize, spacing):
			for y in range(0, gridsize, spacing):
				X.append([x,y])
		
		print("Finished creating grid data")
		return X


##### Preprocessing for filtering images #####

class ImagePreprocessing:
	
	def __init__(self, image, show_difference=True):
		
		self.image = image 
		self.show_difference = show_difference #If we want to show the before and after filter
# 		self.show(image, tag="Unfiltered Image")
	
	def show(self, img, tag="Image", current_fig=False):
		plot_number = 1
		#Plot image with matplotlib
		if self.show_difference == True:
# 			pass
			fig = plt.figure(plot_number)
			fig.suptitle(tag)
			plt.imshow(img)
			plt.show()
			plot_number += 1
	
	#A filter that saturates the whole image> It makes contrasts larger and it removes a bit of noise. This means we can better classify the pixels in the image.
	def saturate(self, satu=1.5):
		
		#Initialize the filtered data that we will return at the end of this method and initialize the filtered image that we will plot.
		filtered_data = []
		filtered_image = []
		
		saturation = satu #Specify level of saturation. 1.1 == +10% saturation., 0.9 == -10% saturation, 1 means that nothing happens.
		threshold = 255 / saturation #Specify threshold where black and white in an image get defined
		
		width = len(self.image[0,:]) #Get width of image
		height = len(self.image[:,0]) #Get height of image
		
		#For every pixel in the image
		for y in range(height):
			row = []
			for x in range(width):
				
				#Get rgb values of the pixel
				rgb = [ self.image[y][x][0], self.image[y][x][1], self.image[y][x][2] ]
				
				#If the pixel is really light, make it white (makes image sharper as well)
				if rgb[0] > threshold and  rgb[1] > threshold and rgb[2] > threshold:
# 					print(type(rgb[0]), type(rgb[1]), type(rgb[2]) )
					rgb = [255,255,255]
					
				
				#If the pixel is really dark, make it black (makes image sharper as well)
				elif rgb[0] < (255 - threshold) and  rgb[1] < (255 - threshold) and rgb[2] < (255 - threshold):
# 					print(type(rgb[0]), type(rgb[1]), type(rgb[2]) )
					rgb = [0, 0, 0]
					
				#Otherwise we scale the max and the min rgb values according to our saturation level
				else:
					mx = max(rgb)
					mn = min(rgb)
					
					#Get the indexes of the brightest and least bright rgb values of the pixel.
					min_rgb_index = rgb.index(mn)
					max_rgb_index = rgb.index(mx)
					
					#Scale the least bright rgb value down with the saturation level
					rgb[min_rgb_index] = max(0, int( round( mn / saturation ) ) )
					
					#Scale the brightest rgb value up with the saturation level
					rgb[max_rgb_index] = min(255, int( round( mx * saturation ) ) )
				
				#Get the RGB values of all pixels and store them together with the coordinates as our data.
				filtered_data.append([ x, y, int(rgb[0]), int(rgb[1]), int(rgb[2]) ] )
				row.append([int(rgb[0]), int(rgb[1]), int(rgb[2])])
				
			#Because of the way matplotlib.pyplot.imshow() works we need to return the filtered image in rows.
			filtered_image.append(row)
		
		#We only plot the filtered image if show_difference is set to True
		if self.show_difference == True:
			self.show(filtered_image, tag="Filtered Image")
		
# 		print("Finished saturating image")
		#Returns a list of pixels with the new RGB values
		return filtered_data, filtered_image

##### Preprocessing for more general applications #####

class DataPreprocessing:
	
	def __init__(self):
		pass
	
	#Scale the data according to the range of it's individual featuresets. We can use different scaling modes. Normalizing means scaling between 0 and 1.
	def scale(data, mode="normalize"):
		#Copy the data so we can safely modify it
		data = copy(data)
		
		if mode == "normalize":
			for column in range(len(data[0])): #For every featureset in the data:
				featureset = np.array(data)[:,column] #Get the featureset as a list
				max_val = max(featureset) #Get the minumum value
				min_val = min(featureset) #Get the maximum value
				for index, feature in enumerate(featureset):
					data[index][column] = (feature - min_val) / (max_val - min_val) * 100 #Scale everything in the featureset according to this formula, which puts it between 0 and 1.
		
# 		print("Finished scaling data")
		
		#Return the scaled data
		return data




"""
def test(data):
	neigh = KNeighborsClassifier(n_neighbors=3)
	neigh.fit(data)


### Let maar niet op deze functie ###
def calculate_eps(data):
	print("Started calculating optimum eps")
# 	print(data[0])
	nr_of_datapoints = len(data)
	dimensions = len(data[0])
	
	space = 100**dimensions
	
	t = time.perf_counter()
	
	noise_points = []
	border_points = []
	core_points = []
# 	min_samples=5
	
# 	nbrs = NearestNeighbors(n_neighbors=min_samples, radius=50.2, algorithm='ball_tree').fit(data)
	
	def k_nearest_query(data, min_samples, eps):
		nbrs = NearestNeighbors(n_neighbors=min_samples, radius=eps, algorithm='ball_tree').fit(data)
		
# 		distances_radius_query, indexes_radius_query = nbrs.radius_neighbors(data)
		
		distances_k_query, indices_k_query = nbrs.kneighbors(data, return_distance=True)
		
	# 	distances_k_query = np.sort(distances_k_query, axis=0)
		
		min_eps_for_all_labeled = []
		for distances in distances_k_query:
			min_eps_for_all_labeled.append(max(distances))
		return max(min_eps_for_all_labeled)
	
	baseline_distance = k_nearest_query(data, 2, 1) 
	min_distances = []
	optima = []
	density = nr_of_datapoints / space
# 	variances = [statistics.variance( [ x[n] for x in data ] ) for n in range(dimensions) ] #5.4 sec voor berekenen dimension_to_split
# 	mean_variance = statistics.mean(variances)
# 	print(mean_variance)
# 	slope = 
# 	occupied = (space / (space - nr_of_datapoints)
# 	print(occupied)
# 	max_eps = (1/density) *
	print(math.log10(density))
# 	baseline_distance = baseline_distance * math.log10(mean_variance)
# 	print(baseline_distance)
	min_samples = range(2,100)
	
	slope = (k_nearest_query(data,  min_samples[-1], 1) - k_nearest_query(data, min_samples[0], 1)) / min_samples[-1]
	
	for min_sample in min_samples:
		print(min_sample)
		min_distances.append(k_nearest_query(data, min_sample, 1))
	
	jumps = []
	for index, min_distance in enumerate(min_distances):
		if index != 0:
			jump = min_distances[index] - min_distances[index - 1]
			if jump > 3*slope:
				jumps.append([jump, min_samples[index-1]])
	
	print(jumps)
# 	jumps = sorted(jumps)
# 	slope = (jumps[-1][0] - jumps[0][0]) / min_samples[-1]
	
# 	for index, min_distance in enumerate(min_distances):
# 		if index != 0:
# 			distance = min_distances[index] - min_distances[index-1]
# 			jumps.append([distance, min_samples[index-1]])
	
# 	print(slope)
			
# 		if min_distances[index] - min_distances[index-1] > (slope*baseline_distance):
# 			optima.append([min_samples[index-1], min_distances[index-1]])
# 			print("Optimum nr of samples: ", min_samples[index-1])
# 			print("Optimum eps range for this sample size: ", slope * min_distances[index-1])
# 			print(min_distances[index-1])
# 			print(min_samples[index-1])
	
# 	print(slope)
# 	print((slope*baseline_distance))
# 	print(space/nr_of_datapoints)
	plt.plot(min_samples, min_distances)
	plt.show()
	
	
	
	print("test")
	
	distances_radius_query, indexes_radius_query = nbrs.radius_neighbors(data)
	
	print("test2")
		
# 		fig = plt.figure()
# 		fig.suptitle(min_samples)
# 		plt.plot(distances_k_query)
# 		plt.show()
		
	noise = 0
	border = 0
	core = 0
# 		
	for index in indexes_radius_query:
		if 1 < len(index) < min_samples:
			border += 1
		elif len(index) == 1:
			noise += 1
		elif len(index) > min_samples:
			core += 1
	print("Noise, ", noise)
	print("Border, ", border)
	print("Core, ", core)
# 	noise_points.append(noise)
# 	border_points.append(border)
# 	core_points.append(core)
# # 	
# # 	print(time.perf_counter() - t)
# 	plt.plot(noise_points, label="noise")
# 	plt.plot(border_points, label="border")
# 	plt.plot(core_points, label="core")
# 	plt.legend()
# 	plt.show()

	
# 	for eps in range(0,10):
# 	nbrs = NearestNeighbors(n_neighbors=min_samples, radius=0.4, algorithm='ball_tree').fit(data)

# 	distances_radius_query, indexes_radius_query = nbrs.radius_neighbors(data)
# 	
# 	for point in distances_radius_query:
# 		for distance in point:
# 			if distance > 0.4:
# 				print("Wtf")
# 	
# 	noise = 0
# 	border = 0
# 	core = 0
# 	
# 	for index in indexes_radius_query:
# 		if 1 < len(index) < 2:
# 			border += 1
# 		elif len(index) == 1:
# 			noise += 1
# 		elif len(index) >= 2:
# 			core += 1
# # 	noise_points.append(noise)
# # 	border_points.append(border)
# # 	core_points.append(core)
# 	
# 	print("Noise, ", noise)
# 	print("Border, ", border)
# 	print("Core, ", core)
		
	
# 	noise = 0
# 	border = 0
# 	core = 0
# 	

# 	
# 	for index in indexes_radius_query:
# 		if 1 < len(index) < min_samples:
# 			border += 1
# 		elif len(index) == 1:
# 			noise += 1
# 		elif len(index) > min_samples:
# 			core += 1
# 	
# 	print("Noise, ", noise)
# 	print("Border, ", border)
# 	print("Core, ", core)
		
	
# 	print(len(distances_test))
# 	print(len(indexes_test))
	
# 	for index in indexes_test:
# 		print(data[index])
	
	
	#We take the distances of all those neighbours and take the average
# 	distances, indices = nbrs.kneighbors(data)
	
# 	mean_distance = statistics.mean([statistics.mean([x for x in distance]) for distance in distances])
# 	distances1 = np.sort(distances, axis=0)
# 	distances1 = distances1[:,1]
# 	plt.plot(distances1)
# 	plt.show()
	
# 	variances = [statistics.variance( [ x[n] for x in data ] ) for n in range(dimensions) ] #5.4 sec voor berekenen dimension_to_split
# 	mean_variance = statistics.mean(variances)
# # 	print(variances)
# 	print(mean_variance)
# 	print(mean_distance)
# 	eps = mean_variance ** (1/1.85)
# 	eps = mean_distance * math.log(mean_variance, mean_distance)
# 	print(eps)

# 	print(mean_distance)math.log(np.sqrt(min_samples), dimensions) 
	### Works up to dimensions = 7 ###
 	# optimum_eps = (min_samples ** (1/6)) * (math.log(mean_variance, min_samples*dimensions)) * mean_distance * ( 2 ** (  math.log(min_samples**(1/2), np.sqrt(mean_variance) ) ) )
# 	optimum_eps = mean_distance * (min_samples ** (1/3) ) * (math.log(mean_variance, np.sqrt(min_samples)*np.sqrt(dimensions) ) )
	
# 	print("Finished calculating optimum eps: ", optimum_eps)
# 	return optimum_eps
# 	mean_distances = [statistics.mean([x for x in distance]) for distance in distances]
# 	optimum_eps = statistics.median(mean_distances) * min_samples
	

	
	#The optimum eps is (roughly) the maximum mean_distance
# 	return eps

"""