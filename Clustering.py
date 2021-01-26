# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 11:21:04 2020

@authors: Sietse Schröder and Matthijs Schrage
"""

#Our own modules
import Clustering
import Preprocessing
import Plotting
import kNN_DBSCAN

#Imported modules
import numpy as np
import random
import copy
import time
import timeit
from operator import itemgetter
from matplotlib import style
from collections import deque
style.use('ggplot')

class DBSCAN:
	
	def __init__(self, eps=2, min_samples=3, metric="euclidian"):
		self.eps = eps #Same as above
		self.min_samples = min_samples #Does not have to be attribute, we can just state the min_samples where we call the function
		self.metric = metric #Useless
		self.labels = None #Useless
		
		#First we check if the eps and min_samples are correct.
		if not self.eps > 0:
			raise ValueError("Eps must be positive.")
		if not min_samples > 1:
			raise ValueError("Minimum samples must be larger than 1.")
		
	def fit(self, data):
		copied_data = copy.deepcopy(data)
		zeros = set([x for x in range(len(data))])
		
		for datapoint in copied_data:
			for feature in datapoint:
				if type(feature) == "numpy.uint8":
					raise TypeError("Data cannot contain type 'numpy.uint8' features")
	
		def fit_kNN(copied_data):
# 			print("Started kNN clustering.")
			
			#Initiate the tree of squares which we use to search for neighbours. Also specify if we want to plot the squares.
			tree = kNN_DBSCAN.kNNTree(copied_data, plotting=False)
			
			#Initialize a time counter to measure how long it takes to label the data.
			t_fit = time.perf_counter()
			
			#For every point in the dataset, we initialize label = 0 values at the end of the datapoint, so we can replace that value later in the algorithm.
			for index, point in enumerate(copied_data):
				point.append(index) #The index value of the datapoint in the actual data.
				point.append(0) #Will become the label of the datapoint.
			
			#Initialize an empty list, add a random starting point to the list.
			random_point = random.choice(copied_data)
			to_check = deque([random_point])
			
			#Initialize the label nr. Clusters get labeled with a positive integere. Noise gets labeled as -1
			label_nr = 1
			nr_of_data_points = len(np.array(copied_data)[:,-1])
			points_checked = 0
			
			time_random = 0
			time_search = 0
			#While we have not labeled all the points, keep labeling points.
			while len(zeros) > 0:
				
				### CHECKEN OF EEN PUNT AL NEIGHBOURS HEEFT DIE WIJ KUNNEN WETEN ###

				if len(to_check) == 0:
					t9 = time.perf_counter()
					to_check.append(copied_data[random.choice(tuple(zeros))])
					time_random += time.perf_counter() - t9
# 					print("Time to get a random point: ", time.perf_counter() - t_random)
				
				#Get the first element from the list
				point = to_check.popleft()
				points_checked += 1
				
				#Search the kNN tree for neighbours of this point
				ts = time.perf_counter()
				points_inside_eps = tree.search(point, self.eps)
				time_search += time.perf_counter() - ts
				
				#The next 40 lines of code are responsible for the labeling part of the process
				if len(points_inside_eps) >= self.min_samples: #If the point qualifies for a core point
					label_core_point = copied_data[point[-2]][-1]
					
					if label_core_point == 0:
						label_core_point = label_nr
						label_nr += 1
						
					copied_data[point[-2]][-1] = label_core_point
					zeros.discard(point[-2])
					
					for neighbour in points_inside_eps: #For every point inside the range
					
						if copied_data[neighbour[-2]][-1] <= 0: #If the point has not been labeled yet
						
							to_check.append(copied_data[neighbour[-2]]) #Add the point to be checked later
							copied_data[neighbour[-2]][-1] = label_core_point
							zeros.discard(neighbour[-2])
							
				elif 1 < len(points_inside_eps) < self.min_samples:
					labels = [copied_data[x[-2]][-1] for x in points_inside_eps]
					
					most_common_label_in_range = max(labels, key=labels.count)
					
					if most_common_label_in_range == 0:
						#To check functie toevoegen
						copied_data[point[-2]][-1] = - 1
						zeros.discard(point[-2])
						
					else:
						
						copied_data[point[-2]][-1] = most_common_label_in_range
						zeros.discard(point[-2])
				
				elif len(points_inside_eps) == 1:
					copied_data[point[-2]][-1] = -1
					zeros.discard(point[-2])
				
				else:
					raise ValueError("Someting went wrong. Received empty list of neighbours")
		
		
		#Run the fit function as a fit_kNN function
		fit_kNN(copied_data)
		
		#Copy the labels back and return them as attributes.
		self.labels = np.array(copied_data, dtype="int_")[:,-1]
		
		return self.labels