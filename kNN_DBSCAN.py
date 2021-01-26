# -- coding: utf-8 --
"""
Created on Mon Apr 27 16:42:55 2020

@authors: Matthijs Schrage en Sietse Schröder

Test implementation of N dimensional kNN search tree
"""
import random
import time
from operator import itemgetter
from collections import deque

##### A square class to handle instances of squares during the tree expansion and searching #####
class Square:
	#Initiate properties of a square instance
	def __init__(self, points, children, minmax):
		self.points = points
		self.children = children
		self.level = 0
		self.minmax = minmax
		self.len = len(points)
	
	def divide(self):
		#Determine branching factor of this square.
		divide_by = 4 if self.len > 3 else self.len
		
		#Determine dimension of data.
		dimensions = len(self.minmax[0])
		
		#Sort points in this square in a random dimension
		sorted_points = sorted(self.points, key=itemgetter(random.randint(0, dimensions - 1)))
		
		#Determine on which indices the points should be split to divide them among children
		split_indices = [int(x * (len(sorted_points) / divide_by)) for x in range(divide_by + 1)]
		
		#Create lists of points for each of the children
		points_children = [self.points[split_indices[x - 1]:split_indices[x]] for x in range(1, len(split_indices))]
		
		#Create the children
		self.children = [Square(points_children[x], None, [[min([n[d] for n in points_children[x]]) for d in range(dimensions)], [max([n[d] for n in points_children[x]]) for d in range(dimensions)]]) for x in range(len(points_children))]
		
		if self.len > 1:
			self.points = []
		
		#Recursively call divide on each of the chilren with more than 1 point
		for child in self.children:
			if child.len > 1:
				child.divide()

##### Initiate a class to make a kNN tree from the dataset #####
class kNNTree:
	def __init__(self, data, plotting=False):

		t_tree = time.perf_counter()
# 		print("Started building tree")
		
		#Initiate the first square, which should exist from the minimum value of a featureset to the maximum value of a featureset
		self.dimensions = len(data[0])
		self.top_square = Square(data, None, [[min([n[d] for n in data]) for d in range(self.dimensions)], [max([n[d] for n in data]) for d in range(self.dimensions)]])
		
		#Recursion in divide function
		self.top_square.divide()
		
# 		print("Tree building time: ", time.perf_counter() - t_tree)
		
	#A method to search for the neighbours of a certain point inside of the eps range
	def search(self, point_to_check, eps):
		#Make a square object from the eps box around the point
		eps_box_minmax = [[point_to_check[i] - eps for i in range(self.dimensions)], [point_to_check[i] + eps for i in range(self.dimensions)]]
		
		#Initiate queue and put the top square in it
		queue = deque([self.top_square])
		
		#Initiate an empty list to store the neighbours in later on
		points_inside_eps = deque([]) ########
		
		#While we have not exhausted the list of potential neighbours yet
		while len(queue) > 0:

			#Get the first alement from the queue
			square = queue.popleft()

			#We check if the eps_box overlaps with the current square in the queue (we start at the top of the kNN tree). It adds it's children if it overlaps, otherwise it discards this square and it's children.
			if self.squares_collide(square, eps_box_minmax):
				
				#If the eps_box overlaps with a square with 1 point in it, it checks if that point actually falls in the eps range (which is actually a circle with radius 3)
				if square.len == 1:
 					points_inside_eps.extend(square.points) ##########
					
				#Else the square still has descendants, add those to the queue
				else:
					queue.extend(square.children)
		
		return points_inside_eps
	
	#Method to check whether two N dimensional squares collide
	def squares_collide(self, square1, square2):
		for i in range(self.dimensions):
			if (square2[0][i] > square1.minmax[1][i] or square1.minmax[0][i] > square2[1][i]):
				return False
		return True