# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 14:40:01 2020

@author: Sietse Schröder and Matthijs Schrage
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Circle, Rectangle

#Get a spcific amount of random colors. This is helpful when plotting a labeled set of datapoints.
def get_colors(n_colors, mode="rgba"):
	
	#Initiate empty list of colors
	colors = []
	
	#This is for scatterplots
	if mode == "rgba":
		#Random color generator (rgba, so we can use it in matplotlib)
		while len(colors) < n_colors:
			
			#Pick a random color between 0 and 255 and divide by 255 to get it in the rgba spectrum
			random_color_rgba = list(np.random.choice(range(0,255), size=3)/255)
			
			#Check if the color is not already in use
			if random_color_rgba not in colors:
				colors.append(random_color_rgba)
	
	#This is for images
	if mode == "rgb":
		#Random color generator (rgba, so we can use it in matplotlib)
		while len(colors) < n_colors:
			
			#Pick a random color between 0 and 255 and divide by 255 to get it in the rgba spectrum
			random_color_rgb = list(np.random.choice(range(0,255), size=3))
			
			#Check if the color is not already in use
			if random_color_rgb not in colors:
				colors.append(random_color_rgb)
	
	return colors

#Create a labeled scatterplot. When we specify an eps range larger than 0, the function also plots eps ranges around very point.
def labeled_scatterplot(data, labels, eps=0):
	
	#Get the randomly generated colors we need
	colors = get_colors(len(np.unique(labels)), mode="rgba")
	currentAxis = plt.gca()
	
	#Create the image with colored labels
	for index, x in enumerate(data):
		
		#In DBSCAN noise is labeled as -1. We want to plot every cluster according to its cluster color, but noise we plot as white.
		if labels[index] != -1:
			plt.scatter(x[0], x[1], color=colors[int(labels[index]) - 1], zorder=100)
			if eps > 0:
				currentAxis.add_patch(Circle((x[0], x[1]), eps,  color=colors[int(labels[index]) - 1], fill=False))
		else:
			plt.scatter(x[0], x[1], color="white", zorder=100)
			if eps > 0:
				currentAxis.add_patch(Circle((x[0], x[1]), eps, color="white", fill=False))

#Create a plot where the color of the pixel in the image reflects the cluster that it is put in.
def labeled_image_plot(image_width, image_height, labels):
	
	#Get the randomly generated colors we need
	colors = get_colors(len(np.unique(labels)), mode="rgb")
	labeled_image = []
	noise = 0
	
	#Navigate to every pixel and attach a color to its label
	for y in range(image_height):
		
		row = []
		
		for x in range(image_width):
			
# 			if labels:
			
			#Get the correct label of the pixel
			label = labels[(y * image_width + x)]
			
			#Check if the label is -1 (noise), or not
			if label != -1:
				row.append(colors[int(label) - 1])
				
			#Noise is plotted as a white pixel
			else:
				noise += 1
				row.append([255,255,255])
			
# 			elif list_of_colors:
# 				color = list_of_colors[(y * image_width + x)]
			
		labeled_image.append(row)
	
	plt.imshow(np.array(labeled_image))
	plt.show()
	print("Finished plotting. ", noise, " noise points found.")
	return labeled_image

def plot_square(x, y, w, h, nr_of_classes, fill=None, width=1, level=1):
	
	#Get the axis we are plotting on currently
	currentAxis = plt.gca()
	
	#Get a list of colors
	colors = get_colors(nr_of_classes, mode="rgba")
	
	#Set the line color according to it's level
	colour = colors[level]
	
	#Plot the rectangle in the figure
	currentAxis.add_patch(Rectangle((x, y), w, h, alpha=1, fill=None, color=colour, lw=width, zorder=level))

def rgb_heatmap(image, color="r"):
	image = np.array(image)
	if color == "r":
		image = np.delete(image,obj=[1,2], axis= 2)
	elif color == "g":
		image = np.delete(image,obj=[0,2], axis= 2)
	elif color == "b":
		image = np.delete(image,obj=[0,1], axis= 2)
	image = np.reshape(image, (64,64))
	plt.imshow(image, cmap="viridis")
	plt.colorbar()
	plt.show()
