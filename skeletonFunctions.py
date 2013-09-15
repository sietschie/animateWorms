#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import vectorFunctions as vf
import numpy as np
import cv2

def polygonFromTangents(tangents):
	pts = [];
	for i in range(len(tangents)):
		pts.append(tangents[i][0])
	for i in reversed(range(len(tangents))):
		pts.append(tangents[i][1])
	return pts
	
def maskFromTangents(image, tangents, margin):
# creates a mask from the tangents that encloses object, a safety margin is added
	poly = polygonFromTangents(tangents)
	
	polyInt = [[int(elem) for elem in pair] for pair in poly]
	
	mask = np.zeros(image.shape, np.uint8);
	
	#print polyInt
	
	cv2.fillPoly(mask, np.array([polyInt]),[255,255,255,255]);

	element = cv2.getStructuringElement(cv2.MORPH_RECT,(margin,margin))
	mask = cv2.dilate(mask,element)

	return mask


def segmentsFromPointsToPolar(segments):
# convert so that each segment is described by a length and an angle
# the starting point is the end of the previous segment
	
	polarSegments = [segments[0]] # start point
	# sequential
	for i in range(1,len(segments)):
		p1 = segments[i-1]
		p2 = segments[i]
		
		distance = vf.distance(p2,p1)
		diff = vf.pDiff(p2, p1)
		theta = math.atan2(diff[0], -diff[1])
		
		polarSegments.append([distance, theta])
		
	return polarSegments
	
	
def segmentsFromPolarToPoints(polarSegments):
# inverse of segmentsFromPointsToPolar
	segments = [polarSegments[0]] # start point
	
	for i in range(1,len(polarSegments)):
		p1 = segments[-1]
		
		distance = polarSegments[i][0]
		theta = polarSegments[i][1]
		
		
		p2 = [p1[0] + distance*math.sin(theta), p1[1] - distance*math.cos(theta)]
		
		segments.append(p2)
		
	return segments 
