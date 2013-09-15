#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import math
import vectorFunctions as vf
import numpy as np
import sys
import os
from skeletonFunctions import *
from sys import stdout
from progress import *
import argparse


nrFrames = 20
alpha = 2
preview = False
smoothing = False
outputDir = 'output/'

def main():	

	parser = argparse.ArgumentParser(description='Animate images of tall, segmented structures.')
	parser.add_argument('images', type=str, nargs='+')
	parser.add_argument('--alpha', type=int, default=2)
	parser.add_argument('--frames', type=int, default=20)
	parser.add_argument('--preview', type=bool, default=False)
	parser.add_argument('--smoothing', type=bool, default=False)
	parser.add_argument('--outputDir', type=str, default='output/')
	
	
	args = parser.parse_args()
	
	
	global nrFrames
	global alpha
	global preview
	global smoothing
	global outputDir
	
	images = args.images
	alpha = args.alpha
	nrFrames = args.frames
	preview = args.preview
	smoothing = args.smoothing
	outputDir = args.outputDir
	
	#print images, alpha, nrFrames, preview, smoothing, outputDir

	progress = Progress( len(images) * (nrFrames+1))
	for image in images:
		process(image,progress)



def process(image, progress):
	filepath = image
	filename = os.path.basename(filepath)
	filebasename = os.path.splitext(filename)[0]
	
	print 'read image %s' % filename
	image=cv2.imread(filepath, cv2.CV_LOAD_IMAGE_UNCHANGED) #Load the image
	if smoothing:
		image2x = cv2.resize(image,(2*image.shape[1],2*image.shape[0]));
		image = image2x;

	print 'compute skeleton points'
	skeletonPoints = getCenterPoints(image)
	segmentPoints = getSegmentsFromImage(image,skeletonPoints)

	# hack to avoid thick boarders in first segment, also: it does not look natural 
	segmentPoints[0][0] = segmentPoints[1][0]

	imageSke = visualizeSkeleton(image, skeletonPoints, [255,0,0,255])
	imageSeg = visualizeSkeleton(image, segmentPoints, [0,255,0,255])
	skeletonTangents = computeSkeletonTangents(segmentPoints)
	extendedSkeletonTangents = extendSkeletonTangentsToObjectBorder(image, skeletonTangents)
	imageSegTan = visualizeSkeletonTangents(imageSeg, extendedSkeletonTangents, [0,0,255,255])

	if preview:
		cv2.imshow('segments tangents', imageSegTan) #Show the image
		cv2.waitKey(1)


	ts = [i/float(nrFrames) for i in range(nrFrames+1)]
	ts_ease = [ math.pow(i,alpha) / (math.pow(i,alpha) + math.pow(1-i,alpha)) for i in ts]

	if preview:
		print 'compute animation preview'
	
		skeletonAniImages = [];
		for i in reversed(range(nrFrames+1)):
			imageDest = np.zeros(image.shape, np.uint8)
	
			t = ts_ease[i]
			segments_t = interpolateSegments(segmentPoints, t)
	
			sourceSkeletonTangents = computeSkeletonTangents(segmentPoints)
			extendedSourceSkeletonTangents = extendSkeletonTangentsToObjectBorder(image, sourceSkeletonTangents)
			
			destinationSkeletonTangents = computeSkeletonTangents(segments_t)
			extendedDestinationSkeletonTangents = extendSkeletonTangentsToMatchSourceTangents( destinationSkeletonTangents, extendedSourceSkeletonTangents)
			
			imageDestSeg = visualizeSkeleton(imageDest, segments_t, [0,255,0,255])
			imageDestSegTan = visualizeSkeletonTangents(imageDestSeg, extendedDestinationSkeletonTangents, [0,0,255,255])
			
			imageMask = maskFromTangents(image, extendedDestinationSkeletonTangents, 10)
			imageMask = visualizeSkeletonTangents(imageMask, destinationSkeletonTangents, [0,0,255,255])
			#imageMask = visualizeSkeletonTangents(imageMask, extendedDestinationSkeletonTangents, [255,0,0,255])
			
			#skeletonAniImages.append(imageDestSegTan)
			skeletonAniImages.append(imageMask)
			
		print 'press key to start rendering'
		
		counter = 0
		delta = 1
		while cv2.waitKey(100) == -1:
			#print counter, delta
			cv2.imshow('skeleton animation', skeletonAniImages[counter]) #Show the image
			if counter + delta >= 21 or counter + delta < 0:
				delta = -delta
	
			counter += delta
		

	for i in reversed(range(nrFrames+1)):
		print 'image %d of %d' % (nrFrames-i, nrFrames)
		t = ts_ease[i]
		segments_t = interpolateSegments(segmentPoints, t)
		imageRemapped = remapImage(image, segmentPoints, segments_t)
		
		if smoothing:
			image2x = cv2.resize(imageRemapped, (image.shape[1]/2,image.shape[0]/2))
			imageRemapped = image2x
		
		print outputDir + filebasename + '_ani_%d.png' % i
		cv2.imwrite(outputDir + filebasename + '_ani_%d.png' % i, imageRemapped)
		
		progress.increment()
		print 'totel progress: '
		progress.print_status_line()
		print
		print



def getCenterPoints(image):
	skeletonPoints = [];
	
	for y in range(image.shape[0]):
		xcenter = -1;
		xsum = 0
		xcount = 0
		for x in range(image.shape[1]):
			#print image[y,x,3]
			if image[y,x,3] != 0:
				xcount += 1
				xsum += x
	      
		if xcount > 0:
			xcenter = float( xsum ) / xcount
			#image[y,int(xcenter)] = [255,0,0,255]
			skeletonPoints.append( (xcenter,y) )
		#print xcenter
		
	skeletonPoints.reverse()
	return skeletonPoints
	
def getSegmentsFromImage(image,skeletonPoints):

	# 0 - start, 1 - white region, 2 - black region
	currentState = 0; 
	startPoint = skeletonPoints[0]
	endPoint = skeletonPoints[0]
	segmentPoints = []
	
	for i in range(len(skeletonPoints)):
		current_pixel = image[skeletonPoints[i][1], int(skeletonPoints[i][0])]
		if current_pixel[0] < 50 and currentState != 1: #beginning black
			currentState = 1
			startPoint = skeletonPoints[i]
		elif current_pixel[0] >= 50 and currentState != 2: #beginning white
			currentState = 2
			endPoint = skeletonPoints[i]
			meanPoint = [ (startPoint[0] + endPoint[0])/2.0, (startPoint[1] + endPoint[1])/2.0 ]
			segmentPoints.append(meanPoint)
	segmentPoints.append(skeletonPoints[-1])
	
	return segmentPoints
	
def visualizeSkeleton(image, skeletonPoints, color):

	image = image.copy()

	for i in range(len(skeletonPoints)-1):
		p1int = (int(skeletonPoints[i][0]),int(skeletonPoints[i][1]))
		p2int = (int(skeletonPoints[i+1][0]),int(skeletonPoints[i+1][1]))
		cv2.line(image,p1int,p2int,color)
	return image

def extendSkeletonTangentsToMatchSourceTangents(tangents, sourceTangents):
	extendedTangents = [];

	for i in range(len(tangents)):
		l1 = tangents[i][0]
		l2 = tangents[i][1]

		lmean = [ (l1[0] + l2[0])/2.0, (l1[1] + l2[1])/2.0 ]

		sl1 = sourceTangents[i][0]
		sl2 = sourceTangents[i][1]

		slmean = [ (sl1[0] + sl2[0])/2.0, (sl1[1] + sl2[1])/2.0 ]
		
		sDistL1 = vf.distance(sl1,slmean)
		distL1 = vf.distance(l1,lmean)
		
		diff = vf.pDiff(l1, lmean)
		extendedL1 = [ lmean[0] + sDistL1 * diff[0]/distL1, lmean[1] + sDistL1 * diff[1]/distL1 ]

		sDistL2 = vf.distance(sl2,slmean)
		distL2 = vf.distance(l2,lmean)
		
		diff = vf.pDiff(l2, lmean)
		extendedL2 = [ lmean[0] + sDistL2 * diff[0]/distL2, lmean[1] + sDistL2 * diff[1]/distL2 ]

		extendedTangents.append( [ extendedL1, extendedL2 ] )
	return extendedTangents



def extendSkeletonTangentsToObjectBorder(image, tangents):
	
	extendedTangents = [];
	
	for i in range(len(tangents)):
		l1 = tangents[i][0]
		l2 = tangents[i][1]
		
		lmean = [ (l1[0] + l2[0])/2.0, (l1[1] + l2[1])/2.0 ]
		
		# for l1
		
		diff = vf.pDiff(l1, lmean)
		dist = vf.pLength(diff)
		delta = [diff[0]/dist, diff[1]/dist]
		
		prolonged_l1 = l1
		while True:
			if not( prolonged_l1[0] >= 0 and prolonged_l1[0] < image.shape[1] and prolonged_l1[1] >= 0 and prolonged_l1[1] < image.shape[0]):
				# point has left the image
				
				prolonged_l1 = l1
				break
				
			pixel = image[int(prolonged_l1[1]),int(prolonged_l1[0])]
			
			if pixel[3] < 10: #entering the transparent zone, leaving the object
				break
				
			prolonged_l1 = [prolonged_l1[0] + delta[0], prolonged_l1[1] + delta[1]] 
		prolonged_l1 = (int(prolonged_l1[0]),int(prolonged_l1[1]))
		
		diff = vf.pDiff(l2, lmean)
		dist = vf.pLength(diff)
		delta = [diff[0]/dist, diff[1]/dist]

		prolonged_l2 = l2
		while True:
			if not( prolonged_l2[0] >= 0 and prolonged_l2[0] < image.shape[1] and prolonged_l2[1] >= 0 and prolonged_l2[1] < image.shape[0]):
				# point has left the image
				
				prolonged_l2 = l2
				#print 'point left the image'
				break
				
			pixel = image[int(prolonged_l2[1]),int(prolonged_l2[0])]
			
			if pixel[3] < 10: #entering the transparent zone, leaving the object
				#print 'found end of object'
				break
				
			prolonged_l2 = [prolonged_l2[0] + delta[0], prolonged_l2[1] + delta[1]] 
			#print prolonged_l2
		prolonged_l2 = (int(prolonged_l2[0]),int(prolonged_l2[1]))
		
		extendedTangents.append( [ prolonged_l1, prolonged_l2 ] )
		
		#print 'prolong: ', [l1,l2], [ prolonged_l1, prolonged_l2 ]
	return extendedTangents

def computeSkeletonTangents(skeletonPoints):

	length = 10;
	skeletonTangents = [  [(int(skeletonPoints[0][0] + length), int(skeletonPoints[0][1])), (int(skeletonPoints[0][0] - length), int(skeletonPoints[0][1])) ]];

	for i in range(len(skeletonPoints)-2):
		p1 = skeletonPoints[i]
		p2 = skeletonPoints[i+1]
		p3 = skeletonPoints[i+2]
		
		theta = vf.getTangentialAngle(p1,p2,p3);
		
		l1 = ( int(p2[0] + length*math.cos(theta)), int(p2[1] + length*math.sin(theta)));
		l2 = ( int(p2[0] - length*math.cos(theta)), int(p2[1] - length*math.sin(theta)));
		
		skeletonTangents.append([l1,l2])
		
	pDiff = vf.pDiff(skeletonPoints[-1], skeletonPoints[-2])
	
	l1 = ( int(skeletonTangents[-1][0][0] + pDiff[0]), int(skeletonTangents[-1][0][1] + pDiff[1]) )
	l2 = ( int(skeletonTangents[-1][1][0] + pDiff[0]), int(skeletonTangents[-1][1][1] + pDiff[1]) )
	skeletonTangents.append( [l1,l2] )
	
	
	return skeletonTangents

def visualizeSkeletonTangents(image, skeletonTangents, color):
	image = image.copy()

		
	for i in range(len(skeletonTangents)):
		#print skeletonTangents[i]
		p1 = ( int(skeletonTangents[i][0][0]), int(skeletonTangents[i][0][1]))
		p2 = ( int(skeletonTangents[i][1][0]), int(skeletonTangents[i][1][1]))
		cv2.line(image,p1,p2,color)
		
	return image


def remapImage(image, sourceSkeletonPoints, destinationSkeletonPoints):
	assert len(sourceSkeletonPoints) == len(destinationSkeletonPoints)

#	imageDest = image.copy()
	imageDest = np.zeros(image.shape, np.uint8)

	
	sourceSkeletonTangents = computeSkeletonTangents(sourceSkeletonPoints)
	extendedSourceSkeletonTangents = extendSkeletonTangentsToObjectBorder(image, sourceSkeletonTangents)
	
	destinationSkeletonTangents = computeSkeletonTangents(destinationSkeletonPoints)
	extendedDestinationSkeletonTangents = extendSkeletonTangentsToMatchSourceTangents( destinationSkeletonTangents, extendedSourceSkeletonTangents)
	
	mask = maskFromTangents(image, extendedDestinationSkeletonTangents, 50)

	
	# precompute perspective transforms
	perspTransforms = [];
	for i in range(len(extendedSourceSkeletonTangents)-1):
		sp1 = extendedSourceSkeletonTangents[i][0];
		sp2 = extendedSourceSkeletonTangents[i][1];
		sp3 = extendedSourceSkeletonTangents[i+1][0];
		sp4 = extendedSourceSkeletonTangents[i+1][1];
		src = np.array([sp1,sp2,sp3,sp4],np.float32)

		dp1 = extendedDestinationSkeletonTangents[i][0];
		dp2 = extendedDestinationSkeletonTangents[i][1];
		dp3 = extendedDestinationSkeletonTangents[i+1][0];
		dp4 = extendedDestinationSkeletonTangents[i+1][1];
		dst = np.array([dp1,dp2,dp3,dp4],np.float32)
	
		ret = cv2.getPerspectiveTransform(dst,src)
		perspTransforms.append(np.array(ret))
	
	
	for x in range(imageDest.shape[1]):
		print '\r %d%% ' % ((100 * x) / imageDest.shape[1]), 
		stdout.flush()

		for y in range(imageDest.shape[0]):
		
			if mask[y,x,3] == 0:
				continue
		
			(segmentId, distance_upper, distance_lower) = determineEnclosingSegments(extendedDestinationSkeletonTangents, [x,y]) 
			#print segmentId, len(perspTransforms), distance_upper, distance_lower
			if segmentId >= 0 and segmentId < len(perspTransforms):
				
				#print perspTransforms[segmentId]
				res = perspTransforms[segmentId].dot(np.array([x,y,1]))
				
				sx = res[0]/res[2]
				sy = res[1]/res[2]
				#print [sx,sy], [x,y]
				
				# lower
				sx_lower = -1
				sy_lower = -1
				if segmentId-1 >= 0 and segmentId-1 < len(perspTransforms) and distance_lower < 10:
					res = perspTransforms[segmentId-1].dot(np.array([x,y,1]))
					
					sx_lower = res[0]/res[2]
					sy_lower = res[1]/res[2]
				else:
					distance_lower = 1e100
					
				# upper
				sx_upper = -1
				sy_upper = -1
				if segmentId+1 >= 0 and segmentId+1 < len(perspTransforms) and distance_upper < 10:
					res = perspTransforms[segmentId+1].dot(np.array([x,y,1]))
					
					sx_upper = res[0]/res[2]
					sy_upper = res[1]/res[2]
				else:
					distance_upper = 1e100


				if distance_upper < 10 and distance_lower < 10:
					if distance_upper < distance_lower:
						distance_lower = 20
					else:
						distance_upper = 20

#				if distance_upper < 10 and distance_lower < 10:
#					ndu = distance_upper / (distance_upper + distance_lower + 5.0)
#					ndl = distance_lower / (distance_upper + distance_lower + 5.0)
#					ndm = 5 / (distance_upper + distance_lower + 5.0)
#
				#	sx = ndu * sx_upper + ndl * sx_lower + ndm * sx
				#	sy = ndu * sy_upper + ndl * sy_lower + ndm * sx
				#	
				#el
				if distance_upper < 10:
					#print sx, sx_upper, distance_upper, 
					sx = (distance_upper / 20.0 + 0.5) * sx + (-distance_upper / 20.0 + 0.5) * sx_upper
					#print sx
					sy = (distance_upper / 20.0 + 0.5) * sy + (-distance_upper / 20.0 + 0.5) * sy_upper
					
				elif distance_lower < 10:
					sx = (distance_lower / 20.0 + 0.5) * sx + (-distance_lower / 20.0 + 0.5) * sx_lower
					sy = (distance_lower / 20.0 + 0.5) * sy + (-distance_lower / 20.0 + 0.5) * sy_lower
				
				
				
				if sx >= 0 and sx < image.shape[1] and sy >= 0 and sy < image.shape[0]:
					imageDest[y,x] = image[sy,sx]
			
#				imageRegions[y,x] = [255,0,0,255]
#			else:
#				imageRegions[y,x] = [0,255,0,255]

	
	#cv2.remap(image, imageDest, mapX, mapY, cv2.INTER_CUBIC)

	print '\r 100%'
	
	return imageDest

def determineEnclosingSegments(skeletonTangents, u):
	for i in range(len(skeletonTangents)):
		p1 = skeletonTangents[i][0]
		p2 = skeletonTangents[i][1]
		
		if not vf.isLeft( p1, p2, u):
		
			distance_upper = vf.shortestDistanceToLine(p1,p2,u)
			if i > 0:
				distance_lower = vf.shortestDistanceToLine(skeletonTangents[i-1][0],skeletonTangents[i-1][1],u)
			else:
				distance_lower = 1e100 #todo: besser sowas wie INF verwenden?

			#print p1, p2, u
			return (i-1, distance_upper, distance_lower)
	
	distance_lower = vf.shortestDistanceToLine(skeletonTangents[-1][0],skeletonTangents[-1][1],u)
	return (len(skeletonTangents)-1, 1e100, distance_lower)


def interpolateSegments(segments, t):

	polar = segmentsFromPointsToPolar(segments)
	
	for i in range(1,len(polar)):
		polar[i][1] = -2*(t-0.5) * polar[i][1] 
		
	return segmentsFromPolarToPoints(polar)


if __name__ == '__main__':
    main()