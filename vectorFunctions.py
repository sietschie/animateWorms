#!/usr/bin/python
# -*- coding: utf-8 -*-

import math

def pDiff(v,w):
	return [v[0]-w[0], v[1]-w[1]]

def distance(v,w):
	return pLength(pDiff(v,w))

def pLength(v):
	return math.sqrt( math.pow(v[0],2) + math.pow(v[1],2) )

def dot(v,w):
	return v[0]*w[0]+v[1]*w[1]
	
def angle(v,w):
	costheta = dot(v,w) / (pLength(v) * pLength(w))
	return math.acos(costheta)
	
def isLeft(a,b,c):
	return ((b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0])) > 0;

	
def getTangentialAngle(p1,p2,p3):
# 
	#alpha1 = angle( pDiff(p2,p1), [0,1] )
	#alpha2 = angle( pDiff(p1,p2), pDiff(p3,p2) )
	
	alpha1 = math.atan2( pDiff(p2,p1)[1], pDiff(p2,p1)[0] )
	#alpha2 = math.atan2( 
	
	alpha2_1 = math.atan2( pDiff(p3,p2)[1], pDiff(p3,p2)[0] )
	alpha2_2 = math.atan2( pDiff(p1,p2)[1], pDiff(p1,p2)[0] )
	
	alpha2 = alpha2_1 - alpha2_2

	#print alpha1, alpha2, alpha2_1, alpha2_2, p1, p2, p3
	
	return alpha2/2 + alpha1 - math.pi
	
def shortestDistanceToLine(v,w,p):
	l2 =  distance(v,w)*distance(v,w);
	assert l2 != 0.0
	
	t = dot( [ p[0]-v[0], p[1]-v[1] ], [ w[0]-v[0], w[1]-v[1] ] ) / l2

	projection = [ v[0] + t*(w[0]-v[0]), v[1] + t*(w[1]-v[1]) ];
	return distance(p,projection)
	
	
def shortestDistanceToSegment(v,w,p):
# Return minimum distance between line segment vw and point p
# adapted from: http://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
	l2 =  distance(v,w)*distance(v,w);
	if l2 == 0:
		return distance(p,v)
		
	t = dot( [ p[0]-v[0], p[1]-v[1] ], [ w[0]-v[0], w[1]-v[1] ] ) / l2
	
	if t < 0.0:
		return distance(p,v)
	elif t > 1.0:
		return distance(p,w)
		
	projection = [ v[0] + t*(w[0]-v[0]), v[1] + t*(w[1]-v[1]) ];
	return distance(p,projection)

def DouglasPeucker(PointList, epsilon):
# Find the point with the maximum distance
# from: http://en.wikipedia.org/wiki/Ramer–Douglas–Peucker_algorithm
	dmax = 0
	index = 0
	for i in range(1,len(PointList)-1):
		d = shortestDistanceToSegment(PointList[0], PointList[-1], PointList[i])
		if d > dmax:
			index = i
			dmax = d

	# If max distance is greater than epsilon, recursively simplify
	if dmax > epsilon:
		# Recursive call
		recResults1 = DouglasPeucker(PointList[0:(index+1)], epsilon)
		recResults2 = DouglasPeucker(PointList[index:], epsilon)
 
		# Build the result list
		ResultList = recResults1 + recResults2[1:]
	else:
		ResultList = [PointList[1], PointList[-1]]

	return ResultList
