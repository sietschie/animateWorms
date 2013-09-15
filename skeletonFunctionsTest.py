import unittest
from skeletonFunctions import *
import math

class TestSkeletonFunctions(unittest.TestCase):

	def setUp(self):
		self.p1 = (0,0)
		self.p2 = (0,1)
		self.p3 = (0,2)
		self.p4 = (1,2)
		self.p5 = (2,2)
		self.p6 = (2,1)
		self.p7 = (2,0)
		self.p8 = (1,0)

	def test_polygonFromTangents(self):
		tangents = [];
		for i in range(5):
			tangents.append([(i,i)])
		for i,j in enumerate(reversed(range(5,10))):
			tangents[i].append( (j,j) )
			
		poly = polygonFromTangents(tangents)
		for i in range(10):
			self.assertEqual( (i,i), poly[i] )


	def test_polarCoordinateTransform(self):

		# straight skeleton
		skel = [ (1,10),(1,9),(1,8),(1,7)]; 
		skelPolar = segmentsFromPointsToPolar(skel)
		self.assertEqual( skelPolar[0], skel[0])
		for i in range(1,len(skelPolar)):
			self.assertEqual( skelPolar[i][1], 0)
		skelPolarPoint = segmentsFromPolarToPoints(skelPolar)
		for i in range(0,len(skel)):
			for j in range(2):
				self.assertAlmostEqual( skelPolarPoint[i][j], skel[i][j] )
				
		# 90 degree skeleton
		skel = [ (1,10),(0,11),(-1,11),(-2,10),(-2,9),(-1,8),(0,8),(1,9),(1,10)]; 
		skelPolar = segmentsFromPointsToPolar(skel)
		self.assertEqual( skelPolar[0], skel[0])
		for i in range(1,len(skelPolar)):
			self.assertEqual( skelPolar[i][1], i*math.pi/4 - math.pi)


if __name__ == '__main__':
	unittest.main(verbosity=2)