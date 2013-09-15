import unittest
import vectorFunctions
import math

class TestVectorFunctions(unittest.TestCase):

	def setUp(self):
		self.p1 = [0,0]
		self.p2 = [0,1]
		self.p3 = [0,2]
		self.p4 = [1,2]
		self.p5 = [2,2]
		self.p6 = [2,1]
		self.p7 = [2,0]
		self.p8 = [1,0]

	def test_dot(self):
		
		# should be zero
		self.assertEqual( vectorFunctions.dot(self.p1,self.p1), 0)
		self.assertEqual( vectorFunctions.dot(self.p3,self.p7), 0)
		
		# 
		self.assertEqual( vectorFunctions.dot(self.p5,self.p5), 8)
		self.assertEqual( vectorFunctions.dot(self.p4,self.p6), 4)

	def test_length(self):
		self.assertEqual( vectorFunctions.pLength(self.p1), 0)
		self.assertEqual( vectorFunctions.pLength(self.p2), 1)
		self.assertEqual( vectorFunctions.pLength(self.p3), 2)
		self.assertEqual( vectorFunctions.pLength(self.p5), math.sqrt(8))

	def test_distance(self):
		self.assertEqual( vectorFunctions.distance(self.p5,self.p5), 0)
		self.assertEqual( vectorFunctions.distance(self.p1,self.p1), 0)
		self.assertEqual( vectorFunctions.distance(self.p8,self.p8), 0)
		self.assertEqual( vectorFunctions.distance(self.p6,self.p6), 0)

		self.assertEqual( vectorFunctions.distance(self.p3,self.p7), math.sqrt(8))
		
	def test_shortestDistanceToSegment(self):
	
		d = vectorFunctions.shortestDistanceToSegment(self.p1,self.p3,self.p2)
		self.assertEqual(d,0)
				
		d = vectorFunctions.shortestDistanceToSegment(self.p1,self.p3,self.p5)
		self.assertEqual(d,2)
		d = vectorFunctions.shortestDistanceToSegment(self.p1,self.p3,self.p6)
		self.assertEqual(d,2)
		d = vectorFunctions.shortestDistanceToSegment(self.p1,self.p3,self.p7)
		self.assertEqual(d,2)
		
		d = vectorFunctions.shortestDistanceToSegment(self.p1,self.p2,self.p5)
		self.assertTrue(d>2)

	def test_pDiff(self):
		rp = vectorFunctions.pDiff(self.p4, self.p6)
		self.assertEqual(rp[0], -1)
		self.assertEqual(rp[1], 1)
	
	def test_angle(self):
		self.assertAlmostEqual( vectorFunctions.angle(self.p3, self.p3), 0 )
		self.assertAlmostEqual( vectorFunctions.angle(self.p5, self.p5), 0 )
		self.assertAlmostEqual( vectorFunctions.angle(self.p7, self.p7), 0 )

		self.assertAlmostEqual( vectorFunctions.angle([2,2], [-2,-2]), math.pi )
		self.assertAlmostEqual( vectorFunctions.angle([2,2], [-2,2]), math.pi/2 )

	def test_douglasPeucker(self):
		res = vectorFunctions.DouglasPeucker([self.p1, self.p2, self.p3, self.p4, self.p5, self.p6, self.p7, self.p8], 0.1)
		self.assertEqual( len(res), 5)
		
	def test_tangetialAngle(self):
		alpha = vectorFunctions.getTangentialAngle([0,0],[0,1],[0,2])
		self.assertAlmostEqual( alpha, 0 )
		
		alpha = vectorFunctions.getTangentialAngle([1,1],[2,2],[3,3])
		self.assertAlmostEqual( alpha, -math.pi/4.0 )

		alpha = vectorFunctions.getTangentialAngle([-1,1],[-2,2],[-3,3])
		self.assertAlmostEqual( alpha, math.pi/4.0 )

		alpha = vectorFunctions.getTangentialAngle([0,0],[0,2],[2,2])
		self.assertAlmostEqual( alpha, -math.pi/4.0 )

		alpha = vectorFunctions.getTangentialAngle([0,0],[0,2],[-2,2])
		self.assertAlmostEqual( alpha, math.pi/4.0 )

	def test_isLeft(self):
		res = vectorFunctions.isLeft([1,1], [10,1], [5,2]);
		self.assertTrue(res)
		
		res = vectorFunctions.isLeft([1,1], [10,1], [5,-2]);
		self.assertFalse(res)
		
		res = vectorFunctions.isLeft([1,1], [1,10], [-2,-2]);
		self.assertTrue(res)

		res = vectorFunctions.isLeft([1,1], [1,10], [-2,-2]);
		self.assertTrue(res)

if __name__ == '__main__':
	unittest.main(verbosity=2)