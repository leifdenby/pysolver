import unittest

def RootNotFoundError(Exception):
	pass

class NewtonRaphson:
	"""
		General purpose Newton-Raphson solver
	"""
	def solve(self, epsilon, initialGuess, f, dfdx):
		"""
			Calculates the root nearest to the initial guess
			Pass in required accuracy through epsilon, initial guess, and lambda functions for f(x) and f'(x)
		"""
		
		x = initialGuess
		while True:
			x_new = x - f(x)/dfdx(x)
			if abs(x - x_new) / abs(x+x_new)*2.0 < epsilon:
				x = x_new
				break
			x = x_new


		if abs(f(x)) < epsilon:
			return x
        	else:
			print f(x)
			raise RootNotFoundException()
    


class NewtonRaphsonTest(unittest.TestCase):
	class PolynomialNewtonRaphson(NewtonRaphson):
		def solve(self, epsilon, initialGuess):
			return NewtonRaphson().solve(epsilon, initialGuess, lambda x: x**2-4, lambda x: 2*x)

	def testRootLeft(self):
		self.assertEqual(self.PolynomialNewtonRaphson().solve(1e-10,-10.0), -2.0)
	def testRootRight(self):
		self.assertEqual(self.PolynomialNewtonRaphson().solve(1e-10,10.0), 2.0)





if __name__ == "__main__":
	unittest.main()


