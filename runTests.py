from unittest import TestLoader, TextTestRunner, TestSuite
import vectorFunctionsTest
import skeletonFunctionsTest

if __name__ == "__main__":

    loader = TestLoader()
    suite = TestSuite((
        loader.loadTestsFromTestCase(vectorFunctionsTest.TestVectorFunctions),
        loader.loadTestsFromTestCase(skeletonFunctionsTest.TestSkeletonFunctions),
        ))

    runner = TextTestRunner(verbosity = 2)
    runner.run(suite)