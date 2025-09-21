# test_runner.py
import unittest
import sys
import os

if __name__ == '__main__':
    # Add current directory to path
    sys.path.insert(0, os.path.dirname(__file__))
    
    # Discover and run tests
    loader = unittest.TestLoader()
    suite = loader.discover('.')
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)