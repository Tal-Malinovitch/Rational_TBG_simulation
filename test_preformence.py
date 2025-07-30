# tests/test_performance.py
import unittest
import time
import numpy as np
import os
from Lattice import TBG

class TestPerformance(unittest.TestCase):
    """Test performance characteristics."""
    
    def test_construction_time(self):
        """Test that TBG construction completes in reasonable time."""
        start_time = time.time()
        
        tbg = TBG(maxsize_n=5, maxsize_m=5, a=5, b=1,
                 interlayer_dist_threshold=1.0, unit_cell_radius_factor=2)
        
        construction_time = time.time() - start_time
        
        # Should complete within 10 seconds for this size
        self.assertLess(construction_time, 10.0, 
                       f"Construction took {construction_time:.2f}s, too slow!")
    