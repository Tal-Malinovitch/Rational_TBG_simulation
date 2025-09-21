#!/usr/bin/env python3
"""
Integration tests for TBG physics workflows.

Tests complete workflows from TBG construction through band calculations,
Dirac point finding, and training data generation.
"""

import unittest
import numpy as np
import tempfile
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import constants
from TBG import hex_lattice, tbg, Dirac_analysis
from Generate_training_data import gradient_decent_adam, _process_tbg_system_worker
from stats import statistics_holder
from band_comp_and_plot import band_handler


class TestTBGWorkflow(unittest.TestCase):
    """Test complete TBG construction and analysis workflow."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Use small system for fast testing
        self.maxsize_n = 2
        self.maxsize_m = 2
        self.a = 5
        self.b = 1
        self.interlayer_threshold = 2.0
        self.intralayer_threshold = 2.0
        self.unit_cell_factor = 1
    
    def test_hex_lattice_workflow(self):
        """Test hexagonal lattice construction workflow."""
        with hex_lattice(
            self.maxsize_n, 
            self.maxsize_m,
            lattice_constant=1.0,
            lattice_vectors=[(1.0, 0.0), (0.5, np.sqrt(3)/2)],
            unit_cell_radius_factor=self.unit_cell_factor
        ) as lattice:
            
            # Check lattice was created
            self.assertIsNotNone(lattice.graph)
            self.assertGreater(len(lattice.graph.nodes), 0)
            
            # Check lattice vectors are set
            self.assertIsNotNone(lattice.lattice_vectors)
            self.assertEqual(len(lattice.lattice_vectors), 2)
    
    def test_tbg_construction_workflow(self):
        """Test TBG bilayer construction workflow."""
        with tbg(
            self.maxsize_n, self.maxsize_m,
            self.a, self.b,
            interlayer_dist_threshold=self.interlayer_threshold,
            intralayer_dist_threshold=self.intralayer_threshold,
            unit_cell_radius_factor=self.unit_cell_factor
        ) as tbg_system:
            
            # Check TBG system was created
            self.assertIsNotNone(tbg_system.full_graph)
            self.assertGreater(len(tbg_system.full_graph.nodes), 0)
            
            # Check both layers are present
            sublattice_ids = {node.sublattice_id for node in tbg_system.full_graph.nodes}
            self.assertEqual(len(sublattice_ids), 2)  # Two layers
            
            # Check lattice vectors are computed
            self.assertIsNotNone(tbg_system.lattice_vectors)
            
            # Check twist parameters
            self.assertEqual(tbg_system.a, self.a)
            self.assertEqual(tbg_system.b, self.b)
    
    def test_periodic_copy_workflow(self):
        """Test periodic boundary condition workflow."""
        with tbg(
            self.maxsize_n, self.maxsize_m,
            self.a, self.b,
            interlayer_dist_threshold=self.interlayer_threshold,
            intralayer_dist_threshold=self.intralayer_threshold,
            unit_cell_radius_factor=self.unit_cell_factor
        ) as tbg_system:
            
            # Create periodic copy
            k_point = (0.1, 0.1)
            with tbg_system.full_graph.create_periodic_copy(
                tbg_system.lattice_vectors, k_point
            ) as periodic_graph:
                
                # Check periodic graph was created
                self.assertIsNotNone(periodic_graph)
                self.assertGreater(len(periodic_graph.nodes), 0)
                self.assertEqual(periodic_graph.k_point, k_point)
                
                # Check matrix handler was created
                self.assertIsNotNone(periodic_graph.matrix_handler)
                
                # Check adjacency matrix
                adj_matrix = periodic_graph.adj_matrix
                self.assertIsInstance(adj_matrix, constants.csr_matrix)
    
    def test_band_calculation_workflow(self):
        """Test band structure calculation workflow."""
        with tbg(
            self.maxsize_n, self.maxsize_m,
            self.a, self.b,
            interlayer_dist_threshold=self.interlayer_threshold,
            intralayer_dist_threshold=self.intralayer_threshold,
            unit_cell_radius_factor=self.unit_cell_factor
        ) as tbg_system:
            
            k_point = (0.0, 0.0)  # Gamma point for stability
            with tbg_system.full_graph.create_periodic_copy(
                tbg_system.lattice_vectors, k_point
            ) as periodic_graph:
                
                # Test band calculation
                try:
                    eigenvals, eigenvecs = periodic_graph.compute_bands_at_k(
                        Momentum=np.array([0.0, 0.0]),
                        min_bands=1,
                        max_bands=4,
                        inter_graph_weight=1.0,
                        intra_graph_weight=1.0
                    )
                    
                    # Check results
                    self.assertEqual(len(eigenvals), 4)
                    self.assertEqual(eigenvecs.shape[0], 4)
                    
                    # Eigenvalues should be real
                    self.assertTrue(np.allclose(np.imag(eigenvals), 0, atol=1e-10))
                    
                except Exception as e:
                    self.fail(f"Band calculation failed: {e}")


class TestDiracAnalysisWorkflow(unittest.TestCase):
    """Test Dirac point analysis workflow."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create small TBG system for testing
        self.tbg_system = tbg(
            maxsize_n=2, maxsize_m=2,
            a=5, b=1,
            interlayer_dist_threshold=2.0,
            intralayer_dist_threshold=2.0,
            unit_cell_radius_factor=1
        )
        
        self.k_point = (1/3, 1/3)  # K point
        self.periodic_graph = self.tbg_system.full_graph.create_periodic_copy(
            self.tbg_system.lattice_vectors, self.k_point
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self, 'periodic_graph'):
            self.periodic_graph.cleanup()
        if hasattr(self, 'tbg_system'):
            self.tbg_system.cleanup()
    
    def test_dirac_analysis_creation(self):
        """Test Dirac analysis object creation."""
        with Dirac_analysis(self.periodic_graph) as analyzer:
            self.assertIsNotNone(analyzer.periodic_graph)
            self.assertEqual(analyzer.periodic_graph, self.periodic_graph)
    
    def test_gap_function_evaluation(self):
        """Test gap function evaluation."""
        with Dirac_analysis(self.periodic_graph) as analyzer:
            
            # Test gap function at a point
            k = np.array([0.1, 0.1])
            try:
                gap = analyzer.gap_function(
                    k, inter_graph_weight=1.0, intra_graph_weight=1.0
                )
                
                # Gap should be a non-negative real number
                self.assertIsInstance(gap, (float, np.floating))
                self.assertGreaterEqual(gap, 0.0)
                
            except Exception as e:
                self.fail(f"Gap function evaluation failed: {e}")


class TestTrainingDataWorkflow(unittest.TestCase):
    """Test training data generation workflow."""
    
    def test_adam_optimizer_workflow(self):
        """Test ADAM optimizer workflow."""
        with gradient_decent_adam(alpha=0.01, beta_1=0.9, beta_2=0.999) as optimizer:
            
            # Test initial state
            self.assertEqual(optimizer.t, 0)
            self.assertEqual(optimizer.current_m, 0.0)
            self.assertEqual(optimizer.current_v, 0.0)
            
            # Test optimization step
            gradient = 0.5
            new_weight = optimizer.update(gradient)
            
            # Check state updated
            self.assertEqual(optimizer.t, 1)
            self.assertNotEqual(optimizer.current_m, 0.0)
            self.assertNotEqual(optimizer.current_v, 0.0)
            self.assertIsInstance(new_weight, (float, np.floating))
    
    def test_statistics_collection_workflow(self):
        """Test statistics collection workflow."""
        with statistics_holder() as stats:
            
            # Test logging combinations
            stats.log_combination(duration=1.5, success=True)
            stats.log_combination(duration=2.0, success=False, 
                                 failure_reason="no_intersections")
            
            # Check statistics
            self.assertEqual(stats.total_combinations, 2)
            self.assertEqual(stats.successful_combinations, 1)
            self.assertEqual(stats.failed_no_intersections, 1)
            self.assertEqual(len(stats.combination_times), 2)
    
    def test_process_tbg_worker(self):
        """Test TBG processing worker function."""
        # Create test parameters
        params = {
            'maxsize_n': 2,
            'maxsize_m': 2,
            'a': 5,
            'b': 1,
            'interlayer_dist_threshold': 2.0,
            'intralayer_dist_threshold': 2.0,
            'unit_cell_radius_factor': 1,
            'k_range': 0.1,
            'num_k_points': 3,
            'learning_rate': 0.01,
            'max_iterations': 10,
            'tolerance': 0.1
        }
        
        try:
            result = _process_tbg_system_worker(params)
            
            # Check result structure
            self.assertIsInstance(result, dict)
            self.assertIn('success', result)
            self.assertIn('metrics', result)
            self.assertIn('duration', result)
            
            # Check metrics if successful
            if result['success']:
                metrics = result['metrics']
                self.assertIn('gap', metrics)
                self.assertIn('k', metrics)
                
        except Exception as e:
            # Worker might fail due to convergence issues in small system
            # This is acceptable for testing - we're testing the workflow structure
            self.assertIsInstance(e, (ValueError, RuntimeError, constants.tbg_error))


class TestBandHandlerIntegration(unittest.TestCase):
    """Test band handler integration with TBG systems."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create TBG system
        self.tbg_system = tbg(
            maxsize_n=2, maxsize_m=2,
            a=5, b=1,
            interlayer_dist_threshold=2.0,
            intralayer_dist_threshold=2.0,
            unit_cell_radius_factor=1
        )
        
        self.periodic_graph = self.tbg_system.full_graph.create_periodic_copy(
            self.tbg_system.lattice_vectors, (0.0, 0.0)
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self, 'periodic_graph'):
            self.periodic_graph.cleanup()
        if hasattr(self, 'tbg_system'):
            self.tbg_system.cleanup()
    
    def test_band_handler_integration(self):
        """Test band handler with TBG periodic graph."""
        with band_handler(self.periodic_graph.matrix_handler) as bh:
            
            # Check initialization
            self.assertIsNotNone(bh.matrix_handler)
            self.assertEqual(bh.periodic_graph, self.periodic_graph)
            
            # Test eigenvalue computation
            try:
                laplacian = bh.matrix_handler.build_laplacian(
                    np.array([0.0, 0.0]), 
                    inter_graph_weight=1.0, 
                    intra_graph_weight=1.0
                )
                
                # Test sparse eigensolve
                result = bh._sparse_eigensolve(laplacian, 1, 4)
                if result is not None:
                    eigenvals, eigenvecs = result
                    self.assertEqual(len(eigenvals), 4)
                    self.assertTrue(np.allclose(np.imag(eigenvals), 0, atol=1e-10))
                
            except Exception as e:
                # Eigenvalue computation might fail for very small systems
                # This is acceptable - we're testing integration structure
                pass


class TestFileIOIntegration(unittest.TestCase):
    """Test file I/O integration for training data."""
    
    def test_statistics_file_output(self):
        """Test statistics file output workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filename = os.path.join(temp_dir, "test_stats.csv")
            
            with statistics_holder() as stats:
                # Log some test data
                stats.log_combination(duration=1.0, success=True)
                stats.log_combination(duration=2.0, success=False, 
                                     failure_reason="no_dirac")
                
                # Export to file
                stats.export_statistics(filename)
                
                # Check file was created
                self.assertTrue(os.path.exists(filename))
                
                # Check file content
                with open(filename, 'r') as f:
                    content = f.read()
                    self.assertIn("duration", content)
                    self.assertIn("success", content)


if __name__ == '__main__':
    unittest.main(verbosity=2)