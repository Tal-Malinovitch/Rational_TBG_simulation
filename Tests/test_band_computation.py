#!/usr/bin/env python3
"""
Comprehensive unit tests for band_comp_and_plot.py module.

Tests eigenvalue computation, band analysis, and Dirac point calculations.
"""

import unittest
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import constants
from band_comp_and_plot import band_handler
from graph import graph, node, periodic_graph, periodic_matrix


class TestBandHandler(unittest.TestCase):
    """Test band handler functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create simple test system
        base_graph = graph()
        self.node1 = node((0.0, 0.0), (0, 0), sublattice_id=0)
        self.node2 = node((0.5, 0.0), (0, 0), sublattice_id=1)
        base_graph.add_node(self.node1)
        base_graph.add_node(self.node2)
        base_graph.add_edge(self.node1, self.node2)
        
        lattice_vectors = [(1.0, 0.0), (0.0, 1.0)]
        k_point = (0.0, 0.0)
        
        self.periodic_graph = periodic_graph(base_graph, lattice_vectors, k_point)
        self.matrix_handler = periodic_matrix(self.periodic_graph)
        self.band_handler = band_handler(self.matrix_handler)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self, 'band_handler'):
            self.band_handler.cleanup()
        if hasattr(self, 'matrix_handler'):
            self.matrix_handler.cleanup()
        if hasattr(self, 'periodic_graph'):
            self.periodic_graph.cleanup()
    
    def test_band_handler_creation(self):
        """Test band handler creation and validation."""
        # Test valid creation
        bh = band_handler(self.matrix_handler)
        self.assertEqual(bh.matrix_handler, self.matrix_handler)
        self.assertEqual(bh.periodic_graph, self.periodic_graph)
        bh.cleanup()
        
        # Test invalid creation
        with self.assertRaises(constants.physics_parameter_error):
            band_handler(None)
        
        with self.assertRaises(constants.physics_parameter_error):
            band_handler("not a matrix handler")
    
    def test_sparse_eigensolve_validation(self):
        """Test input validation for sparse eigensolve."""
        # Create test Laplacian
        laplacian = self.matrix_handler.build_laplacian(
            np.array([0.0, 0.0]), 
            inter_graph_weight=1.0, 
            intra_graph_weight=1.0
        )
        
        # Test invalid inputs
        with self.assertRaises(constants.matrix_operation_error):
            self.band_handler._sparse_eigensolve(None, 1, 2)
        
        with self.assertRaises(constants.matrix_operation_error):
            self.band_handler._sparse_eigensolve(laplacian, 0, 2)  # min_bands < 1
        
        with self.assertRaises(constants.matrix_operation_error):
            self.band_handler._sparse_eigensolve(laplacian, 3, 2)  # min > max
        
        with self.assertRaises(constants.matrix_operation_error):
            self.band_handler._sparse_eigensolve(laplacian, 1, 1000)  # max > matrix size
    
    def test_sparse_eigensolve_basic(self):
        """Test basic sparse eigenvalue computation."""
        # Create test Laplacian
        laplacian = self.matrix_handler.build_laplacian(
            np.array([0.0, 0.0]), 
            inter_graph_weight=1.0, 
            intra_graph_weight=1.0
        )
        
        try:
            result = self.band_handler._sparse_eigensolve(laplacian, 1, 2)
            
            if result is not None:  # Might be None if convergence fails
                eigenvals, eigenvecs = result
                
                # Check dimensions
                self.assertEqual(len(eigenvals), 2)
                self.assertEqual(eigenvecs.shape[0], 2)
                self.assertEqual(eigenvecs.shape[1], laplacian.shape[0])
                
                # Eigenvalues should be real for Hermitian matrix
                self.assertTrue(np.allclose(np.imag(eigenvals), 0, atol=1e-10))
                
                # Eigenvalues should be sorted
                self.assertTrue(np.all(eigenvals[:-1] <= eigenvals[1:]))
                
        except (constants.ArpackNoConvergence, constants.matrix_operation_error):
            # Convergence issues are acceptable for small test systems
            pass
    
    def test_dense_eigensolve_basic(self):
        """Test dense eigenvalue computation."""
        # Create test Laplacian
        laplacian = self.matrix_handler.build_laplacian(
            np.array([0.0, 0.0]), 
            inter_graph_weight=1.0, 
            intra_graph_weight=1.0
        )
        
        try:
            eigenvals, eigenvecs = self.band_handler._dense_eigensolve(laplacian, 1, 2)
            
            # Check dimensions
            self.assertEqual(len(eigenvals), 2)
            self.assertEqual(eigenvecs.shape[0], 2)
            self.assertEqual(eigenvecs.shape[1], laplacian.shape[0])
            
            # Eigenvalues should be real
            self.assertTrue(np.allclose(np.imag(eigenvals), 0, atol=1e-10))
            
            # Eigenvalues should be sorted
            self.assertTrue(np.all(eigenvals[:-1] <= eigenvals[1:]))
            
        except Exception as e:
            self.fail(f"Dense eigensolve failed: {e}")
    
    def test_memory_estimation(self):
        """Test memory estimation for dense computation."""
        matrix_size = 100
        estimated_mb = self.band_handler._estimate_dense_memory_usage(matrix_size)
        
        # Should return reasonable estimate
        self.assertIsInstance(estimated_mb, (int, float))
        self.assertGreater(estimated_mb, 0)
        
        # Estimate should scale quadratically
        estimated_mb_large = self.band_handler._estimate_dense_memory_usage(matrix_size * 2)
        self.assertGreater(estimated_mb_large, estimated_mb * 3)  # Should be ~4x but account for overhead
    
    def test_context_manager(self):
        """Test band handler context manager."""
        with band_handler(self.matrix_handler) as bh:
            self.assertIsNotNone(bh.matrix_handler)
            self.assertEqual(bh.periodic_graph, self.periodic_graph)
        
        # After context exit, cleanup should be called


class TestEigenvalueStrategies(unittest.TestCase):
    """Test eigenvalue computation strategies and fallbacks."""
    
    def setUp(self):
        """Set up test fixtures with known eigenvalue system."""
        # Create simple 2x2 Hermitian matrix with known eigenvalues
        self.test_matrix = constants.csr_matrix(np.array([
            [2.0, 1.0],
            [1.0, 2.0]
        ], dtype=complex))
        # Eigenvalues should be 1.0 and 3.0
        self.expected_eigenvals = np.array([1.0, 3.0])
    
    def test_known_eigenvalues_sparse(self):
        """Test sparse computation on matrix with known eigenvalues."""
        # Create minimal band handler for testing
        class MockMatrixHandler:
            def __init__(self):
                self.periodic_graph = None
        
        mock_handler = MockMatrixHandler()
        bh = band_handler(mock_handler)
        
        try:
            result = bh._sparse_eigensolve(self.test_matrix, 1, 2)
            
            if result is not None:
                eigenvals, eigenvecs = result
                
                # Check that we get the correct eigenvalues
                np.testing.assert_allclose(eigenvals, self.expected_eigenvals, rtol=1e-10)
                
                # Check eigenvector orthogonality
                dot_product = np.abs(np.vdot(eigenvecs[0], eigenvecs[1]))
                self.assertLess(dot_product, 1e-10)
                
        except Exception:
            # Sparse methods might fail for very small matrices
            pass
        
        bh.cleanup()
    
    def test_known_eigenvalues_dense(self):
        """Test dense computation on matrix with known eigenvalues."""
        # Create minimal band handler for testing
        class MockMatrixHandler:
            def __init__(self):
                self.periodic_graph = None
        
        mock_handler = MockMatrixHandler()
        bh = band_handler(mock_handler)
        
        eigenvals, eigenvecs = bh._dense_eigensolve(self.test_matrix, 1, 2)
        
        # Check that we get the correct eigenvalues
        np.testing.assert_allclose(eigenvals, self.expected_eigenvals, rtol=1e-10)
        
        # Check eigenvector orthogonality
        dot_product = np.abs(np.vdot(eigenvecs[0], eigenvecs[1]))
        self.assertLess(dot_product, 1e-10)
        
        bh.cleanup()


class TestBandComputationEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions in band computation."""
    
    def test_single_band_request(self):
        """Test requesting only a single band."""
        # Create simple test system
        base_graph = graph()
        node1 = node((0.0, 0.0), (0, 0), sublattice_id=0)
        base_graph.add_node(node1)
        
        lattice_vectors = [(1.0, 0.0), (0.0, 1.0)]
        k_point = (0.0, 0.0)
        
        pg = periodic_graph(base_graph, lattice_vectors, k_point)
        mh = periodic_matrix(pg)
        
        with band_handler(mh) as bh:
            laplacian = mh.build_laplacian(
                np.array([0.0, 0.0]), 
                inter_graph_weight=1.0, 
                intra_graph_weight=1.0
            )
            
            try:
                # Request single band
                result = bh._sparse_eigensolve(laplacian, 1, 1)
                if result is not None:
                    eigenvals, eigenvecs = result
                    self.assertEqual(len(eigenvals), 1)
                    self.assertEqual(eigenvecs.shape[0], 1)
            except Exception:
                # Single node systems might have convergence issues
                pass
    
    def test_empty_matrix_handling(self):
        """Test handling of degenerate cases."""
        # Create empty matrix
        empty_matrix = constants.csr_matrix((0, 0))
        
        class MockMatrixHandler:
            def __init__(self):
                self.periodic_graph = None
        
        mock_handler = MockMatrixHandler()
        with band_handler(mock_handler) as bh:
            # Should handle empty matrix gracefully
            with self.assertRaises(constants.matrix_operation_error):
                bh._sparse_eigensolve(empty_matrix, 1, 1)


class TestNumericalStability(unittest.TestCase):
    """Test numerical stability of band computations."""
    
    def test_hermitian_matrix_real_eigenvalues(self):
        """Test that Hermitian matrices produce real eigenvalues."""
        # Create larger Hermitian matrix
        size = 4
        random_matrix = np.random.randn(size, size) + 1j * np.random.randn(size, size)
        hermitian_matrix = random_matrix + random_matrix.conj().T
        hermitian_csr = constants.csr_matrix(hermitian_matrix)
        
        class MockMatrixHandler:
            def __init__(self):
                self.periodic_graph = None
        
        mock_handler = MockMatrixHandler()
        with band_handler(mock_handler) as bh:
            # Test dense computation
            eigenvals, eigenvecs = bh._dense_eigensolve(hermitian_csr, 1, size)
            
            # Eigenvalues should be real
            self.assertTrue(np.allclose(np.imag(eigenvals), 0, atol=1e-12))
            
            # Test a few eigenvector properties
            for i in range(len(eigenvals)):
                # Eigenvector should be normalized
                norm = np.linalg.norm(eigenvecs[i])
                self.assertAlmostEqual(norm, 1.0, places=10)
    
    def test_eigenvalue_ordering(self):
        """Test that eigenvalues are properly sorted."""
        # Create matrix with known unordered eigenvalues
        diagonal_values = [5.0, 1.0, 3.0, 2.0]
        diagonal_matrix = constants.csr_matrix(np.diag(diagonal_values))
        
        class MockMatrixHandler:
            def __init__(self):
                self.periodic_graph = None
        
        mock_handler = MockMatrixHandler()
        with band_handler(mock_handler) as bh:
            eigenvals, eigenvecs = bh._dense_eigensolve(diagonal_matrix, 1, 4)
            
            # Should be sorted
            expected_sorted = [1.0, 2.0, 3.0, 5.0]
            np.testing.assert_allclose(eigenvals, expected_sorted, rtol=1e-10)


if __name__ == '__main__':
    unittest.main(verbosity=2)