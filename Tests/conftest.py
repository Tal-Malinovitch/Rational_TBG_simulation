"""
Pytest configuration and fixtures for TBG project tests.

This module provides common fixtures, test utilities, and configuration
for the entire test suite.
"""

import pytest
import numpy as np
import tempfile
import os
import constants
from graph import node, graph, periodic_graph, periodic_matrix
from TBG import hex_lattice, tbg


@pytest.fixture(scope="session")
def temp_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def simple_nodes():
    """Create simple test nodes."""
    nodes = {
        'origin': node((0.0, 0.0), (0, 0), sublattice_id=0),
        'unit_x': node((1.0, 0.0), (1, 0), sublattice_id=0),
        'unit_y': node((0.0, 1.0), (0, 1), sublattice_id=0),
        'other_sublattice': node((0.5, 0.5), (0, 0), sublattice_id=1),
    }
    return nodes


@pytest.fixture
def simple_graph(simple_nodes):
    """Create simple test graph with connected nodes."""
    g = graph()
    
    # Add all nodes
    for node_obj in simple_nodes.values():
        g.add_node(node_obj)
    
    # Add some edges
    g.add_edge(simple_nodes['origin'], simple_nodes['unit_x'])
    g.add_edge(simple_nodes['origin'], simple_nodes['unit_y'])
    g.add_edge(simple_nodes['origin'], simple_nodes['other_sublattice'])
    
    yield g
    g.cleanup()


@pytest.fixture
def square_lattice_vectors():
    """Standard square lattice vectors."""
    return [(1.0, 0.0), (0.0, 1.0)]


@pytest.fixture
def hex_lattice_vectors():
    """Standard hexagonal lattice vectors."""
    return [(np.sqrt(3)/2, 0.5), (np.sqrt(3)/2, -0.5)]


@pytest.fixture
def periodic_graph_simple(simple_graph, square_lattice_vectors):
    """Create simple periodic graph for testing."""
    k_point = (0.0, 0.0)
    pg = periodic_graph(simple_graph, square_lattice_vectors, k_point)
    yield pg
    pg.cleanup()


@pytest.fixture
def matrix_handler_simple(periodic_graph_simple):
    """Create simple matrix handler for testing."""
    mh = periodic_matrix(periodic_graph_simple)
    yield mh
    mh.cleanup()


@pytest.fixture
def small_tbg_system():
    """Create small TBG system for integration tests."""
    system = tbg(
        maxsize_n=2, maxsize_m=2,
        a=5, b=1,
        interlayer_dist_threshold=2.0,
        intralayer_dist_threshold=2.0,
        unit_cell_radius_factor=1
    )
    yield system
    system.cleanup()


@pytest.fixture
def test_hermitian_matrix():
    """Create a known Hermitian matrix for testing."""
    # 2x2 Hermitian matrix with known eigenvalues [1, 3]
    matrix_data = np.array([
        [2.0, 1.0],
        [1.0, 2.0]
    ], dtype=complex)
    return constants.csr_matrix(matrix_data)


@pytest.fixture
def random_hermitian_matrix():
    """Create random Hermitian matrix for testing."""
    size = 4
    np.random.seed(42)  # For reproducibility
    random_matrix = np.random.randn(size, size) + 1j * np.random.randn(size, size)
    hermitian_matrix = random_matrix + random_matrix.conj().T
    return constants.csr_matrix(hermitian_matrix)


# Test markers for categorization
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests for individual components")
    config.addinivalue_line("markers", "integration: Integration tests for workflows")
    config.addinivalue_line("markers", "performance: Performance and timing tests")
    config.addinivalue_line("markers", "physics: Physics calculation validation tests")
    config.addinivalue_line("markers", "numerical: Numerical stability tests")
    config.addinivalue_line("markers", "slow: Tests that take longer to run")


# Custom test utilities
class TestUtils:
    """Utility functions for tests."""
    
    @staticmethod
    def assert_matrix_hermitian(matrix, rtol=1e-12):
        """Assert that a matrix is Hermitian within tolerance."""
        from utils import is_hermitian_sparse
        assert is_hermitian_sparse(matrix, rtol=rtol), "Matrix is not Hermitian"
    
    @staticmethod
    def assert_eigenvalues_real(eigenvalues, atol=1e-12):
        """Assert that eigenvalues are real within tolerance."""
        assert np.allclose(np.imag(eigenvalues), 0, atol=atol), "Eigenvalues are not real"
    
    @staticmethod
    def assert_eigenvalues_sorted(eigenvalues):
        """Assert that eigenvalues are sorted in ascending order."""
        assert np.all(eigenvalues[:-1] <= eigenvalues[1:]), "Eigenvalues are not sorted"
    
    @staticmethod
    def assert_vectors_orthogonal(vec1, vec2, atol=1e-12):
        """Assert that two vectors are orthogonal."""
        dot_product = np.abs(np.vdot(vec1, vec2))
        assert dot_product < atol, f"Vectors are not orthogonal: dot product = {dot_product}"
    
    @staticmethod
    def assert_vector_normalized(vector, atol=1e-12):
        """Assert that a vector is normalized."""
        norm = np.linalg.norm(vector)
        assert abs(norm - 1.0) < atol, f"Vector is not normalized: norm = {norm}"


@pytest.fixture
def test_utils():
    """Provide test utility functions."""
    return TestUtils


# Performance testing utilities
@pytest.fixture
def performance_timer():
    """Timer for performance tests."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
            return self.elapsed_time()
        
        def elapsed_time(self):
            if self.start_time is None or self.end_time is None:
                return 0.0
            return self.end_time - self.start_time
    
    return Timer()


# Skip markers for optional dependencies
def pytest_runtest_setup(item):
    """Setup function to handle conditional skipping."""
    # Skip tests that require optional dependencies
    if 'requires_plotting' in item.keywords:
        try:
            import matplotlib
        except ImportError:
            pytest.skip("matplotlib not available")
    
    if 'requires_gui' in item.keywords:
        try:
            import PyQt6
        except ImportError:
            pytest.skip("PyQt6 not available")


# Cleanup after tests
def pytest_runtest_teardown(item, nextitem):
    """Cleanup after each test."""
    # Force garbage collection to prevent memory issues
    import gc
    gc.collect()