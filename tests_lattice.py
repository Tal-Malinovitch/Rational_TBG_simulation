import unittest
import numpy as np
from .. import Lattice 

class TestNode(unittest.TestCase):
    """Test Lattice.Node class functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.node1 = Lattice.Node((0.0, 0.0), (0, 0), sublattice_id=0)
        self.node2 = Lattice.Node((1.0, 0.0), (1, 0), sublattice_id=0)
    
    def test_node_creation(self):
        """Test node creation with valid parameters."""
        self.assertEqual(self.node1.position, (0.0, 0.0))
        self.assertEqual(self.node1.lattice_index, (0, 0))
        self.assertEqual(self.node1.sublattice_id, 0)
        self.assertEqual(len(self.node1.neighbors), 0)
    
    def test_add_neighbor(self):
        """Test adding neighbors to nodes."""
        self.node1.add_neighbor(self.node2, (0, 0))
        self.assertEqual(len(self.node1.neighbors), 1)
        self.assertEqual(self.node1.neighbors[0][0], self.node2)
        self.assertEqual(self.node1.neighbors[0][1], (0, 0))
    
    def test_node_copy(self):
        """Test node copying functionality."""
        node_copy = self.node1.copy(sublattice_id=1)
        self.assertEqual(node_copy.position, self.node1.position)
        self.assertEqual(node_copy.lattice_index, self.node1.lattice_index)
        self.assertEqual(node_copy.sublattice_id, 1)  # Changed
        self.assertNotEqual(node_copy.sublattice_id, self.node1.sublattice_id)

class TestTwistConstants(unittest.TestCase):
    """Test twist angle calculations."""
    
    def test_valid_parameters(self):
        """Test computation with valid a, b parameters."""
        N, alpha, factor, k_point = Lattice.compute_twist_constants(5, 1)
        
        # Check that N is computed correctly: sqrt(a^2 + 3*b^2) / alpha
        expected_N_raw = np.sqrt(5**2 + 3*1**2)  # sqrt(28)
        self.assertAlmostEqual(N * alpha, expected_N_raw, places=10)
        
        # Check that alpha includes the factor of 2 for odd a*b
        self.assertEqual(alpha, 2)  # Since 5*1 = 5 is odd
    
    def test_a_divisible_by_3(self):
        """Test special case when a is divisible by 3."""
        N, alpha, factor, k_point = Lattice.compute_twist_constants(6, 1)
        
        # Should include factor of 4π when a % 3 == 0
        self.assertAlmostEqual(alpha, 2 * 4 * np.pi, places=10)
        
    def test_parameter_validation(self):
        """Test that invalid parameters raise appropriate errors."""
        with self.assertRaises(ValueError):
            Lattice.validate_ab(0, 1)  # a must be positive
        
        with self.assertRaises(ValueError):
            Lattice.validate_ab(2, 0)  # b cannot be zero
        
        with self.assertRaises(ValueError):
            Lattice.validate_ab(4, 6)  # not coprime (gcd = 2)
        
        with self.assertRaises(ValueError):
            Lattice.validate_ab(3, 5)  # |b| > a

class TestTBGConstruction(unittest.TestCase):
    """Test TBG system construction."""
    
    def setUp(self):
        """Create a small TBG system for testing."""
        self.tbg = Lattice.TBG(maxsize_n=2, maxsize_m=2, a=5, b=1, 
                      interlayer_dist_threshold=2.0, unit_cell_radius_factor=1)
    
    def test_tbg_creation(self):
        """Test that TBG system is created successfully."""
        self.assertEqual(self.tbg.a, 5)
        self.assertEqual(self.tbg.b, 1)
        self.assertIsNotNone(self.tbg.full_graph)
        self.assertGreater(len(self.tbg.full_graph.nodes), 0)
    
    def test_bilayer_structure(self):
        """Test that both layers are present with correct sublattice IDs."""
        sublattice_ids = {node.sublattice_id for node in self.tbg.full_graph.nodes}
        self.assertEqual(sublattice_ids, {0, 1})  # Both layers present
    
    def test_interlayer_connections(self):
        """Test that interlayer connections exist."""
        has_interlayer_edge = False
        for node in self.tbg.full_graph.nodes:
            for neighbor, _ in node.neighbors:
                if neighbor.sublattice_id != node.sublattice_id:
                    has_interlayer_edge = True
                    break
            if has_interlayer_edge:
                break
        self.assertTrue(has_interlayer_edge, "No interlayer connections found")
    
    def test_periodic_copy(self):
        """Test creation of periodic boundary conditions."""
        periodic_graph = self.tbg.full_graph.create_periodic_copy(
            self.tbg.lattice_vectors, (1/3, 1/3)
        )
        self.assertIsNotNone(periodic_graph)
        self.assertGreater(len(periodic_graph.nodes), 0)

class TestNumericalStability(unittest.TestCase):
    """Test numerical stability and edge cases."""
    
    def test_small_system_eigenvalues(self):
        """Test eigenvalue computation on a small, well-conditioned system."""
        tbg = Lattice.TBG(maxsize_n=1, maxsize_m=1, a=3, b=1, 
                 interlayer_dist_threshold=1.5, unit_cell_radius_factor=1)
        
        periodic_graph = tbg.full_graph.create_periodic_copy(
            tbg.lattice_vectors, (0, 0)  # Gamma point
        )
        
        try:
            eigenvals = periodic_graph.Lattice.compute_bands_at_k(
                Momentum=np.array([0.0, 0.0]), min_bands=1, max_bands=3,
                inter_graph_weight=1.0, intra_graph_weight=1.0
            )
            
            # Check that we got the right number of eigenvalues
            self.assertEqual(len(eigenvals), 3)
            
            # Check that eigenvalues are real (imaginary part should be ~0)
            self.assertTrue(np.allclose(np.imag(eigenvals), 0, atol=1e-10))
            
        except Exception as e:
            self.fail(f"Eigenvalue computation failed: {e}")
