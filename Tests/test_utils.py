#!/usr/bin/env python3
"""
Comprehensive unit tests for utils.py module.

Tests all utility functions including mathematical operations,
coordinate transformations, physics calculations, and error handling.
"""

import unittest
import numpy as np
import hashlib
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import constants
from utils import (
    calculate_distance, canonical_position, compute_twist_constants,
    validate_ab, is_hermitian_sparse, edge_color_hash
)
from graph import node


class TestCalculateDistance(unittest.TestCase):
    """Test distance calculation functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.node1 = node((0.0, 0.0), (0, 0))
        self.node2 = node((1.0, 1.0), (1, 1))
        self.node3 = node((3.0, 4.0), (3, 4))
    
    def test_distance_to_origin(self):
        """Test distance calculation from node to origin."""
        distance = calculate_distance(self.node3)
        expected = np.sqrt(3.0**2 + 4.0**2)  # 5.0
        self.assertAlmostEqual(distance, expected, places=10)
    
    def test_distance_between_nodes(self):
        """Test distance calculation between two nodes."""
        distance = calculate_distance(self.node1, self.node2)
        expected = np.sqrt((1.0-0.0)**2 + (1.0-0.0)**2)  # sqrt(2)
        self.assertAlmostEqual(distance, expected, places=10)
    
    def test_distance_with_offset(self):
        """Test distance calculation with coordinate offset."""
        offset = (1.0, 1.0)
        distance = calculate_distance(self.node2, offset=offset)
        expected = np.sqrt((1.0-1.0)**2 + (1.0-1.0)**2)  # 0.0
        self.assertAlmostEqual(distance, expected, places=10)
    
    def test_distance_invalid_node(self):
        """Test error handling for nodes without position attribute."""
        class BadNode:
            pass
        
        bad_node = BadNode()
        with self.assertRaises(constants.physics_parameter_error):
            calculate_distance(bad_node)


class TestCanonicalPosition(unittest.TestCase):
    """Test canonical position transformations."""
    
    def setUp(self):
        """Set up lattice vectors."""
        self.v1 = [1.0, 0.0]
        self.v2 = [0.0, 1.0]  # Square lattice
        self.hex_v1 = [np.sqrt(3)/2, 0.5]  # Hexagonal lattice
        self.hex_v2 = [np.sqrt(3)/2, -0.5]
    
    def test_square_lattice_origin(self):
        """Test canonical position for origin in square lattice."""
        pos = (0.0, 0.0)
        canonical_pos, shift = canonical_position(pos, self.v1, self.v2)
        
        self.assertAlmostEqual(canonical_pos[0], 0.0, places=10)
        self.assertAlmostEqual(canonical_pos[1], 0.0, places=10)
        self.assertEqual(shift, (0, 0))
    
    def test_square_lattice_wrapping(self):
        """Test position wrapping in square lattice."""
        pos = (1.5, 0.5)
        canonical_pos, shift = canonical_position(pos, self.v1, self.v2)
        
        # Should wrap to (-0.5, -0.5) due to centered unit cell wrapping
        self.assertAlmostEqual(canonical_pos[0], -0.5, places=10)
        self.assertAlmostEqual(canonical_pos[1], -0.5, places=10)
    
    def test_hexagonal_lattice(self):
        """Test canonical position in hexagonal lattice."""
        pos = (0.1, 0.1)
        canonical_pos, shift = canonical_position(pos, self.hex_v1, self.hex_v2)
        
        # Should stay within unit cell
        self.assertIsInstance(canonical_pos, tuple)
        self.assertIsInstance(shift, tuple)
        self.assertEqual(len(canonical_pos), 2)
        self.assertEqual(len(shift), 2)
    
    def test_invalid_lattice_vectors(self):
        """Test error handling for invalid lattice vectors."""
        # Linearly dependent vectors
        v1_bad = [1.0, 0.0]
        v2_bad = [2.0, 0.0]
        
        with self.assertRaises(constants.physics_parameter_error):
            canonical_position((0, 0), v1_bad, v2_bad)


class TestComputeTwistConstants(unittest.TestCase):
    """Test twist angle constant calculations."""
    
    def test_basic_case_5_1(self):
        """Test standard case (5,1)."""
        N, alpha, factor, k_point = compute_twist_constants(5, 1)
        
        # Check NFactor calculation: sqrt(a^2 + 3*b^2) / alpha
        expected_raw = np.sqrt(5**2 + 3*1**2)  # sqrt(28)
        self.assertAlmostEqual(N * alpha, expected_raw, places=10)
        
        # alpha should be 2 since (5*1) % 2 = 1 (odd)
        self.assertEqual(alpha, 2)
        
        # factor should be 1 since 5 % 3 != 0
        self.assertEqual(factor, 1)
        
        # k_point should be regular
        self.assertEqual(k_point, constants.K_POINT_REG)
    
    def test_a_divisible_by_3(self):
        """Test case when a is divisible by 3."""
        N, alpha, factor, k_point = compute_twist_constants(6, 1)
        
        # For (6,1): a%3==0 so alpha gets 4π factor
        # (6*1)%2==0 so alpha does NOT get *2 factor (even product)
        expected_alpha = 4 * np.pi  # Just 4π from a%3==0, no *2 since ab is even
        self.assertAlmostEqual(alpha, expected_alpha, places=10)
        
        # factor should be reciprocal constant
        self.assertEqual(factor, constants.reciprocal_constant)
        
        # k_point should be dual
        self.assertEqual(k_point, constants.K_POINT_DUAL)
    
    def test_even_product(self):
        """Test case when a*b is even."""
        N, alpha, factor, k_point = compute_twist_constants(4, 1)
        
        # alpha should NOT include factor of 2 since (4*1) % 2 = 0 (even)
        self.assertEqual(alpha, 1)
    
    def test_invalid_parameters(self):
        """Test error handling for invalid parameters."""
        with self.assertRaises(constants.physics_parameter_error):
            compute_twist_constants(-1, 1)  # negative a
        
        with self.assertRaises(constants.physics_parameter_error):
            compute_twist_constants(5, 0)  # b cannot be zero


class TestValidateAB(unittest.TestCase):
    """Test parameter validation for rational TBG."""
    
    def test_valid_parameters(self):
        """Test that valid parameters pass validation."""
        # Should not raise any exception
        validate_ab(5, 1)
        validate_ab(7, 2)
        validate_ab(13, 5)
    
    def test_invalid_a(self):
        """Test validation of parameter a."""
        with self.assertRaises(constants.physics_parameter_error):
            validate_ab(0, 1)  # a must be positive
        
        with self.assertRaises(constants.physics_parameter_error):
            validate_ab(-5, 1)  # a must be positive
    
    def test_invalid_b(self):
        """Test validation of parameter b."""
        with self.assertRaises(constants.physics_parameter_error):
            validate_ab(5, 0)  # b cannot be zero
    
    def test_non_coprime(self):
        """Test validation of coprimality."""
        with self.assertRaises(constants.physics_parameter_error):
            validate_ab(4, 6)  # gcd(4,6) = 2, not coprime
        
        with self.assertRaises(constants.physics_parameter_error):
            validate_ab(10, 15)  # gcd(10,15) = 5, not coprime
    
    def test_b_greater_than_a(self):
        """Test validation of |b| <= a constraint."""
        with self.assertRaises(constants.physics_parameter_error):
            validate_ab(3, 5)  # |b| > a
        
        with self.assertRaises(constants.physics_parameter_error):
            validate_ab(2, -3)  # |b| > a
    
    def test_non_integer_types(self):
        """Test validation of integer types."""
        with self.assertRaises(constants.physics_parameter_error):
            validate_ab(5.5, 1)  # a must be integer
        
        with self.assertRaises(constants.physics_parameter_error):
            validate_ab(5, 1.2)  # b must be integer


class TestIsHermitianSparse(unittest.TestCase):
    """Test sparse matrix hermiticity checking."""
    
    def test_hermitian_matrix(self):
        """Test recognition of Hermitian matrix."""
        # Create a simple Hermitian matrix
        data = np.array([1+0j, 2+1j, 2-1j, 3+0j])
        row = np.array([0, 0, 1, 1])
        col = np.array([0, 1, 0, 1])
        matrix = constants.csr_matrix((data, (row, col)), shape=(2, 2))
        
        self.assertTrue(is_hermitian_sparse(matrix))
    
    def test_non_hermitian_matrix(self):
        """Test recognition of non-Hermitian matrix."""
        # Create a non-Hermitian matrix
        data = np.array([1+0j, 2+1j, 3+1j, 4+0j])  # (1,0) != conj((0,1))
        row = np.array([0, 0, 1, 1])
        col = np.array([0, 1, 0, 1])
        matrix = constants.csr_matrix((data, (row, col)), shape=(2, 2))
        
        self.assertFalse(is_hermitian_sparse(matrix))
    
    def test_non_square_matrix(self):
        """Test that non-square matrices return False."""
        data = np.array([1, 2, 3])
        row = np.array([0, 0, 1])
        col = np.array([0, 1, 0])
        matrix = constants.csr_matrix((data, (row, col)), shape=(2, 3))
        
        self.assertFalse(is_hermitian_sparse(matrix))
    
    def test_invalid_input(self):
        """Test error handling for invalid input."""
        with self.assertRaises(constants.matrix_operation_error):
            is_hermitian_sparse("not a matrix")


class TestEdgeColorHash(unittest.TestCase):
    """Test edge color hash generation."""
    
    def test_basic_color_generation(self):
        """Test basic color generation from key."""
        key = (1, 2, 3, 4, (5, 6))
        r, g, b = edge_color_hash(key)
        
        # Colors should be in range [0, 1]
        self.assertGreaterEqual(r, 0.0)
        self.assertLessEqual(r, 1.0)
        self.assertGreaterEqual(g, 0.0)
        self.assertLessEqual(g, 1.0)
        self.assertGreaterEqual(b, 0.0)
        self.assertLessEqual(b, 1.0)
    
    def test_reproducible_colors(self):
        """Test that same key produces same color."""
        key = (1, 2, 3, 4, (5, 6))
        color1 = edge_color_hash(key)
        color2 = edge_color_hash(key)
        
        self.assertEqual(color1, color2)
    
    def test_different_keys_different_colors(self):
        """Test that different keys produce different colors."""
        key1 = (1, 2, 3, 4, (5, 6))
        key2 = (1, 2, 3, 4, (7, 8))
        
        color1 = edge_color_hash(key1)
        color2 = edge_color_hash(key2)
        
        self.assertNotEqual(color1, color2)
    
    def test_invalid_key(self):
        """Test error handling for invalid key."""
        with self.assertRaises(constants.physics_parameter_error):
            edge_color_hash((1, 2, 3))  # Too few elements
        
        with self.assertRaises(constants.physics_parameter_error):
            edge_color_hash("not a tuple")


if __name__ == '__main__':
    unittest.main(verbosity=2)