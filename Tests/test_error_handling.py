#!/usr/bin/env python3
"""
Test script to verify error handling implementation.
Tests various error conditions to ensure proper exception handling.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import constants
from utils import calculate_distance, validate_ab, compute_twist_constants, canonical_position
from graph import node, graph
from TBG import hex_lattice, tbg

def test_utils_error_handling():
    """Test error handling in utils.py functions."""
    print("Testing utils.py error handling...")
    
    # Test calculate_distance with invalid input
    try:
        calculate_distance(None)
        print("[FAIL] calculate_distance: Failed to catch None input")
    except constants.physics_parameter_error as e:
        print(f"[OK] calculate_distance: Correctly caught None input - {e}")
    except Exception as e:
        print(f"[FAIL] calculate_distance: Wrong exception type - {e}")
    
    # Test validate_ab with invalid parameters
    try:
        validate_ab(0, 1)  # a must be positive
        print("[FAIL] validate_ab: Failed to catch a=0")
    except constants.physics_parameter_error as e:
        print(f"[OK] validate_ab: Correctly caught a=0 - {e}")
    except Exception as e:
        print(f"[FAIL] validate_ab: Wrong exception type - {e}")
    
    try:
        validate_ab(4, 6)  # not coprime
        print("[FAIL] validate_ab: Failed to catch non-coprime")
    except constants.physics_parameter_error as e:
        print(f"[OK] validate_ab: Correctly caught non-coprime - {e}")
    except Exception as e:
        print(f"[FAIL] validate_ab: Wrong exception type - {e}")
    
    # Test compute_twist_constants with invalid input
    try:
        compute_twist_constants(-1, 1)  # negative a
        print("[FAIL] compute_twist_constants: Failed to catch negative a")
    except constants.physics_parameter_error as e:
        print(f"[OK] compute_twist_constants: Correctly caught negative a - {e}")
    except Exception as e:
        print(f"[FAIL] compute_twist_constants: Wrong exception type - {e}")
    
    # Test canonical_position with invalid lattice vectors
    try:
        canonical_position((0, 0), [1, 0], [1, 0])  # linearly dependent vectors
        print("[FAIL] canonical_position: Failed to catch linearly dependent vectors")
    except constants.physics_parameter_error as e:
        print(f"[OK] canonical_position: Correctly caught linearly dependent vectors - {e}")
    except Exception as e:
        print(f"[FAIL] canonical_position: Wrong exception type - {e}")

def test_graph_error_handling():
    """Test error handling in graph.py functions."""
    print("\nTesting graph.py error handling...")
    
    # Test node creation with invalid position
    try:
        test_node = node("invalid_position", (0, 0), sublattice_id=0)
        print("[FAIL] node: Failed to catch invalid position type")
    except constants.graph_construction_error as e:
        print(f"[OK] node: Correctly caught invalid position - {e}")
    except Exception as e:
        print(f"[FAIL] node: Wrong exception type - {e}")
    
    # Test graph node addition with duplicate
    try:
        test_graph = graph()
        test_node1 = node((0.0, 0.0), (0, 0), sublattice_id=0)
        test_node2 = node((0.0, 0.0), (0, 0), sublattice_id=0)
        test_graph.add_node(test_node1)
        test_graph.add_node(test_node2)  # Same lattice index and sublattice
        print("[FAIL] graph.add_node: Failed to catch duplicate node")
    except constants.graph_construction_error as e:
        print(f"[OK] graph.add_node: Correctly caught duplicate node - {e}")
    except Exception as e:
        print(f"[FAIL] graph.add_node: Wrong exception type - {e}")

def test_tbg_error_handling():
    """Test error handling in TBG.py functions."""
    print("\nTesting TBG.py error handling...")
    
    # Test hex_lattice with invalid parameters
    try:
        test_lattice = hex_lattice(-1, 5, 1.0, [(1, 0), (0, 1)], 1.0)
        print("[FAIL] hex_lattice: Failed to catch negative maxsize_n")
    except constants.physics_parameter_error as e:
        print(f"[OK] hex_lattice: Correctly caught negative maxsize_n - {e}")
    except Exception as e:
        print(f"[FAIL] hex_lattice: Wrong exception type - {e}")
    
    # Test tbg with invalid parameters
    try:
        test_tbg = tbg(-1, 5, 5, 1)  # negative maxsize_n
        print("[FAIL] tbg: Failed to catch negative maxsize_n")
    except constants.physics_parameter_error as e:
        print(f"[OK] tbg: Correctly caught negative maxsize_n - {e}")
    except Exception as e:
        print(f"[FAIL] tbg: Wrong exception type - {e}")
    
    try:
        test_tbg = tbg(5, 5, 4, 6, interlayer_dist_threshold=-1.0)  # negative threshold
        print("[FAIL] tbg: Failed to catch negative threshold")
    except constants.physics_parameter_error as e:
        print(f"[OK] tbg: Correctly caught negative threshold - {e}")
    except Exception as e:
        print(f"[FAIL] tbg: Wrong exception type - {e}")

def test_import_functionality():
    """Test that all imports are working correctly."""
    print("\nTesting import functionality...")
    
    try:
        # Test constants access
        test_val = constants.np.array([1, 2, 3])
        print(f"[OK] constants.np.array: Working - {test_val}")
    except Exception as e:
        print(f"[FAIL] constants.np.array: Failed - {e}")
    
    try:
        # Test logging
        logger = constants.logging.getLogger("test")
        logger.info("Test log message")
        print("[OK] constants.logging: Working")
    except Exception as e:
        print(f"[FAIL] constants.logging: Failed - {e}")
    
    try:
        # Test scipy sparse
        test_matrix = constants.csr_matrix([[1, 0], [0, 1]])
        print(f"[OK] constants.csr_matrix: Working - shape {test_matrix.shape}")
    except Exception as e:
        print(f"[FAIL] constants.csr_matrix: Failed - {e}")

def main():
    """Run all error handling tests."""
    print("*** Starting Error Handling Verification Tests ***")
    print("=" * 50)
    
    try:
        test_utils_error_handling()
        test_graph_error_handling()
        test_tbg_error_handling() 
        test_import_functionality()
    except ImportError as e:
        print(f"[X] Import Error: {e}")
        print("Please ensure all modules are properly configured.")
    except Exception as e:
        print(f"[X] Unexpected Error: {e}")
    
    print("\n" + "=" * 50)
    print("*** Error Handling Test Complete! ***")

if __name__ == "__main__":
    main()