"""
Utility functions for TBG (Twisted Bilayer Graphene) project.

This module contains standalone mathematical and computational utilities
that are used across multiple modules in the TBG project. Moving these
utilities here helps eliminate circular dependencies and improves code
organization.

Functions:
    calculate_distance: Distance calculations between nodes or to origin
    canonical_position: Coordinate transformations to lattice unit cells  
    compute_twist_constants: TBG physics parameter calculations
    validate_ab: Input validation for TBG rotation parameters
    is_hermitian_sparse: Matrix validation utilities
    edge_color_hash: Color generation for visualization
"""

import constants
from constants import hashlib
from constants import List, Tuple, Optional, Union, Any

# Configure logging
logger = constants.logging.getLogger(__name__)


def calculate_distance(node1: Any, node2: Optional[Any] = None, offset: Tuple[int, int] = (0, 0)) -> float:
    """
    Calculate distance between nodes or from a node to the origin.

    Parameters:
        node1: The first node (with .position attribute).
        node2: The second node (with .position attribute). Defaults to None.
        offset (tuple, optional): Coordinate offset to apply. Defaults to (0, 0).

    Returns:
        float: Euclidean distance.
        
    Raises:
        constants.physics_parameter_error: If nodes don't have position attribute or invalid positions.
    """
    try:
        if not hasattr(node1, 'position'):
            raise constants.physics_parameter_error("node1 must have a 'position' attribute")
        
        pos1 = constants.np.array(node1.position)
        offset_array = constants.np.array(offset)
        
        if node2 is None:
            return float(constants.np.linalg.norm(pos1 - offset_array))
        
        if not hasattr(node2, 'position'):
            raise constants.physics_parameter_error("node2 must have a 'position' attribute")
        
        pos2 = constants.np.array(node2.position)
        return float(constants.np.linalg.norm(pos1 - offset_array - pos2))
        
    except (AttributeError, TypeError, ValueError) as e:
        raise constants.physics_parameter_error(f"Error calculating distance: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in calculate_distance: {str(e)}")
        raise


def canonical_position(pos: Tuple[float, float], latticevector1: List[float], latticevector2: List[float], verbosity: bool = False) -> List[Tuple[float, ...]]:
    """
    Convert a Cartesian coordinate into its canonical position in the lattice.

    Parameters:
        pos (tuple): Cartesian position.
        latticevector1 (list): First lattice vector.
        latticevector2 (list): Second lattice vector.
        verbosity (bool): If true, a more detailed log will be supplied.

    Returns:
        list: Canonical position and shift in lattice units.
        
    Raises:
        constants.physics_parameter_error: If lattice vectors are invalid.
        constants.matrix_operation_error: If matrix operations fail.
    """
    try:
        if len(latticevector1) != len(latticevector2):
            raise constants.physics_parameter_error("Lattice vectors must have same dimension")
        
        A = constants.np.array([latticevector1, latticevector2]).T
        if constants.np.linalg.det(A) == 0:
            raise constants.physics_parameter_error("Lattice vectors are linearly dependent")
            
        Ainv = constants.np.linalg.inv(A)
        lattice_coords = Ainv @ constants.np.array(pos)
        wrapped = (lattice_coords + 0.5) % 1.0 - 0.5  # We want centered unit cell
        shift = constants.np.round(lattice_coords - wrapped).astype(int)
        canonical_pos = A @ wrapped
        canonical_pos = constants.np.round(canonical_pos, 12)
    except constants.np.linalg.LinAlgError as e:
        raise constants.matrix_operation_error(f"Linear algebra error in canonical_position: {str(e)}")
    except (ValueError, TypeError) as e:
        raise constants.physics_parameter_error(f"Invalid parameters in canonical_position: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in canonical_position: {str(e)}")
        raise
    if verbosity:
        logger.debug(f"Canonicalizing pos={pos} to {canonical_pos} with shift={shift} "
                    f"using lattice vectors {latticevector1}, {latticevector2}")
    return [tuple(canonical_pos), tuple(shift)]  # Return both the fractional and rounded positions in Cartesian coordinates


def compute_twist_constants(a: int, b: int) -> Tuple[float, float, float, Tuple[float, float]]:
    """
    Calculate constants alpha and NFactor for rational TBG twist angle.
    If a%3=0, then by the theory we are working with reciprocal lattice
    and we need to change the factor of the lattice, as well as alpha

    Args:
        a (int): Integer parameter a of the twist.
        b (int): Integer parameter b of the twist.

    Returns:
        tuple: (NFactor, alpha, factor, kpoint):
            NFactor: The scaling factor of the new lattice = sqrt(a^2+3b^2), taken from [malinovitch2024twisted].
            alpha: The correction factor to the scaling = 2^eps (4pi)^rho, where eps=1 if ab%2=1, and 0 else, and rho=1 if a%3=0 and 0 else.
            factor: The scaling factor of the underlying lattice, if a%3=0, then we have that our lattice is N*Lambda^*, not N*Lambda.
            kpoint: The coordinate of the k point, depending on whether we have Lambda or Lambda^*.

    Raises:
        ValueError: If a is not positive, or b is 0.
    """
    try:
        # Validate inputs using constants utilities
        constants.validate_positive_number(a, "parameter a")
        if b == 0:
            raise constants.physics_parameter_error("parameter b cannot be zero")
        if not isinstance(a, int) or not isinstance(b, int):
            raise constants.physics_parameter_error("parameters a and b must be integers")
            
        alpha = 1
        factor = 1
        if a % 3 == 0:
            alpha *= 4 * constants.np.pi
            factor = constants.reciprocal_constant
            k_point = constants.K_POINT_DUAL
        else:
            k_point = constants.K_POINT_REG
        if (a * b) % 2 == 1:
            alpha *= 2
        
        NFactor = constants.safe_divide(constants.np.sqrt(a**2 + 3*b**2), alpha, 0.0)
        if NFactor == 0.0:
            raise constants.physics_parameter_error(f"Invalid twist constants for a={a}, b={b}")
            
        return NFactor, alpha, factor, k_point
    except constants.physics_parameter_error:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in compute_twist_constants: {str(e)}")
        raise constants.physics_parameter_error(f"Failed to compute twist constants: {str(e)}")


def validate_ab(a: int, b: int) -> None:
    """
    Validates the parameters a and b for rational TBG rotation.

    Args:
        a (int): Integer rotation parameter.
        b (int): Integer rotation parameter.

    Raises:
        constants.physics_parameter_error: If the conditions for rational TBG are not met (a,b, co-prime integer, 0<|b|<=a).
    """
    try:
        if not isinstance(a, int) or not isinstance(b, int):
            raise constants.physics_parameter_error("Both a and b must be integers.")
        constants.validate_positive_number(a, "parameter a")
        if b == 0:
            raise constants.physics_parameter_error("b must be a non-zero integer.")
        if constants.np.gcd(a, b) != 1:
            raise constants.physics_parameter_error("a and b must be coprime (gcd(a, b) == 1).")
        if abs(b) > a:
            raise constants.physics_parameter_error("a must be greater than or equal to |b|.")
    except constants.physics_parameter_error:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in validate_ab: {str(e)}")
        raise constants.physics_parameter_error(f"Failed to validate parameters: {str(e)}")


def is_hermitian_sparse(matrix: constants.csr_matrix, rtol: float = constants.NUMERIC_TOLERANCE) -> bool:
    """
    Check if a sparse matrix is Hermitian within numerical tolerance.
    
    Args:
        matrix (csr_matrix): The sparse matrix to check
        rtol (float): Relative tolerance for comparison
        
    Returns:
        bool: True if matrix is Hermitian, False otherwise
        
    Raises:
        constants.matrix_operation_error: If matrix operations fail.
    """
    try:
        if not isinstance(matrix, constants.csr_matrix):
            raise constants.matrix_operation_error("Input must be a csr_matrix")
        if matrix.shape[0] != matrix.shape[1]:
            return False
        
        # Compare matrix with its conjugate transpose
        matrix_h = matrix.conj().T.tocsr()
        diff = matrix - matrix_h
        max_diff = constants.np.abs(diff.data).max() if diff.nnz > 0 else 0.0
        max_val = max(constants.np.abs(matrix.data).max(), constants.np.abs(matrix_h.data).max())
        
        return max_diff <= rtol * max_val
    except (AttributeError, ValueError) as e:
        raise constants.matrix_operation_error(f"Error checking matrix hermiticity: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in is_hermitian_sparse: {str(e)}")
        raise constants.matrix_operation_error(f"Failed to check matrix hermiticity: {str(e)}")


def edge_color_hash(key: Tuple) -> Tuple[float, float, float]:
    """
    Generate unique RGB color from edge key using hash.
    
    Args:
        key: Tuple containing edge identification data
        
    Returns:
        Tuple[float, float, float]: RGB color values between 0 and 1
    """
    try:
        if not isinstance(key, tuple) or len(key) < 5:
            raise constants.physics_parameter_error("Key must be a tuple with at least 5 elements")
        
        flat_key = tuple(key[:4]) + tuple(key[4])
        h = hashlib.md5(str(flat_key).encode()).hexdigest()
        r = constants.safe_divide(int(h[0:2], 16), 255, 0.0)
        g = constants.safe_divide(int(h[2:4], 16), 255, 0.0)
        b = constants.safe_divide(int(h[4:6], 16), 255, 0.0)
        return (r, g, b)
    except (ValueError, TypeError, IndexError) as e:
        raise constants.physics_parameter_error(f"Error generating color hash: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in edge_color_hash: {str(e)}")
        raise constants.physics_parameter_error(f"Failed to generate color hash: {str(e)}")