"""
Constants and configuration parameters for TBG (Twisted Bilayer Graphene) simulations.

This module contains physical constants, default plotting parameters, simulation
configuration, common imports, and error handling utilities.
"""

# Common imports - imported here so other modules can access them via constants
import numpy as np
import dataclasses
import logging
import os
import time
import csv
import glob
import sys
import hashlib
import gc
import json
from typing import Mapping, List, Tuple, Optional, Union, Dict, Any, Callable, DefaultDict
from collections import defaultdict
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix
from scipy.sparse.linalg import eigs, eigsh, ArpackNoConvergence
from scipy.spatial import cKDTree
import matplotlib.axes
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.figure import Figure

# Make imports available as module attributes for easy access
# This allows other modules to do: from constants import np, List, etc.
# Common standard library imports

# Configure logging for the entire project
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Plotting defaults
MARKERSIZE: int = 50 
LINEWIDTH: float = 1.5
LINEWIDTHOFUNITCELLBDRY: float = 2.5
DEFAULT_COLORS: List[str] = ['b', 'r', 'k']

# Periodic boundary conditions
MAX_ADJACENT_CELLS: int = 1  # Maximum cells to consider for periodic edges
"""
When considering a periodic structure we only take one unit cell into account. 
So you shouldn't have edges crossing multiple unit cells, so we limit to edges crossing a single unit cell.
"""

# Numerical parameters
NUMERIC_TOLERANCE: float = 1e-10  # Tolerance for eigenvalue convergence
INITIAL_RADIUS_DEFAULT: float = 2.0  # Initial radius to search for neighbors when connecting layers
EXPANSION_FACTOR: int = 3  # The factor which the system will expand if there is only one node

# Standard lattice vectors for triangular lattice of graphene
v1: np.ndarray = np.array([np.sqrt(3)/2, 0.5])  # First lattice vector
v2: np.ndarray = np.array([np.sqrt(3)/2, -0.5])  # Second lattice vector

# Reciprocal lattice parameters
reciprocal_constant: float = 4.0 * np.pi / np.sqrt(3)  # Reciprocal lattice constant
k1: np.ndarray = np.array([0.5, np.sqrt(3)/2]) * reciprocal_constant  # First reciprocal vector
k2: np.ndarray = np.array([0.5, -np.sqrt(3)/2]) * reciprocal_constant  # Second reciprocal vector

# Optimization parameters
attempt_num: int = 5  # Number of attempts to converge by jitter
direction_num: int = 3  # Number of directions to check the Dirac point
points_num: int = 3  # Number of points to check the Dirac point (per direction)
NUMERIC_RANGE_TO_CHECK: float = 0.005  # Range for checking Dirac point convergence

# High symmetry points
K_POINT_DUAL: Tuple[float, float] = (1/3, 1/3)  # K point for dual lattice
K_POINT_REG: Tuple[float, float] = (1/3, -1/3)  # K point for regular lattice

# Convergence tolerances
DEFAULT_K_TOLERANCE: float = 0.05  # Distance threshold to treat two k-points as identical
DEFAULT_E_TOLERANCE: float = 0.001  # Energy threshold for band touching
DEFAULT_TOLERANCE: float = 0.1  # General optimization tolerance
MAX_ITERATIONS: int = 300  # Maximum optimization iterations
DEFAULT_SEARCH_RADIUS=0.5 #the search radius of Dirac point
DEFAULT_NUM_OF_POINTS=5 #the numberof points to check in the search of Dirac point

# ADAM optimization constants
ADAM_LEARNING_RATE: float = 0.01  # Default ADAM learning rate
LEARNING_RATE_REDUCTION_FACTOR: float = 0.1  # Factor to reduce learning rate on improvement
IMPROVEMENT_THRESHOLD: float = 0.1  # Threshold for significant improvement
TRAINING_SAMPLE_MIN_FACTOR: int = 1  # Minimum scaling factor for training samples
TRAINING_SAMPLE_MAX_FACTOR: int = 10  # Maximum scaling factor for training samples

# Performance and memory thresholds
MATRIX_SIZE_SPARSE_THRESHOLD: int = 100  # Matrix size threshold for using sparse vs dense solvers
MEMORY_WARNING_THRESHOLD_MB: int = 500  # Memory warning threshold in MB
MEMORY_LIMIT_MB: int = 1000  # Memory limit threshold in MB
COMPLEX128_BYTE_SIZE: int = 16  # Byte size of Complex128 data type

# Physics simulation defaults
DEFAULT_K_RANGE: float = 0.5  # Default k-space range for band calculations
DEFAULT_UNIT_CELL_RADIUS_FACTOR: float = 2.0  # Default unit cell radius scaling factor

# Neural network and training defaults
DEFAULT_NN_LOSS_WEIGHTS: List[float] = [0.6, 0.3, 0.1]  # [gap, R2, isotropy] loss weights
DEFAULT_BATCH_SIZE: int = 100  # Default batch size for training data generation
DEFAULT_NN_LAYER_SIZE: int = 8  # Default neural network layer size

# Neural network activation constants
LEAKY_RELU_ALPHA: float = 0.01  # Leaky ReLU negative slope
SIGMOID_CLIPPING_RANGE: Tuple[float, float] = (-50.0, 50.0)  # Range for sigmoid stability
LOG_ACTIVATION_THRESHOLD: float = 10.0  # Threshold for log activation function
LOG_ACTIVATION_MIN_ARG: float = 1e-8  # Minimum argument for log function
SCALED_SIGMOID_MAX: float = 10.0  # Maximum value for scaled sigmoid

# Neural network training constants
DEFAULT_LEARNING_RATE: float = 0.001  # Default learning rate
DEFAULT_TRAINING_BATCH_SIZE: int = 128  # Production batch size
DEFAULT_VALIDATION_SPLIT: float = 0.15  # Validation split ratio
DEFAULT_EARLY_STOPPING_PATIENCE: int = 100  # Early stopping patience
DEFAULT_CHECKPOINT_INTERVAL: int = 10  # Checkpoint save interval (very frequent for cloud)
DEFAULT_MAX_EPOCHS: int = 1000  # Maximum training epochs
BATCH_LOGGING_FREQUENCY: int = 100  # How often to log batch progress
EPOCH_LOGGING_FREQUENCY: int = 10  # How often to log epoch progress

# Gradient clipping and training stability constants
GRADIENT_CLIP_VALUE: float = 5.0  # Maximum gradient magnitude for clipping
LOSS_EXPLOSION_THRESHOLD: float = 100.0  # Threshold for detecting loss explosion
LEARNING_RATE_REDUCTION_FACTOR: float = 0.5  # Factor to reduce learning rate when loss explodes
MIN_LEARNING_RATE: float = 1e-6  # Minimum allowed learning rate
LOSS_HISTORY_WINDOW: int = 5  # Number of epochs to look back for loss explosion detection

# Neural network numerical constants
WEIGHT_CHANGE_THRESHOLD: float = 1e-10  # Threshold for detecting weight changes
DEAD_GRADIENT_THRESHOLD: float = 1e-8  # Threshold for detecting dead gradients
EXTREME_GRADIENT_THRESHOLD: float = 10.0  # Threshold for extreme gradients
VANISHING_GRADIENT_THRESHOLD: float = 1e-6  # Threshold for vanishing gradients
NETWORK_COLLAPSE_THRESHOLD: float = 1e-10  # Threshold for network output collapse

# Neural network architecture constants
INPUT_TRANSFORMATION_RECIPROCAL_MIN: float = 1e-8  # Minimum value for 1/x transformation
VELOCITY_CONVERSION_MIN_NU: float = 0.0001  # Minimum ν for velocity conversion
VELOCITY_CONVERSION_FALLBACK: float = 10000.0  # Fallback velocity for small ν

# He weight initialization constants
HE_INIT_FACTOR: float = 2.0  # Factor for He initialization: sqrt(2/n_in)

# GUI display constants
DEFAULT_GUI_SIZE: int = 10  # Default GUI element sizing
DEFAULT_WINDOW_SIZE: int = 600  # Default window dimensions

# File paths
PATH: str = os.path.dirname(os.path.abspath(__file__))

# Error handling utilities and custom exceptions
class tbg_error(Exception):
    """Base exception class for TBG-related errors."""
    pass

class physics_parameter_error(tbg_error):
    """Raised when invalid physics parameters are provided."""
    pass

class graph_construction_error(tbg_error):
    """Raised when graph/lattice construction fails."""
    pass

class matrix_operation_error(tbg_error):
    """Raised when matrix operations fail."""
    pass

def safe_divide(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    """Safely divide two numbers, returning fallback if division by zero."""
    try:
        return numerator / denominator if denominator != 0 else fallback
    except (ZeroDivisionError, TypeError):
        return fallback

def validate_positive_number(value: Union[int, float], name: str) -> None:
    """Validate that a number is positive."""
    if not isinstance(value, (int, float)):
        raise physics_parameter_error(f"{name} must be a number, got {type(value)}")
    if value <= 0:
        raise physics_parameter_error(f"{name} must be positive, got {value}")

def validate_array_shape(array: np.ndarray, expected_shape: tuple, name: str) -> None:
    """Validate that an array has the expected shape."""
    if array.shape != expected_shape:
        raise matrix_operation_error(f"{name} shape {array.shape} doesn't match expected {expected_shape}")

def handle_convergence_failure(operation_name: str, max_attempts: int = 3):
    """Decorator for handling convergence failures with retries."""
    def decorator(func: Any) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise ValueError(f"{operation_name} failed to converge after {max_attempts} attempts: {str(e)}")
                    logging.warning(f"{operation_name} attempt {attempt + 1} failed: {str(e)}, retrying...")
            return None
        return wrapper
    return decorator

# data classes:
@dataclasses.dataclass(frozen=True)  # frozen=True makes instances immutable
class simulation_parameters:
    """
    Default simulation parameters for TBG calculations.
    
    This dataclass contains all the default parameters needed for TBG simulations,
    including twist parameters, plotting options, and physical constants.
    
    Attributes:
        a: Integer parameter for twist angle (co-prime to b, a >= |b|).
        b: Integer parameter for twist angle (co-prime to a, a >= |b|).
        unit_cell_radius_factor: Scaling factor for plotted unit cell radius.
        unit_cell_flag: Whether to plot unit cell only or entire graph.
        interlayer_dist_threshold: Distance threshold for interlayer connections.
        intralayer_dist_threshold (float): Threshold for connecting intralayer nodes.
        min_band: Index of minimum band to compute (1-indexed).
        max_band: Index of maximum band to compute (1-indexed).
        num_of_points: Number of sampling points for band structure.
        inter_graph_weight: Laplacian weight for edges between sublattices.
        intra_graph_weight: Laplacian weight for edges within sublattices.
        k_min: Minimum k-value to plot (in natural units).
        k_max: Maximum k-value to plot (in natural units).
        k_flag: Whether to plot around K point (True) or origin (False).
    """
    # Twist parameters (must satisfy: gcd(a,b)=1 and a >= |b|)
    a: int = 5  # Integer twist parameter
    b: int = 1  # Integer twist parameter
    
    # Geometry parameters
    unit_cell_radius_factor: float = 3.0  # Scaling factor for unit cell radius
    unit_cell_flag: bool = False  # Plot unit cell only vs entire graph
    interlayer_dist_threshold: float = 1.0  # Interlayer connection threshold
    intralayer_dist_threshold: float=1.0 # Intralayer connection threshold
    
    # Band structure parameters
    min_band: int = 1  # Minimum band index (1-indexed)
    max_band: int = 3  # Maximum band index (1-indexed)
    num_of_points: int = 50  # Number of k-points for band sampling
    
    # Hamiltonian weights
    inter_graph_weight: float = 0.5  # Weight for inter-sublattice edges
    intra_graph_weight: float = 1.0  # Weight for intra-sublattice edges
    
    # Plotting range
    k_min: float = -0.25  # Minimum k-value (natural units)
    k_max: float = 0.25  # Maximum k-value (natural units)
    k_flag: bool = True  # Plot around K point (True) or origin (False) 
@dataclasses.dataclass(frozen=True)
class simulation_parameters_adam:
    """
    Default simulation parameters for ADAM optimizer.
    
    This dataclass contains all the default parameters needed for ADAM optimizer.
    
    Attributes:
        alpha (float, optional): Learning rate. Defaults to 0.001.
        beta_1 (float, optional): Exponential decay rate for first moment. Defaults to 0.9.
        beta_2 (float, optional): Exponential decay rate for second moment. Defaults to 0.999.
        eps (float, optional): Small constant for numerical stability. Defaults to 1e-8.
        initial_weight (float, optional): Initial weight value. Defaults to 0.
    """
    alpha: float = 0.001
    beta_1: float = 0.9
    beta_2: float = 0.999 
    eps: float = 1e-8  
    initial_weight: float = 0