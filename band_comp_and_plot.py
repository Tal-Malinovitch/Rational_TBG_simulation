"""
Band computation and analysis module for twisted bilayer graphene (TBG) simulations.

This module provides specialized classes for eigenvalue computation, band structure
analysis, and energy band visualization:

Classes:
    band_handler: Handles eigenvalue solving, band computation, and analysis

The module separates band computation logic from core graph structures,
providing clean interfaces for different types of band analysis.
"""

import constants
from typing import Optional, Tuple, Any

# Configure logging
logger = constants.logging.getLogger(__name__)

# Default parameters - will be imported from constants when needed
try:
    DEFAULT_PARAMS = constants.dataclasses.asdict(constants.simulation_parameters())
except (ImportError, AttributeError, TypeError) as e:
    DEFAULT_PARAMS = {'inter_graph_weight': 1.0, 'intra_graph_weight': 1.0}
    constants.logging.warning(f"Could not load default parameters: {e}, using fallback")


class band_handler:
    """
    Handles eigenvalue solving, band computation, and Dirac point analysis.
    Focuses on energy bands and electronic properties.
    """
    
    def __init__(self, matrix_handler: 'periodic_matrix') -> None:
        """
        Initialize band handler with matrix operations.
        
        Args:
            matrix_handler: periodic_matrix object with all matrix operations
            
        Raises:
            constants.physics_parameter_error: If invalid matrix handler provided.
        """
        try:
            if matrix_handler is None:
                raise constants.physics_parameter_error("matrix_handler cannot be None")
            
            # Check that matrix_handler has required attributes
            if not hasattr(matrix_handler, 'periodic_graph'):
                raise constants.physics_parameter_error("matrix_handler must have periodic_graph attribute")
            
            self.matrix_handler = matrix_handler
            # Access periodic_graph through matrix_handler when needed
            self.periodic_graph = matrix_handler.periodic_graph
            
        except constants.physics_parameter_error:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in band_handler.__init__: {str(e)}")
            raise constants.physics_parameter_error(f"Failed to initialize band handler: {str(e)}")

    def _sparse_eigensolve(self, Laplacian: constants.csr_matrix, min_bands: int, max_bands: int) -> Optional[Tuple[constants.np.ndarray, constants.np.ndarray]]:
        """
        Attempt sparse eigenvalue computation with multiple strategies.
        
        Uses multiple fallback strategies to compute eigenvalues robustly:
        1. Smallest real eigenvalues (SR) - most reliable for lowest bands
        2. Smallest magnitude eigenvalues (SM) - backup approach  
        3. Shift-invert around zero (LM) - last resort
        
        Args:
            Laplacian (constants.csr_matrix): The Laplacian matrix in CSR format
            min_bands (int): Index of the lowest band to compute (1-indexed)
            max_bands (int): Index of the highest band to compute (1-indexed)
            
        Returns:
            Optional[Tuple[constants.np.ndarray, constants.np.ndarray]]: If successful, returns tuple containing:
                - eigenvalues: Sorted eigenvalues for requested bands
                - eigenvectors: Corresponding eigenvectors (transposed)
            Returns None if all strategies fail.
            
        Raises:
            constants.matrix_operation_error: If input validation fails.
        """
        try:
            # Input validation
            if Laplacian is None:
                raise constants.matrix_operation_error("Laplacian matrix cannot be None")
            if not isinstance(Laplacian, constants.csr_matrix):
                raise constants.matrix_operation_error("Laplacian must be a csr_matrix")
            if Laplacian.shape[0] != Laplacian.shape[1]:
                raise constants.matrix_operation_error("Laplacian matrix must be square")
            if min_bands < 1 or max_bands < min_bands:
                raise constants.matrix_operation_error("Invalid band indices: must have 1 <= min_bands <= max_bands")
            if max_bands > Laplacian.shape[0]:
                raise constants.matrix_operation_error(f"max_bands ({max_bands}) cannot exceed matrix size ({Laplacian.shape[0]})")
            
            strategies = [
                # Strategy 1: Smallest real eigenvalues (most reliable for lowest eigenvalues)
                {'sigma': None, 'which': 'SR'},
                # Strategy 2: Smallest magnitude eigenvalues
                {'sigma': None, 'which': 'SM'},
                # Strategy 3: Shift-invert around zero (fallback)
                {'sigma': 0.0, 'which': 'LM'},
            ]
            
            for i, strategy in enumerate(strategies):
                try:
                    logger.debug(f"Trying sparse strategy {i+1}: {strategy}")
                    
                    # Request more eigenvalues than needed for stability
                    k_request = min(max_bands + 5, Laplacian.shape[0] - 2)
                    
                    eigvals, eigfun = constants.eigsh(Laplacian,k=k_request,sigma=strategy['sigma'],which=strategy['which'],maxiter=1000,tol=constants.NUMERIC_TOLERANCE)
                    
                    # Sort and extract requested range
                    sorted_indices= constants.np.argsort(eigvals.real)
                    eigfun = eigfun[:, sorted_indices]
                    eigvals = eigvals[sorted_indices]
                    if len(eigvals) >= max_bands:
                        return eigvals[min_bands-1:max_bands],eigfun[:, min_bands-1:max_bands].T
                    
                    else:
                        logger.warning(f"Insufficient eigenvalues computed: {len(eigvals)} < {max_bands}")
                        
                except (constants.ArpackNoConvergence, constants.np.linalg.LinAlgError) as e:
                    logger.debug(f"Sparse strategy {i+1} failed: {e}")
                    continue
            
            return None  # All sparse strategies failed
            
        except constants.matrix_operation_error:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in _sparse_eigensolve: {str(e)}")
            raise constants.matrix_operation_error(f"Failed in sparse eigenvalue computation: {str(e)}")
    
    def _dense_eigensolve(self, Laplacian: constants.csr_matrix, min_bands: int, max_bands: int) -> Tuple[constants.np.ndarray, constants.np.ndarray]:
        """
        Dense eigenvalue computation with memory management.
        
        Args:
            Laplacian (constants.csr_matrix): The Laplacian matrix in CSR format
            min_bands (int): Index of the lowest band to compute (1-indexed)
            max_bands (int): Index of the highest band to compute (1-indexed)
            
        Returns:
            Tuple[constants.np.ndarray, constants.np.ndarray]: Tuple containing:
                - eigenvalues: Sorted eigenvalues for requested bands
                - eigenvectors: Corresponding eigenvectors (transposed)
        """
        H = Laplacian.toarray()
        
        # Use scipy's optimized eigenvalue solver
        try:
            E, V = constants.np.linalg.eigh(H)
            sorted_indices= constants.np.argsort(E)
            E= E[sorted_indices]
            V = V[:, sorted_indices]
            # Check bounds to avoid index errors
            if max_bands > len(E):
                logger.warning(f"Requested max_bands {max_bands} exceeds available eigenvalues {len(E)}")
                max_bands = len(E)
            if min_bands < 1:
                logger.warning(f"min_bands {min_bands} < 1, setting to 1")
                min_bands = 1
                
            eigvals = E[min_bands-1:max_bands]
            eigfun = V[:, min_bands-1:max_bands].T

            return eigvals, eigfun
        except constants.np.linalg.LinAlgError as e:
            logger.error(f"Dense eigenvalue computation failed: {e}")
            return constants.np.full(max_bands - min_bands + 1, constants.np.nan),constants.np.full([len(self.periodic_graph.nodes),max_bands - min_bands + 1], constants.np.nan)

    def compute_bands_at_k(self, Momentum: tuple[float, float], min_bands: int, max_bands: int, 
                          inter_graph_weight: float = DEFAULT_PARAMS['inter_graph_weight'],
                          intra_graph_weight: float = DEFAULT_PARAMS['intra_graph_weight']) -> tuple[constants.np.ndarray, constants.np.ndarray]:
        """
        Solve the eigenvalue problem for a given momentum.

        Args:
            Momentum (tuple[float,float]): Quasimomentum vector.
            min_bands (int): the index of the lowest band to plot.
            max_bands (int): the index of the highest band to plot.
            inter_graph_weight (float): Inter-subgraph coupling strength.
            intra_graph_weight (float): Intra-subgraph coupling strength.
        Returns:
            constants.np.ndarray: Sorted list of computed eigenvalues.
            constants.np.ndarray: Eigenfunctions corresponding to the eigenvalues.
        
        Raises:
            ValueError: if laplacian provided without periodic edges
        """
        if self.matrix_handler.adj_matrix is None:
            if self.matrix_handler.periodic_edges is None:
                compute_adj_flag = True
            else:
                raise ValueError("Missing input! when providing the Laplacian one should also provide periodic edges list!")
        else:
            compute_adj_flag = False
            if self.matrix_handler.periodic_edges is None:
                raise ValueError("Missing input! when providing the Laplacian one should also provide periodic edges list!")
        
        Laplacian, _ = self.matrix_handler.build_laplacian(inter_graph_weight=inter_graph_weight, 
                                                         intra_graph_weight=intra_graph_weight,
                                                         Momentum=Momentum, compute_adj_flag=compute_adj_flag)
        # Convert to CSR format for better performance
        if not isinstance(Laplacian, constants.csr_matrix):
            Laplacian = Laplacian.tocsr()
        matrix_size = Laplacian.shape[0]
        num_bands = max_bands - min_bands + 1
        
        # Handle trivial matrix cases
        if matrix_size < 2:
            logger.warning(f"Matrix too small ({matrix_size}x{matrix_size}) for meaningful eigenvalue computation.")
            if matrix_size == 0:
                return constants.np.array([]), constants.np.array([]).reshape(0, 0)
            else:  # matrix_size == 1
                eigenvalue = Laplacian[0, 0].real
                eigenvector = constants.np.array([[1.0]], dtype=complex)
                return constants.np.array([eigenvalue]), eigenvector
        if matrix_size > constants.MATRIX_SIZE_SPARSE_THRESHOLD and num_bands < matrix_size // 2:
            try:
                # Use shift-invert mode around zero for better convergence
                eigvals, eigfun = self._sparse_eigensolve(Laplacian, min_bands, max_bands)
                if eigvals is not None:
                    return eigvals, eigfun
                    
            except (constants.ArpackNoConvergence, constants.np.linalg.LinAlgError) as e:
                logger.warning(f"Sparse solver failed at k={Momentum} (relative coordinates): {e}")
        
        # Strategy 2: Fallback to dense solver with memory check
        memory_estimate_mb = (matrix_size ** 2 * constants.COMPLEX128_BYTE_SIZE) / (1024 ** 2)  # Complex128 bytes
        
        if memory_estimate_mb > constants.MEMORY_WARNING_THRESHOLD_MB:  # Memory warning threshold
            logger.warning(f"Large matrix ({matrix_size}x{matrix_size}, ~{memory_estimate_mb:.1f}MB). "
                         f"Consider reducing system size or using sparse methods.")
        if memory_estimate_mb > constants.MEMORY_LIMIT_MB:  # Memory limit
            logger.error(f"Matrix too large({memory_estimate_mb:.1f}MB). Aborting.")
            return constants.np.full(num_bands, constants.np.nan)    
        try:
            eigvals, eigfun = self._dense_eigensolve(Laplacian, min_bands, max_bands)
            return eigvals, eigfun
        except MemoryError:
            logger.error(f"Out of memory for matrix size {matrix_size}x{matrix_size}")
            return constants.np.full(num_bands, constants.np.nan)


    def compute_energy_and_derivative_of_energy_at_a_point(self, k_point: tuple[float, float], num_of_band: int = 1) -> tuple[constants.np.ndarray, float, float]:
        """
        Compute eigenvalues and energy derivatives at a given k-point using Feynman-Hellmann theorem.
        
        The derivatives are computed as:
        ∂E_n/∂k_α = ⟨ψ_n(k)| ∂H(k)/∂k_α |ψ_n(k)⟩
        
        Args:
            k_point (tuple): 2D quasimomentum vector [kx, ky]
            num_of_band (int): Starting band index for eigenvalue computation
            
        Returns:
            tuple: (eigenvalues, ∂E/∂kx, ∂E/∂ky) where derivatives are for the energy gap
        """
        der_of_H = self.matrix_handler.compute_laplacian_derivative(k_point)
            
        # Compute eigenvalues and eigenvectors for the bands of interest
        eigvals, eigfun = self.compute_bands_at_k(Momentum=k_point, min_bands=num_of_band, max_bands=num_of_band + 1)
        # Extract the two bands we're interested in
        if len(eigvals) < 2:
            raise ValueError(f"Not enough eigenvalues computed at k={k_point}")
        
        # Compute energy derivatives for each band using Feynman-Hellmann theorem
        # ∂E_n/∂k_α = ⟨ψ_n| ∂H/∂k_α |ψ_n⟩
        der_x_band0 = constants.np.real(eigfun[0,:].conj().T @ der_of_H[0] @ eigfun[0,:])
        der_x_band1 = constants.np.real(eigfun[1,:].conj().T @ der_of_H[0] @ eigfun[1,:])
        der_y_band0 = constants.np.real(eigfun[0,:].conj().T @ der_of_H[1] @ eigfun[0,:])
        der_y_band1 = constants.np.real(eigfun[1,:].conj().T @ der_of_H[1] @ eigfun[1,:])
        
        # Return derivatives of the energy gap (upper band - lower band)
        der_x_of_E = der_x_band1 - der_x_band0
        der_y_of_E = der_y_band1 - der_y_band0
        
        return eigvals, der_x_of_E, der_y_of_E


    def cleanup(self) -> None:
        """
        Clean up band handler references.
        """
        self.matrix_handler = None
        self.periodic_graph = None

    def __enter__(self) -> 'band_handler':
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Optional[type], exc_value: Optional[Exception], traceback: Optional[Any]) -> bool:
        """Context manager exit with cleanup."""
        self.cleanup()
        return False