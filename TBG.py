"""
High-level physics systems for twisted bilayer graphene (TBG) simulations.

This module contains the physics orchestration classes that compose the fundamental
building blocks from graph.py to implement complete TBG physics simulations:

Classes:
    hex_lattice: Specialized lattice generator for graphene-like structures
    TBG: Complete twisted bilayer graphene system with interlayer coupling

Functions:
    compute_twist_constants: Calculate rational TBG twist angle parameters
    validate_ab: Validate rational TBG rotation parameters

These classes implement the high-level physics logic and coordinate the building
blocks to perform complex TBG simulations including Dirac point analysis,
band structure computations, and lattice construction.
"""
import constants
from scipy.spatial import cKDTree
from typing import List, Tuple, Optional, Union, Dict, Any
from graph import (node, graph, periodic_matrix,periodic_graph, DEFAULT_PARAMS)
from utils import calculate_distance, canonical_position, compute_twist_constants, validate_ab
from plotting import dirac_plotter, band_plotter
from band_comp_and_plot import band_handler

# Configure global logging
logger = constants.logging.getLogger(__name__)


class hex_lattice() : 
    """
    hex_lattice class to create a hexagonal lattice of nodes arranged in a grid.

    Attributes:
        graph (graph): Underlying graph storing nodes and their connections.
        max_rows (int): Maximum range in the N-direction.
        max_cols (int): Maximum range in the M-direction.
        lattice_vectors (List[Tuple[float, float]]): Lattice vectors defining the hexagonal basis.
    """
    def __init__(self, maxsize_n: int, maxsize_m: int, 
                 radius: Optional[float] = None, 
                 lattice_vectors: Optional[List[Tuple[float, float]]] = None,
                 intralayer_dist_threshold: float = 1.0) -> None:
        """
        Initialize a hex_lattice instance.

        Args:
            maxsize_n (int): Range of lattice nodes in the N-direction.
            maxsize_m (int): Range of lattice nodes in the M-direction.
            radius (Optional[float]): Optional max_distance cutoff to exclude distant nodes.
            lattice_vectors (Optional[List[Tuple[float, float]]]): List of lattice vectors defining the hexagonal grid.
                Defaults to constants.v1 and constants.v2 if None.
            intralayer_dist_threshold (float): Threshold for connecting intralayer nodes.
            
        Raises:
            constants.physics_parameter_error: If parameters are invalid.
            constants.graph_construction_error: If lattice construction fails.
        """
        try:
            constants.validate_positive_number(maxsize_n, "maxsize_n")
            constants.validate_positive_number(maxsize_m, "maxsize_m")
            constants.validate_positive_number(intralayer_dist_threshold, "intralayer_dist_threshold")
            
            if radius is not None:
                constants.validate_positive_number(radius, "radius")
            
            if lattice_vectors is None:
                lattice_vectors = [constants.v1, constants.v2]
            elif len(lattice_vectors) != 2:
                raise constants.physics_parameter_error("lattice_vectors must contain exactly 2 vectors")
            
            self.graph = graph()  # Create a new graph instance
            self.max_rows = maxsize_n  # Number of nodes in the N direction
            self.max_cols = maxsize_m  # Number of nodes in the M direction
            self.lattice_vectors = lattice_vectors
            self.create_lattice(radius, intralayer_dist_threshold)
        except (constants.physics_parameter_error, constants.graph_construction_error):
            raise
        except Exception as e:
            logger.error(f"Unexpected error in hex_lattice initialization: {str(e)}")
            raise constants.graph_construction_error(f"Failed to initialize hex_lattice: {str(e)}")

    def create_lattice(self, radius: Optional[float] = None, intralayer_dist_threshold: float = 1.0) -> None:
        """
        Populate the graph with hexagonally arranged nodes and add neighbors that are of distance smaller than intralayer_dist_threshold.
        When connecting the nodes, we will be checking all the neighbors of the form i*v_1 + j*v_2,
        for i and j smaller than r/(min vector length), where min vector length is the smallest
        vector (in norm) out of v_1, v_2, v_1+v_2, v_1-v_2.
        
        Args:
            radius (Optional[float]): Only nodes within this radius are included.
            intralayer_dist_threshold (float): Threshold for connecting intralayer nodes.
            
        Raises:
            constants.physics_parameter_error: If parameters are invalid.
            constants.graph_construction_error: If lattice construction fails.
        """
        try:
            if radius is not None:
                constants.validate_positive_number(radius, "radius")
            constants.validate_positive_number(intralayer_dist_threshold, "intralayer_dist_threshold")
            
            lattice_vector_array = [constants.np.array(self.lattice_vectors[0]), constants.np.array(self.lattice_vectors[1])]
            min_vector_length = constants.np.min([constants.np.linalg.norm(lattice_vector_array[0]),
                                                 constants.np.linalg.norm(lattice_vector_array[1]),
                                                 constants.np.linalg.norm(lattice_vector_array[0]+lattice_vector_array[1]),
                                                 constants.np.linalg.norm(lattice_vector_array[0]-lattice_vector_array[1])])
            
            if min_vector_length <= 0:
                raise constants.physics_parameter_error("Lattice vectors are too small or invalid")
            max_index = int(constants.np.ceil(constants.safe_divide(intralayer_dist_threshold, min_vector_length, 1.0)))
            directions = []
            for ind_i in range(-max_index, max_index+1):
                for ind_j in range(-max_index, max_index+1):
                    vector = ind_i*lattice_vector_array[0] + ind_j*lattice_vector_array[1]
                    if constants.np.linalg.norm(vector) <= intralayer_dist_threshold:
                        directions.append((ind_i, ind_j))
            
            for y in range(-self.max_rows,self.max_rows+1):
                for x in range(-self.max_cols,self.max_cols+1):
                    position = (x*self.lattice_vectors[0][0] + y*self.lattice_vectors[1][0], x*self.lattice_vectors[0][1] + y*self.lattice_vectors[1][1])
                    current_node = node(position, (x,y))
                    if radius is None or calculate_distance(current_node) <= radius:  # Check if the node is within the specified radius
                        self.graph.add_node(current_node)
            valid_indices = set()
            for graph_node in self.graph.nodes:
                valid_indices.add(graph_node.lattice_index)
            for graph_node in self.graph.nodes:
                x, y = graph_node.lattice_index  # Get the index of the node
                for dx, dy in directions:
                    neighbor_index = (x + dx, y + dy)
                    if neighbor_index in valid_indices:
                        neighbor = self.graph.get_node_by_index(neighbor_index)
                        self.graph.add_edge(graph_node, neighbor)
        except (constants.physics_parameter_error, constants.graph_construction_error):
            raise
        except Exception as e:
            logger.error(f"Unexpected error in create_lattice: {str(e)}")
            raise constants.graph_construction_error(f"Failed to create lattice: {str(e)}")
    def rotate(self, angle: float) -> None:
        """
        Rotate the entire lattice by a specified angle.

        Args:
            angle (float): Angle in radians to rotate the lattice.
        """
        cos_angle = constants.np.cos(angle)
        sin_angle = constants.np.sin(angle)
        for node in self.graph.nodes:
            x, y = node.position
            new_x = x * cos_angle - y * sin_angle
            new_y = x * sin_angle + y * cos_angle
            node.position = (new_x, new_y)

    def rotate_rational_TBG(self, a: int, b: int, N: float, alpha: float) -> None: 
        """
        Rotate the lattice by a rational TBG angle derived from integer parameters.

        Args:
            a (int): Integer parameter for the cosine term.
            b (int): Integer parameter for the sine term.
            N (float): Norm derived from a and b.
            alpha (float): Scaling factor based on (a, b).

        Raises:
            ValueError: If parameters violate validity conditions.
        """
        validate_ab(a,b)
        cos_angle = a / (alpha * N)
        sin_angle = constants.np.sqrt(3) * b / (alpha * N)
        
        for node in self.graph.nodes:
            x, y = node.position
            new_x = x * cos_angle - y * sin_angle
            new_y = x * sin_angle + y * cos_angle
            node.position = (new_x, new_y)
    def cleanup(self) -> None:
        """Clean up hex_lattice by cleaning its graph."""
        if hasattr(self, 'graph') and self.graph is not None:
            self.graph.cleanup()
            self.graph = None
        self.lattice_vectors = None
        self.max_rows = None
        self.max_cols = None

    def __enter__(self) -> 'hex_lattice':
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Optional[type], exc_value: Optional[Exception], traceback: Optional[Any]) -> bool:
        """Context manager exit with cleanup."""
        self.cleanup()
        return False
class tbg :
    """
    TBG (Twisted Bilayer Graphene) class that constructs a bilayer structure
    with adjustable rotation angle and interlayer coupling.

    Attributes:
        a (int): Integer controlling the rational rotation angle.
        b (int): Integer controlling the rational rotation angle.
        N (float): Computed normalization factor for rotation.
        alpha (float): Scale factor based on arithmetic conditions.
        factor (float): Adjustment factor depending on a.
        lattice_vectors (List[Tuple[float, float]]): Lattice vectors for periodicity.
        dual_vectors (List[Tuple[float, float]]): Reciprocal lattice vectors.
        top_layer (hex_lattice): Rotated hexagonal top layer.
        bottom_layer (hex_lattice): Rotated hexagonal bottom layer.
        full_graph (graph): Combined graph of top and bottom layers.
    """
    def __init__(self, maxsize_n: int, maxsize_m: int, a: int, b: int,
                 interlayer_dist_threshold: float = 1.0,
                 intralayer_dist_threshold: float = 1.0,
                 unit_cell_radius_factor: int = 1) -> None:
        """Initialize the TBG structure with a given size and rational rotation.

        Args:
            maxsize_n (int): Number of nodes in the N direction.
            maxsize_m (int): Number of nodes in the M direction.
            a (int): Integer rotation parameter.
            b (int): Integer rotation parameter.
            interlayer_dist_threshold (float): Threshold for connecting interlayer nodes.
            intralayer_dist_threshold (float): Threshold for connecting intralayer nodes.
            unit_cell_radius_factor (int): Controls radius scaling of node interaction.

        Raises:
            constants.physics_parameter_error: If invalid parameters are provided.
            constants.graph_construction_error: If TBG construction fails.
        """
        try:
            # Validate inputs
            constants.validate_positive_number(maxsize_n, "maxsize_n")
            constants.validate_positive_number(maxsize_m, "maxsize_m")
            constants.validate_positive_number(interlayer_dist_threshold, "interlayer_dist_threshold")
            constants.validate_positive_number(intralayer_dist_threshold, "intralayer_dist_threshold")
            constants.validate_positive_number(unit_cell_radius_factor, "unit_cell_radius_factor")
            
            if not isinstance(maxsize_n, int) or not isinstance(maxsize_m, int):
                raise constants.physics_parameter_error("maxsize_n and maxsize_m must be integers")
            if not isinstance(unit_cell_radius_factor, float):
                raise constants.physics_parameter_error("unit_cell_radius_factor must be a float")
            
            validate_ab(a, b)
            
            self.a = a
            self.b = b
            
            # Calculate the constants N and alpha for the rational TBG angle
            self.N, self.alpha, self.factor, _ = compute_twist_constants(self.a, self.b)
            logger.info(f"Calculated N: {self.N}, alpha: {self.alpha} for a={self.a}, b={self.b}")
            
            # Create the top layer of the TBG
            self.top_layer = hex_lattice(maxsize_n, maxsize_m,
                                       unit_cell_radius_factor*self.factor*self.N,
                                       [constants.v1, constants.v2],
                                       intralayer_dist_threshold)
            self.top_layer.rotate_rational_TBG(self.a, self.b, self.N, self.alpha)
            
            # Create the bottom layer of the TBG  
            self.bottom_layer = hex_lattice(maxsize_n, maxsize_m,
                                          unit_cell_radius_factor*self.factor*self.N,
                                          [constants.v1, constants.v2],
                                          intralayer_dist_threshold)
            self.bottom_layer.rotate_rational_TBG(self.a, -self.b, self.N, self.alpha)

            # Create a copy of the top layer graph
            self.full_graph = self.top_layer.graph.copy(0)
            self.full_graph.append(self.bottom_layer.graph, 1)

            self._connect_layers(interlayer_dist_threshold)

            # Calculate the primitive vectors of the TBG structure
            if self.a % 3 == 0:
                self.lattice_vectors = [tuple(item * self.N for item in constants.k1), 
                                      tuple(item * self.N for item in constants.k2)]
                self.dual_vectors = [tuple(constants.safe_divide(item, self.N, 0.0) for item in constants.v1), 
                                   tuple(constants.safe_divide(item, self.N, 0.0) for item in constants.v2)]
            else:
                # If a is not a multiple of 3, use the hexagonal lattice vectors
                self.lattice_vectors = [tuple(item * self.N for item in constants.v1), 
                                      tuple(item * self.N for item in constants.v2)]
                self.dual_vectors = [tuple(constants.safe_divide(item, self.N, 0.0) for item in constants.k1), 
                                   tuple(constants.safe_divide(item, self.N, 0.0) for item in constants.k2)]
            
            logger.info(f"Lattice vectors: ({self.lattice_vectors[0][0]:.2f},{self.lattice_vectors[0][1]:.2f})"
                        f"({self.lattice_vectors[1][0]:.2f},{self.lattice_vectors[1][1]:.2f}),"
                        f" and the duals are ({self.dual_vectors[0][0]:.2f},{self.dual_vectors[0][1]:.2f})"
                        f"({self.dual_vectors[1][0]:.2f},{self.dual_vectors[1][1]:.2f})")
                        
        except (constants.physics_parameter_error, constants.graph_construction_error):
            raise
        except Exception as e:
            logger.error(f"Unexpected error in tbg.__init__: {str(e)}")
            raise constants.graph_construction_error(f"Failed to initialize TBG system: {str(e)}")

    def _connect_layers(self, interlayer_dist_threshold: float = 1.0) -> None:
        """
        A symmetric subfunction to connect the two layers, using direct distance checks.
        Args:
            interlayer_dist_threshold (float): Threshold for connecting interlayer nodes.
            
        Raises:
            constants.graph_construction_error: If layer connection fails.
        """
        try:
            constants.validate_positive_number(interlayer_dist_threshold, "interlayer_dist_threshold")
            
            if not hasattr(self, 'top_layer') or not hasattr(self, 'bottom_layer'):
                raise constants.graph_construction_error("Layers must be created before connecting them")
            
            top_nodes = self.top_layer.graph.nodes
            bottom_nodes = self.bottom_layer.graph.nodes
            
            # If there is an empty layer- exit
            if not top_nodes or not bottom_nodes:
                logger.warning("One or both layers are empty, no interlayer connections made")
                return
            
            # Create spatial index for both layers
            top_positions = constants.np.array([node.position for node in top_nodes])
            bottom_positions = constants.np.array([node.position for node in bottom_nodes])

            # Create KD-tree for top layer to search for neighbors
            top_tree = cKDTree(top_positions)

            connections_made = 0
            connections_set = set()  # To avoid duplicate connections
            
            # Process in batches for memory efficiency
            batch_size = min(1000, len(bottom_nodes))
            logger.info(f"Processing {len(bottom_nodes)} bottom nodes against {len(top_nodes)} top nodes")

            for batch_start in range(0, len(bottom_nodes), batch_size):
                batch_end = min(batch_start + batch_size, len(bottom_nodes))
                batch_positions = bottom_positions[batch_start:batch_end]
                
                # Find all top neighbors within threshold for this batch
                neighbor_indices = top_tree.query_ball_point(batch_positions, r=interlayer_dist_threshold)
                
                # Process connections for this batch
                for i, top_neighbors in enumerate(neighbor_indices):
                    if not top_neighbors:
                        continue
                        
                    bottom_idx = batch_start + i
                    bottom_node = bottom_nodes[bottom_idx]
                    bottom_node_in_graph = self.full_graph.get_node_by_index(bottom_node.lattice_index, 1)
                    
                    for top_idx in top_neighbors:
                        top_node = top_nodes[top_idx]
                        top_node_in_graph = self.full_graph.get_node_by_index(top_node.lattice_index, 0)
                        
                        # Create a unique connection identifier to avoid duplicates
                        connection_id = (min(bottom_idx, top_idx), max(bottom_idx, top_idx), 
                                       0 if bottom_idx < top_idx else 1)
                        
                        if connection_id not in connections_set:
                            self.full_graph.add_edge(bottom_node_in_graph, top_node_in_graph)
                            connections_set.add(connection_id)
                            connections_made += 1
                
                # Progress logging for large systems
                if len(bottom_nodes) > 1000:
                    progress = (batch_end / len(bottom_nodes)) * 100
                    if progress % 20 < (batch_size / len(bottom_nodes)) * 100:
                        logger.info(f"Interlayer connections: {progress:.0f}% complete")
            
            logger.info(f"Created {connections_made} unique interlayer connections")
            
        except (constants.physics_parameter_error, constants.graph_construction_error):
            raise
        except Exception as e:
            logger.error(f"Unexpected error in _connect_layers: {str(e)}")
            raise constants.graph_construction_error(f"Failed to connect layers: {str(e)}")


    def __repr__(self) -> str:
        s = "Upper Layer:\n"
        s += self.top_layer.graph.__repr__()
        s += "Bottom Layer:\n"
        s += self.bottom_layer.graph.__repr__()
        return s
    
    def plot(self, ax: Optional[constants.matplotlib.axes.Axes] = None, plot_color_top: str = 'b',
              plot_color_bottom: str = 'r', plot_color_full: str = 'g',
              radius: Optional[float] = None, plot_separate_layers: bool = False,
              lattice_vectors: Optional[List[Tuple[float, float]]] = None) -> None:
        """
        Plot the full TBG structure or its layers separately.

        Args:
            ax (Optional[matplotlib.axes.Axes]): Axis object to draw the plot. If None, creates new figure.
            plot_color_top (str): Color for the top layer.
            plot_color_bottom (str): Color for the bottom layer.
            plot_color_full (str): Color for edges connecting the two layers.
            radius (Optional[float]): Optional cutoff for visible nodes.
            plot_separate_layers (bool): Whether to plot layers separately or together.
            lattice_vectors (Optional[List[Tuple[float, float]]]): The list of the lattice vectors.
            
        Raises:
            constants.tbg_error: If plotting fails.
        """
        try:
            if radius is not None:
                constants.validate_positive_number(radius, "radius")
            
            if not hasattr(self, 'full_graph') or self.full_graph is None:
                raise constants.tbg_error("TBG system not properly initialized for plotting")
                
            dirac_plotter.plot_tbg_structure(ax, self, plot_color_top, plot_color_bottom, 
                                           plot_color_full, radius, plot_separate_layers, lattice_vectors)
        except constants.tbg_error:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in plot: {str(e)}")
            raise constants.tbg_error(f"Failed to plot TBG structure: {str(e)}")

    def plot_unit_cell(self, plot_color_top: str = 'b', plot_color_bottom: str = 'r',
                        plot_color_full: str = 'g', lattice_vectors: Optional[List[Tuple[float, float]]] = None) -> None:
        """
        Plot only the unit cell of the TBG structure.

        Args:
            plot_color_top (str): Color for the top layer.
            plot_color_bottom (str): Color for the bottom layer.
            plot_color_full (str): Color for full graph.
            lattice_vectors (Optional[List[Tuple[float, float]]]): The list of the lattice vectors.
            
        Raises:
            constants.tbg_error: If plotting fails.
        """
        try:
            if not hasattr(self, 'N') or self.N is None:
                raise constants.tbg_error("TBG system not properly initialized - missing N parameter")
            return self.plot(plot_color_top=plot_color_top, plot_color_bottom=plot_color_bottom, 
                            plot_color_full=plot_color_full, radius=self.N/2, lattice_vectors=lattice_vectors)
        except constants.tbg_error:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in plot_unit_cell: {str(e)}")
            raise constants.tbg_error(f"Failed to plot unit cell: {str(e)}")
    def cleanup(self) -> None:
        """Clean up TBG by properly cleaning all layers and graphs.
        
        Raises:
            constants.tbg_error: If cleanup encounters errors.
        """
        try:
            # Clean in reverse order of creation
            if hasattr(self, 'full_graph') and self.full_graph is not None:
                self.full_graph.cleanup()
                self.full_graph = None

            if hasattr(self, 'top_layer') and self.top_layer is not None:
                self.top_layer.cleanup()
                self.top_layer = None

            if hasattr(self, 'bottom_layer') and self.bottom_layer is not None:     
                self.bottom_layer.cleanup()
                self.bottom_layer = None

            self.lattice_vectors = None
            self.dual_vectors = None
            self.a = None
            self.b = None
            self.N = None
            self.alpha = None
            self.factor = None
            
            logger.debug("TBG cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during TBG cleanup: {str(e)}")
            raise constants.tbg_error(f"Failed to cleanup TBG system: {str(e)}")

    def __enter__(self) -> 'tbg':
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Optional[type], exc_value: Optional[Exception], traceback: Optional[Any]) -> bool:
        """Context manager exit with cleanup."""
        self.cleanup()
        return False


class Dirac_analysis:
    """
    Specialized class for Dirac point analysis in periodic graphs.
    
    This class contains all methods related to analyzing the quality of Dirac points
    by computing band gap, R² loss, and isotropy metrics. This is physics-specific
    analysis relevant to TBG and other condensed matter systems.
    
    Attributes:
        periodic_graph: The periodic_graph object this analyzer operates on
    """
    
    def __init__(self, periodic_graph_obj: 'periodic_graph'):
        """
        Initialize Dirac analyzer with a periodic graph.
        
        Args:
            periodic_graph_obj: The periodic_graph object to analyze
            
        Raises:
            constants.physics_parameter_error: If invalid periodic graph is provided.
        """
        try:
            if periodic_graph_obj is None:
                raise constants.physics_parameter_error("periodic_graph_obj cannot be None")
            
            # Basic validation that the object has required methods
            required_methods = ['compute_energy_and_derivative_of_energy_at_a_point']
            for method in required_methods:
                if not hasattr(periodic_graph_obj, method):
                    raise constants.physics_parameter_error(f"periodic_graph_obj missing required method: {method}")
            
            self.periodic_graph = periodic_graph_obj
            
        except constants.physics_parameter_error:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in Dirac_analysis.__init__: {str(e)}")
            raise constants.physics_parameter_error(f"Failed to initialize Dirac analyzer: {str(e)}")

    def _compute_band_gap_metric(self, momentum: tuple[float, float], lower_band_index: int) -> tuple[float, float, float]:
        """
        Compute band gap and its gradient at a point.
        
        Args:
            momentum: K-point coordinates in relative coordinates
            lower_band_index: Index of the lower band
            
        Returns:
            tuple: (band_gap, gap_derivative_x, gap_derivative_y)
            
        Raises:
            constants.physics_parameter_error: If invalid parameters are provided.
            constants.matrix_operation_error: If band calculations fail.
        """
        try:
            if not isinstance(momentum, tuple) or len(momentum) != 2:
                raise constants.physics_parameter_error("momentum must be a tuple of 2 floats")
            
            if not isinstance(lower_band_index, int) or lower_band_index < 0:
                raise constants.physics_parameter_error("lower_band_index must be a non-negative integer")
            
            eigvals, der_x_gap, der_y_gap = self.periodic_graph.compute_energy_and_derivative_of_energy_at_a_point(momentum, lower_band_index)
            
            if len(eigvals) < 2:
                raise constants.matrix_operation_error(f"Need at least 2 eigenvalues for gap calculation, got {len(eigvals)}")
            
            band_gap = abs(eigvals[1] - eigvals[0])
            
            if eigvals[1] > eigvals[0]:
                sign_of_gap = 1
            else:
                sign_of_gap = -1
            
            gap_der_x = sign_of_gap * constants.np.real(der_x_gap)
            gap_der_y = sign_of_gap * constants.np.real(der_y_gap)
            
            return band_gap, gap_der_x, gap_der_y
            
        except (constants.physics_parameter_error, constants.matrix_operation_error):
            raise
        except Exception as e:
            logger.error(f"Unexpected error in _compute_band_gap_metric: {str(e)}")
            raise constants.matrix_operation_error(f"Failed to compute band gap metric: {str(e)}")

    def _transform_gradients_to_lattice(self, result_der_x: list, result_der_y: list, 
                                      mu_v_der_x: float, mu_v_der_y: float) -> tuple[list, list, constants.np.ndarray]:
        """
        Transform Cartesian gradients to lattice coordinates.
        
        Args:
            result_der_x: List of x-derivatives in Cartesian coordinates
            result_der_y: List of y-derivatives in Cartesian coordinates
            mu_v_der_x: Mean velocity x-derivative
            mu_v_der_y: Mean velocity y-derivative
            
        Returns:
            tuple: (result_der_n_1, result_der_n_2, grad_mu_v_lattice)
        """
        B = constants.np.array([self.periodic_graph.dual_vectors[0], self.periodic_graph.dual_vectors[1]]).T
        B_inv = constants.np.linalg.inv(B)
        
        result_der_n_1 = []
        result_der_n_2 = []
        for i in range(len(result_der_x)):
            grad_cartesian = constants.np.array([result_der_x[i], result_der_y[i]])
            grad_lattice = B_inv.T @ grad_cartesian
            result_der_n_1.append(grad_lattice[0])
            result_der_n_2.append(grad_lattice[1])
        
        grad_mu_v_cartesian = constants.np.array([mu_v_der_x, mu_v_der_y])
        grad_mu_v_lattice = B_inv.T @ grad_mu_v_cartesian
        
        return result_der_n_1, result_der_n_2, grad_mu_v_lattice

    def _compute_r2_loss_metric(self, all_r2_values: list, all_r2_der_x: list, all_r2_der_y: list) -> tuple[float, float, float]:
        """
        Compute R² loss metric and its derivatives.
        
        Args:
            all_r2_values: List of R² values from all directions  
            all_r2_der_x: List of R² x-derivatives from all directions
            all_r2_der_y: List of R² y-derivatives from all directions
            
        Returns:
            tuple: (R2_loss, R2_loss_der_x, R2_loss_der_y)
        """
        mean_R2 = constants.np.mean(all_r2_values)
        R2_loss = 1 - mean_R2
        R2_loss_der_x = -constants.np.mean(all_r2_der_x)
        R2_loss_der_y = -constants.np.mean(all_r2_der_y)
        
        return R2_loss, R2_loss_der_x, R2_loss_der_y

    def _compute_isotropy_metric(self, all_velocities: list, all_vel_der_x: list, all_vel_der_y: list) -> tuple[float, float, float, float, float, float]:
        """
        Compute isotropy loss (σ_v/μ_v) and its derivatives.
        
        Args:
            all_velocities: List of velocity values from all directions
            all_vel_der_x: List of velocity x-derivatives from all directions  
            all_vel_der_y: List of velocity y-derivatives from all directions
            
        Returns:
            tuple: (iso_loss, iso_loss_der_x, iso_loss_der_y, mu_v, mu_v_der_x, mu_v_der_y)
        """
        velocity_array = constants.np.array(all_velocities)
        vel_der_x_array = constants.np.array(all_vel_der_x)
        vel_der_y_array = constants.np.array(all_vel_der_y)
        
        # Basic statistics
        mu_v = constants.np.mean(velocity_array)
        sigma_v = constants.np.std(velocity_array, ddof=0)  # Population std dev
        
        if mu_v < 1e-15:  # Essentially zero mean velocity
            iso_loss = 1.0  # Maximum isotropy loss
            iso_loss_der_x = 0.0
            iso_loss_der_y = 0.0
            mu_v_der_x = 0.0
            mu_v_der_y = 0.0
        else:
            iso_loss = sigma_v / mu_v
            
            # Compute derivatives using analytical formula from documentation
            # ∇σ involves terms with (vⱼ - μᵥ), so we need careful computation
            N_total = len(velocity_array)
            
            # Mean derivatives
            mu_v_der_x = constants.np.mean(vel_der_x_array)
            mu_v_der_y = constants.np.mean(vel_der_y_array)
            
            if sigma_v < 1e-15:  # All velocities are equal
                iso_loss_der_x = 0.0
                iso_loss_der_y = 0.0
            else:
                # From the analytical derivation:
                # ∇(σᵥ/μᵥ) = (1/σᵥ)[1/μᵥ * (1/N)Σvⱼ∇vⱼ - ((σᵥ/μᵥ)² + 1) * (1/N)Σ∇vⱼ]
                
                v_times_der_x_mean = constants.np.sum(velocity_array * vel_der_x_array)/N_total
                v_times_der_y_mean = constants.np.sum(velocity_array * vel_der_y_array)/N_total
                
                iso_loss_der_x = (1/sigma_v) * (
                    (1/mu_v) * v_times_der_x_mean - (iso_loss**2 + 1) * mu_v_der_x
                )
                iso_loss_der_y = (1/sigma_v) * (
                    (1/mu_v) * v_times_der_y_mean - (iso_loss**2 + 1) * mu_v_der_y
                )
        
        return iso_loss, iso_loss_der_x, iso_loss_der_y, mu_v, mu_v_der_x, mu_v_der_y

    def _sample_energies_along_direction(self, momentum: tuple[float, float], direction_lattice: constants.np.ndarray, 
                                       delta_k: float, lower_band_index: int) -> tuple[constants.np.ndarray, constants.np.ndarray, constants.np.ndarray, constants.np.ndarray, constants.np.ndarray]:
        """
        Sample band energies and derivatives along a specific direction.
        
        Args:
            momentum: Center momentum point in relative coordinates
            direction_lattice: Direction vector in lattice coordinates
            delta_k: Step size for sampling
            lower_band_index: Index of the lower band
            
        Returns:
            tuple: (k_array, e1_array, e2_array, der_x_array, der_y_array)
        """
        k_samples = []
        e1_samples = []  # Lower band energies
        e2_samples = []  # Upper band energies
        der_x_samples = []
        der_y_samples = []
        
        for i in range(-constants.points_num, constants.points_num+1):
            delta = i * delta_k 
            k_point = (momentum[0] + delta * direction_lattice[0], momentum[1] + delta * direction_lattice[1]) 
            eigvals_i, der_x_i, der_y_i = self.periodic_graph.compute_energy_and_derivative_of_energy_at_a_point(k_point, lower_band_index)
            
            if delta >= 0:
                eigvals_i = eigvals_i[::-1]  # make sure that we use the same band - even across the dirac point (the upper band becomes the lower one - as they cross linearly)
                # Note: derivatives are for the gap, so sign flips but magnitude preserved
                der_x_i = -der_x_i  
                der_y_i = -der_y_i
                
            k_samples.append(delta)
            e1_samples.append(eigvals_i[0])  # Lower band
            e2_samples.append(eigvals_i[1])  # Upper band
            der_x_samples.append(constants.np.real(der_x_i))
            der_y_samples.append(constants.np.real(der_y_i))

        # Convert to numpy arrays
        k_array = constants.np.array(k_samples)
        e1_array = constants.np.array(e1_samples)
        e2_array = constants.np.array(e2_samples)
        der_x_array = constants.np.array(der_x_samples)
        der_y_array = constants.np.array(der_y_samples)
        
        return k_array, e1_array, e2_array, der_x_array, der_y_array

    def _analyze_direction_fits(self, directions: list, momentum: tuple[float, float], delta_k: float, 
                              lower_band_index: int, visual_flag: bool) -> dict:
        """
        Analyze linear fits in all crystallographic directions.
        
        Args:
            directions: List of direction vectors in Cartesian coordinates
            momentum: Center momentum point in relative coordinates
            delta_k: Step size for sampling
            lower_band_index: Index of the lower band
            visual_flag: Whether to plot dispersion fits
            
        Returns:
            dict: Dictionary containing all velocities, R² values, and derivatives
        """
        # Storage for all direction results
        all_velocities = []
        all_R2_values = []
        all_R2_der_x = []
        all_R2_der_y = []
        all_vel_der_x = []
        all_vel_der_y = []
        
        for direction in directions: 
            B = constants.np.array([self.periodic_graph.dual_vectors[0], self.periodic_graph.dual_vectors[1]]).T  # dual vectors in Cartesian
            B_inv = constants.np.linalg.inv(B)  # inverse to go from Cartesian to lattice
            direction_lattice = B_inv @ direction  # Convert to lattice coordinates - since we work in relative coordinates. 
            
            # Sample energies along this direction
            k_array, e1_array, e2_array, der_x_array, der_y_array = self._sample_energies_along_direction(
                momentum, direction_lattice, delta_k, lower_band_index)
            
            # Compute linear fits and derivatives for both bands
            try:
                # Lower band analysis
                (slope1, intercept1), R2_1, R2_der_x_1, R2_der_y_1, vel1, vel_der_x_1, vel_der_y_1 = self.compute_R2_and_velocity_and_der(k_array, e1_array, der_x_array, der_y_array)
                
                # Upper band analysis  
                (slope2, intercept2), R2_2, R2_der_x_2, R2_der_y_2, vel2, vel_der_x_2, vel_der_y_2 = self.compute_R2_and_velocity_and_der(k_array, e2_array, der_x_array, der_y_array)
                
                # Store results for both bands
                all_velocities.extend([vel1, vel2])
                all_R2_values.extend([R2_1, R2_2])
                all_R2_der_x.extend([R2_der_x_1, R2_der_x_2])
                all_R2_der_y.extend([R2_der_y_1, R2_der_y_2])
                all_vel_der_x.extend([vel_der_x_1, vel_der_x_2])
                all_vel_der_y.extend([vel_der_y_1, vel_der_y_2])
                
                # Optional visualization
                if visual_flag:
                    self._plot_fit_of_dirac_point(slope1, intercept1, slope2, intercept2,
                                                k_array, e1_array, e2_array, momentum, direction)
            
            except Exception as e:
                print(f"Warning: Failed to fit direction {direction}: {e}")
                # Add placeholder values to maintain array structure
                all_velocities.extend([0.0, 0.0])
                all_R2_values.extend([0.0, 0.0])
                all_R2_der_x.extend([0.0, 0.0])
                all_R2_der_y.extend([0.0, 0.0])
                all_vel_der_x.extend([0.0, 0.0])
                all_vel_der_y.extend([0.0, 0.0])
        
        return {
            'velocities': all_velocities,
            'r2_values': all_R2_values,
            'r2_der_x': all_R2_der_x,
            'r2_der_y': all_R2_der_y,
            'vel_der_x': all_vel_der_x,
            'vel_der_y': all_vel_der_y
        }

    def check_Dirac_point(self, Momentum:tuple[float,float],lower_band_index:int=1,inter_graph_weight:float=DEFAULT_PARAMS['inter_graph_weight'],
                          intra_graph_weight:float=DEFAULT_PARAMS['intra_graph_weight'],
                          delta_k:float=constants.NUMERIC_RANGE_TO_CHECK,visual_flag=False)->tuple:
        """
        Analyze the quality of a Dirac point by computing three metrics and their derivatives.
        
        The three metrics are:
        1. Band gap: |E₁(k) - E₀(k)| at the suspected Dirac point
        2. R² loss:  1-mean(R²) of linear fits in all directions  
        3. Isotropy loss: σ_v/μ_v (coefficient of variation of velocities)
        
        Args:
            Momentum (tuple): Suspected Dirac point coordinates in relative coordinates
            lower_band_index (int): Index of lower band (Dirac between this and next band)
            inter_graph_weight (float): Inter-layer coupling strength
            intra_graph_weight (float): Intra-layer coupling strength  
            delta_k (float): Momentum step size for linear fits (default from constants)
            visual_flag (bool): Whether to plot dispersion fits
            
        Returns:
            tuple: (metrics, ∂metrics/∂k1, ∂metrics/∂k2,mean_velocity) where each of the first 3 is a 3-element list
        """
        # Ensure adjacency matrix is built
        if self.periodic_graph.adj_matrix is None or self.periodic_graph.periodic_edges is None:
            self.periodic_graph.build_adj_matrix(inter_graph_weight,intra_graph_weight)
        result=[]
        result_der_x=[]
        result_der_y=[]

        # === Metric 1: Band gap at the Dirac point ===
        band_gap, gap_der_x, gap_der_y = self._compute_band_gap_metric(Momentum, lower_band_index)
        result.append(band_gap)
        result_der_x.append(gap_der_x)
        result_der_y.append(gap_der_y)
        # === Metrics 2 & 3: Linear fits in multiple directions ===
        # We need to check only up to pi/3 due to the symmetry
        directions = [constants.np.array([constants.np.cos(θ), constants.np.sin(θ)]) for θ in constants.np.linspace(0, constants.np.pi/3, constants.direction_num, endpoint=False)] 
        
        direction_results = self._analyze_direction_fits(directions, Momentum, delta_k, lower_band_index, visual_flag)
        all_velocities = direction_results['velocities']
        all_R2_values = direction_results['r2_values']
        all_R2_der_x = direction_results['r2_der_x']
        all_R2_der_y = direction_results['r2_der_y']
        all_vel_der_x = direction_results['vel_der_x']
        all_vel_der_y = direction_results['vel_der_y']
    
        # === Metric 2: Mean R² of linear fits ===
        R2_loss, R2_loss_der_x, R2_loss_der_y = self._compute_r2_loss_metric(all_R2_values, all_R2_der_x, all_R2_der_y)
        result.append(R2_loss)
        result_der_x.append(R2_loss_der_x)
        result_der_y.append(R2_loss_der_y)
        # === Metric 3: Isotropy loss ===
        iso_loss, iso_loss_der_x, iso_loss_der_y, mu_v, mu_v_der_x, mu_v_der_y = self._compute_isotropy_metric(all_velocities, all_vel_der_x, all_vel_der_y)
        result.append(iso_loss)
        result_der_x.append(iso_loss_der_x)
        result_der_y.append(iso_loss_der_y)

        
        # Transform gradients from Cartesian to lattice coordinates  
        result_der_n_1, result_der_n_2, grad_mu_v_lattice = self._transform_gradients_to_lattice(
            result_der_x, result_der_y, mu_v_der_x, mu_v_der_y)
        
        return result, result_der_n_1, result_der_n_2, mu_v, grad_mu_v_lattice

    def _plot_fit_of_dirac_point(self, slope_1: float, intercept_1: float, slope_2: float, intercept_2: float, k_array: constants.np.ndarray, 
                                 e1_array: constants.np.ndarray, e2_array: constants.np.ndarray, Momentum: tuple[float, float], direction: constants.np.ndarray):
        """
        Plot the linear fits of both bands around a Dirac point.
        
        Args:
            slope_1: Slope of lower band fit
            intercept_1: Intercept of lower band fit  
            slope_2: Slope of upper band fit
            intercept_2: Intercept of upper band fit
            k_array: Array of momentum offsets
            e1_array: Lower band energies
            e2_array: Upper band energies
            Momentum: Dirac point coordinates in relative coordinates
            direction: Direction vector for the fit
        """
        dirac_plotter.plot_dispersion_fit(slope_1, intercept_1, slope_2, intercept_2,
                                       k_array, e1_array, e2_array, Momentum, direction)

    def compute_R2_and_velocity_and_der(self, k_values: constants.np.ndarray, e_values: constants.np.ndarray, der_x_of_E_array: constants.np.ndarray, der_y_of_E_array: constants.np.ndarray) -> tuple[tuple[float, float], float, float, float, float, float, float]:
        """
        Compute the R² value and its derivative for a linear fit to the dispersion relation.
    
        Uses the simplified analytical formulas derived in the documentation:
        - For 2N+1 data points sampled from -N to N
        - ρ = N(N+1)/3 where N is the half-width of the sampling range
        - E[f(k,E)] = (1/(2N+1)) * Σ_{i=-N}^N f(i, E_i)
        - SS_tot = E[(E - E[E])²]
        - R² = (1/ρ) * (E[kE])² / SS_tot
        - ∇R² = (2/SS_tot) * ((1/ρ)E[kE]E[k∇E] - R²(E[E∇E] - E[E]E[∇E]))
        - ∇v = sgn(slope)/ρ * E[k∇E]
        
        Args:
            k_values (constants.np.ndarray): Array of momentum offsets from -N*δ to N*δ
            e_values (constants.np.ndarray): Corresponding energy values
            der_x_of_E_array (constants.np.ndarray): Derivatives of energy w.r.t. kx
            der_y_of_E_array (constants.np.ndarray): Derivatives of energy w.r.t. ky
        
        Returns:
            tuple: ((slope, intercept), R², ∂R²/∂kx, ∂R²/∂ky, velocity, ∂v/∂kx, ∂v/∂ky)
        """
        # Input validation
        arrays = [k_values, e_values, der_x_of_E_array, der_y_of_E_array]
        if any(len(arr) == 0 for arr in arrays):
            raise ValueError("All input arrays must be non-empty")
    
        if not all(len(arr) == len(k_values) for arr in arrays):
            raise ValueError("All input arrays must have the same length")
        
        # Verify odd number of points (2N+1 structure)
        k_value_length = len(k_values)
        if k_value_length % 2 == 0:
            raise ValueError("Number of k-points must be odd (2N+1 structure)")
        # Compute basic linear fit using numpy (for validation/backup)
        A = constants.np.vstack([k_values, constants.np.ones(len(k_values))]).T
        result = constants.np.linalg.lstsq(A, e_values, rcond=None)
        slope, intercept = result[0]

        # === Core analytical computation ===
        N = (k_value_length - 1) // 2  # Half-width of sampling range
        # Create index array: [-N, -N+1, ..., -1, 0, 1, ..., N-1, N]
        i_values = constants.np.arange(-N, N + 1)
        # Verify this matches our k_values structure
        assert len(i_values) == k_value_length, "Index array length mismatch"
        # Fundamental parameters
        rho = N * (N + 1) / 3  # Variance of uniform distribution on [-N, N]
        # Expectation values (all normalized by 2N+1 = k_value_length)
        expected_E = constants.np.mean(e_values)
        expected_kE = constants.np.sum(i_values * e_values) / k_value_length

        # Derivatives expectations
        expected_der_x_E = constants.np.mean(der_x_of_E_array)
        expected_k_der_x_E = constants.np.sum(i_values * der_x_of_E_array) / k_value_length
        expected_der_y_E = constants.np.mean(der_y_of_E_array)
        expected_k_der_y_E = constants.np.sum(i_values * der_y_of_E_array) / k_value_length

        # Covariances: Cov(E, ∇E) = E[E∇E] - E[E]E[∇E]
        cov_E_der_E_x = constants.np.mean(e_values * der_x_of_E_array) - expected_E * expected_der_x_E
        cov_E_der_E_y = constants.np.mean(e_values * der_y_of_E_array) - expected_E * expected_der_y_E
        # Total sum of squares
        SS_tot = constants.np.sum((e_values - expected_E) ** 2)/k_value_length

        # === R² computation ===
        if SS_tot < 1e-15:  # Essentially zero variance
            R2 = 1.0  # Perfect fit to constant
            R2_der_x = 0.0
            R2_der_y = 0.0
        else:
            # R² = (1/ρ) * (E[kE])² / SS_tot
            R2 = (expected_kE**2) / (rho * SS_tot)
            
            # ∇R² = (2/SS_tot) * [(1/ρ)E[kE]E[k∇E] - R²*Cov(E,∇E)]
            R2_der_x = (2 / SS_tot) * (expected_kE * expected_k_der_x_E / rho - R2 * cov_E_der_E_x)
            R2_der_y = (2 / SS_tot) * (expected_kE * expected_k_der_y_E / rho - R2 * cov_E_der_E_y)
        # === Velocity computation ===
        velocity = constants.np.abs(slope)
        
        # Handle the sign function carefully for small slopes
        if constants.np.abs(slope) < 1e-15:
            # For very small slopes, set derivatives to zero to avoid numerical issues
            velocity_der_x = 0.0
            velocity_der_y = 0.0
        else:
            # ∇v = sgn(slope)/ρ * E[k∇E]
            sign_slope = constants.np.sign(slope)
            velocity_der_x = sign_slope * expected_k_der_x_E / rho
            velocity_der_y = sign_slope * expected_k_der_y_E / rho
        
        return (slope, intercept), R2, R2_der_x, R2_der_y, velocity, velocity_der_x, velocity_der_y

    def cleanup(self) -> None:
        """Clean up DiracAnalysis resources."""
        self.periodic_graph = None

    def __enter__(self) -> 'Dirac_analysis':
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Optional[type], exc_value: Optional[Exception], traceback: Optional[Any]) -> bool:
        """Context manager exit with cleanup."""
        self.cleanup()
        return False


