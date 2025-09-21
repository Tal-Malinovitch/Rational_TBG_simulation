"""
Fundamental building blocks for lattice structures and graph operations.

This module provides the core data structures and mathematical utilities used
by higher-level TBG physics simulations:

Classes:
    node: Individual lattice site with position and connectivity
    graph: Collection of nodes with edges and periodic boundary conditions
    periodic_graph: Extended graph with momentum-space operations and periodicity
    periodic_matrix: Sparse matrix operations and Laplacian construction

Functions:
    calculate_distance: Distance calculation utility
    canonical_position: Coordinate transformation to the unit cell torus utility- 

These components provide reusable mathematical and computational building blocks
that can be composed by higher-level physics classes.
"""

import constants
from typing import List, Tuple, Optional, Union, Dict, Any
from plotting import graph_plotter, band_plotter
from utils import calculate_distance, canonical_position, is_hermitian_sparse
from band_comp_and_plot import band_handler

# Configure logging
logger = constants.logging.getLogger(__name__)

DEFAULT_PARAMS = constants.dataclasses.asdict(constants.simulation_parameters())

class node: 
    """
    Class representing a node in a graph.
    
    Attributes:
        position (Tuple[float, float]): A tuple representing the real position in plane.
        lattice_index (Tuple[int, int]): A tuple of integers that describe the position in the lattice coordinates.
        sublattice_id (int): An index to distinguish different graphs or lattices.
        neighbors (List[Tuple[node, Tuple[int, int]]]): List of neighboring nodes with periodic offsets.
    """
    def __init__(self, position: Tuple[float, float], lattice_index: Tuple[int, int], sublattice_id: int = 0) -> None:
        try:
            if not isinstance(position, (tuple, list)) or len(position) != 2:
                raise constants.graph_construction_error("Position must be a tuple/list of length 2")
            if not isinstance(lattice_index, (tuple, list)) or len(lattice_index) != 2:
                raise constants.graph_construction_error("Lattice index must be a tuple/list of length 2")
            if not isinstance(sublattice_id, int):
                raise constants.graph_construction_error("Sublattice ID must be an integer")
            
            self.lattice_index = tuple(lattice_index)
            self.position = tuple(position)
            self.neighbors = []
            self.sublattice_id = sublattice_id
        except constants.graph_construction_error:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in node initialization: {str(e)}")
            raise constants.graph_construction_error(f"Failed to initialize node: {str(e)}")  

    def add_neighbor(self, Neighbor_node: "node", periodic_offset: Tuple[int, int] = (0, 0)) -> None:
        """
        Add a neighboring node with periodic offset.
        
        Args:
            Neighbor_node: The node to add as neighbor
            periodic_offset: Tuple representing periodic boundary offset (default: (0,0))
        
        Raises:
            constants.graph_construction_error: If neighbor node or offset is invalid.
        """
        try:
            if not isinstance(Neighbor_node, node):
                raise constants.graph_construction_error("Neighbor must be a node instance")
            if not isinstance(periodic_offset, (tuple, list)) or len(periodic_offset) != 2:
                raise constants.graph_construction_error("Periodic offset must be a tuple/list of length 2")
            
            self.neighbors.append((Neighbor_node, tuple(periodic_offset)))
        except constants.graph_construction_error:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in add_neighbor: {str(e)}")
            raise constants.graph_construction_error(f"Failed to add neighbor: {str(e)}")

    def __repr__(self) -> str:
        return f"node({self.lattice_index}) at {self.position} from graph of index {self.sublattice_id}\n"
    
    def copy(self, sublattice_id: Optional[int] = None) -> "node":
        """
        Returns a copy of the node, optionally with a different subgraph index.
        
        Args:
            sublattice_id (Optional[int]): An integer to override the sublattice index of the node (if copied to another graph).
            
        Returns:
            node: A copy of the node with the specified sublattice_id.
            
        Raises:
            constants.graph_construction_error: If copying fails.
        """
        try:
            if sublattice_id is not None:
                if not isinstance(sublattice_id, int):
                    raise constants.graph_construction_error("Sublattice ID must be an integer")
                return node(self.position, self.lattice_index, sublattice_id)
            return node(self.position, self.lattice_index, self.sublattice_id)
        except constants.graph_construction_error:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in node copy: {str(e)}")
            raise constants.graph_construction_error(f"Failed to copy node: {str(e)}")
    def cleanup(self) -> None:
        """
        Clean up the node object by breaking neighbor references.
        
        Raises:
            constants.graph_construction_error: If cleanup fails.
        """
        try:
            if hasattr(self, 'neighbors') and self.neighbors is not None:
                for i in range(len(self.neighbors)):
                    self.neighbors[i] = None
                self.neighbors.clear()
                self.neighbors = None
            self.position = None
            self.lattice_index = None
        except Exception as e:
            logger.error(f"Error during node cleanup: {str(e)}")
            raise constants.graph_construction_error(f"Failed to cleanup node: {str(e)}")

    def __enter__(self) -> 'node':
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Optional[type], exc_value: Optional[Exception], traceback: Optional[Any]) -> bool:
        """Context manager exit with cleanup."""
        self.cleanup()
        return False


class graph:
    """
    graph class representing a collection of nodes with connectivity and periodic relations.

    Attributes:
        nodes (list): List of node objects.
        node_dict (dict): Dictionary mapping (lattice_index, sublattice_id) to nodes, for better search.
        number_of_subgraphs (int): Counter for appended subgraphs.
    """
    def __init__(self) -> None:
        self.nodes = [] # nodes list
        self.node_dict= {}  # Dictionary to quickly find nodes by their index
        self.number_of_subgraphs = 0  # Initialize the number of subgraphs
    def add_node(self, node_to_add: node) -> None:
        """
        Add a node to the graph.

        Args:
            node_to_add: The node to add.

        Raises:
            constants.graph_construction_error: If a node with the same lattice_index and sublattice_id exists or node is invalid.
        """
        try:
            if not isinstance(node_to_add, node):
                raise constants.graph_construction_error("Input must be a node instance")
            
            key = (node_to_add.lattice_index, node_to_add.sublattice_id)
            if self.node_dict.get(key) is not None:
                raise constants.graph_construction_error(f"node with lattice_index {node_to_add.lattice_index} and sublattice_id {node_to_add.sublattice_id} already exists.")
            
            self.nodes.append(node_to_add)
            self.node_dict[key] = node_to_add  # Add node to lookup dictionary
        except constants.graph_construction_error:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in add_node: {str(e)}")
            raise constants.graph_construction_error(f"Failed to add node: {str(e)}")

    def get_node_by_index(self, lattice_index: Tuple[int, int], sublattice_id: int = 0) -> node:
        """
        Retrieve a node from the graph by its lattice index and sublattice ID.

        Args:
            lattice_index (tuple): Index in lattice coordinates (e.g., (x, y)).
            sublattice_id (int): Sublattice ID, defaults to 0.

        Returns:
            node: The corresponding node.

        Raises:
            KeyError: If the node is not found.
        """
        try:
            return self.node_dict[(lattice_index, sublattice_id)]
        except KeyError as e:
            raise KeyError(f"node with index {lattice_index} and sublattice {sublattice_id} not found.") from e
        
    def add_edge(self, node1: node, node2: node, periodic_offset: Tuple[int, int] = (0, 0)) -> None:
        """
        Add an edge between two nodes with an optional periodic offset.

        Args:
            node1 (node): First node.
            node2 (node): Second node.
            periodic_offset (tuple): The periodic offset between node1 and node2 (i.e. are they in the same unit cell).

        Raises:
            ValueError: If either node is not in the graph.
        """
        if node1 not in self.nodes or node2 not in self.nodes:
            raise ValueError("Both nodes must be part of the graph.")
        if (node2,periodic_offset) not in node1.neighbors:
            node1.add_neighbor(node2,periodic_offset)
        periodic_offset_inverse=(-periodic_offset[0],-periodic_offset[1])# inverse offset
        if (node1,periodic_offset_inverse) not in node2.neighbors:
            node2.add_neighbor(node1,periodic_offset_inverse) # Add the inverse offset for the neighbor

    def delete_edge(self, node1: node, node2: node, periodic_offset: Tuple[int, int] = (0, 0)) -> None:
        """
        Delete an edge between two nodes.
        First checks if one node is a neighbor of the other, and removes it. Then checks the inverse offset,
        and removes the node1 from the neighbors of node2.

        Args:
            node1 (node): First node.
            node2 (node): Second node.
            periodic_offset (tuple): Periodic offset of the edge.

        Raises:
            ValueError: If either node is not in the graph.
        """
        if node1 not in self.nodes or node2 not in self.nodes:
            raise ValueError("Both nodes must be part of the graph.")
        if (node2,periodic_offset) in node1.neighbors:
            node1.neighbors.remove((node2, periodic_offset))
        periodic_offset_inverse=(-periodic_offset[0],-periodic_offset[1]) # inverse offset
        if (node1,periodic_offset_inverse) in node2.neighbors:
            node2.neighbors.remove((node1, periodic_offset_inverse))

    def delete_node(self, node: node) -> None:
        """Delete a node from the graph, including its references from neighbors.

        Args:
            node (node): node to remove.

        Raises:
            ValueError: If node is not part of the graph.
        """
        if node not in self.nodes:
            raise ValueError("node not found in the graph.")
        self.nodes.remove(node)
        for nodes in self.nodes:
            nodes.neighbors = [neighbor for neighbor in nodes.neighbors if neighbor[0] != node]
        del self.node_dict[(node.lattice_index, node.sublattice_id)]

    def __repr__(self) -> str:
        s = f"graph with {len(self.nodes)} nodes:\n"
        for node in self.nodes:
            s += f"  node {node.lattice_index}: position = {node.position}, sublattice_id = {node.sublattice_id}\n"
        return s
    
    def copy(self, additional_index: Optional[int] = None) -> "graph":
        """
        Create a deep copy of the graph with optional sublattice re-indexing.

        Args:
            additional_index (int, optional): Subgraph index to assign to the copied nodes.

        Returns:
            graph: A new graph instance containing copies of nodes and edges.

        Raises:
            KeyError: If trying to add a neighbor that is not in the graph.
        """
        new_graph = graph()
        for node in self.nodes:
            new_node = node.copy(additional_index)
            new_graph.add_node(new_node)
        for node in self.nodes:
            new_node = new_graph.node_dict[node.lattice_index,node.sublattice_id]  # Get the copied node
            for neighbor, periodic_offset in node.neighbors:
                try:
                    neighbor_in_new_graph = new_graph.get_node_by_index(neighbor.lattice_index, additional_index)
                    new_graph.add_edge(new_node, neighbor_in_new_graph, periodic_offset)
                except KeyError:
                    raise KeyError("Neighbor not in the graph.")
        return new_graph
    
    def append(self, other_graph: "graph", additional_index: Optional[int] = None) -> None:
        """
        Append nodes and edges from another graph to this graph.

        Args:
            other_graph (graph): Another graph instance to merge.
            additional_index (int, optional): Subgraph index for appended nodes.

        Raises:
            ValueError: If other_graph is not a graph instance, or if the additional index collides with an index of an existing graph.
        """
        if not isinstance(other_graph, graph):
            raise ValueError("Can only append another graph instance.")
        if additional_index<self.number_of_subgraphs:
            raise ValueError("The additional index is of an already existing graph!")
        self.number_of_subgraphs += 1  # Increment the number of subgraphs
        copied_nodes={}
        for node in other_graph.nodes:
            copied=node.copy(additional_index)
            self.add_node(copied)
            copied_nodes[(node.lattice_index,node.sublattice_id)]=copied


        for node in other_graph.nodes:
            source = copied_nodes[(node.lattice_index, node.sublattice_id)]
            for neighbor, periodic_offset in node.neighbors:
                target = copied_nodes.get((neighbor.lattice_index, neighbor.sublattice_id))
                if target:
                    self.add_edge(source, target, periodic_offset)

    def _create_periodic_add_nodes(self, periodic_vectors: List[Tuple[float, float]], K_point: Tuple[float, float]) -> Tuple["periodic_graph", Dict]:
        """
        Subfunction to add the nodes to the periodic graph

        Args:
            periodic_vectors (list): List of two tuples defining the lattice primitive vectors.
            K_point (tuple): The position of the K_point, the high symmetry point.

        Returns:
            graph_copy (graph): A graph object that respects the periodic boundary conditions.
            new_nodes (list): A list of the new nodes.
        """
        A = constants.np.array([periodic_vectors[0], periodic_vectors[1]]).T
        B =2 * constants.np.pi * constants.np.linalg.inv(A)
        dual_vectors=[tuple(B[0, :]), tuple(B[1, :])]
        graph_copy = periodic_graph(periodic_vectors,dual_vectors,K_point)  # Create a copy of the graph as a periodic graph
        new_nodes = dict()
        for node in self.nodes:
            node_position,node_shift=canonical_position(node.position, periodic_vectors[0], periodic_vectors[1])  # Get the canonical position of the node  
            if abs(node_shift[0])>constants.MAX_ADJACENT_CELLS or abs(node_shift[1])>constants.MAX_ADJACENT_CELLS:
                continue
            
            # Create unique key combining position and sublattice
            position_sublattice_key = (node_position, node.sublattice_id)
            
            if position_sublattice_key not in new_nodes:  # Check if this position+sublattice combination exists
                new_node = node.copy()
                new_node.position = node_position  # Set the position of the new node to the canonical
                graph_copy.add_node(new_node)  # Add the new node to the copy
                new_nodes[position_sublattice_key] = new_node  # Store with position+sublattice key
        
        # Check for trivial unit cell (insufficient nodes for meaningful computation-which has 2 nodes- one for each layer)
        if len(new_nodes) <= 2:
            logger.warning(f"Trivial unit cell with {len(new_nodes)} nodes detected for the given twist parameters. "
                          f"This corresponds to a fundamental symmetry case. Expanding system to maintain physics.")
            del graph_copy,new_nodes
            return self._create_expanded_system(periodic_vectors, K_point, expansion_factor=constants.EXPANSION_FACTOR)
        
        return graph_copy,new_nodes
    
    def _create_expanded_system(self, periodic_vectors: List[Tuple[float, float]], K_point: Tuple[float, float], expansion_factor: int = 3) -> Tuple["periodic_graph", Dict]:
        """
        Create an expanded system for trivial unit cell cases.
        
        This method creates a larger supercell that preserves the same physics but
        provides enough nodes for meaningful Laplacian computation.
        
        Args:
            periodic_vectors: Original lattice vectors.
            K_point: High symmetry point.
            expansion_factor: Factor by which to expand the unit cell.
            
        Returns:
            Tuple of expanded periodic graph and node dictionary.
        """
        # Scale the lattice vectors to create a larger unit cell
        expanded_vectors = [
            (periodic_vectors[0][0] * expansion_factor, periodic_vectors[0][1] * expansion_factor),
            (periodic_vectors[1][0] * expansion_factor, periodic_vectors[1][1] * expansion_factor)
        ]
        
        A = constants.np.array([expanded_vectors[0], expanded_vectors[1]]).T
        B = 2 * constants.np.pi * constants.np.linalg.inv(A)
        dual_vectors = [tuple(B[0, :]), tuple(B[1, :])]
        
        # Create expanded periodic graph
        graph_copy = periodic_graph(expanded_vectors, dual_vectors, K_point)
        new_nodes = dict()
        
        # Add nodes with expanded unit cell
        for node in self.nodes:
            node_position, node_shift = canonical_position(node.position, expanded_vectors[0], expanded_vectors[1])
            
            # More lenient boundary check for expanded system
            if abs(node_shift[0]) > constants.MAX_ADJACENT_CELLS * expansion_factor or abs(node_shift[1]) > constants.MAX_ADJACENT_CELLS * expansion_factor:
                continue
            
            # Create unique key combining position and sublattice
            position_sublattice_key = (node_position, node.sublattice_id)
            if position_sublattice_key not in new_nodes:  # Check if this position+sublattice combination exists
                new_node = node.copy()
                new_node.position = node_position  # Set the position of the new node to the canonical
                graph_copy.add_node(new_node)  # Add the new node to the copy
                new_nodes[position_sublattice_key] = new_node  # Store with position+sublattice key
        
        
        logger.info(f"Expanded system created with {len(new_nodes)} nodes "
                   f"(expansion factor: {expansion_factor}x)")
        
        return graph_copy, new_nodes
    
    def create_periodic_copy(self,periodic_vectors:list,K_point : tuple)-> "periodic_graph":
        """
        Create a periodic version of the graph based on the given primitive vectors.

        This method wraps node positions into a unit cell and reconstructs connectivity 
        considering periodic boundary conditions.

        Args:
            periodic_vectors (list): List of two tuples defining the lattice primitive vectors.
            K_point (tuple): The position of the K_point, the high symmetry point.

        Returns:
            graph_copy: A graph object that respects the periodic boundary conditions.
        
        Raises:
            ValueError: if the periodic vectors given or of the wrong shape (not two 2d vectors)
        """
        if constants.np.array(periodic_vectors).shape != (2,2):
            raise ValueError("The periodic vectors should be two dimensional!")
        graph_copy,new_nodes=self._create_periodic_add_nodes(periodic_vectors,K_point)
        # if the periodic vector were changed it means that we are in a degenrate
        if constants.np.linalg.norm(constants.np.array(graph_copy.lattice_vectors[0])-constants.np.array(periodic_vectors[0]))>1e-6:
            periodic_vectors=graph_copy.lattice_vectors
        existing_edges = set()
        for node in self.nodes:
            node_position,node_Shift=canonical_position(node.position, periodic_vectors[0], periodic_vectors[1])  # Get the canonical position of the node
            node_key = (node_position, node.sublattice_id)
            if node_key not in new_nodes:
                continue  # node was filtered out
            new_node= new_nodes[node_key]  # Get the new node from the dictionary                
            for neighbor, _ in node.neighbors:
                neighbor_position,neighbor_shift=canonical_position(neighbor.position, periodic_vectors[0], periodic_vectors[1])
                if (constants.np.allclose(neighbor_position,node_position,atol=1e-6) and neighbor.sublattice_id== node.sublattice_id):
                    continue
                neighbor_key = (neighbor_position, neighbor.sublattice_id)
                if neighbor_key not in new_nodes:
                    continue  # Neighbor was filtered out
                new_neighbor=new_nodes[neighbor_key]  # Get the new neighbor from the dictionary
                shift = (neighbor_shift[0] - node_Shift[0], neighbor_shift[1] - node_Shift[1])
                if abs(shift[0]) > 1 or abs(shift[1]) > 1: #only count things in adjecent unit cells, not ones that are further removed
                    logger.debug(f"Skipping edge with long-range shift: {shift}")
                    continue
                edge_key = (new_node.lattice_index,new_node.sublattice_id, new_neighbor.lattice_index,new_neighbor.sublattice_id, shift)
                inverse_shift=(-shift[0],-shift[1])
                Inverse_edge_key=( new_neighbor.lattice_index,new_neighbor.sublattice_id, new_node.lattice_index,new_node.sublattice_id,inverse_shift)
                if edge_key not in existing_edges and Inverse_edge_key not in existing_edges:  # Check if the neighbor is already in the new node's neighbors, with this weight
                   
                    graph_copy.add_edge(new_node, new_neighbor, shift)  # Add the edge to the copy
                    if shift[0]!=0 or shift[1]!=0:
                        logger.debug(f"Adding edge: {new_node.lattice_index} at { node_position}  to"
                            f"{new_neighbor.lattice_index} at {neighbor_position}, shift={shift}")
                        graph_copy.num_of_periodic_edges+=1
                    existing_edges.add(edge_key)
        graph_copy.edges_list=existing_edges 
        return graph_copy  # Return the periodic copy of the graph
    

    def plot(self, ax: constants.matplotlib.axes.Axes, node_colors: List[str], max_distance: Optional[float] = None, differentiate_subgraphs: bool = False,
              lattice_vectors: Optional[List[Tuple[float, float]]] = None) -> None:
        """
        Plot the graph with optional coloring and filtering.

        Args:
            ax (constants.matplotlib.axes.Axes): The matplotlib axis on which to draw the plot.
            node_colors (list or dict, optional): Specifies the colors for nodes. 
                Can be a list of colors (applied cyclically to nodes) or a dictionary mapping 'sublattice_id' to color for differentiation. 
                Defaults to a default color scheme.
            max_distance (float, optional): If provided, only nodes and their connected edges whose *absolute position* is within
                this distance from the origin (0,0) will be plotted. This helps focus the visualization on a central region of the lattice.
                Defaults to plotting all available nodes and edges.
            differentiate_subgraphs (bool, optional): If True, nodes belonging to different sublattices (based on their sublattice_id) 
                    will be plotted with distinct colors or markers to visually distinguish them. Defaults to False.
            lattice_vectors (numpy.ndarray, optional): A 2x2 NumPy array representing the primitive lattice vectors of the unit cell.
                Used to draw the boundary of a single unit cell on the plot, providing a visual reference for periodicity.
                Defaults to the graph's internal basis vectors.

        Raises:
            ValueError: If differentiate_subgraphs is True but node_colors does not have enough entries for all subgraphs.
        """
        # Validate subgraph colors if needed
        if differentiate_subgraphs and len(node_colors) < self.number_of_subgraphs + 1:
            raise ValueError("Not enough colors provided for the number of subgraphs.")
        
        # Delegate to plotting module
        graph_plotter.plot_graph(ax, self.nodes, node_colors, max_distance, 
                               differentiate_subgraphs, lattice_vectors)
        
    def cleanup(self) -> None:
        """
        Clean up Graph by breaking all circular references.
        """
        if hasattr(self, 'nodes') and self.nodes is not None:
            # First pass: clear all neighbor references
            for node in self.nodes:
                if node is not None and hasattr(node, 'cleanup'):
                    node.cleanup()

            # Second pass: clear the nodes list
            self.nodes.clear()
            self.nodes = None

        if hasattr(self, 'node_dict') and self.node_dict is not None:
            self.node_dict.clear()
            self.node_dict = None

    def __enter__(self) -> 'graph':
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Optional[type], exc_value: Optional[Exception], traceback: Optional[Any]) -> bool:
        """Context manager exit with cleanup."""
        self.cleanup()
        return False

class periodic_graph(graph):
    """
    A subclass of graph that includes periodicity by incorporating lattice vectors and momentum space structure.

    This class is designed to support physics applications like tight-binding models on periodic lattices. It allows
    construction of Hermitian Laplacians that incorporate both intra-cell and inter-cell connections with momentum-dependent
    phase factors.
    
    Attributes:
        lattice_vectors (List[Tuple[float, float]]): Primitive lattice vectors of the graph.
        dual_vectors (List[Tuple[float, float]]): Reciprocal lattice vectors derived from lattice_vectors.
        K_point (Tuple[float, float]): The K-point in momentum space.
    """
    def __init__(self, lattice_vectors: Optional[List[Tuple[float, float]]] = None,
                 dual_vectors: Optional[List[Tuple[float, float]]] = None,
                 K_point: Tuple[float, float] = constants.K_POINT_REG) -> None: 
        """
        Initialize a periodic graph with given lattice and reciprocal vectors.

        Args:
            lattice_vectors (Optional[List[Tuple[float, float]]]): Real space primitive vectors of the unit cell.
                Defaults to constants.v1 and constants.v2 if None.
            dual_vectors (Optional[List[Tuple[float, float]]]): Reciprocal space basis vectors in cartesian coordinates.
                Defaults to constants.k1 and constants.k2 if None.
            K_point (Tuple[float, float]): The lattice coordinates of the K point that will be used for the band computations.  
        """
        super().__init__()
        if lattice_vectors is None:
            lattice_vectors = [constants.v1, constants.v2]
        if dual_vectors is None:
            dual_vectors = [constants.k1, constants.k2]
        self.lattice_vectors = lattice_vectors
        self.dual_vectors = dual_vectors
        self.K_point = K_point
        self.num_of_periodic_edges = 0
        self.edges_list = []
        
        # Use composition pattern - delegate matrix operations to periodic_matrix
        self.matrix_handler = periodic_matrix(self)

        self.band_handler = band_handler(self.matrix_handler)
    
    # Properties for backward compatibility
    @property
    def adj_matrix(self) -> constants.csr_matrix:
        """Access the adjacency matrix."""
        return self.matrix_handler.adj_matrix
    
    @property
    def periodic_edges(self) -> List[Tuple]:
        """Access the periodic edges."""
        return self.matrix_handler.periodic_edges
    
    
    # Delegate matrix methods to matrix_handler
    def build_adj_matrix(self, inter_graph_weight: float = DEFAULT_PARAMS['inter_graph_weight'], 
                        intra_graph_weight: float = DEFAULT_PARAMS['intra_graph_weight']) -> None:
        return self.matrix_handler.build_adj_matrix(inter_graph_weight, intra_graph_weight)
    
    def update_weights(self, inter_graph_weight: float = DEFAULT_PARAMS['inter_graph_weight'],
                      intra_graph_weight: float = DEFAULT_PARAMS['intra_graph_weight']) -> None:
        return self.matrix_handler.update_weights(inter_graph_weight, intra_graph_weight)
    
    def build_laplacian(self, inter_graph_weight: float = DEFAULT_PARAMS['inter_graph_weight'], 
                       intra_graph_weight: float = DEFAULT_PARAMS['intra_graph_weight'], 
                       Momentum: Tuple[float, float] = (0.0, 0.0), 
                       compute_adj_flag: bool = True) -> Tuple[constants.csr_matrix, constants.csr_matrix]:
        return self.matrix_handler.build_laplacian(inter_graph_weight, intra_graph_weight, Momentum, compute_adj_flag)
    
    # Delegate band methods to band_handler
    
    def compute_bands_at_k(self, Momentum: tuple[float, float], min_bands: int, max_bands: int, 
                          inter_graph_weight: float = DEFAULT_PARAMS['inter_graph_weight'],
                          intra_graph_weight: float = DEFAULT_PARAMS['intra_graph_weight']) -> tuple[constants.np.ndarray, constants.np.ndarray]:
        return self.band_handler.compute_bands_at_k(Momentum, min_bands, max_bands, inter_graph_weight, intra_graph_weight)


    def compute_energy_and_derivative_of_energy_at_a_point(self, k_point: tuple[float, float], num_of_band: int = 1) -> tuple[constants.np.ndarray, float, float]:
        return self.band_handler.compute_energy_and_derivative_of_energy_at_a_point(k_point, num_of_band)


    def plot_band_structure(self, ax: constants.matplotlib.axes.Axes, num_of_points: int, min_bands: int, max_bands: int,
                            inter_graph_weight: float, intra_graph_weight: float, k_max: float = constants.DEFAULT_K_RANGE,
                            k_min: float = -constants.DEFAULT_K_RANGE, k_flag: bool = False, build_adj_matrix_flag: bool = False) -> None:
        """Plot 3D band structure using the plotting module."""
        band_plotter.plot_band_structure_3d(ax, self.band_handler, num_of_points, min_bands, max_bands,
                                          inter_graph_weight, intra_graph_weight, k_max, k_min, k_flag, build_adj_matrix_flag)

    # Note: check_Dirac_point method moved to TBG.DiracAnalysis class
    # This is physics-specific functionality that belongs in the TBG module

    # End of delegation methods - cleanup method follows  
    def cleanup(self) -> None:
        """
        Clean up periodic_graph and its matrices.
        """
            
        # Clean up band handler
        if hasattr(self, 'band_handler') and self.band_handler is not None:
            self.band_handler.cleanup()
            self.band_handler = None
            
        # Clean up matrix handler
        if hasattr(self, 'matrix_handler') and self.matrix_handler is not None:
            self.matrix_handler.cleanup()
            self.matrix_handler = None

        if hasattr(self, 'edges_list'):
            self.edges_list = None

        self.lattice_vectors = None
        self.dual_vectors = None
        self.K_point = None
        self.num_of_periodic_edges = None

        # Call parent cleanup
        super().cleanup()

    def __enter__(self) -> 'periodic_graph':
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Optional[type], exc_value: Optional[Exception], traceback: Optional[Any]) -> bool:
        """Context manager exit with cleanup."""
        self.cleanup()
        return False



class periodic_matrix():
    """
    Handles matrix operations for periodic graphs
    including adjacency matrix construction,
    Laplacian building, and matrix validation.
    """

    def __init__(self, periodic_graph_obj: periodic_graph):
        """
        Initialize periodic matrix handler.

        Args:
            periodic_graph_obj: Periodic graph object containing nodes and connectivity
        """
        self.periodic_graph = periodic_graph_obj
        self.lattice_vectors = periodic_graph_obj.lattice_vectors
        self.dual_vectors = periodic_graph_obj.dual_vectors

        # Matrix storage
        self.adj_matrix: Optional[constants.csr_matrix] = None      
        self.periodic_edges: Optional[List] = None        
        self.inter_edges: Optional[List] = None
        self.intra_edges: Optional[List] = None
    def build_adj_matrix(self, inter_graph_weight: float = DEFAULT_PARAMS['inter_graph_weight'], 
                         intra_graph_weight: float = DEFAULT_PARAMS['intra_graph_weight']) -> None: 
        """
        Construct the Laplacian matrix ignoring periodic phase factors.

        Args:
            inter_graph_weight (float): Weight of connections between subgraphs.
            intra_graph_weight (float): Weight of connections within a subgraph.

        Updates:
            - adj_matrix (scipy.sparse.constants.csr_matrix): Initial Laplacian matrix.
            - periodic_edges (list): List of periodic edges needing phase factors.
        Raises:
            ValueError- if the matrix is not Hermitian
        """
        n = len(self.periodic_graph.nodes)
        
        # Handle trivial cases with insufficient nodes
        if n < 2:
            logger.warning(f"Cannot build meaningful Laplacian with only {n} nodes. "
                         f"Creating minimal valid matrix structure.")
            # Create minimal valid sparse matrix
            if n == 0:
                self.adj_matrix = constants.csr_matrix((0, 0), dtype=complex)
            else:  # n == 1
                self.adj_matrix = constants.csr_matrix(([0.0+0.0j], ([0], [0])), shape=(1, 1), dtype=complex)
            
            self.periodic_edges = []
            self.inter_edges = []
            self.intra_edges = []
            logger.info(f"Minimal matrix structure created for {n} nodes")
            return
        self.periodic_graph.nodes.sort(key=lambda node:(node.lattice_index, node.sublattice_id))
        #precompute that hashing node graph index to index
        node_key_to_index = {(node.lattice_index,node.sublattice_id): i for i, node in enumerate(self.periodic_graph.nodes)}
        row_indices = []
        col_indices = []
        data = []
        processed_edges = set()
        periodic_edges=[]
        intra_edges=[]
        inter_edges=[]
        for i, node in enumerate(self.periodic_graph.nodes):
            #For now solving the free particle case, so the diagonal is zero.  
            row_indices.append(i)
            col_indices.append(i)
            data.append(0.0 + 0.0j) 
            for neighbor, periodic_offset in node.neighbors:
                j = node_key_to_index[(neighbor.lattice_index,neighbor.sublattice_id)]
                edge_key = (min(i, j), max(i, j), periodic_offset)
                reverse_offset = (-periodic_offset[0], -periodic_offset[1])
                edge_key_inverse=(min(i, j), max(i, j), reverse_offset)
                if edge_key in processed_edges or edge_key_inverse in processed_edges:
                    continue
                processed_edges.add(edge_key)
                # Intra-cell edge
                if node.sublattice_id == neighbor.sublattice_id:
                    weight = intra_graph_weight
                    intra_edges.append([i, j])  
                    intra_edges.append([j, i])  
                else:
                    weight = inter_graph_weight
                    inter_edges.append([i, j])  
                    inter_edges.append([j, i])  
                # Add both (i,j) and (j,i) for symmetry
                row_indices.extend([i, j])
                col_indices.extend([j, i])
                data.extend([weight, weight])
                if periodic_offset != (0, 0):
                    # Periodic edge - store for later phase factor application
                    periodic_edges.append([i, j, periodic_offset])  
                    # Also store the reverse edge
                    periodic_edges.append([j, i, reverse_offset])
        #more efficient construction in constants.coo_matrix
        laplacian_coo = constants.coo_matrix((data, (row_indices, col_indices)),shape=(n, n), dtype=complex)
        #manipulation is better in csr 
        laplacian = laplacian_coo.tocsr()
        if not is_hermitian_sparse(laplacian):
            raise ValueError("Laplacian matrix is not Hermitian.")
    
        logger.info(f"Matrix construction: {n}×{n}, {len(periodic_edges)} periodic edges")
        logger.info(f"Matrix density: {laplacian.nnz / (n*n) * 100:.2f}%")
        logger.info(f"Total periodic edges: {len(periodic_edges)}")
        self.adj_matrix=laplacian
        self.periodic_edges=periodic_edges
        self.inter_edges=inter_edges
        self.intra_edges=intra_edges
    def build_laplacian(self, inter_graph_weight: float = DEFAULT_PARAMS['inter_graph_weight'],
                         intra_graph_weight: float = DEFAULT_PARAMS['intra_graph_weight'],
                           Momentum: Tuple[float, float] = (0.0, 0.0),
                             compute_adj_flag: bool = True) -> Tuple[constants.csr_matrix, constants.csr_matrix]: 
        """
        Construct the full Laplacian matrix with periodic phase factors applied.
        
        This method builds the complete TBG Hamiltonian by applying momentum-dependent 
        phase factors to periodic edges. The implementation is memory-optimized, working
        directly with CSR matrix data arrays to avoid expensive format conversions.

        Args:
            inter_graph_weight (float): Weight of inter-sublattice connections.
            intra_graph_weight (float): Weight of intra-sublattice connections.
            Momentum (Tuple[float, float]): 2D quasimomentum vector in relative coordinates.
            compute_adj_flag (bool): If True, build base adjacency matrix first. If False, use existing.
            
        Returns:
            Tuple[constants.csr_matrix, constants.csr_matrix]: Tuple containing:
                - laplacian_csr: Complete Hermitian Laplacian with phase factors
                - phase_csr: Phase factor matrix only (for analysis)
                
        Raises:
            ValueError: If Laplacian matrix is not Hermitian after phase factor application.
        """
        if compute_adj_flag:
            self.build_adj_matrix(inter_graph_weight,intra_graph_weight)
        
        n = len(self.periodic_graph.nodes)
        
        if not self.periodic_edges:
            # No periodic edges - return base matrix and empty phase matrix
            empty_phase = constants.csr_matrix((n, n), dtype=complex)
            return self.adj_matrix, empty_phase
        
        # Pre-allocate arrays with known size for better performance
        num_periodic = len(self.periodic_edges)
        i_indices = constants.np.empty(num_periodic, dtype=int)
        j_indices = constants.np.empty(num_periodic, dtype=int)
        offsets = constants.np.empty((num_periodic, 2), dtype=float)
        
        # Vectorized extraction - much faster than list comprehensions
        for idx, edge in enumerate(self.periodic_edges):
            i_indices[idx] = edge[0]
            j_indices[idx] = edge[1]
            offsets[idx, 0] = edge[2][0]
            offsets[idx, 1] = edge[2][1]
        
        # Vectorized phase calculation
        phases = constants.np.dot(offsets, constants.np.array(Momentum)) * 2 * constants.np.pi
        phase_factors = constants.np.exp(1j * phases)
        
        # Apply phase factors directly to CSR matrix data - OPTIMIZED: single copy
        laplacian_data = self.adj_matrix.data.copy()  # Only copy data array we need to modify
        
        # Apply phase factors to periodic edges
        for idx in range(num_periodic):
            i, j = i_indices[idx], j_indices[idx]
            # Find the data index for element (i,j) in CSR format
            start_idx = self.adj_matrix.indptr[i]
            end_idx = self.adj_matrix.indptr[i + 1]
            col_indices = self.adj_matrix.indices[start_idx:end_idx]
            data_pos = constants.np.where(col_indices == j)[0]
            if len(data_pos) > 0:
                data_idx = start_idx + data_pos[0]
                laplacian_data[data_idx] *= phase_factors[idx]
        
        # Create modified laplacian with updated data - reuse original indices/indptr
        laplacian_csr = constants.csr_matrix((laplacian_data, self.adj_matrix.indices, self.adj_matrix.indptr), shape=(n, n))
        
        # Build phase matrix efficiently
        phase_csr = constants.csr_matrix((phase_factors, (i_indices, j_indices)), shape=(n, n), dtype=complex)
        
        # Hermiticity check on the final matrix
        if not is_hermitian_sparse(laplacian_csr):
            raise ValueError("Laplacian matrix is not Hermitian.")
        
        return laplacian_csr, phase_csr

    def update_weights(self, inter_graph_weight: float = DEFAULT_PARAMS['inter_graph_weight'],
                      intra_graph_weight: float = DEFAULT_PARAMS['intra_graph_weight']) -> None:
        """
        Update edge weights in the existing adjacency matrix without rebuilding.
        
        Args:
            inter_graph_weight (float): Weight of connections between subgraphs.
            intra_graph_weight (float): Weight of connections within a subgraph.

        Updates:
            - adj_matrix (scipy.sparse.constants.csr_matrix): Updated adjacency matrix.
        Raises:
            ValueError: if adjacency matrix or edge lists haven't been built
        """
        if self.adj_matrix is None or self.inter_edges is None or self.intra_edges is None:
            raise ValueError("Must build adjacency matrix first before updating weights")
        
        # Create new data array with updated weights
        adj_data = self.adj_matrix.data.copy()
        
        # Update intra-edges (same sublattice connections)
        for edge in self.intra_edges:
            i, j = edge
            # Find corresponding entries in sparse matrix data
            for idx in range(self.adj_matrix.indptr[i], self.adj_matrix.indptr[i+1]):
                if self.adj_matrix.indices[idx] == j:
                    adj_data[idx] = intra_graph_weight
                    break
        
        # Update inter-edges (different sublattice connections)  
        for edge in self.inter_edges:
            i, j = edge
            # Find corresponding entries in sparse matrix data
            for idx in range(self.adj_matrix.indptr[i], self.adj_matrix.indptr[i+1]):
                if self.adj_matrix.indices[idx] == j:
                    adj_data[idx] = inter_graph_weight
                    break
        
        # Create new matrix with updated weights
        self.adj_matrix = constants.csr_matrix((adj_data, self.adj_matrix.indices, self.adj_matrix.indptr), 
                                    shape=self.adj_matrix.shape, dtype=complex)

    def _is_hermitian_sparse(self, matrix: constants.csr_matrix, rtol: float = constants.NUMERIC_TOLERANCE) -> bool:
        """
        Check if a sparse matrix is Hermitian within numerical tolerance.
        
        Args:
            matrix (constants.csr_matrix): The sparse matrix to check
            rtol (float): Relative tolerance for comparison
            
        Returns:
            bool: True if matrix is Hermitian, False otherwise
        """
        if matrix.shape[0] != matrix.shape[1]:
            return False
        
        # Compare matrix with its conjugate transpose
        matrix_h = matrix.conj().T.tocsr()
        diff = matrix - matrix_h
        max_diff = constants.np.abs(diff.data).max() if diff.nnz > 0 else 0.0
        max_val = max(constants.np.abs(matrix.data).max(), constants.np.abs(matrix_h.data).max())
        
        return max_diff <= rtol * max_val
    def compute_laplacian_derivative(self, Momentum:tuple[float,float])->tuple[constants.lil_matrix,constants.lil_matrix]: 
        """
        Compute the derivative of the Laplacian with respect to momentum.
        Args:
            Momentum (tuple): 2D quasimomentum vector for phase factor calculation in relative coordinates.
            inter_graph_weight (float): Weight of inter-subgraph connections.
            intra_graph_weight (float): Weight of intra-subgraph connections.
        Returns:
            tuple: A tuple containing:
                - derivative_x_lil (scipy.sparse matrix):Derivative of the Laplacian with respect to the first momentum component.
                - derivative_y_lil (scipy.sparse matrix):Derivative of the Laplacian with respect to the second momentum component.
        Raises:
            ValueError: If laplacian or periodic edges are not  provided.
        """
        if self.adj_matrix is None or self.periodic_edges is None:
                raise ValueError("Missing input! Provide the laplacian and periodic edges list!")
        _,phase_matrix = self.build_laplacian(Momentum=Momentum,compute_adj_flag=False)  # Get the phase matrix without recomputing adjacency
        # Build offset matrices for x and y components
        offset_x_matrix = constants.lil_matrix(phase_matrix.shape, dtype=complex)
        offset_y_matrix = constants.lil_matrix(phase_matrix.shape, dtype=complex)
        for edge in self.periodic_edges:
            i, j, offset = edge
            offset_in_space=offset[0]*constants.np.array(self.lattice_vectors[0])+offset[1]*constants.np.array(self.lattice_vectors[1])
            if offset[0] != 0 or offset[1] != 0:
                offset_x_matrix[i, j] = 1j  * offset_in_space[0] 
                offset_x_matrix[j, i] = -1j  * offset_in_space[0]
                offset_y_matrix[i, j] = 1j * offset_in_space[1]
                offset_y_matrix[j, i] = -1j * offset_in_space[1]
        # Convert to CSR for efficient operations after construction
        offset_x_matrix = offset_x_matrix.tocsr()
        offset_y_matrix = offset_y_matrix.tocsr()
        # Compute derivatives
        derivative_x = phase_matrix.multiply(offset_x_matrix)
        derivative_y = phase_matrix.multiply(offset_y_matrix)
        # Keep in CSR format - no need for LIL conversion
        derivative_x_csr = derivative_x.tocsr()
        derivative_y_csr = derivative_y.tocsr()
        if not is_hermitian_sparse(derivative_x_csr) or not is_hermitian_sparse(derivative_y_csr):
            raise ValueError("Laplacian derivative matrix is not Hermitian.")  
        return derivative_x_csr, derivative_y_csr


    def cleanup(self) -> None:
        """
        Clean up matrix storage.
        """
        matrix_attrs = ['adj_matrix', 'periodic_edges', 'inter_edges', 'intra_edges']
        for attr in matrix_attrs:
            if hasattr(self, attr):
                setattr(self, attr, None)

    def __enter__(self) -> 'periodic_matrix':
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Optional[type], exc_value: Optional[Exception], traceback: Optional[Any]) -> bool:
        """Context manager exit with cleanup."""
        self.cleanup()
        return False

