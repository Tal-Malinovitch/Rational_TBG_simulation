import numpy as np
from scipy.sparse import lil_matrix,csr_matrix,coo_matrix
from scipy.sparse.linalg import eigs, eigsh, ArpackNoConvergence
from scipy.spatial import cKDTree
from matplotlib.patches import Polygon
import matplotlib.axes 
from matplotlib.figure import Figure
from copy import deepcopy
import constants
import logging
import hashlib


# Configure global logging
logger = logging.getLogger(__name__)


def compute_twist_constants(a:int,b:int)->tuple[float,float,float,tuple[float,float]]:
    
    """
    Calculate constants alpha and NFactor for rational TBG twist angle.
    If a%3=0, then by the theory we are working with reciprocal lattice
    and we need to change the factor of the lattice, as well as alpha

    Args:
        a (int): Integer parameter a of the twist.
        b (int): Integer parameter b of the twist.

    Returns:
        tuple: (NFactor, alpha, factor,kpoint) :
        NFactor- The scaling factor of the new lattice = sqrt(a^2+3b^2), taken from [malinovitch2024twisted]
        alpha- the corrrection factor to the scaling= 2^eps (4pi)^rho, where eps=1 if ab%2=1,and 0 else, and rho=1 if a%3 =0 and 0 else
        factor- the scaluing factor of the underlying lattice, if a%3=0, then we have that our lattice is N*Lambda^*, not N*Lambda
        kpoint- the coordiante of the k point- depending on whether we have Lambda or Lambda^*,

    Raises:
         ValueError: If a  is not positive, or b is 0 
    """
    alpha = 1
    factor=1
    if a<=0:
        raise ValueError("a have to be positive!")
    if b==0:
        raise ValueError("b can't be 0!")
    if a%3 == 0:
        alpha*=4*np.pi
        factor=constants.reciprocal_constant
        k_point=constants.K_POINT_DUAL
    else:
        k_point=constants.K_POINT_REG
    if (a*b)%2 ==1:
        alpha *= 2
    NFactor = np.sqrt(a**2 + 3*b**2)/alpha
    return NFactor,alpha,factor,k_point
class Node: 
    """Class representing a node in a graph.
    Args:
        position- a tuple of float representing the real position in plane.
        lattice_index- a tuple of integers that decribe the position in the lattice ocoordinates
        sublattice_id- an index to distiguish diffeerent graphs or lattices. 
    """
    def __init__(self, position:tuple,lattice_index:tuple,sublattice_id:int=0):
        self.lattice_index = lattice_index
        self.position = position
        self.neighbors = [] 
        self.sublattice_id = sublattice_id  

    def add_neighbor(self, Neighbor_node: "Node",periodic_offset:tuple[int,int]=(0,0)):
        """Add a neighboring node with periodic offset."""
        self.neighbors.append((Neighbor_node, periodic_offset))  # Add a neighbor with its periodic offset

    def __repr__(self):
        return f"Node({self.lattice_index}) at {self.position} from graph of index {self.sublattice_id}\n"
    
    def copy(self,sublattice_id:int=None)->"Node":
        """Returns a copy of the node, optionally with a different subgraph index.
        Args:
            sublattice_id- an integer to overide the sublattice index of the node ( if copied to another graph)"""
        if sublattice_id is not None:
            return Node(self.position, self.lattice_index, sublattice_id)
        return Node(self.position, self.lattice_index, self.sublattice_id)

def calculate_distance(node1: Node, node2:Node= None,offset:tuple[int,int]=(0,0))->float:
    """Calculate distance between nodes or from a node to the origin.

    Parameters:
        node1 (Node): The first node.
        node2 (Node, optional): The second node. Defaults to None.
        offset (tuple, optional): Coordinate offset to apply. Defaults to (0, 0).

    Returns:
        float: Euclidean distance.
    """
    if node2 is None:
        return np.linalg.norm(np.array(node1.position)-offset)  # distance from origin
    return np.linalg.norm(np.array(node1.position)- offset- np.array(node2.position))   

def canonical_position(pos:tuple[float,float], latticevector1:list[float], latticevector2:list[float], verbosity:bool=False)->list[tuple[float]]: 
        """Convert a Cartesian coordinate into its canonical position in the lattice.

        Parameters:
            pos (tuple): Cartesian position.
            latticevector1 (list): First lattice vector.
            latticevector2 (list): Second lattice vector.
            verbosity  (bool): If true- a more detailed log will be supplied

        Returns:
            list: Canonical position and shift in lattice units.
        """
        A = np.array([latticevector1, latticevector2]).T
        Ainv = np.linalg.inv(A)
        lattice_coords = Ainv @ np.array(pos)
        wrapped = (lattice_coords + 0.5) % 1.0 - 0.5 # We want centered unit cell
        shift = np.round(lattice_coords - wrapped).astype(int)
        canonical_pos = A @ wrapped
        canonical_pos = np.round(canonical_pos, 8)
        if  verbosity:
            logger.debug(f"Canonicalizing pos={pos} to {canonical_pos} with shift={shift} "
                 f"using lattice vectors {latticevector1}, {latticevector2}")
        return [tuple(canonical_pos), tuple(shift)]  # Return both the fractional and rounded positions in Cartesian coordinates

def validate_ab(a: int, b: int):
    """
    Validates the parameters a and b for rational TBG rotation.

    Args:
        a (int): Integer rotation parameter.
        b (int): Integer rotation parameter.

    Raises:
        ValueError: If the conditions for rational TBG are not met (a,b, co-prime integer, 0<|b|<=a).
    """
    if not isinstance(a, int) or not isinstance(b, int):
        raise ValueError("Both a and b must be integers.")
    if a <= 0:
        raise ValueError("a must be a positive integer.")
    if b == 0:
        raise ValueError("b must be a non-zero integer.")
    if np.gcd(a, b) != 1:
        raise ValueError("a and b must be coprime (gcd(a, b) == 1).")
    if abs(b) > a:
        raise ValueError("a must be greater than or equal to |b|.")
    
class graph:
    """
    graph class representing a collection of nodes with connectivity and periodic relations.

    Attributes:
        nodes (list): List of Node objects.
        node_dict (dict): Dictionary mapping (lattice_index, sublattice_id) to nodes, for better sreach.
        number_of_subgraphs (int): Counter for appended subgraphs.
    """
    def __init__(self):
        self.nodes = [] # nodes list
        self.node_dict= {}  # Dictionary to quickly find nodes by their index
        self.number_of_subgraphs = 0  # Initialize the number of subgraphs
    def add_node(self, node: Node):
        """Add a node to the graph.

        Parameters:
            node (Node): The node to add.

        Raises:
            ValueError: If a node with the same lattice_index and sublattice_id exists.
        """
        self.nodes.append(node)
        if self.node_dict.get((node.lattice_index, node.sublattice_id)) is not None:
            raise ValueError(f"Node with lattice_index {node.lattice_index} and sublattice_id {node.sublattice_id} already exists.")  
        self.node_dict[node.lattice_index,node.sublattice_id] = node  # Add node to lookup dictionary

    def get_node_by_index(self, lattice_index:tuple[int,int], sublattice_id:int=0)-> Node:
        """
        Retrieve a node from the graph by its lattice index and sublattice ID.

        Args:
            lattice_index (tuple): Index in lattice coordinates (e.g., (x, y)).
            sublattice_id (int): Sublattice ID, defaults to 0.

        Returns:
            Node: The corresponding node.

        Raises:
            KeyError: If the node is not found.
        """
        try:
            return self.node_dict[(lattice_index, sublattice_id)]
        except KeyError as e:
            raise KeyError(f"Node with index {lattice_index} and sublattice {sublattice_id} not found.") from e
        
    def add_edge(self, node1: Node, node2: Node,periodic_offset:tuple[int,int]=(0,0)):
        """Add an edge between two nodes with an optional periodic offset.

        Parameters:
            node1 (Node): First node.
            node2 (Node): Second node.
            periodic_offset (tuple): The periodic offset between node1 and node2 ( i.e. are they in the same unit cell).

        Raises:
            ValueError: If either node is not in the graph."""
        if node1 not in self.nodes or node2 not in self.nodes:
            raise ValueError("Both nodes must be part of the graph.")
        if (node2,periodic_offset) not in node1.neighbors:
            node1.add_neighbor(node2,periodic_offset)
        periodic_offset_inverse=(-periodic_offset[0],-periodic_offset[1])# inverse offset
        if (node1,periodic_offset_inverse) not in node2.neighbors:
            node2.add_neighbor(node1,periodic_offset_inverse) # Add the inverse offset for the neighbor

    def delete_edge(self, node1: Node, node2: Node,periodic_offset:tuple[int,int]=(0,0)):
        """Delete an edge between two nodes.
        first checks if one node in a neighbor of the other, and removes it- then checks the inverse offset,
          and removes the node1 from the neighbors of node2

        Parameters:
            node1 (Node): First node.
            node2 (Node): Second node.
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

    def delete_node(self, node: Node):
        """Delete a node from the graph, including its references from neighbors.

        Parameters:
            node (Node): Node to remove.

        Raises:
            ValueError: If node is not part of the graph.
        """
        if node not in self.nodes:
            raise ValueError("Node not found in the graph.")
        self.nodes.remove(node)
        for nodes in self.nodes:
            nodes.neighbors = [neighbor for neighbor in nodes.neighbors if neighbor[0] != node]
        del self.node_dict[(node.lattice_index, node.sublattice_id)]

    def __repr__(self) -> str:
        s = f"graph with {len(self.nodes)} nodes:\n"
        for node in self.nodes:
            s += f"  Node {node.lattice_index}: position = {node.position}, sublattice_id = {node.sublattice_id}\n"
        return s
    
    def copy(self,additional_index:int= None)->"graph":
        """Create a deep copy of the graph with optional sublattice re-indexing.

        Parameters:
            additional_index (int, optional): Subgraph index to assign to the copied nodes.

        Returns:
            graph: A new graph instance containing copies of nodes and edges.

        Raises
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
                    raise KeyError("Neighbor not in the garph")  
        return new_graph
    
    def append(self, other_graph: "graph",additional_index:int= None):
        """Append nodes and edges from another graph to this graph.

        Parameters:
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

    def _create_periodic_add_nodes(self,periodic_vectors:list,K_point : tuple)->tuple["graph",list[Node]]:
        """
        Subfunction to add the nodes to the periodic graph

        Parameters:
            periodic_vectors (list): List of two tuples defining the lattice primitive vectors.
            K_point (tuple) : The position of the K_point- the high symmetry point

        Returns:
            graph_copy (graph): A graph object that respects the periodic boundary conditions.
            new_nodes (list): a list of the new nodes
        """
        A = np.array([periodic_vectors[0], periodic_vectors[1]]).T
        B =2 * np.pi * np.linalg.inv(A)
        dual_vectors=[tuple(B[0, :]), tuple(B[1, :])]
        graph_copy = periodic_graph(periodic_vectors,dual_vectors,K_point)  # Create a copy of the graph as a periodic graph
        new_nodes = dict()
        for node in self.nodes:
            node_position,node_shift=canonical_position(node.position, periodic_vectors[0], periodic_vectors[1])  # Get the canonical position of the node  
            if abs(node_shift[0])>constants.MAX_ADJACENT_CELLS or abs(node_shift[1])>constants.MAX_ADJACENT_CELLS:
                continue
            if node_position not in new_nodes:  # Check if the node already exists in the copy
                new_node = node.copy()
                new_node.position = node_position  # Set the position of the new node to the canonical
                graph_copy.add_node(new_node)  # Add the new node to the copy
                new_nodes[node_position] = new_node  # Store the new node in the dictionary
        return graph_copy,new_nodes
    
    def create_periodic_copy(self,periodic_vectors:list,K_point : tuple)-> "periodic_graph":
        """Create a periodic version of the graph based on the given primitive vectors.

        This method wraps node positions into a unit cell and reconstructs connectivity 
        considering periodic boundary conditions.

        Parameters:
            periodic_vectors (list): List of two tuples defining the lattice primitive vectors.
            K_point (tuple) : The position of the K_point- the high symmetry point

        Returns:
            graph_copy: A graph object that respects the periodic boundary conditions.
        
        Raises:
            ValueError: if the periodic vectors given or of the wrong shape (not two 2d vectors)
        """
        if np.array(periodic_vectors).shape != (2,2):
            raise ValueError("The periodic vectors should be two dimensional!")
        graph_copy,new_nodes=self._create_periodic_add_nodes(periodic_vectors,K_point)
        existing_edges = set()
        for node in self.nodes:
            node_position,node_Shift=canonical_position(node.position, periodic_vectors[0], periodic_vectors[1])  # Get the canonical position of the node
            new_node= new_nodes[node_position]  # Get the new node from the dictionary                
            for neighbor, _ in node.neighbors:
                neighbor_position,neighbor_shift=canonical_position(neighbor.position, periodic_vectors[0], periodic_vectors[1])
                if np.allclose(neighbor_position,node_position,atol=1e-6):
                    continue
                new_neighbor=new_nodes[neighbor_position]  # Get the new neighbor from the dictionary
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
    
    def edge_color_hash(self,key: tuple)->tuple:
        """Generate a unique color from an edge key using a hash.
        
        Args:
        key (tuple)- the key to turn intro color
        
        Returns:
        (r,g,b)- color coding
        """
        
        flat_key = tuple(key[:4]) + tuple(key[4])
        h = hashlib.md5(str(flat_key).encode()).hexdigest()
        r = int(h[0:2], 16) / 255
        g = int(h[2:4], 16) / 255
        b = int(h[4:6], 16) / 255
        return (r, g, b)
    def _plot_periodic_edges(self,ax,node: Node,node2: Node,periodic_offset: tuple,plotted_periodic_edges: list,lattice_vectors: list):
        """
        Subfunction to plot the periodic edges. Adds them to the list of plotted_periodic_edges.

        Args:
            ax (matplotlib.axes.Axes): The matplotlib axis on which to draw the plot.
            node(Node): One of the nodes in the edge
            node2(Node): One of the nodes in the edge
            periodic_offset (tuple of integers): a tuple of integers with the period offset between these neighbors. 
            lattice_vectors(list): Is the list of the lattice vectors used to compute the location of 
                                the edge that crosses boundary


        """
        if periodic_offset==(0,0):
            return 
        key = (node.lattice_index, node.sublattice_id,node2.lattice_index,node2.sublattice_id, tuple(periodic_offset))
        inv_key = (node2.lattice_index,node2.sublattice_id,node.lattice_index, node.sublattice_id, tuple(-np.array(periodic_offset)))
        if key in plotted_periodic_edges or inv_key in plotted_periodic_edges:
            return  # Already plotted this edge or its inverse
        color = self.edge_color_hash(key)
        A = np.array([lattice_vectors[0], lattice_vectors[1]]).T
        Shift=A@np.array(periodic_offset)
        new_position=(node2.position[0]+Shift[0],node2.position[1]+Shift[1])
        new_position_inv=(node.position[0]-Shift[0],node.position[1]-Shift[1])
        ax.plot([node.position[0], new_position[0]], [node.position[1], new_position[1]],color=color,linestyle='-',linewidth=constants.LINEWIDTH,alpha=0.7)
        ax.plot([node2.position[0], new_position_inv[0]], [node2.position[1], new_position_inv[1]],color=color,linestyle='-',linewidth=constants.LINEWIDTH,alpha=0.7)
        plotted_periodic_edges.add(key)

    def _plot_unitcell(self,ax:matplotlib.axes.Axes,lattice_vectors:list[tuple[float,float],tuple[float,float]]):
        """
        Subfunction that plots the unit cell
        Args:
            ax (matplotlib.axes.Axes): The matplotlib axis on which to draw the plot.
            lattice_vectors(list): the list of the lattice vectors used to compute the boundary of the unit cell.
        """
        periodic_vector1 = np.array(lattice_vectors[0])
        periodic_vector2 = np.array(lattice_vectors[1])
        corners=[-(periodic_vector1 + periodic_vector2)/2,(periodic_vector1 - periodic_vector2)/2,(periodic_vector1 + periodic_vector2)/2,(-periodic_vector1 + periodic_vector2)/2]
        polygon = Polygon(corners, closed=True, fill=False, edgecolor='black', linestyle='--', linewidth=constants.LINEWIDTHOFUNITCELLBDRY)
        ax.add_patch(polygon)

    def _plot_single_graph(self,ax:matplotlib.axes.Axes, node_colors:list,max_distance:float=None):

        """
        Subfunction that plots the graph without distiguishing the different subgraphs
        Args:
            ax (matplotlib.axes.Axes): The matplotlib axis on which to draw the plot.
            node_colors (list or dict, optional): Specifies the colors for nodes. 
                Can be a list of colors (applied cyclically to nodes) or a dictionary mapping 'sublattice_id' to color for differentiation. 
                Defaults to a default color scheme.
            max_distance (float, optional): If provided, only nodes and their connected edges whose *absolute position* is within
                this distance from the origin (0,0) will be plotted. This helps focus the visualization on a central region of the lattice.
                Defaults to plotting all available nodes and edges.
        """
        for node in self.nodes: # if we don't want to distinguish subgraphs, just plot all the nodes in the max_distance
            node_max_distance=calculate_distance(node)
            if max_distance is None or node_max_distance < max_distance: 
                ax.scatter(node.position[0], node.position[1], color=node_colors[0],marker="o",s=constants.MARKERSIZE)
                for neighbor, periodic_offset in node.neighbors: #and plot  all edges all the same.
                    lineType= '-' if periodic_offset == (0, 0) else '--'
                    neighbor_max_distance=calculate_distance(neighbor)
                    if max_distance is None or neighbor_max_distance < max_distance:
                        ax.plot([node.position[0], neighbor.position[0]], [node.position[1], neighbor.position[1]], lineType, color=node_colors[0], linewidth=constants.LINEWIDTH)

    def plot(self,ax:matplotlib.axes.Axes, node_colors:list,max_distance:float=None,differentiate_subgraphs:bool=False,lattice_vectors:list[tuple[float,float],tuple[float,float]]=None):
        """
        Plot the graph with optional coloring and filtering.

        Args:
            ax (matplotlib.axes.Axes): The matplotlib axis on which to draw the plot.
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
        ax.set_aspect('equal')
        ax.axis('off')
        if lattice_vectors is not None: # if we have lattic, vector we want to plot the unit cell
            self._plot_unitcell(ax,lattice_vectors)
            plotted_periodic_edges = set()
        if differentiate_subgraphs: #if we want to differentiate subgraph, we:
            if len(node_colors) < self.number_of_subgraphs+1: #make sure there  are enough colors
                raise ValueError("Not enough colors provided for the number of subgraphs.")
            for node in self.nodes: # we plot only the nodes that are inside the unit cell
                node_max_distance=calculate_distance(node) 
                if max_distance is None or node_max_distance < max_distance:
                    ax.scatter(node.position[0], node.position[1], color=node_colors[node.sublattice_id],marker="o",s=constants.MARKERSIZE)
                    for neighbor, periodic_offset in node.neighbors: #we plot all the edges
                        if periodic_offset != (0, 0): 
                            self._plot_periodic_edges(ax,node,neighbor,periodic_offset,plotted_periodic_edges,lattice_vectors) #plotting the periodic edges is more complicated- so it is in auxilary function
                        else: #otherwise, choose the color, if it is an interlayer edge or a intralayer edge
                            if neighbor.sublattice_id == node.sublattice_id:
                                linecolor=node_colors[node.sublattice_id]
                            else:
                                linecolor=node_colors[-1]
                            ax.plot([node.position[0], neighbor.position[0]], [node.position[1], neighbor.position[1]], '-', color=linecolor, linewidth=constants.LINEWIDTH)
            
            return
        self._plot_single_graph(self,ax, node_colors,max_distance)
        

class periodic_graph(graph):
    """
    A subclass of graph that includes periodicity by incorporating lattice vectors and momentum space structure.

    This class is designed to support physics applications like tight-binding models on periodic lattices. It allows
    construction of Hermitian Laplacians that incorporate both intra-cell and inter-cell connections with momentum-dependent
    phase factors.
    
    Attributes:
        lattice_vectors (list): Primitive lattice vectors of the graph.
        dual_vectors (list): Reciprocal lattice vectors derived from lattice_vectors.
    """
    def __init__(self,lattice_vectors:list[tuple[float,float],tuple[float,float]]=[constants.v1, constants.v2],
                 dual_vectors:list[tuple[float,float],tuple[float,float]]=[constants.k1, constants.k2],
                 K_point:tuple[float,float]= constants.K_POINT_REG): 
        """
        Initialize a periodic graph with given lattice and reciprocal vectors.

        Args:
            lattice_vectors (list): Real space primitive vectors of the unit cell.
            dual_vectors (list): Reciprocal space basis vectors.
            K_point (tuple): The lattice coordinates f the K point that will be used for the band computations.  
        """
        super().__init__()
        self.lattice_vectors=lattice_vectors
        self.dual_vectors=dual_vectors
        self.K_point=K_point
        self.num_of_periodic_edges=0
        self.edges_list=[]

    def build_adj_matrix(self,inter_graph_weight:float=1.0,intra_graph_weight:float=1.0)->tuple[csr_matrix,list[int,int,tuple[int,int]]]: 
        """
        Construct the Laplacian matrix ignoring periodic phase factors.

        Args:
            inter_graph_weight (float): Weight of connections between subgraphs.
            intra_graph_weight (float): Weight of connections within a subgraph.

        Returns:
            tuple: A tuple containing:
                - laplacian (scipy.sparse.csr_matrix): Initial Laplacian matrix.
                - periodic_edges (list): List of periodic edges needing phase factors.
        Raises:
            ValueError- if the matrix is not Hermitian
        """
        n = len(self.nodes)
        #precompute that hashing node graph index to index
        node_key_to_index = {(node.lattice_index,node.sublattice_id): i for i, node in enumerate(self.nodes)}
        row_indices = []
        col_indices = []
        data = []
        processed_edges = set()
        # laplacian = lil_matrix((n, n), dtype=complex)
        periodic_edges=[]
        for i, node in enumerate(self.nodes):
            #For now solving the free particle case, so the diagonal is zero.  
            row_indices.append(i)
            col_indices.append(i)
            data.append(0.0 + 0.0j) 
            for neighbor, periodic_offset in node.neighbors:
                j = node_key_to_index[(neighbor.lattice_index,neighbor.sublattice_id)]
                edge_key = (min(i, j), max(i, j), periodic_offset)
                if edge_key in processed_edges:
                    continue
                processed_edges.add(edge_key)
                if periodic_offset == (0, 0):
                # Intra-cell edge
                    if node.sublattice_id == neighbor.sublattice_id:
                        weight = intra_graph_weight
                    else:
                        weight = inter_graph_weight
                    # Add both (i,j) and (j,i) for symmetry
                    row_indices.extend([i, j])
                    col_indices.extend([j, i])
                    data.extend([weight, weight])
                else:
                    # Periodic edge - store for later phase factor application
                    periodic_edges.append([i, j, periodic_offset])  
                    # Also store the reverse edge
                    reverse_offset = (-periodic_offset[0], -periodic_offset[1])
                    periodic_edges.append([j, i, reverse_offset])
        #more efficient construction in coo_matrix
        laplacian_coo = coo_matrix((data, (row_indices, col_indices)),shape=(n, n), dtype=complex)
        #manipulation is better in csr 
        laplacian = laplacian_coo.tocsr()
        if not self._is_hermitian_sparse(laplacian):
            raise ValueError("Laplacian matrix is not Hermitian.")
    
        logger.info(f"Matrix construction: {n}×{n}, {len(periodic_edges)} periodic edges")
        logger.info(f"Matrix density: {laplacian.nnz / (n*n) * 100:.2f}%")
        logger.info(f"Total periodic edges: {len(periodic_edges)}")
        return laplacian ,periodic_edges
    def _is_hermitian_sparse(self,matrix: csr_matrix, rtol: float = constants.NUMERIC_TOLERANCE) -> bool:
        """
        Efficiently check if sparse matrix is Hermitian.
        
        Args:
            matrix: Sparse matrix to check
            rtol: Relative tolerance for comparison
            
        Returns:
            bool: True if matrix is Hermitian within tolerance
        """
        # For sparse matrices, this is much faster than full comparison
        diff = matrix - matrix.getH()
        return diff.max() < rtol and abs(diff.min()) < rtol

    def build_laplacian(self,inter_graph_weight:float=1.0,intra_graph_weight:float=1.0, Momentum:tuple[float,float]=[0.0,0.0],
                        laplacian:csr_matrix=None,periodic_edges:list=None, verbosity:bool=False)->lil_matrix: 
        """
        Construct the full Laplacian matrix with periodic phase factors applied.

        Args:
            inter_graph_weight (float): Weight of inter-subgraph connections.
            intra_graph_weight (float): Weight of intra-subgraph connections.
            Momentum (list): 2D quasimomentum vector for phase factor calculation.
            laplacian (scipy.sparse matrix, optional): Precomputed base Laplacian.
            periodic_edges (list, optional): List of periodic edges (i, j, shift).
            verbosity (bool):  If true, log more infomation for debugging. 
        Returns:
            scipy.sparse matrix: Hermitian Laplacian with phase factors.
        
        Raises:
            ValueError if laplacian provided without periodic edges, or the laplacian is not Hermitian
        """
        if laplacian is None:
            if periodic_edges is None:
                laplacian_copy,periodic_edges = self.build_adj_matrix(inter_graph_weight,intra_graph_weight)
            else:
                raise ValueError("Missing input! when providing the Laplacian one should also provide periodic edges list!")
        else:
            laplacian_copy=laplacian.copy()
        
        # Vectorized phase factor computation
        i_indices = np.array([edge[0] for edge in periodic_edges], dtype=int)
        j_indices = np.array([edge[1] for edge in periodic_edges], dtype=int)
        
        # Handle offset tuples properly
        offsets = np.array([[edge[2][0], edge[2][1]] for edge in periodic_edges], dtype=float)
   
        
        # Vectorized phase calculation
        phases = np.dot(offsets, np.array(Momentum)) * 2 * np.pi
        phase_factors = np.exp(1j * phases)
        
        # Batch update of matrix elements
        laplacian_copy[i_indices, j_indices] += phase_factors
        if not (laplacian_copy != laplacian_copy.getH()).nnz == 0:
            raise ValueError("Laplacian matrix is not Hermitian.")
        return laplacian_copy   
    def _sparse_eigensolve(self, Laplacian: lil_matrix, min_bands:int, max_bands:int):
        """
        Attempt sparse eigenvalue computation with multiple strategies.
        Args:
            min_bands (int): the index of the lowest band to plot.
            max_bands (int): the index of the highest band to plot.
            laplacian (scipy.sparse matrix):   The Laplacian matrix
        Returns:
            np.ndarray: Sorted list of computed eigenvalues.

        """
        num_bands=max_bands-min_bands+1
        strategies = [
            # Strategy 1: Shift-invert around zero
            {'sigma': 0.0, 'which': 'LM'},
            # Strategy 2: Smallest magnitude eigenvalues
            {'sigma': None, 'which': 'SM'},
            # Strategy 3: Smallest real eigenvalues
            {'sigma': None, 'which': 'SR'},
        ]
        
        for i, strategy in enumerate(strategies):
            try:
                logger.debug(f"Trying sparse strategy {i+1}: {strategy}")
                
                # Request more eigenvalues than needed for stability
                k_request = min(num_bands + 5, Laplacian.shape[0] - 2)
                
                eigvals, _ = eigsh(Laplacian,k=k_request,sigma=strategy['sigma'],which=strategy['which'],maxiter=1000,tol=constants.NUMERIC_TOLERANCE)
                
                # Sort and extract requested range
                eigvals = np.sort(eigvals.real)
                if len(eigvals) >= max_bands:
                    return eigvals[min_bands-1:max_bands]
                else:
                    logger.warning(f"Insufficient eigenvalues computed: {len(eigvals)} < {max_bands}")
                    
            except (ArpackNoConvergence, np.linalg.LinAlgError) as e:
                logger.debug(f"Sparse strategy {i+1} failed: {e}")
                continue
        
        return None  # All sparse strategies failed
    
    def _dense_eigensolve(self, Laplacian:lil_matrix, min_bands:int, max_bands:int):
        """
        Dense eigenvalue computation with memory management.
        Args:
            min_bands (int): the index of the lowest band to plot.
            max_bands (int): the index of the highest band to plot.
            laplacian (scipy.sparse matrix):   The Laplacian matrix
        Returns:
            np.ndarray: Sorted list of computed eigenvalues.
        """
        H = Laplacian.toarray()
        
        # Use scipy's optimized eigenvalue solver
        try:
            E, V = np.linalg.eigh(H)
            eigvals = E[min_bands-1:max_bands]
            return np.sort(eigvals)
        except np.linalg.LinAlgError as e:
            logger.error(f"Dense eigenvalue computation failed: {e}")
            return np.full(max_bands - min_bands + 1, np.nan)

    def compute_bands_at_k (self,Momentum:tuple[float,float],min_bands:int,max_bands:int,inter_graph_weight:float=1.0,
                            intra_graph_weight:float=1.0,laplacian:lil_matrix=None,periodic_edges:list=None)->np.ndarray:
        """
        Solve the eigenvalue problem for a given momentum.

        Args:
            Momentum (tuple[float,float]): Quasimomentum vector.
            min_bands (int): the index of the lowest band to plot.
            max_bands (int): the index of the highest band to plot.
            inter_graph_weight (float): Inter-subgraph coupling strength.
            intra_graph_weight (float): Intra-subgraph coupling strength.
            laplacian (scipy.sparse matrix, optional): Optional precomputed Laplacian.
            periodic_edges (list, optional): Optional list of periodic edges.
        Returns:
            np.ndarray: Sorted list of computed eigenvalues.
        
        Raises:
            ValueError: if laplacian provided without periodic edges
        """
        if laplacian is None:
            if periodic_edges is None:
                Laplacian=self.build_laplacian(inter_graph_weight=inter_graph_weight,intra_graph_weight=intra_graph_weight,Momentum=Momentum)
            else:
                raise ValueError("Missing input! when providing the Laplacian one should also provide periodic edges list!")
        else:
            Laplacian=self.build_laplacian(Momentum=Momentum,laplacian=laplacian,periodic_edges=periodic_edges)
        # Convert to CSR format for better performance
        if not isinstance(Laplacian, csr_matrix):
            Laplacian = Laplacian.tocsr()
        matrix_size = Laplacian.shape[0]
        num_bands=max_bands-min_bands+1
        if matrix_size > 100 and num_bands < matrix_size // 2:
            try:
                # Use shift-invert mode around zero for better convergence
                eigvals = self._sparse_eigensolve(Laplacian, num_bands, min_bands, max_bands)
                if eigvals is not None:
                    return eigvals
                    
            except (ArpackNoConvergence, np.linalg.LinAlgError) as e:
                logger.warning(f"Sparse solver failed at k={Momentum}: {e}")
        
        # Strategy 2: Fallback to dense solver with memory check
        memory_estimate_mb = (matrix_size ** 2 * 16) / (1024 ** 2)  # Complex128 = 16 bytes
        
        if memory_estimate_mb > 500:  # More than 500MB
            logger.warning(f"Large matrix ({matrix_size}x{matrix_size}, ~{memory_estimate_mb:.1f}MB). "
                         f"Consider reducing system size or using sparse methods.")
            
        try:
            return self._dense_eigensolve(Laplacian, min_bands, max_bands)
        except MemoryError:
            logger.error(f"Out of memory for matrix size {matrix_size}x{matrix_size}")
            return np.full(num_bands, np.nan)
    
    
    def _plot_band_compute_k_grid(self,k1_vals:np.ndarray,k2_vals:np.ndarray,center: tuple[float,float])->tuple[np.ndarray,np.ndarray]:
        """
        Subfunction to compute the k_grid around the center given in center.

        Args:
            k1_vals(ndarray) : The array of values along the first axis.
            k2_vals(ndarray) : The array of values along the second axis.
            center (tuple): The center ot the grid.
        
        Returns:
            tuple: A tuple containing:
                - kx_grid (ndarray): The Cartesian x coordiante of the grid.
                - ky_grid (ndarray): The Cartesian y coordiante of the grid.
        """
        k1_grid, k2_grid = np.meshgrid(k1_vals, k2_vals, indexing='ij')
        b1 = np.array(self.dual_vectors[0])
        b2 = np.array(self.dual_vectors[1])
        kx_grid = (center[0]+k1_grid) * b1[0] + (center[1]+k2_grid) * b2[0]
        ky_grid = (center[0]+k1_grid) * b1[1] + (center[1]+k2_grid) * b2[1]
        return kx_grid,ky_grid
    
    def _plot_band_get_bands(self,k1_vals:np.ndarray,k2_vals:np.ndarray,num_of_points:int,min_bands:int,max_bands:int,center: tuple[float,float],
                             laplacian: lil_matrix,periodic_edges:list)->np.array:
        """
        Subfunction to compute the bands on the grid given by kx_grid,ky_grid.

        Args:
            k1_vals(ndarray) : the array of values along the first axis.
            k2_vals(ndarray) : the array of values along the second axis.
            num_of_points (int): Number of points per momentum axis.
            min_bands (int): the index of the lowest band to plot.
            max_bands (int): the index of the highest band to plot.
            center (tuple): the center ot the grid.
            laplacian (scipy.sparse.lil_matrix): The adjency matrix.
            periodic_edges (list): List of periodic edges needing phase factors.
        Returns:
            bands(np.array) : a 3d array that have the the bands for each point on the grid. 
        """
        num_of_bands=max_bands-min_bands+1
        bands = np.zeros((num_of_points, num_of_points, num_of_bands))
        for i in range(num_of_points):
            for j in range(num_of_points):
                k = (center[0]+k1_vals[i], center[1]+k2_vals[j])
                success = False
                for attempt in range(constants.attemp_num):  # Try up to attemp_num perturbations
                    try:

                        bands[i, j, :]=self.compute_bands_at_k (Momentum=k,min_bands=min_bands,max_bands=max_bands,laplacian=laplacian,periodic_edges=periodic_edges)
                        logger.debug(f"Laplacian shape: {laplacian.shape}, nnz: {laplacian.nnz}")
                        logger.debug(f"the adj matrix is {laplacian}")
                        success = True
                        break    
                    except Exception as e:
                        k += np.random.normal(scale=1e-4, size=2)  # Small jitter
                if not success:
                    logger.warning(f"WARNING: Failed at k=({k}) even after retries.")
                    bands[i, j, :] = np.nan  # or some default/fallback value
        return bands
    
    def plot_band_structure(self,ax:matplotlib.axes.Axes, num_of_points:int,min_bands:int,max_bands:int,
                            inter_graph_weight:float,intra_graph_weight:float,k_max:float=0.5,
                            k_min:float=-0.5,K_flag:bool=False,laplacian:lil_matrix=None,
                            periodic_edges:list=None)->tuple[lil_matrix,list]:
        """
        Plot the band structure over a Brillouin zone grid.

        Args:
            ax (matplotlib.axes): Matplotlib 3D axes to plot on.
            num_of_points (int): Number of points per momentum axis.
            min_bands (int): the index of the lowest band to plot.
            max_bands (int): the index of the highest band to plot.
            inter_graph_weight (float): Inter-subgraph edge weight.
            intra_graph_weight (float): Intra-subgraph edge weight.
            k_max (float): Maximum value in k-space.
            k_min (float): Minimum value in k-space.
            K_flag(bool):  If True, center the plot around the K( = 1/3 (k_1-k_2) or =1/3(v_1+v_2)) point 
                                If False, center the plot around the origin
            laplacian (scipy.sparse.lil_matrix): If exists, it is the adjency matrix.
            periodic_edges (list): if exists, the list of periodic edges needing phase factors.

        Returns:
            tuple: A tuple containing:
                - laplacian (scipy.sparse.lil_matrix): Initial Laplacian matrix.
                - periodic_edges (list): List of periodic edges needing phase factors.
        
        Raises:
            ValueEror if ax is not a 3d Axes object
        """
        if ax.name!='3d':
            raise ValueError("Expecting a 3d Axes object")
        if K_flag:
            center=self.K_point
        else:
            center=(0,0)
            
        k1_vals = np.linspace(k_min, k_max, num_of_points)
        k2_vals = np.linspace(k_min, k_max, num_of_points)
        kx_grid,ky_grid=self._plot_band_compute_k_grid(k1_vals,k2_vals,center)
        if laplacian is None  or periodic_edges is None:
            logger.info(f"recomputing the laplacian ")
            laplacian,periodic_edges = self.build_adj_matrix(inter_graph_weight,intra_graph_weight)
        logger.debug(f"the adj matrix is {laplacian}")
        bands=self._plot_band_get_bands(k1_vals,k2_vals,num_of_points,min_bands,max_bands,center,laplacian,periodic_edges)
        for n in range(bands.shape[2]):
            ax.plot_surface(kx_grid, ky_grid, bands[:, :, n], cmap='viridis', alpha=0.8)
        ax.set_xlabel('$k_x$')
        ax.set_ylabel('$k_y$')
        ax.set_zlabel('Energy $E_n(k)$')
        return laplacian,periodic_edges

class hex_lattice() : 
    """
    hex_lattice class to create a hexagonal lattice of nodes arranged in a grid.

    Attributes:
        graph (raph): Underlying graph storing nodes and their connections.
        maxsize_n (int): Maximum range in the N-direction.
        maxsize_m (int): Maximum range in the M-direction.
        lattice_vectors (list): Lattice vectors defining the hexagonal basis.
    """
    def __init__(self, maxsize_n:int, maxsize_m:int,radius:float=None,lattice_vectors:list[tuple[float,float],tuple[float,float]]=[constants.v1,constants.v2]):
        """
        Initialize a hex_lattice instance.

        Args:
            maxsize_n (int): Range of lattice nodes in the N-direction.
            maxsize_m (int): Range of lattice nodes in the M-direction.
            radius (float, optional): Optional max_distance cutoff to exclude distant nodes.
            lattice_vectors (list): List of lattice vectors defining the hexagonal grid.
        """
        self.graph= graph() # Create a new graph instance
        self.max_rows = maxsize_n # Number of nodes in the N direction
        self.max_cols = maxsize_m # Number of nodes in the M direction 
        self.lattice_vectors=lattice_vectors
        self.create_lattice(radius)

    def create_lattice(self,radius:float=None):
        """
        Populate the graph with hexagonally arranged nodes and add neighbors in cardinal directions.

        Args:
            radius (float, optional): Only nodes within this radius are included.
        """
        for y in range(-self.max_rows,self.max_rows+1):
            for x in range(-self.max_cols,self.max_cols+1):
                position = (x*self.lattice_vectors[0][0] + y*self.lattice_vectors[1][0], x*self.lattice_vectors[0][1] + y*self.lattice_vectors[1][1])
                node = Node(position, (x,y))
                if radius is None or calculate_distance(node) <= radius:  # Check if the node is within the specified radius
                    self.graph.add_node(node)
        for node in self.graph.nodes:
            x, y = node.lattice_index  # Get the index of the node
            # Add neighbors in eacj of the natural directions
            # Using periodic boundary conditions
            directions = [(1, 0), (0, 1),(-1,0),(0,-1)]
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                try:
                    neighbor = self.graph.get_node_by_index((nx, ny))
                    self.graph.add_edge(node, neighbor)
                except KeyError:
                    pass  # Neighbor doesn't exist in the current graph — safely ignore
    def rotate(self, angle:float):
        """
        Rotate the entire lattice by a specified angle.

        Args:
            angle (float): Angle in radians to rotate the lattice.
        """
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        for node in self.graph.nodes:
            x, y = node.position
            new_x = x * cos_angle - y * sin_angle
            new_y = x * sin_angle + y * cos_angle
            node.position = (new_x, new_y)

    def rotate_rational_TBG(self,a:int,b:int,N:float,alpha:float):
            
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
        sin_angle = np.sqrt(3) * b / (alpha * N)
        
        for node in self.graph.nodes:
            x, y = node.position
            new_x = x * cos_angle - y * sin_angle
            new_y = x * sin_angle + y * cos_angle
            node.position = (new_x, new_y)

class TBG :
    """
    TBG (Twisted Bilayer graphene) class that constructs a bilayer structure
    with adjustable rotation angle and interlayer coupling.

    Attributes:
        a (int): Integer controlling the rational rotation angle.
        b (int): Integer controlling the rational rotation angle.
        N (float): Computed normalization factor for rotation.
        alpha (float): Scale factor based on arithmetic conditions.
        factor (float): Adjustment factor depending on a.
        lattice_vectors(list): Lattice vectors for periodicity.
        dual_vectors (list): Reciprocal lattice vectors.
        top_layer (hex_lattice): Rotated hexagonal top layer.
        bottom_layer (hex_lattice): Rotated hexagonal bottom layer.
        full_graph (graph): Combined graph of top and bottom layers.
    """
    def __init__(self, maxsize_n:int, maxsize_m:int, a:int, b:int,interlayer_dist_threshold:float= 1.,
                 unit_cell_radius_factor:int=1): # Initialize the TBG with the given parameters
        """Initialize the TBG structure with a given size and rational rotation.

        Args:
            maxsize_n (int): Number of nodes in the N direction.
            maxsize_m (int): Number of nodes in the M direction.
            a (int): Integer rotation parameter.
            b (int): Integer rotation parameter.
            interlayer_dist_threshold (float): Threshold for connecting interlayer nodes.
            unit_cell_radius_factor (int): Controls radius scaling of node interaction.

        Raises:
            ValueError: If invalid configuration is passed.
        """
        validate_ab(a,b)
        if interlayer_dist_threshold <= 0:
            raise ValueError("DistPara must be a positive number.")
        self.a=a
        self.b=b
        self.N, self.alpha,self.factor,_ = compute_twist_constants(self.a, self.b) # Calculate the constants N and alpha for the rational TBG angle
        logger.info(f"Calculated N: {self.N}, alpha: {self.alpha} for a={self.a}, b={self.b}")
        self.lattice_vectors=[constants.v1,constants.v2]
        self.top_layer=hex_lattice(maxsize_n, maxsize_m,unit_cell_radius_factor*self.factor*self.N,self.lattice_vectors) # create the top layer of the TBG
        self.top_layer.rotate_rational_TBG(self.a, self.b,self.N,self.alpha)
        self.bottom_layer=hex_lattice(maxsize_n, maxsize_m,unit_cell_radius_factor*self.factor*self.N,self.lattice_vectors) # create the bottom layer of the TBG
        self.bottom_layer.rotate_rational_TBG(self.a, -self.b,self.N,self.alpha)

        self.full_graph =self.top_layer.graph.copy(0) # Create a copy of the top layer graph
        self.full_graph.append(self.bottom_layer.graph,1)

        self._connect_layers(interlayer_dist_threshold)

        if self.a%3==0: # Calculate the primitive vectors of the TBG structure,
            self.lattice_vectors= [tuple(item * self.N for item in constants.k1), tuple(item * self.N for item in constants.k2)]
            self.dual_vectors= [tuple(item * self.N for item in constants.v1), tuple(item * self.N for item in constants.v2)]
        else: # If a is not a multiple of 3, use the hexagonal lattice vectors and the duals are  the regular duals
            self.lattice_vectors= [tuple(item * self.N for item in constants.v1), tuple(item * self.N for item in constants.v2)]
            self.dual_vectors= [tuple(item * self.N for item in constants.k1), tuple(item * self.N for item in constants.k2)]
        logger.info(f"Lattice vectors: {self.lattice_vectors}, and the duals are {self.dual_vectors}")

    def _connect_layers(self,interlayer_dist_threshold:float= 1.):
        """
        A subfunction to connect the two layers, using spatial indexing and vectorization.
        Args:
            interlayer_dist_threshold (float): Threshold for connecting interlayer nodes.
            
        """
        top_nodes = self.top_layer.graph.nodes
        bottom_nodes = self.bottom_layer.graph.nodes
        # If there is an empty layer- exit
        if not top_nodes or not bottom_nodes:
            return
        
        # Create spatial index for top layer and bottom
        top_positions = np.array([node.position for node in top_nodes])
        bottom_positions = np.array([node.position for node in bottom_nodes])

        #create a box around the top layer,
        top_bbox = np.array([top_positions.min(axis=0) - interlayer_dist_threshold,
                        top_positions.max(axis=0) + interlayer_dist_threshold])
        # Remove the nodes that are too far in the bottom layer
        in_bbox = ((bottom_positions >= top_bbox[0]) & 
                (bottom_positions <= top_bbox[1])).all(axis=1)
        #in the unlikely case of no nodes conneections:
        if not in_bbox.any():
            logger.info("No nodes within connection threshold - skipping interlayer connections")
            return
        # Work only with potentially connecting nodes
        candidate_bottom_indices = np.where(in_bbox)[0]
        candidate_bottom_positions = bottom_positions[candidate_bottom_indices]
        candidate_bottom_nodes = [bottom_nodes[i] for i in candidate_bottom_indices]

        logger.info(f"Filtered to {len(candidate_bottom_indices)} candidate bottom nodes "
                f"from {len(bottom_nodes)} total")
        
        initial_radius = min(interlayer_dist_threshold, constants.INITIAL_RADIUS_DEFAULT)  # Don't search too far initially

        #create a cDkTree for efficient search
        top_tree = cKDTree(top_positions)

        connections_made = 0
        # Process in batches for better memory usage and progress tracking
        batch_size = min(1000, len(candidate_bottom_nodes))

        for batch_start in range(0, len(bottom_nodes), batch_size):
            batch_end = min(batch_start + batch_size, len(bottom_nodes))
            batch_positions = candidate_bottom_positions[batch_start:batch_end]
            
            # Find all neighbors within threshold for this batch
            neighbor_indices = top_tree.query_ball_point(batch_positions, r=initial_radius)
            # If initial radius was too small, expand search
            if initial_radius < interlayer_dist_threshold:
                for i, neighbors in enumerate(neighbor_indices):
                    if not neighbors:  # No neighbors found, try larger radius
                        expanded_neighbors = top_tree.query_ball_point(
                            batch_positions[i], r=interlayer_dist_threshold
                        )
                        neighbor_indices[i] = expanded_neighbors
        
            # Process connections for this batch
            for i, neighbors in enumerate(neighbor_indices):
                if not neighbors:
                    continue
                bottom_idx = candidate_bottom_indices[batch_start + i]
                bottom_node = bottom_nodes[bottom_idx]
                bottom_node_in_graph = self.full_graph.get_node_by_index(bottom_node.lattice_index, 1)
                
                for top_idx in neighbors:
                    top_node = top_nodes[top_idx]
                    top_node_in_graph = self.full_graph.get_node_by_index(top_node.lattice_index, 0)
                    
                    self.full_graph.add_edge(bottom_node_in_graph, top_node_in_graph)
                    connections_made += 1
            
            # Progress logging for large systems
            if len(bottom_nodes) > 1000:
                progress = (batch_end / len(bottom_nodes)) * 100
                if progress % 20 < (batch_size / len(bottom_nodes)) * 100:
                    logger.info(f"Interlayer connections: {progress:.0f}% complete")
        
        logger.info(f"Created {connections_made} interlayer connections")


    def __repr__(self):
        s="Upper Layer:\n"
        s+=self.top_layer.graph.__repr__()
        s+="Bottom Layer:\n"
        s+=self.bottom_layer.graph.__repr__()
        return s
    
    def plot(self,ax:matplotlib.axes.Axes,plot_color_top:str='b',
              plot_color_bottom:str='r',plot_color_full:str='g',
              radius:float=None,plot_seperate_layers:bool=False,
              lattice_vectors:list[tuple[float,float],tuple[float,float]]=None): #Plots the TBG structure
        """
        Plot the full TBG structure or its layers separately.

        Args:
            ax (matplotlib.axes.Axes): Axis object to draw the plot.
            plot_color_top (str): Color for the top layer.
            plot_color_bottom (str): Color for the bottom layer.
            plot_color_full (str): Color for edges connecting the two layers.
            radius (float): Optional cutoff for visible nodes.
            plot_seperate_layers (bool): Whether to plot layers separately or together.
            lattice_vectors (list): the list of the lattice vectors
        """
        if ax is None:
            fig = Figure(figsize=(6, 4))
            ax= fig.subplots(1,1)  
        if plot_seperate_layers==True:
            self.top_layer.graph.plot(ax,plot_color_top,radius)
            self.bottom_layer.graph.plot(ax,plot_color_bottom,radius) 
            return 
        self.full_graph.plot(ax,node_colors=[plot_color_top, plot_color_bottom, plot_color_full],max_distance=radius, differentiate_subgraphs=True,lattice_vectors=lattice_vectors) # Plot the full graph with the specified color

    def plot_unit_cell(self, plot_color_top:str='b', plot_color_bottom:str='r',
                        plot_color_full:str='g',lattice_vectors:list[tuple[float,float],tuple[float,float]]=None):
        """
        Plot only the unit cell of the TBG structure.

        Args:
            plot_color_top (str): Color for the top layer.
            plot_color_bottom (str): Color for the bottom layer.
            plot_color_full (str): Color for full graph.
            lattice_vectors (list): the list of the lattice vectors
        """
        return self.plot(plot_color_top, plot_color_bottom, plot_color_full, radius=self.N/2,lattice_vectors=lattice_vectors) # Plot the unit cell of the TBG structure


