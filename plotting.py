"""
Plotting utilities for twisted bilayer graphene (TBG) structures and band analysis.

This module provides specialized plotting functions for:
- Graph structures with periodic boundaries
- TBG lattice visualization  
- Band structure plotting
- Dirac point analysis visualization

Classes:
    graph_plotter: Handles lattice and graph structure visualization
    band_plotter: Manages band structure and 3D plotting
    dirac_plotter: Specialized plotting for Dirac point analysis

The module separates visualization logic from core physics calculations,
providing clean interfaces for different types of plots.
"""

import constants
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.figure import Figure
from typing import List, Tuple, Optional, Set, Dict, Any
from utils import calculate_distance, edge_color_hash

# Configure logging
logger = constants.logging.getLogger(__name__)


class graph_plotter:
    """
    Handles plotting of graph structures, lattices, and periodic boundaries.
    
    This class provides methods for visualizing:
    - Basic graph structures with nodes and edges
    - Periodic boundary conditions
    - Unit cell boundaries
    - Multi-layer structures with different colors
    """
    
    @staticmethod
    def plot_periodic_edges(ax: constants.matplotlib.axes.Axes, node1, node2, 
                           periodic_offset: Tuple[int, int], 
                           plotted_periodic_edges: Set,
                           lattice_vectors: List[Tuple[float, float]]) -> None:
        """
        Plot edges that cross periodic boundaries with unique colors.
        
        Args:
            ax: Matplotlib axes object
            node1: Source node
            node2: Target node  
            periodic_offset: Lattice offset between nodes
            plotted_periodic_edges: Set to track already plotted edges
            lattice_vectors: Lattice primitive vectors
        """
        if periodic_offset == (0, 0):
            return
            
        # Create unique edge identifier
        key = (node1.lattice_index, node1.sublattice_id, 
               node2.lattice_index, node2.sublattice_id, tuple(periodic_offset))
        inv_key = (node2.lattice_index, node2.sublattice_id,
                   node1.lattice_index, node1.sublattice_id, 
                   tuple(-constants.np.array(periodic_offset)))
        
        if key in plotted_periodic_edges or inv_key in plotted_periodic_edges:
            return
            
        # Generate unique color from edge key
        color = edge_color_hash(key)
        
        # Calculate positions with periodic shifts
        A = constants.np.array([lattice_vectors[0], lattice_vectors[1]]).T
        shift = A @ constants.np.array(periodic_offset)
        new_position = (node2.position[0] + shift[0], node2.position[1] + shift[1])
        new_position_inv = (node1.position[0] - shift[0], node1.position[1] - shift[1])
        
        # Plot both segments of the periodic edge
        ax.plot([node1.position[0], new_position[0]], 
                [node1.position[1], new_position[1]], 
                color=color, linestyle='-', linewidth=constants.LINEWIDTH, alpha=0.7)
        ax.plot([node2.position[0], new_position_inv[0]], 
                [node2.position[1], new_position_inv[1]], 
                color=color, linestyle='-', linewidth=constants.LINEWIDTH, alpha=0.7)
        
        plotted_periodic_edges.add(key)


    @staticmethod
    def plot_unit_cell(ax: constants.matplotlib.axes.Axes, 
                      lattice_vectors: List[Tuple[float, float]]) -> None:
        """
        Draw unit cell boundary as dashed polygon.
        
        Args:
            ax: Matplotlib axes object
            lattice_vectors: List of two lattice primitive vectors
        """
        v1 = constants.np.array(lattice_vectors[0])
        v2 = constants.np.array(lattice_vectors[1])
        
        # Define unit cell corners
        corners = [-(v1 + v2)/2, (v1 - v2)/2, (v1 + v2)/2, (-v1 + v2)/2]
        
        polygon = Polygon(corners, closed=True, fill=False, 
                         edgecolor='black', linestyle='--', 
                         linewidth=constants.LINEWIDTHOFUNITCELLBDRY)
        ax.add_patch(polygon)

    @staticmethod
    def plot_single_layer_graph(ax: constants.matplotlib.axes.Axes, nodes: List, 
                               node_color: str, max_distance: Optional[float] = None) -> None:
        """
        Plot a single graph layer without subgraph differentiation.
        
        Args:
            ax: Matplotlib axes object
            nodes: List of nodes to plot
            node_color: Color for nodes and edges
            max_distance: Optional distance cutoff from origin
        """
        for node in nodes:
            node_distance = calculate_distance(node)
            if max_distance is None or node_distance < max_distance:
                # Plot node
                ax.scatter(node.position[0], node.position[1], 
                          color=node_color, marker="o", s=constants.MARKERSIZE)
                
                # Plot edges
                for neighbor, periodic_offset in node.neighbors:
                    neighbor_distance = calculate_distance(neighbor)
                    if max_distance is None or neighbor_distance < max_distance:
                        line_style = '-' if periodic_offset == (0, 0) else '--'
                        ax.plot([node.position[0], neighbor.position[0]], 
                               [node.position[1], neighbor.position[1]], 
                               line_style, color=node_color, 
                               linewidth=constants.LINEWIDTH)

    @staticmethod
    def plot_multi_layer_graph(ax: constants.matplotlib.axes.Axes, nodes: List, 
                              node_colors: List[str], max_distance: Optional[float] = None,
                              lattice_vectors: Optional[List[Tuple[float, float]]] = None) -> None:
        """
        Plot multi-layer graph with sublattice differentiation.
        
        Args:
            ax: Matplotlib axes object
            nodes: List of all nodes
            node_colors: Colors for different sublattices and interlayer edges
            max_distance: Optional distance cutoff
            lattice_vectors: For plotting periodic edges and unit cell
        """
        plotted_periodic_edges = set()
        
        for node in nodes:
            node_distance = calculate_distance(node)
            if max_distance is None or node_distance < max_distance:
                # Plot node with sublattice-specific color
                ax.scatter(node.position[0], node.position[1], 
                          color=node_colors[node.sublattice_id], 
                          marker="o", s=constants.MARKERSIZE)
                
                # Plot edges
                for neighbor, periodic_offset in node.neighbors:
                    if periodic_offset != (0, 0) and lattice_vectors is not None:
                        # Periodic edge - use special plotting
                        graph_plotter.plot_periodic_edges(
                            ax, node, neighbor, periodic_offset, 
                            plotted_periodic_edges, lattice_vectors)
                    else:
                        # Regular edge - choose color based on layer connection
                        if neighbor.sublattice_id == node.sublattice_id:
                            line_color = node_colors[node.sublattice_id]  # Intralayer
                        else:
                            line_color = node_colors[-1]  # Interlayer
                        
                        ax.plot([node.position[0], neighbor.position[0]], 
                               [node.position[1], neighbor.position[1]], 
                               '-', color=line_color, linewidth=constants.LINEWIDTH)

    @staticmethod
    def plot_graph(ax: constants.matplotlib.axes.Axes, nodes: List, node_colors: List[str],
                  max_distance: Optional[float] = None, 
                  differentiate_subgraphs: bool = False,
                  lattice_vectors: Optional[List[Tuple[float, float]]] = None) -> None:
        """
        Main graph plotting interface.
        
        Args:
            ax: Matplotlib axes object
            nodes: List of nodes to plot
            node_colors: Colors for nodes/edges
            max_distance: Optional distance cutoff
            differentiate_subgraphs: Whether to use different colors per sublattice  
            lattice_vectors: For unit cell and periodic edge plotting
        """
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Plot unit cell if vectors provided
        if lattice_vectors is not None:
            graph_plotter.plot_unit_cell(ax, lattice_vectors)
        
        # Choose plotting mode
        if differentiate_subgraphs:
            graph_plotter.plot_multi_layer_graph(ax, nodes, node_colors, 
                                               max_distance, lattice_vectors)
        else:
            graph_plotter.plot_single_layer_graph(ax, nodes, node_colors[0], max_distance)


class band_plotter:
    """
    Handles 3D band structure plotting and k-space visualization.
    
    This class provides methods for:
    - Computing k-space grids
    - Band structure calculation on grids  
    - 3D surface plotting of energy bands
    """
    
    @staticmethod
    def compute_k_grid(k1_vals: constants.np.ndarray, k2_vals: constants.np.ndarray, 
                      center: Tuple[float, float], 
                      dual_vectors: List[Tuple[float, float]]) -> Tuple[constants.np.ndarray, constants.np.ndarray]:
        """
        Compute Cartesian k-space grid from lattice coordinates.
        
        Args:
            k1_vals: Values along first reciprocal lattice direction
            k2_vals: Values along second reciprocal lattice direction  
            center: Center point in lattice coordinates
            dual_vectors: Reciprocal lattice vectors
            
        Returns:
            Tuple of (kx_grid, ky_grid) in Cartesian coordinates
        """
        k1_grid, k2_grid = constants.np.meshgrid(k1_vals, k2_vals, indexing='ij')
        b1 = constants.np.array(dual_vectors[0])
        b2 = constants.np.array(dual_vectors[1])
        
        kx_grid = (center[0] + k1_grid) * b1[0] + (center[1] + k2_grid) * b2[0]
        ky_grid = (center[0] + k1_grid) * b1[1] + (center[1] + k2_grid) * b2[1]
        
        return kx_grid, ky_grid

    @staticmethod
    def compute_bands_on_grid(k1_vals: constants.np.ndarray, k2_vals: constants.np.ndarray,
                             num_of_points: int, min_bands: int, max_bands: int,
                             center: Tuple[float, float], band_handler,
                             inter_graph_weight: float, intra_graph_weight: float) -> constants.np.ndarray:
        """
        Compute energy bands on a k-space grid.
        
        Args:
            k1_vals: Array of k-values along first direction
            k2_vals: Array of k-values along second direction  
            num_of_points: Number of grid points per axis
            min_bands, max_bands: Band indices to compute
            center: Grid center in lattice coordinates
            band_handler: Object with compute_bands_at_k method
            inter_graph_weight, intra_graph_weight: Coupling strengths
            
        Returns:
            3D array of band energies [i, j, band_index]
        """
        num_of_nodes = len(band_handler.periodic_graph.nodes)
        if max_bands > num_of_nodes:
            logger.warning(f"Requested max_bands {max_bands} exceeds available eigenvalues {num_of_nodes}")
            max_bands = num_of_nodes
            
        num_of_bands = max_bands - min_bands + 1
        bands = constants.np.zeros((num_of_points, num_of_points, num_of_bands))
        
        for i in range(num_of_points):
            for j in range(num_of_points):
                k = (center[0] + k1_vals[i], center[1] + k2_vals[j])
                success = False
                
                # Try with small perturbations if eigenvalue computation fails
                for attempt in range(constants.attempt_num):
                    try:
                        eigvals, _ = band_handler.compute_bands_at_k(
                            Momentum=k, min_bands=min_bands, max_bands=max_bands,
                            inter_graph_weight=inter_graph_weight, 
                            intra_graph_weight=intra_graph_weight)
                        bands[i, j, :] = eigvals
                        success = True
                        break
                    except Exception as e:
                        k = (k[0] + constants.np.random.normal(scale=1e-4), 
                             k[1] + constants.np.random.normal(scale=1e-4))
                
                if not success:
                    logger.warning(f"Failed to compute bands at k={k} after {constants.attempt_num} attempts")
                    bands[i, j, :] = constants.np.nan
        
        return bands

    @staticmethod 
    def plot_band_structure_3d(ax: constants.matplotlib.axes.Axes, band_handler,
                              num_of_points: int, min_bands: int, max_bands: int,
                              inter_graph_weight: float, intra_graph_weight: float,
                              k_max: float = constants.DEFAULT_K_RANGE, k_min: float = -constants.DEFAULT_K_RANGE, 
                              k_flag: bool = False, build_adj_matrix_flag: bool = False) -> None:
        """
        Plot 3D band structure over Brillouin zone.
        
        Args:
            ax: 3D matplotlib axes
            band_handler: Object containing band computation methods
            num_of_points: Grid resolution per axis  
            min_bands, max_bands: Band range to plot
            inter_graph_weight, intra_graph_weight: Coupling parameters
            k_max, k_min: k-space range 
            k_flag: Whether to center at K-point vs Gamma point
            build_adj_matrix_flag: Whether to rebuild adjacency matrix
        """
        if ax.name != '3d':
            raise ValueError("Expecting a 3D Axes object")
        
        # Determine center point
        center = band_handler.periodic_graph.K_point if k_flag else (0, 0)
        
        # Generate k-space grid
        k1_vals = constants.np.linspace(k_min, k_max, num_of_points)
        k2_vals = constants.np.linspace(k_min, k_max, num_of_points)
        kx_grid, ky_grid = band_plotter.compute_k_grid(
            k1_vals, k2_vals, center, band_handler.periodic_graph.dual_vectors)
        
        # Build adjacency matrix if needed
        if (band_handler.matrix_handler.adj_matrix is None or 
            band_handler.matrix_handler.periodic_edges is None or 
            build_adj_matrix_flag):
            logger.info("Rebuilding adjacency matrix for band computation")
            band_handler.matrix_handler.build_adj_matrix(inter_graph_weight, intra_graph_weight)
        
        # Compute bands on grid
        bands = band_plotter.compute_bands_on_grid(
            k1_vals, k2_vals, num_of_points, min_bands, max_bands, center,
            band_handler, inter_graph_weight, intra_graph_weight)
        
        # Plot each band as a surface
        for n in range(bands.shape[2]):
            ax.plot_surface(kx_grid, ky_grid, bands[:, :, n], 
                           cmap='viridis', alpha=0.8)
        
        ax.set_xlabel('$k_x$')
        ax.set_ylabel('$k_y$')
        ax.set_zlabel('Energy $E_n(k)$')


def plot_band_structure(band_handler, ax: constants.matplotlib.axes.Axes, num_of_points: int, min_bands: int, max_bands: int,
                       inter_graph_weight: float, intra_graph_weight: float, k_max: float = constants.DEFAULT_K_RANGE,
                       k_min: float = -constants.DEFAULT_K_RANGE, k_flag: bool = False, build_adj_matrix_flag: bool = False) -> None:
    """
    Plot the band structure over a Brillouin zone grid.
    
    Args:
        band_handler: The band_handler object to plot bands for
        ax (constants.matplotlib.axes): Matplotlib 3D axes to plot on.
        num_of_points (int): Number of points per momentum axis.
        min_bands (int): the index of the lowest band to plot.
        max_bands (int): the index of the highest band to plot.
        inter_graph_weight (float): Inter-subgraph edge weight.
        intra_graph_weight (float): Intra-subgraph edge weight.
        k_max (float): Maximum value in k-space.
        k_min (float): Minimum value in k-space.
        k_flag(bool):  If True, center the plot around the K( = 1/3 (k_1-k_2) or =1/3(v_1+v_2)) point 
                            If False, center the plot around the origin
        build_adj_matrix_flag (bool): If True, recompute the adjacency matrix before plotting.
    
    Raises:
        ValueError if ax is not a 3d Axes object
    """
    band_plotter.plot_band_structure_3d(ax, band_handler, num_of_points, min_bands, max_bands,
                                      inter_graph_weight, intra_graph_weight, k_max, k_min, k_flag, build_adj_matrix_flag)


class dirac_plotter:
    """
    Specialized plotting for Dirac point analysis and linear fitting.
    
    Provides visualization tools for:
    - Linear fits of dispersion relations
    - Energy vs momentum plots around Dirac points
    - Analysis of fitting quality
    """
    
    @staticmethod
    def plot_dispersion_fit(slope1: float, intercept1: float, 
                           slope2: float, intercept2: float,
                           k_array: constants.np.ndarray, e1_array: constants.np.ndarray, e2_array: constants.np.ndarray,
                           momentum_point: Tuple[float, float], direction: constants.np.ndarray) -> None:
        """
        Plot linear fits of both bands around a Dirac point.
        
        Args:
            slope1, intercept1: Linear fit parameters for lower band
            slope2, intercept2: Linear fit parameters for upper band  
            k_array: Momentum offsets from Dirac point
            e1_array, e2_array: Energy values for lower/upper bands
            momentum_point: Dirac point coordinates
            direction: Direction vector for the cut
        """
        # Compute fit predictions
        e1_pred = slope1 * k_array + intercept1
        e2_pred = slope2 * k_array + intercept2
        
        # Create plot
        plt.figure(figsize=(8, 6))
        
        # Plot data points and fits
        plt.plot(k_array, e1_array, 'o-', label='Lower Band Data', color='blue')
        plt.plot(k_array, e1_pred, '-', label='Lower Band Fit', color='lightblue')
        plt.plot(k_array, e2_array, 'o-', label='Upper Band Data', color='red') 
        plt.plot(k_array, e2_pred, '-', label='Upper Band Fit', color='lightcoral')
        
        # Mark Dirac point
        plt.axvline(x=0, color='black', linestyle='--', 
                   linewidth=constants.LINEWIDTH, label='Dirac Point')
        
        # Labels and formatting
        plt.xlabel('k (distance from Dirac point)')
        plt.ylabel('Energy')
        angle_deg = constants.np.atan2(direction[1], direction[0]) * 180 / constants.np.pi
        plt.title(f'Dispersion at Dirac Point {momentum_point}\nDirection: {angle_deg:.1f}°')
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_tbg_structure(ax: Optional[constants.matplotlib.axes.Axes], tbg_system,
                          plot_color_top: str = 'b', plot_color_bottom: str = 'r', 
                          plot_color_interlayer: str = 'g', radius: Optional[float] = None,
                          plot_separate_layers: bool = False,
                          lattice_vectors: Optional[List[Tuple[float, float]]] = None) -> constants.matplotlib.axes.Axes:
        """
        Plot TBG structure with proper layer differentiation.
        
        Args:
            ax: Matplotlib axes (creates new if None)
            tbg_system: TBG object with layer information
            plot_color_top, plot_color_bottom: Colors for each layer
            plot_color_interlayer: Color for interlayer connections
            radius: Distance cutoff for plotting  
            plot_separate_layers: Whether to plot layers separately
            lattice_vectors: For unit cell visualization
            
        Returns:
            The axes object used for plotting
            
        Raises:
            constants.tbg_error: If plotting fails.
        """
        try:
            if radius is not None:
                constants.validate_positive_number(radius, "radius")
            
            if tbg_system is None:
                raise constants.tbg_error("tbg_system cannot be None")
            
            # Check required attributes
            required_attrs = ['top_layer', 'bottom_layer', 'full_graph']
            for attr in required_attrs:
                if not hasattr(tbg_system, attr):
                    raise constants.tbg_error(f"tbg_system missing required attribute: {attr}")
            
            if ax is None:
                fig = Figure(figsize=(6, 4))
                ax = fig.subplots(1, 1)
        
            if plot_separate_layers:
                # Plot layers separately without connections
                graph_plotter.plot_graph(ax, tbg_system.top_layer.graph.nodes, 
                                       [plot_color_top], radius, False)
                graph_plotter.plot_graph(ax, tbg_system.bottom_layer.graph.nodes, 
                                       [plot_color_bottom], radius, False)
            else:
                # Plot combined structure with layer differentiation
                node_colors = [plot_color_top, plot_color_bottom, plot_color_interlayer]
                graph_plotter.plot_graph(ax, tbg_system.full_graph.nodes, node_colors,
                                       radius, True, lattice_vectors)
            return ax
            
        except constants.tbg_error:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in plot_tbg_structure: {str(e)}")
            raise constants.tbg_error(f"Failed to plot TBG structure: {str(e)}")


# Convenience functions for backward compatibility
def plot_graph(graph_obj, ax: constants.matplotlib.axes.Axes, node_colors: List[str],
              max_distance: Optional[float] = None, differentiate_subgraphs: bool = False,
              lattice_vectors: Optional[List[Tuple[float, float]]] = None) -> None:
    """Convenience wrapper for graph plotting."""
    graph_plotter.plot_graph(ax, graph_obj.nodes, node_colors, max_distance, 
                           differentiate_subgraphs, lattice_vectors)

def plot_band_structure(band_handler, ax: constants.matplotlib.axes.Axes, *args, **kwargs) -> None:
    """Convenience wrapper for band structure plotting."""  
    band_plotter.plot_band_structure_3d(ax, band_handler, *args, **kwargs)

def plot_tbg(tbg_system, ax: Optional[constants.matplotlib.axes.Axes] = None, *args, **kwargs) -> None:
    """Convenience wrapper for TBG plotting."""
    dirac_plotter.plot_tbg_structure(ax, tbg_system, *args, **kwargs)