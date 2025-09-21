"""
Interactive GUI Application for Twisted Bilayer Graphene (TBG) Simulation and Visualization.

This module provides a comprehensive PyQt6-based graphical user interface for exploring
twisted bilayer graphene systems. It enables real-time parameter adjustment, lattice
structure visualization, and band structure computation with immediate visual feedback.

Key Features:
    - Interactive parameter controls for TBG system configuration
    - Real-time lattice structure visualization with customizable display options
    - 3D band structure plotting with adjustable energy ranges
    - Live parameter validation and error handling
    - Professional matplotlib integration with navigation tools
    - Responsive layout with tabbed interface for different views

Main Components:
    - tbg_main_window: Primary application window with parameter controls
    - Parameter input widgets with validation and real-time updates
    - Matplotlib canvas integration for lattice and band structure plots
    - Status bar for user feedback and operation progress

Usage:
    Run as standalone application:
        python GuiFile.py

    Or integrate into larger application:
        from GuiFile import tbg_main_window
        app = QApplication(sys.argv)
        window = tbg_main_window()
        window.show()

Dependencies:
    - PyQt6: GUI framework
    - matplotlib: Plotting and visualization
    - TBG module: Core physics simulation
    - constants: Configuration and shared parameters

Author: Tal Malinovitch
License: MIT (Academic Use)
"""

import sys
import constants
from typing import Union
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QCheckBox, QLineEdit, QFormLayout, QGroupBox, QWidget,
    QSizePolicy, QSplitter, QTabWidget, QMessageBox
)
from PyQt6.QtGui import QFont, QPixmap
from PyQt6.QtCore import Qt, pyqtSignal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from TBG import tbg
from utils import compute_twist_constants, validate_ab
from graph import graph,periodic_graph

# Default simulation parameters
DEFAULT_PARAMS = constants.dataclasses.asdict(constants.simulation_parameters())

TOOLTIPS = {
        "a": "Twist parameter a (a,b co-prime, b<a, positive integer)",
        "b": "Twist parameter b (a,b co-prime,  non-negative integer, b < a)",
        "unit_cell_radius_factor": "Scaling factor for the plotted unit cell radius (positive intger)",
        "unit_cell_flag": "Toggle to plot only the unit cell (bool)",
        "interlayer_dist_threshold": "Maximum distance for interlayer edges (float)",
        "intralayer_dist_threshold": "Maximum distance for interlayer edges (float)",
        "min_band": "The index of the lowest eigenvalue (band) to plot (positive intger)",
        "max_band":  "The index of the highest eigenvalue (band) to plot(positive intger)",
        "num_of_points": "Number of k-points in each direction (positive integer)",
        "inter_graph_weight": "Coupling weight between layers (float)",
        "intra_graph_weight": "Coupling weight within a layer (float)",
        "k_min": "Minimum k value for band structure (float)",
        "k_max": "Maximum k value for band structure (float)",
        "k_flag": "Toggle to plot the band structure around the K point (bool)"
}
GRAPH_KEYS={"a", "b","unit_cell_radius_factor","unit_cell_flag","interlayer_dist_threshold","intralayer_dist_threshold"} #the paramters, that if changed,requirte reconstruction of the graph
BAND_KEYS={"a", "b","interlayer_dist_threshold","intralayer_dist_threshold","min_band","max_band","num_of_points",
           "inter_graph_weight","intra_graph_weight","k_min","k_max", "k_flag"} #the paramters, that if changed,requirte reconstruction of the band
REBUILD_KEYS={"a","b","interlayer_dist_threshold","intralayer_dist_threshold","inter_graph_weight","intra_graph_weight"} #the paramters, that if changed,requirte reconstruction of the adj matrix
constants.logging.basicConfig(
    level=constants.logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        constants.logging.FileHandler("tbg_debug.log", mode='w'),  # overwrite on each run
    ]
)
logger = constants.logging.getLogger(__name__)
""" 
This is the GUI for the TBG simulation. 
It runs a main window with the parameters and the plot
"""
def calc_n(factor:float,N_num:float,Conversionfactor:float)-> int:
    """
    Helper function to compute the maximum number of unit vector in each direction. Note that it calculate the same number in both directions.

    Args:
        factor (float): scaling factor determined by the user
        N_num (float): the scaling factor of the TBG
        Conversionfactor (float):the scaling factor of the lattice ( whether it is dual of not)
    returns:
        the integer closest to product of all three
    """
    
    return int(constants.np.round(factor*N_num*Conversionfactor))

class main_window(QMainWindow):
    """
    Main GUI window class for simulating and visualizing
    twisted bilayer graphene (TBG) systems and their band structures.
    """
    def __init__(self)->None:
        super().__init__()
        N_from_computation,_,factor,k_point=compute_twist_constants(DEFAULT_PARAMS["a"],DEFAULT_PARAMS["b"]) 
        self.current_params = DEFAULT_PARAMS #set up the paramters
        self.current_params["N"]=N_from_computation
        self.current_params["factor"]=factor
        self.current_params["k_point"]=k_point
        self._last_graph_params = {k: self.current_params[k] for k in GRAPH_KEYS}
        self._last_band_params  = {k: self.current_params[k] for k in BAND_KEYS}
        self._last_rebuild_params  ={k: self.current_params[k] for k in REBUILD_KEYS}
        self.current_params["build_laplacian_flag"]=True
        n=calc_n(self.current_params["unit_cell_radius_factor"],N_from_computation,factor) # computes an estimate of how many vecotrs of the original lattice are needed
        try:
            self.graph = tbg(n,n, self.current_params["a"], self.current_params["b"],
                                     self.current_params["interlayer_dist_threshold"],
                                     self.current_params["intralayer_dist_threshold"],
                                     self.current_params["unit_cell_radius_factor"]) # computes the TBG graph with the parameters
        except ValueError as e:
            QMessageBox.critical(self, "Invalid Input", f"Failed to generate TBG:\n{str(e)}")
            return  # Exit early on failure
        self.periodic_graph = self.graph.full_graph.create_periodic_copy(self.graph.lattice_vectors,k_point)  # Create a periodic copy of the full graph
        self.init_ui()

    def init_ui(self)->None:
        """Initializes the GUI layout and widgets."""
        
        
        central_widget = QWidget()
        hbox = QHBoxLayout()
        # We have 2 widgets now: the parameters and the plot graph
        self.param_widget = graph_parameter(self)
        self.param_widget.parameter_changed.connect(self.handle_param_update)
        self.param_widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.param_widget.setMaximumWidth(constants.DEFAULT_WINDOW_SIZE)

        #put the plots in a seperate tabs
        self.plot_graph_widget = plot_graph(self.graph,self.periodic_graph,self.current_params) 
        self.plot_band_widget = plot_bands(self.periodic_graph,self.current_params) 

        tab_widget=QTabWidget()
        tab_widget.addTab(self.plot_graph_widget, "The graph")
        tab_widget.addTab(self.plot_band_widget, "The bands")

        # we split the scrren for them
        splitter = QSplitter(Qt.Orientation.Horizontal)

        splitter.addWidget(self.param_widget)
        splitter.addWidget(tab_widget)
        splitter.setStretchFactor(0, 1)  # param_widget
        splitter.setStretchFactor(1, 3)  # tab_widget (graphs)
        hbox.addWidget(splitter)
        central_widget.setLayout(hbox)
        self.setCentralWidget(central_widget)

        #some styling 
        self.setWindowTitle('TBG in Rational angles')
        self.statusBar().showMessage('Ready')
        self.show()

    def handle_param_update(self, param_dict: dict)->None:
        """
        Update internal parameter state from GUI input and refresh plots.

        Args:
            param_dict (dict): Updated parameters from graph_parameter widget.
        """
        
        self.current_params.update(param_dict)
        # extract slices
        graph_slice = {k: param_dict[k] for k in GRAPH_KEYS}
        band_slice  = {k: param_dict[k] for k in BAND_KEYS}
        rebuild_slice  = {k: param_dict[k] for k in REBUILD_KEYS}
        # only rebuild real‐space plot if graph params changed
        if graph_slice != self._last_graph_params:
            self._last_graph_params = graph_slice
            self.update_plot()

        # only rebuild bands if band params changed
        if band_slice != self._last_band_params :
            self._last_band_params = band_slice
            self.update_Bands()

        # If one of the rebuild params changed, we need to rebuild the laplacian
        if rebuild_slice != self._last_rebuild_params :
            self._last_rebuild_params = rebuild_slice
            self.current_params["build_laplacian_flag"]=True

    def update_plot(self)->None:
        """
        Recompute TBG structure and update the lattice plot.
        Handles validation errors and GUI messaging.
        """
        n=calc_n(self.current_params["unit_cell_radius_factor"],self.current_params["N"],self.current_params["factor"])
        try:
            self.graph = tbg(n,n, self.current_params["a"], self.current_params["b"],
                                     self.current_params["interlayer_dist_threshold"],
                                     self.current_params["intralayer_dist_threshold"],
                                     unit_cell_radius_factor=self.current_params["unit_cell_radius_factor"]) # recomputes the TBG graph with the new parameters
        except ValueError as e:
            QMessageBox.critical(self, "Invalid Input", f"Failed to generate TBG:\n{str(e)}")
            return  # Exit early on failure
        
        self.periodic_graph = self.graph.full_graph.create_periodic_copy(self.graph.lattice_vectors,self.current_params["k_point"]) #update the periodic Graph
        self.plot_graph_widget.update_plot(self.graph,self.periodic_graph) # and now plots it

        self.statusBar().showMessage("Graph updated", 3000)
        
    def update_Bands(self)->None:
        """Refreshes the band structure view."""
        self.plot_band_widget.update_plot(self.periodic_graph)# and now plots it
        self.statusBar().showMessage("Bands plot updated", 3000)

class plot_bands(QWidget):
    """
    Manages the visualization of the electronic band structure of the
    Twisted Bilayer Graphene (TBG) system.

    This class computes and plots the energy eigenvalues (bands) against
    momentum (k-points) along a high-symmetry path in the Brillouin zone.
    It uses the Laplacian matrix computed from the graph to find the bands.

    Args:
        parent_widget (QWidget): The parent Qt widget to which this plot
                                 widget will be added.
        periodic_graph_data (periodic_graph): An instance of the
                                                      periodic_graph object,
                                                      used to build the
                                                      Laplacian for band
                                                      structure calculation.
    """
    def __init__(self,Periodic_graph_obj: periodic_graph,params: dict)->None:
        super().__init__()
        self.periodic_graph=Periodic_graph_obj
        self.params = params
        self.laplacian_cache = {}  # Cache for different parameter combinations
        self.periodic_edges_cache = {}
        self.init_ui()

    def init_ui(self)->None:
        """Initializes the layout and draws the initial band plot."""
        
        self.layout = QVBoxLayout()
        self.layout.setSpacing(constants.DEFAULT_GUI_SIZE)
        self.layout.setContentsMargins(constants.DEFAULT_GUI_SIZE, constants.DEFAULT_GUI_SIZE, constants.DEFAULT_GUI_SIZE, constants.DEFAULT_GUI_SIZE)
        self.fig = Figure(figsize=(6, 4))
        self.axs = self.fig.add_subplot(111, projection='3d')
        
        self.canvas= FigureCanvas(self.fig)
        # Add the navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.update_plot( self.periodic_graph)
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)
        self.setLayout(self.layout)

    def update_plot(self, Periodic_graph_obj: periodic_graph)->None:
        """
        Refresh the band structure plot using an updated periodic graph.

        Args:
            Periodic_graph_obj (PeriodicGraph): The new periodic graph to compute bands from.
        """
        self.periodic_graph=Periodic_graph_obj
        if hasattr(self, "axs"):
            self.axs.clear()
        try:
            self.periodic_graph.plot_band_structure(self.axs,self.params["num_of_points"],self.params["min_band"],self.params["max_band"],
                                                inter_graph_weight=self.params["inter_graph_weight"],
                                                intra_graph_weight=self.params["intra_graph_weight"],
                                                k_max=self.params["k_max"],k_min=self.params["k_min"],k_flag=self.params["k_flag"],
                                                build_adj_matrix_flag=self.params["build_laplacian_flag"])
            self.params["build_laplacian_flag"]=False
        except (constants.ArpackNoConvergence, constants.np.linalg.LinAlgError) as e:
            QMessageBox.warning(self, "Convergence Error", f"Numerical error: {str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Band Plot Error", f"Error plotting bands: {str(e)}\nType: {type(e).__name__}")
        self.fig.tight_layout()
        self.canvas.draw()


class plot_graph(QWidget):
    """
    Manages the visualization of the Twisted Bilayer Graphene (TBG) graph
    within the GUI, including nodes, edges, and periodic connections.

    This class handles the creation and updating of the graph plot canvas
    using Matplotlib, displaying the lattice structure based on provided
    graph data.

    Args:
        parent_widget (QWidget): The parent Qt widget to which this plot
                                 widget will be added.
        graph_data (graph): An instance of the main Graph object
                                   containing nodes, edges, and structural info.
        periodic_graph_data (periodic_graph): An instance of the
                                                      periodic_graph object
                                                      for plotting periodic
                                                      connections and unit cells.
    """
    def __init__(self,full_graph_obj:graph,periodic_graph_obj:periodic_graph,params:dict)->None:
        super().__init__()
        self.graph = full_graph_obj
        self.periodic_graph=periodic_graph_obj
        self.params = params
        self.init_ui()
    
    def init_ui(self)->None:
        """Initialize the layout and draw the initial graph plot."""
        self.layout = QVBoxLayout()
        self.layout.setSpacing(constants.DEFAULT_GUI_SIZE)
        self.layout.setContentsMargins(constants.DEFAULT_GUI_SIZE, constants.DEFAULT_GUI_SIZE, constants.DEFAULT_GUI_SIZE, constants.DEFAULT_GUI_SIZE)
        self.fig = Figure(figsize=(6, 4))
        self.axs = self.fig.add_subplot()
        self.canvas= FigureCanvas(self.fig)
        # Add the navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.update_plot(self.graph,self.periodic_graph)
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)
        self.setLayout(self.layout)

    def update_plot(self, Full_graph_obj: graph,Periodic_graph_obj:periodic_graph)->None:
        """
        Update the displayed graph with a new one.

        Args:
            Full_graph_obj (Graph): Updated full graph.
            Periodic_graph_obj (PeriodicGraph): Updated periodic graph.
        """
        self.graph=Full_graph_obj
        self.periodic_graph=Periodic_graph_obj
        if hasattr(self, "axs"):
            self.axs.clear()
        if self.params["unit_cell_flag"]== True:
            self.periodic_graph.plot(self.axs,node_colors=constants.DEFAULT_COLORS, max_distance=None, 
                                      differentiate_subgraphs=True,lattice_vectors=self.periodic_graph.lattice_vectors)
            self.fig.tight_layout()
        else:
            self.graph.plot(self.axs,plot_color_top=constants.DEFAULT_COLORS[0], plot_color_bottom=constants.DEFAULT_COLORS[1], plot_color_full=constants.DEFAULT_COLORS[2])
            self.fig.tight_layout()
        self.canvas.draw()

class graph_parameter(QWidget):
    """
    GUI widget to input and modify graph-related parameters.

    Emits updated values to the main window.

    """
    parameter_changed = pyqtSignal(dict)
    def __init__(self,main_window:QMainWindow)->None:
        super().__init__()
        self.main_window=main_window
        self.init_ui()
    def create_input(self, key:str)->Union[QCheckBox, QLineEdit]:
        """
        Creates the inputs lines based on the type, with the tooltips if provided. 
        Most of the lines are QlineEdit, except form the flags
        Args:
            key- the nanme to create, 
        """
        value = DEFAULT_PARAMS[key]
        if key == "unit_cell_flag" or key == 'k_flag':
            widget = QCheckBox()
            widget.setChecked(value)
        else:
            widget = QLineEdit(str(value))
        tooltip = TOOLTIPS.get(key)
        if tooltip:
            widget.setToolTip(tooltip)
        self.inputs[key] = widget
        return widget
    def init_ui(self)->None:
        """Initialize the input fields and layout for parameter editing."""

        self.inputs = {}
        # --- Geometry Group ---
        geometry_group = QGroupBox("Lattice Geometry")
        geometry_layout = QFormLayout()
        for key in ["a", "b", "unit_cell_radius_factor", "unit_cell_flag"]:
            widget = self.create_input(key)
            geometry_layout.addRow(QLabel(key), widget)
        geometry_group.setLayout(geometry_layout)

        # --- Graph Group ---
        graph_group = QGroupBox("Graph Connectivity")
        graph_layout = QFormLayout()
        for key in ["interlayer_dist_threshold","intralayer_dist_threshold", "inter_graph_weight", "intra_graph_weight"]:
            widget = self.create_input(key)
            graph_layout.addRow(QLabel(key), widget)
        graph_group.setLayout(graph_layout)

        # --- Band Structure Group ---
        band_group = QGroupBox("Band Structure")
        band_layout = QFormLayout()
        for key in ["min_band","max_band", "k_min", "k_max", "num_of_points", "k_flag"]:
            widget = self.create_input(key)
            band_layout.addRow(QLabel(key), widget)
        band_group.setLayout(band_layout)
        # --- Assemble all groups ---
        update_btn = QPushButton("Update Parameters")
        update_btn.clicked.connect(self.emit_params)

        main_layout = QVBoxLayout()
        main_layout.addWidget(geometry_group)
        main_layout.addWidget(graph_group)
        main_layout.addWidget(band_group)
        main_layout.addWidget(update_btn)

        self.setLayout(main_layout)        
        self.setWindowTitle('Tooltips')

    def emit_params(self)->None:
        """
        Validate and emit updated parameter values to the main window.
        
        Raises:
            ValueError- if the value is not postive for positive keys (a,b,min/max_band,num_of_points,interlayer_dist_threshold,intralayer_dist_threshold,unit_cell_radius_factor)
            or if the min_band>=max_band
        """
        param_dict = {}
        try:
            for key, widget in self.inputs.items():
                if key == "unit_cell_flag" or key == 'k_flag':
                    value = widget.isChecked()
                else:
                    raw_value = float(widget.text())
                    if key in ["a", "b", "min_band", "max_band", "num_of_points"]:
                        value = int(raw_value)
                        if value<0:
                            raise ValueError(f"{key} needs to be positive")
                    else:
                        value=raw_value
                        if key in ['interlayer_dist_threshold','intralayer_dist_threshold','unit_cell_radius_factor'] and value<0:
                            raise ValueError(f"{key} needs to be positive")

                param_dict[key] = value
            validate_ab(param_dict["a"],param_dict["b"])
            if param_dict["min_band"]>=param_dict["max_band"]:
                raise ValueError(f"the minimal band index should be smaller then the maximal band")
        except ValueError as e:
            QMessageBox.warning(self, "Invalid input", f"Invalid value for {key}. error: {e}.")
            return
        param_dict["N"],_,param_dict["factor"],param_dict["k_point"]=compute_twist_constants(param_dict["a"],param_dict["b"]) 
        self.parameter_changed.emit(param_dict)

def main() -> None:
    """Main application entry point."""
    app = QApplication(sys.argv)
    ex = main_window()
    sys.exit(app.exec())
main()
# main_Dirac()
# main_diagnostic()