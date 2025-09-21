"""
Plotting widget factory for TBG simulation training data analysis GUI.

This module contains the factory class for creating analysis tabs with different configurations.
Uses the concrete implementations from sim_data_analysis.py to avoid circular imports.

Classes:
    tab_factory: Factory class for creating analysis tabs
    analysis_plot_widget: Legacy wrapper for velocity vs theta plot widget
    velocity_vs_weight_plot_widget: Legacy wrapper for velocity vs weight ratio plot widget
"""

# Common imports from data_structures_for_training_data
from data_structures_for_training_data import (
    constants, sys, ABC, abstractmethod,
    List, Dict, Optional, Union, Tuple,
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QWidget, QGroupBox,
    QFormLayout, QLineEdit, QCheckBox, QSizePolicy, QComboBox, QMessageBox,
    QFileDialog, QSplitter, Qt, QTimer,
    FigureCanvas, NavigationToolbar, Figure,
    PlotMode, plot_config, filter_config, simulation_data_point
)
from simulation_data_loader import simulation_data_analyzer, get_k_space_unit_cell_limits
# Import concrete base classes from sim_data_analysis  
from sim_data_analysis import base_plot_widget, base_parameter_control_widget

# Configure global logging
logger = constants.logging.getLogger(__name__)


class tab_factory:
    """Factory class for creating analysis tabs with different configurations."""
    
    @staticmethod
    def create_velocity_vs_theta_tab(analyzer: simulation_data_analyzer) -> Tuple[QWidget, base_plot_widget, base_parameter_control_widget]:
        """Create velocity vs theta analysis tab."""
        # Configure plot
        plot_cfg = plot_config(
            x_param="theta",
            y_param="velocity", 
            x_label="Theta (degrees)",
            y_label="Dirac Velocity",
            title="Dirac Velocity vs Twist Angle (Theta)"
        )
        
        # Configure filters
        filter_configs = [
            filter_config("interlayer_dist_threshold", "Interlayer Dist Threshold", precision=3),
            filter_config("intralayer_dist_threshold", "Intralayer Dist Threshold", precision=3),
            filter_config("weight_ratio", "Weight Ratio (Intra/Inter)", precision=3)
        ]
        
        # Create widgets
        plot_widget = base_plot_widget(analyzer, plot_cfg)
        param_widget = base_parameter_control_widget(analyzer, plot_widget, filter_configs)
        
        # Create tab layout
        tab = QWidget()
        layout = QHBoxLayout()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(param_widget)
        splitter.addWidget(plot_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter)
        tab.setLayout(layout)
        
        return tab, plot_widget, param_widget
    
    @staticmethod
    def create_velocity_vs_weight_ratio_tab(analyzer: simulation_data_analyzer) -> Tuple[QWidget, base_plot_widget, base_parameter_control_widget]:
        """Create velocity vs weight ratio analysis tab."""
        # Configure plot
        plot_cfg = plot_config(
            x_param="weight_ratio",
            y_param="velocity",
            x_label="Weight Ratio (Intra/Inter)",
            y_label="Dirac Velocity", 
            title="Dirac Velocity vs Weight Ratio"
        )
        
        # Configure filters with theta formatter
        theta_formatter = lambda x: f"{x:.2f}°"
        filter_configs = [
            filter_config("interlayer_dist_threshold", "Interlayer Dist Threshold", precision=3),
            filter_config("intralayer_dist_threshold", "Intralayer Dist Threshold", precision=3),
            filter_config("theta", "Theta (Twist Angle)", theta_formatter, precision=2)
        ]
        
        # Create widgets
        plot_widget = base_plot_widget(analyzer, plot_cfg)
        param_widget = base_parameter_control_widget(analyzer, plot_widget, filter_configs)
        
        # Create tab layout
        tab = QWidget()
        layout = QHBoxLayout()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(param_widget)
        splitter.addWidget(plot_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter)
        tab.setLayout(layout)
        
        return tab, plot_widget, param_widget
    
    @staticmethod
    def create_velocity_vs_interlayer_threshold_tab(analyzer: simulation_data_analyzer) -> Tuple[QWidget, base_plot_widget, base_parameter_control_widget]:
        """Create velocity vs interlayer threshold analysis tab."""
        # Configure plot
        plot_cfg = plot_config(
            x_param="interlayer_dist_threshold",
            y_param="velocity",
            x_label="Interlayer Distance Threshold", 
            y_label="Dirac Velocity",
            title="Dirac Velocity vs Interlayer Distance Threshold"
        )
        
        # Configure filters
        filter_configs = [
            filter_config("intralayer_dist_threshold", "Intralayer Dist Threshold", precision=3),
            filter_config("weight_ratio", "Weight Ratio (Intra/Inter)", precision=3),
            filter_config("theta", "Theta (Twist Angle)", lambda x: f"{x:.2f}°", precision=2)
        ]
        
        # Create widgets
        plot_widget = base_plot_widget(analyzer, plot_cfg)
        param_widget = base_parameter_control_widget(analyzer, plot_widget, filter_configs)
        
        # Create tab layout
        tab = QWidget()
        layout = QHBoxLayout()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(param_widget)
        splitter.addWidget(plot_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter)
        tab.setLayout(layout)
        
        return tab, plot_widget, param_widget
    
    @staticmethod
    def create_velocity_vs_intralayer_threshold_tab(analyzer: simulation_data_analyzer) -> Tuple[QWidget, base_plot_widget, base_parameter_control_widget]:
        """Create velocity vs intralayer threshold analysis tab."""
        # Configure plot
        plot_cfg = plot_config(
            x_param="intralayer_dist_threshold", 
            y_param="velocity",
            x_label="Intralayer Distance Threshold",
            y_label="Dirac Velocity",
            title="Dirac Velocity vs Intralayer Distance Threshold"
        )
        
        # Configure filters
        filter_configs = [
            filter_config("interlayer_dist_threshold", "Interlayer Dist Threshold", precision=3),
            filter_config("weight_ratio", "Weight Ratio (Intra/Inter)", precision=3), 
            filter_config("theta", "Theta (Twist Angle)", lambda x: f"{x:.2f}°", precision=2)
        ]
        
        # Create widgets
        plot_widget = base_plot_widget(analyzer, plot_cfg)
        param_widget = base_parameter_control_widget(analyzer, plot_widget, filter_configs)
        
        # Create tab layout
        tab = QWidget()
        layout = QHBoxLayout()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(param_widget)
        splitter.addWidget(plot_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter)
        tab.setLayout(layout)
        
        return tab, plot_widget, param_widget
    
    @staticmethod
    def create_data_coverage_tab(analyzer: simulation_data_analyzer) -> Tuple[QWidget, QWidget, QWidget]:
        """Create data coverage analysis tab."""
        # Import here to avoid circular imports
        from widgets_data_analysis import data_coverage_widget, coverage_parameter_widget
        
        # Create the coverage analysis widget
        coverage_widget = data_coverage_widget(analyzer)
        
        # Create a placeholder for parameter controls (simpler for coverage analysis)
        param_widget = coverage_parameter_widget(analyzer, coverage_widget)
        
        # Create tab layout
        tab = QWidget()
        layout = QHBoxLayout()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(param_widget)
        splitter.addWidget(coverage_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter)
        tab.setLayout(layout)
        
        return tab, coverage_widget, param_widget


# Legacy wrapper classes for backward compatibility
class analysis_plot_widget(base_plot_widget):
    """Legacy wrapper for velocity vs theta plot widget."""
    def __init__(self, analyzer: simulation_data_analyzer):
        config = plot_config("theta", "velocity", "Theta (degrees)", "Dirac Velocity", "Dirac Velocity vs Twist Angle (Theta)")
        super().__init__(analyzer, config)
    
    def plot_velocity_vs_theta(self, filtered_data: List[simulation_data_point] = None) -> None:
        """Legacy method name for compatibility."""
        self.plot_data(filtered_data)


class velocity_vs_weight_plot_widget(base_plot_widget):
    """Legacy wrapper for velocity vs weight ratio plot widget.""" 
    def __init__(self, analyzer: simulation_data_analyzer):
        config = plot_config("weight_ratio", "velocity", "Weight Ratio (Intra/Inter)", "Dirac Velocity", "Dirac Velocity vs Weight Ratio")
        super().__init__(analyzer, config)
    
    def plot_velocity_vs_weight_ratio(self, filtered_data: List[simulation_data_point] = None) -> None:
        """Legacy method name for compatibility."""
        self.plot_data(filtered_data)