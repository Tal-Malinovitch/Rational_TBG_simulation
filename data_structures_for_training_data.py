"""
Data structures and configuration classes for TBG simulation training data analysis.

This module contains the core data classes and configuration structures used 
throughout the training data analysis system. Also serves as the common import 
hub for shared dependencies used across the analysis system.

Classes:
    simulation_data_point: Data structure for individual simulation results
    PlotMode: Enumeration of available plotting modes  
    plot_config: Configuration for plot appearance and behavior
    filter_config: Configuration for data filtering precision
"""

# Common imports - imported here so other modules can access them via this module
import sys
import os  
import csv
import numpy as np
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional, Union, Tuple, Callable
from collections import defaultdict
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QWidget, QGroupBox, QFormLayout, QLineEdit, QCheckBox,
    QSizePolicy, QSplitter, QTabWidget, QComboBox, QMessageBox,
    QListWidget, QListWidgetItem, QFileDialog
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import constants

# Configure global logging
logger = constants.logging.getLogger(__name__)

@dataclass
class simulation_data_point:
    """
    Data structure for a single simulation result point.
    
    Attributes:
        a (int): Twist angle parameter a
        b (int): Twist angle parameter b  
        theta (float): Twist angle in degrees, calculated as atan(sqrt(3)*b/a) * 180/pi
        interlayer_dist_threshold (float): Interlayer coupling distance threshold
        intralayer_dist_threshold (float): Intralayer coupling distance threshold
        inter_graph_weight (float): Inter-sublattice coupling weight
        intra_graph_weight (float): Intra-sublattice coupling weight
        weight_ratio (float): Ratio intra_graph_weight/inter_graph_weight
        k_x (float): Dirac point x-coordinate in relative reciprocal space coordinates
        k_y (float): Dirac point y-coordinate in relative reciprocal space coordinates
        k_x_abs (float): Dirac point x-coordinate in absolute reciprocal space coordinates
        k_y_abs (float): Dirac point y-coordinate in absolute reciprocal space coordinates
        velocity (float): Dirac velocity at the point
        n_scale (Optional[float]): System scaling factor (optional)
        num_nodes (Optional[int]): Number of nodes in the system (optional)
        source_file (Optional[str]): Source CSV file name (optional)
        file_index (Optional[int]): Index of source file for coloring (optional)
    """
    a: int
    b: int
    theta: float
    interlayer_dist_threshold: float
    intralayer_dist_threshold: float
    inter_graph_weight: float
    intra_graph_weight: float
    weight_ratio: float
    k_x: float
    k_y: float
    k_x_abs: float
    k_y_abs: float
    velocity: float
    n_scale: Optional[float] = None
    num_nodes: Optional[int] = None
    source_file: Optional[str] = None
    file_index: Optional[int] = None

class PlotMode(Enum):
    """Enumeration for different plot modes."""
    VELOCITY = "velocity"
    KPOINT = "kpoint"

@dataclass
class plot_config:
    """Configuration for a plot widget."""
    x_param: str  # Parameter name for x-axis
    y_param: str  # Parameter name for y-axis
    x_label: str  # Display label for x-axis
    y_label: str  # Display label for y-axis
    title: str    # Plot title
    x_formatter: Optional[callable] = None  # Optional formatter for x values

@dataclass
class filter_config:
    """Configuration for filter parameters."""
    param_name: str  # Parameter name
    display_name: str  # Display name in UI
    formatter: Optional[callable] = None  # Optional formatter for display values
    precision: int = 3  # Decimal precision for rounding