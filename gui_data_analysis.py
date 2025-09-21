"""
GUI Main Window for TBG Simulation Data Analysis.

This module contains the main application window and GUI logic
for the TBG simulation data analysis tool. Handles file management,
tab creation, and overall application structure.

Classes:
    analysis_main_window: Main application window class

Functions:
    load_simulation_data: Convenience function to quickly load simulation data
    quick_summary: Quick summary of simulation data  
    run_analysis_gui: Run the analysis GUI application
    test_data_structure: Test function to verify data structure
"""

# Common imports from data_structures_for_training_data
from data_structures_for_training_data import (
    sys, os, constants, Dict,
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QWidget, QTabWidget, QMessageBox, QListWidget, QListWidgetItem,
    QFileDialog, Qt, QColor, simulation_data_point
)
from simulation_data_loader import simulation_data_analyzer
from analysis_plot_widgets import tab_factory

logger = constants.logging.getLogger(__name__)


class analysis_main_window(QMainWindow):
    """
    Main window for TBG simulation data analysis.
    """
    
    def __init__(self, data_folder: str = None):
        super().__init__()
        self.analyzer = simulation_data_analyzer(data_folder)
        self.init_ui()
        self.load_data()
    
    def init_ui(self) -> None:
        """Initialize the main window UI."""
        central_widget = QWidget()
        
        # Create main layout
        main_layout = QVBoxLayout()
        
        # Add file selection toolbar at the top
        self.create_file_selection_toolbar(main_layout)
        
        # Create content layout for tabs
        self.content_layout = QHBoxLayout()
        
        # Tabs and widgets will be initialized after data loading
        self.tabs = {}  # Store tab widgets for cleanup
        
        main_layout.addLayout(self.content_layout)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # Window properties
        self.setWindowTitle('TBG Simulation Data Analysis')
        self.setGeometry(100, 100, 1400, 800)
        self.statusBar().showMessage('Initializing...')
    
    def create_file_selection_toolbar(self, parent_layout: QVBoxLayout) -> None:
        """Create toolbar for file selection."""
        toolbar_widget = QWidget()
        toolbar_layout = QHBoxLayout()
        
        # Add files button
        add_files_btn = QPushButton("Add CSV Files")
        add_files_btn.clicked.connect(self.add_csv_files)
        toolbar_layout.addWidget(add_files_btn)
        
        # Clear files button
        clear_files_btn = QPushButton("Clear All Files")
        clear_files_btn.clicked.connect(self.clear_all_files)
        toolbar_layout.addWidget(clear_files_btn)
        
        # File list display
        self.file_list_widget = QListWidget()
        self.file_list_widget.setMaximumHeight(60)
        self.file_list_widget.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        toolbar_layout.addWidget(QLabel("Loaded Files:"))
        toolbar_layout.addWidget(self.file_list_widget)
        
        toolbar_layout.addStretch()
        toolbar_widget.setLayout(toolbar_layout)
        parent_layout.addWidget(toolbar_widget)
    
    def add_csv_files(self) -> None:
        """Open file dialog to add CSV files."""
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        dialog.setNameFilter("CSV files (*.csv)")
        dialog.setDirectory(self.analyzer.data_folder)
        
        if dialog.exec() == QFileDialog.DialogCode.Accepted:
            file_paths = dialog.selectedFiles()
            file_names = [os.path.basename(path) for path in file_paths]
            
            try:
                self.analyzer.load_multiple_csv_files(file_names, clear_existing=False)
                self.update_file_list()
                self.refresh_all_plots()
                self.statusBar().showMessage(f'Loaded {len(self.analyzer.loaded_files)} files with {len(self.analyzer.data_points)} data points')
                
            except Exception as e:
                logger.error(f"Error loading files: {e}")
                QMessageBox.critical(self, "Error Loading Files", f"Failed to load CSV files:\n{str(e)}")
    
    def clear_all_files(self) -> None:
        """Clear all loaded files."""
        self.analyzer.data_points = []
        self.analyzer.raw_data = []
        self.analyzer.loaded_files = []
        self.analyzer.file_colors = {}
        self.update_file_list()
        self.refresh_all_plots()
        self.statusBar().showMessage('All files cleared')
    
    def update_file_list(self) -> None:
        """Update the file list widget."""
        self.file_list_widget.clear()
        for i, file_name in enumerate(self.analyzer.loaded_files):
            item = QListWidgetItem(file_name)
            # Set item color to match the file color
            if file_name in self.analyzer.file_colors:
                color_hex = self.analyzer.file_colors[file_name]
                try:
                    # Convert hex color to QColor
                    qcolor = QColor(color_hex)
                    qcolor.setAlpha(100)  # Make it semi-transparent for better readability
                    item.setBackground(qcolor)
                except Exception as e:
                    logger.debug(f"Error setting color for {file_name}: {e}")
            self.file_list_widget.addItem(item)
    
    def refresh_all_plots(self) -> None:
        """Refresh all plots in all tabs."""
        for tab_info in self.tabs.values():
            if hasattr(tab_info['plot'], 'plot_data'):
                tab_info['plot'].plot_data()
            # Update file coloring button visibility
            if hasattr(tab_info['plot'], 'update_file_button_visibility'):
                tab_info['plot'].update_file_button_visibility()
    
    def create_tabs_and_widgets(self) -> None:
        """Create tab widgets and plots."""
        # Clear existing tabs if any
        if hasattr(self, 'tab_widget'):
            self.content_layout.removeWidget(self.tab_widget)
            self.tab_widget.deleteLater()
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Create first tab: Velocity vs Theta using factory
        theta_tab, theta_plot, theta_param = tab_factory.create_velocity_vs_theta_tab(self.analyzer)
        self.tab_widget.addTab(theta_tab, "Velocity vs Theta")
        self.tabs['theta'] = {'tab': theta_tab, 'plot': theta_plot, 'param': theta_param}
        
        # Create second tab: Velocity vs Weight Ratio using factory
        weight_tab, weight_plot, weight_param = tab_factory.create_velocity_vs_weight_ratio_tab(self.analyzer)
        self.tab_widget.addTab(weight_tab, "Velocity vs Weight Ratio")
        self.tabs['weight'] = {'tab': weight_tab, 'plot': weight_plot, 'param': weight_param}
        
        # Create third tab: Velocity vs Interlayer Threshold using factory
        interlayer_tab, interlayer_plot, interlayer_param = tab_factory.create_velocity_vs_interlayer_threshold_tab(self.analyzer)
        self.tab_widget.addTab(interlayer_tab, "Velocity vs Interlayer Threshold")
        self.tabs['interlayer'] = {'tab': interlayer_tab, 'plot': interlayer_plot, 'param': interlayer_param}
        
        # Create fourth tab: Velocity vs Intralayer Threshold using factory
        intralayer_tab, intralayer_plot, intralayer_param = tab_factory.create_velocity_vs_intralayer_threshold_tab(self.analyzer)
        self.tab_widget.addTab(intralayer_tab, "Velocity vs Intralayer Threshold")
        self.tabs['intralayer'] = {'tab': intralayer_tab, 'plot': intralayer_plot, 'param': intralayer_param}
        
        # Create fifth tab: Data Coverage Analysis using factory
        coverage_tab, coverage_plot, coverage_param = tab_factory.create_data_coverage_tab(self.analyzer)
        self.tab_widget.addTab(coverage_tab, "Data Coverage")
        self.tabs['coverage'] = {'tab': coverage_tab, 'plot': coverage_plot, 'param': coverage_param}
        
        self.content_layout.addWidget(self.tab_widget)
    
    def load_data(self) -> None:
        """Load simulation data and initialize widgets using factory pattern."""
        try:
            self.analyzer.load_csv_data()
            self.create_tabs_and_widgets()
            self.update_file_list()
            self.statusBar().showMessage(f'Loaded {len(self.analyzer.data_points)} data points')
            
        except FileNotFoundError:
            self.statusBar().showMessage('Data file not found - check Data_from_run folder')
            logger.error("Data file not found")
            QMessageBox.warning(self, "Data File Not Found", 
                              "Could not find the simulation data file.\n"
                              "Please ensure 'dirac_training_data.csv' exists in the Data_from_run folder.")
        except Exception as e:
            self.statusBar().showMessage(f'Error loading data: {str(e)}')
            logger.error(f"Error loading data: {e}")
            QMessageBox.critical(self, "Error Loading Data", 
                               f"Failed to load simulation data:\n{str(e)}")
    
    def cleanup(self) -> None:
        """Clean up all tabs and widgets properly."""
        for tab_info in self.tabs.values():
            if hasattr(tab_info['param'], 'cleanup'):
                tab_info['param'].cleanup()
            if hasattr(tab_info['plot'], 'cleanup'):  
                tab_info['plot'].cleanup()
        self.tabs.clear()
    
    def closeEvent(self, event) -> None:
        """Handle window close event with proper cleanup."""
        self.cleanup()
        super().closeEvent(event)


# Convenience functions for quick analysis
def load_simulation_data(data_folder: str = None, filename: str = "dirac_training_data.csv") -> simulation_data_analyzer:
    """
    Convenience function to quickly load simulation data.
    
    Args:
        data_folder: Path to data folder
        filename: CSV filename to load
        
    Returns:
        Initialized and loaded simulation_data_analyzer instance
    """
    analyzer = simulation_data_analyzer(data_folder)
    analyzer.load_csv_data(filename)
    return analyzer


def quick_summary(data_folder: str = None, filename: str = "dirac_training_data.csv") -> Dict:
    """
    Quick summary of simulation data.
    
    Args:
        data_folder: Path to data folder  
        filename: CSV filename to analyze
        
    Returns:
        Dictionary with summary statistics
    """
    analyzer = load_simulation_data(data_folder, filename)
    return analyzer.get_data_summary()


def run_analysis_gui(data_folder: str = None) -> None:
    """
    Run the analysis GUI application.
    
    Args:
        data_folder: Path to data folder (optional)
    """
    app = QApplication(sys.argv)
    window = analysis_main_window(data_folder)
    window.show()
    sys.exit(app.exec())


def test_data_structure() -> None:
    """Test function to verify data structure is working correctly."""
    try:
        # Test creating a sample data point
        test_point = simulation_data_point(
            a=5, b=1, theta=30.0, 
            interlayer_dist_threshold=1.0, intralayer_dist_threshold=1.5,
            inter_graph_weight=0.5, intra_graph_weight=1.0, weight_ratio=2.0,
            k_x=0.1, k_y=0.2, k_x_abs=0.15, k_y_abs=0.25, velocity=1000.0
        )
        
        # Test that all expected parameters exist
        expected_params = ['a', 'b', 'theta', 'interlayer_dist_threshold', 'intralayer_dist_threshold',
                          'inter_graph_weight', 'intra_graph_weight', 'weight_ratio', 'k_x', 'k_y', 
                          'k_x_abs', 'k_y_abs', 'velocity', 'source_file', 'file_index']
        
        for param in expected_params:
            if not hasattr(test_point, param):
                logger.error(f"Missing parameter: {param}")
                return False
                
        logger.info("Data structure test passed")
        return True
        
    except Exception as e:
        logger.error(f"Data structure test failed: {e}")
        return False


if __name__ == "__main__":
    # Test data structure first
    if not test_data_structure():
        logger.error("Data structure test failed - check logs")
        sys.exit(1)
    
    # Example usage
    try:
        # Run GUI application
        run_analysis_gui()
        
    except Exception as e:
        logger.error(f"Error running application: {e}")
        sys.exit(1)