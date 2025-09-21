"""
Coverage Analysis Widgets for TBG Simulation Data Analysis.

This module contains specialized widgets for analyzing parameter space 
coverage and identifying gaps in the simulation data. Provides visualization
tools for understanding data distribution and coverage completeness.

Classes:
    data_coverage_widget: Main coverage analysis and visualization widget
    coverage_parameter_widget: Parameter control widget for coverage analysis
"""

# Common imports from data_structures_for_training_data
from data_structures_for_training_data import (
    constants, List,
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QWidget, QGroupBox,
    QFormLayout, QComboBox, QMessageBox, QFileDialog, Qt,
    FigureCanvas, Figure, simulation_data_point
)
from simulation_data_loader import simulation_data_analyzer

logger = constants.logging.getLogger(__name__)


class data_coverage_widget(QWidget):
    """Widget for visualizing parameter space coverage and identifying gaps."""
    
    def __init__(self, analyzer: simulation_data_analyzer):
        super().__init__()
        self.analyzer = analyzer
        self.current_plot_type = "2d_heatmap"  # Default plot type
        self.figure = None
        self.canvas = None
        self.init_ui()
        self.update_coverage_plot()
    
    def init_ui(self) -> None:
        """Initialize the coverage widget UI."""
        layout = QVBoxLayout()
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems([
            "Theta vs Weight Ratio Coverage",
            "Threshold Coverage Map", 
            "Parameter Distribution",
            "Coverage Statistics"
        ])
        self.plot_type_combo.currentTextChanged.connect(self.on_plot_type_changed)
        button_layout.addWidget(QLabel("Plot Type:"))
        button_layout.addWidget(self.plot_type_combo)
        
        # Export button
        export_btn = QPushButton("Export Coverage")
        export_btn.clicked.connect(self.export_plot)
        button_layout.addWidget(export_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(12, 8), facecolor='white')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(800, 600)
        
        layout.addWidget(self.canvas)
        self.setLayout(layout)
    
    def on_plot_type_changed(self, plot_type: str) -> None:
        """Handle plot type change."""
        type_mapping = {
            "Theta vs Weight Ratio Coverage": "theta_weight_heatmap",
            "Threshold Coverage Map": "threshold_heatmap",
            "Parameter Distribution": "distributions",
            "Coverage Statistics": "statistics"
        }
        self.current_plot_type = type_mapping.get(plot_type, "theta_weight_heatmap")
        self.update_coverage_plot()
    
    def update_coverage_plot(self) -> None:
        """Update the coverage plot based on current type."""
        if not self.analyzer.data_points:
            return
            
        self.figure.clear()
        
        if self.current_plot_type == "theta_weight_heatmap":
            self.plot_theta_weight_coverage()
        elif self.current_plot_type == "threshold_heatmap":
            self.plot_threshold_coverage()
        elif self.current_plot_type == "distributions":
            self.plot_parameter_distributions()
        elif self.current_plot_type == "statistics":
            self.plot_coverage_statistics()
        
        self.canvas.draw()
    
    def plot_theta_weight_coverage(self) -> None:
        """Plot theta vs weight ratio coverage heatmap."""
        
        # Extract data
        theta_values = [dp.theta for dp in self.analyzer.data_points]
        weight_values = [dp.weight_ratio for dp in self.analyzer.data_points]
        
        ax = self.figure.add_subplot(111)
        
        # Create 2D histogram
        counts, theta_edges, weight_edges = constants.np.histogram2d(
            theta_values, weight_values, bins=[20, 20], density=False
        )
        
        # Plot heatmap
        im = ax.imshow(counts.T, origin='lower', aspect='auto', cmap='YlOrRd',
                      extent=[theta_edges[0], theta_edges[-1], weight_edges[0], weight_edges[-1]])
        
        # Add colorbar
        cbar = self.figure.colorbar(im, ax=ax)
        cbar.set_label('Number of Data Points')
        
        ax.set_xlabel('Theta (degrees)')
        ax.set_ylabel('Weight Ratio (Intra/Inter)')
        ax.set_title('Parameter Space Coverage: Theta vs Weight Ratio')
        ax.grid(True, alpha=0.3)
        
        # Mark empty regions
        zero_mask = counts.T == 0
        if constants.np.any(zero_mask):
            theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2
            weight_centers = (weight_edges[:-1] + weight_edges[1:]) / 2
            theta_mesh, weight_mesh = constants.np.meshgrid(theta_centers, weight_centers)
            ax.scatter(theta_mesh[zero_mask], weight_mesh[zero_mask], 
                      c='blue', marker='x', s=30, alpha=0.7, label='No data')
            ax.legend()
    
    def plot_threshold_coverage(self) -> None:
        """Plot threshold parameter coverage."""
        
        # Extract threshold data
        interlayer_values = [dp.interlayer_dist_threshold for dp in self.analyzer.data_points]
        intralayer_values = [dp.intralayer_dist_threshold for dp in self.analyzer.data_points]
        
        ax = self.figure.add_subplot(111)
        
        # Create 2D histogram
        counts, inter_edges, intra_edges = constants.np.histogram2d(
            interlayer_values, intralayer_values, bins=[15, 15], density=False
        )
        
        # Plot heatmap
        im = ax.imshow(counts.T, origin='lower', aspect='auto', cmap='viridis',
                      extent=[inter_edges[0], inter_edges[-1], intra_edges[0], intra_edges[-1]])
        
        # Add colorbar
        cbar = self.figure.colorbar(im, ax=ax)
        cbar.set_label('Number of Data Points')
        
        ax.set_xlabel('Interlayer Distance Threshold')
        ax.set_ylabel('Intralayer Distance Threshold')
        ax.set_title('Parameter Space Coverage: Distance Thresholds')
        ax.grid(True, alpha=0.3)
        
        # Add constraint line (interlayer should be < intralayer)
        x_line = constants.np.linspace(inter_edges[0], inter_edges[-1], 100)
        ax.plot(x_line, x_line, 'r--', alpha=0.8, linewidth=2, label='Interlayer = Intralayer')
        ax.legend()
    
    def plot_parameter_distributions(self) -> None:
        """Plot parameter value distributions."""
        # Create subplots for each parameter
        fig_subplots = self.figure.subplots(2, 2)
        axes = fig_subplots.flatten()
        
        parameters = [
            ('theta', 'Theta (degrees)'),
            ('weight_ratio', 'Weight Ratio'),
            ('interlayer_dist_threshold', 'Interlayer Threshold'),
            ('intralayer_dist_threshold', 'Intralayer Threshold')
        ]
        
        for i, (param, label) in enumerate(parameters):
            values = [getattr(dp, param) for dp in self.analyzer.data_points]
            
            axes[i].hist(values, bins=20, alpha=0.7, edgecolor='black')
            axes[i].set_xlabel(label)
            axes[i].set_ylabel('Count')
            axes[i].set_title(f'Distribution of {label}')
            axes[i].grid(True, alpha=0.3)
        
        self.figure.tight_layout()
    
    def plot_coverage_statistics(self) -> None:
        """Plot coverage statistics and gaps."""
        ax = self.figure.add_subplot(111)
        
        # Calculate statistics
        total_points = len(self.analyzer.data_points)
        unique_theta = len(set(round(dp.theta, 2) for dp in self.analyzer.data_points))
        unique_weights = len(set(round(dp.weight_ratio, 3) for dp in self.analyzer.data_points))
        unique_interlayer = len(set(round(dp.interlayer_dist_threshold, 3) for dp in self.analyzer.data_points))
        unique_intralayer = len(set(round(dp.intralayer_dist_threshold, 3) for dp in self.analyzer.data_points))
        
        # Expected ranges based on generation code
        expected_a = list(range(2, 9))  # [2,3,4,5,6,7,8]
        expected_intralayer = [round(x, 1) for x in constants.np.arange(1.0, 2.0, 0.2)]
        
        # Create bar chart
        categories = ['Total Points', 'Unique θ', 'Unique W.Ratio', 'Unique Inter', 'Unique Intra']
        values = [total_points, unique_theta, unique_weights, unique_interlayer, unique_intralayer]
        
        bars = ax.bar(categories, values, color=['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon', 'lightpink'])
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Count')
        ax.set_title('Data Coverage Statistics')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add text box with additional info
        info_text = f"""Coverage Summary:
        • Total data points: {total_points}
        • θ coverage: {unique_theta} values
        • Weight ratio coverage: {unique_weights} values
        • Parameter space density: {total_points/(unique_theta*unique_weights):.1f} pts/combination"""
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def export_plot(self) -> None:
        """Export the current coverage plot."""
        try:
            plot_type_name = self.current_plot_type.replace('_', '-')
            default_name = f"coverage-{plot_type_name}"
            
            file_dialog = QFileDialog()
            file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
            file_dialog.setNameFilters([
                "PNG Image (*.png)",
                "PDF Document (*.pdf)",
                "SVG Vector (*.svg)"
            ])
            file_dialog.setDefaultSuffix("png")
            file_dialog.selectFile(default_name)
            
            if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
                file_path = file_dialog.selectedFiles()[0]
                
                self.figure.savefig(file_path, dpi=300, bbox_inches='tight',
                                   facecolor='white', edgecolor='none')
                logger.info(f"Coverage plot exported to: {file_path}")
        
        except Exception as e:
            logger.error(f"Error exporting coverage plot: {e}")


class coverage_parameter_widget(QWidget):
    """Simple parameter widget for coverage analysis."""
    
    def __init__(self, analyzer: simulation_data_analyzer, coverage_widget: data_coverage_widget):
        super().__init__()
        self.analyzer = analyzer
        self.coverage_widget = coverage_widget
        self.init_ui()
    
    def init_ui(self) -> None:
        """Initialize the parameter control UI."""
        layout = QVBoxLayout()
        
        # Data summary
        summary_group = QGroupBox("Data Summary")
        summary_layout = QFormLayout()
        
        if self.analyzer.data_points:
            summary = self.analyzer.get_data_summary()
            summary_layout.addRow("Total Points:", QLabel(str(summary.get("total_points", 0))))
            
            # Parameter ranges
            theta_range = summary.get("theta_range", (0, 0))
            summary_layout.addRow("θ Range:", QLabel(f"{theta_range[0]:.2f}° - {theta_range[1]:.2f}°"))
            
            weight_range = summary.get("weight_ratio_range", (0, 0))
            summary_layout.addRow("Weight Range:", QLabel(f"{weight_range[0]:.2f} - {weight_range[1]:.2f}"))
        
        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)
        
        # Coverage info
        coverage_group = QGroupBox("Coverage Analysis")
        coverage_layout = QVBoxLayout()
        
        info_label = QLabel("""
        This tab analyzes parameter space coverage:
        
        • Heatmaps show data density
        • Blue X marks indicate gaps
        • Distributions show sampling patterns
        • Statistics provide coverage metrics
        """)
        info_label.setWordWrap(True)
        info_label.setStyleSheet("QLabel { color: #666; }")
        coverage_layout.addWidget(info_label)
        
        coverage_group.setLayout(coverage_layout)
        layout.addWidget(coverage_group)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def cleanup(self) -> None:
        """Clean up resources."""
        pass