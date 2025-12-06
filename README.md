# Twisted Bilayer Graphene (TBG) Neural Network Simulation Framework

A comprehensive Python framework for simulating twisted bilayer graphene at commensurate angles, with integrated neural network capabilities for Dirac point prediction and analysis.

## 🔬 Scientific Context

This project implements the theoretical framework described in:

**Malinovitch, Tal.** "Twisted Bilayer Graphene in Commensurate Angles." arXiv preprint arXiv:2409.12344 (2024).  
📄 [arxiv.org/abs/2409.12344](https://arxiv.org/abs/2409.12344)

The framework combines:
- **Physics-based modeling** of TBG structures with rational twist angles
- **Neural network prediction** of Dirac points using hybrid training approaches
- **Interactive visualization** through GUI interfaces
- **Comprehensive data analysis** tools for band structure and electronic properties

---

## 🏗️ Project Architecture

### Core Physics Engine
- **`TBG.py`** - Main TBG construction class with lattice generation and interlayer coupling
- **`graph.py`** - Graph theory foundation with periodic boundary conditions and Floquet Laplacian
- **`band_comp_and_plot.py`** - Band structure computation and eigenvalue analysis
- **`plotting.py`** - Specialized plotting utilities for lattice structures and band diagrams
- **`utils.py`** - Mathematical utilities for twist angle calculations and parameter validation

### Neural Network System
- **`NN_Dirac_point.py`** - Main orchestrator for neural network-based Dirac point prediction
- **`dirac_network_builder.py`** - Network architecture construction with specialized activation functions
- **`dirac_network_trainer.py`** - Hybrid training system (data pretraining + physics optimization)
- **`dirac_network_benchmark.py`** - Performance analysis and acceleration factor measurement
- **`dirac_network_persistence.py`** - Model saving/loading and checkpoint management
- **`neural_network_base.py`** - Base neural network implementation with custom activations

### Data Management
- **`Generate_training_data.py`** - Automated training data generation from physics simulations
- **`simulation_data_loader.py`** - Data loading and preprocessing with validation
- **`data_structures_for_training_data.py`** - Structured data containers and batch processing
- **`stats.py`** - Statistical analysis and performance metrics

### GUI Applications
- **`GuiFile.py`** - Main interactive GUI for TBG parameter exploration and visualization
- **`gui_data_analysis.py`** - Data analysis GUI with plotting and statistical tools
- **`widgets_data_analysis.py`** - Custom PyQt6 widgets for data visualization
- **`sim_data_analysis.py`** - Simulation data analysis backend
- **`analysis_plot_widgets.py`** - Specialized plotting widgets for analysis

### Configuration
- **`constants.py`** - Centralized constants, configuration parameters, and common imports
- **`Training_data/`** - Directory for training datasets and model checkpoints

---

## ✨ Key Features

### Physics Simulation
- **Commensurate TBG Construction**: Create bilayer graphene with rational twist angles (a,b parameters)
- **Periodic Boundary Conditions**: Full implementation of Floquet-Bloch theory for infinite systems
- **Band Structure Analysis**: Compute and visualize electronic band structures with focus on flat bands
- **Dirac Point Detection**: Physics-based optimization to locate band touching points

### Neural Network Capabilities
- **Hybrid Training**: Combines data-driven pretraining with physics-based fine-tuning
- **Custom Architecture**: Specialized network design for k-point prediction with wrapped coordinates
- **Performance Optimization**: Achieve 10-100x speedup over direct physics calculations
- **Automated Benchmarking**: Built-in acceleration factor measurement and performance analysis

### Interactive Tools
- **Real-time Visualization**: Interactive GUI for parameter exploration and immediate feedback
- **Data Analysis Suite**: Comprehensive tools for analyzing training data and model performance
- **Automated Data Generation**: Generate large-scale training datasets from physics simulations
- **Model Persistence**: Robust saving/loading system with metadata and version control

---

## 🚀 Quick Start

### Recommended Workflow: Generate Data + Train Network

**Step 1: Generate training data with automatic checkpointing**
```bash
cd "C:\Users\talth\Dropbox\Professional\pythings\TBG project"
python NN_Dirac_point.py --generate-data
```

This will:
- ✅ Generate high-quality training data using 12 CPU cores (4 reserved for you)
- ✅ Save checkpoints every 100 samples (crash recovery)
- ✅ Automatically resume if interrupted
- ✅ Proceed to network training when complete
- ✅ Save final trained model

**Step 2: Use the trained model**
```bash
python NN_Dirac_point.py  # Loads trained model and runs benchmarks
```

**Features**:
- **Parallel processing**: Uses 4 parallel workers (8 cores reserved + hard limit to prevent freeze)
- **Process priority**: Workers run at BELOW_NORMAL priority (user apps get priority)
- **Checkpoint system**: Automatic save/resume for both data generation and training
- **Quality validation**: Strict thresholds (gap<0.001, R²<0.05, weighted<0.05)
- **Expanded coverage**: 13 a-values [2,3,4,5,6,7,8,9,10,13,15,17,19]

---

### Alternative Options

**Option 1: Interactive GUI**
```bash
python GuiFile.py
```
Launch the main GUI for interactive TBG exploration with real-time parameter adjustment.

**Option 2: Data Analysis GUI**
```bash
python gui_data_analysis.py
```
Open the data analysis interface for examining training datasets and model performance.

**Option 3: Direct Neural Network Training (if data exists)**
```python
from NN_Dirac_point import nn_dirac_point

# Create and train neural network
nn = nn_dirac_point()

# Train on existing data
nn.train_general(epochs=150, batch_size=32)

# Make predictions
k_x, k_y, velocity = nn.predict([5, 1, 1.0, 0.8, 0.5, 1.0])
print(f"Dirac point: k=({k_x:.4f}, {k_y:.4f}), velocity={velocity:.2f}")
```

### Option 4: Physics Simulation
```python
from TBG import tbg
from utils import compute_twist_constants
import matplotlib.pyplot as plt

# Create TBG system
system = tbg(n=10, m=10, a=5, b=1, 
            interlayer_dist_threshold=1.0,
            intralayer_dist_threshold=1.0,
            unit_cell_radius_factor=3.0)

# Visualize structure
fig, ax = plt.subplots(figsize=(12, 10))
system.plot(ax)
plt.title(f'TBG Structure: a={system.a}, b={system.b}')
plt.show()

# Create periodic version for band structure
n_scale, alpha, factor, k_point = compute_twist_constants(system.a, system.b)
periodic_system = system.full_graph.create_periodic_copy(
    system.lattice_vectors, k_point
)

# Compute and plot band structure
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')
periodic_system.plot_band_structure(
    ax, num_of_points=30, min_band=1, max_band=8,
    inter_graph_weight=0.5, intra_graph_weight=1.0
)
plt.show()
```

---

## 📦 Installation

### Prerequisites
- **Python 3.8+** (recommended: 3.9-3.12, tested on 3.12)
- **Git** (for cloning repository)
- **8+ GB RAM** (recommended: 16+ GB for large system simulations)

### Dependencies
The project uses a comprehensive set of scientific computing and GUI libraries:
```bash
# Install all dependencies from requirements.txt
pip install -r requirements.txt

# Or install individual packages
pip install numpy>=1.21.0 scipy>=1.7.0 matplotlib>=3.5.0 PyQt6>=6.2.0 scikit-learn>=1.0.0 pandas>=1.3.0 seaborn>=0.11.0 pytest>=7.0.0
```

### Full Installation
```bash
# Clone repository
git clone https://github.com/yourusername/tbg-neural-network.git
cd tbg-neural-network

# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🧠 Neural Network Architecture

The neural network system uses a specialized architecture designed for multi-valued Dirac point prediction:

### Current Network Design (Dec 2025)
- **Input Layer**: 6 features with critical transformation:
  - Raw parameters: (a, b, interlayer_threshold, intralayer_threshold, inter_weight, intra_weight)
  - **Input transformation**: (a,b) → (1/a, 1/b) to normalize scale (a,b ∈ [2,333] → [0.003, 0.5])
  - Other parameters already in [0.15, 1.8] range - no transformation needed
- **Hidden Layers**: 3 layers × 30 neurons with softsign activation (default configuration)
  - Currently testing optimal capacity: 10-10-10 (too small), 20-20-20 (good), 30-30-30 (testing), 40-40-40 (collapsed)
- **Output Layer**: 6 features predicting TWO Dirac points simultaneously
  - Outputs: [k_x1, k_y1, nu1, k_x2, k_y2, nu2]
  - k-points use wrapped coordinates (periodic boundary conditions)
  - nu converted to velocity: v = (1-nu)/nu

### Multi-Valued Prediction Challenge
TBG systems have **2-8 different Dirac points** for the same parameters. The network must:
- Learn to predict multiple valid solutions (not just one)
- Avoid mode collapse (predicting constant values)
- Balance Coulomb repulsion (force predictions apart) with accuracy

### Training Strategy
1. **Minimum Distance Loss**: Each prediction compared to CLOSEST target among all valid Dirac points
2. **Coulomb Repulsion**: Penalty term forces two predictions to be different
3. **Early Stopping**: Monitors validation loss with patience=50 epochs
4. **Gradient Clipping**: Clip to ±2.0 to prevent explosion
5. **Learning Rate Reduction**: Halve LR when loss plateaus (min: 1e-6)

### Current Training Results (Dec 2025)
- **Training data**: 25,362 parameter sets → 49,248 Dirac points (avg 1.94 per parameter set)
- **Data variance**: k_x std=0.261, k_y std=0.236 (target for network to achieve)
- **Best architecture so far**: 20-20-20 network
  - Achieves 78% of training data variance (Pred1 k_x std=0.203 vs 0.261 target)
  - Validation loss: 0.092368
  - Shows network is learning to explore k-space, not fully capturing all modes
- **Challenges**: Larger networks (40-40-40) show training instability and mode collapse
- **Status**: Testing 30-30-30 as optimal middle ground

*Note: This is an active research project. Performance metrics updated as training experiments complete.*

---

## 📊 Data Generation and Analysis

### Training Data Generation
```bash
python Generate_training_data.py
```
This script:
- Systematically samples TBG parameter space
- Performs physics-based Dirac point calculations
- Saves results in structured CSV format
- Validates data quality and completeness

### Data Analysis Tools
The framework provides comprehensive analysis capabilities:
- **Statistical Analysis**: Distribution analysis, correlation studies
- **Performance Benchmarking**: Speed and accuracy comparisons
- **Visualization**: Interactive plots, 3D band structures, parameter sweeps
- **Model Diagnostics**: Training curves, loss landscapes, prediction accuracy

---

## 🔧 Configuration

### Physics Parameters (constants.py)
```python
# Twist angle parameters
DEFAULT_K_TOLERANCE = 0.05        # K-point identification threshold
DEFAULT_E_TOLERANCE = 0.001       # Energy threshold for band touching
DEFAULT_SEARCH_RADIUS = 0.5       # Dirac point search radius

# Numerical parameters  
NUMERIC_TOLERANCE = 1e-10         # Eigenvalue convergence tolerance
MATRIX_SIZE_SPARSE_THRESHOLD = 100 # Switch to sparse solvers

# Neural network defaults
DEFAULT_NN_LAYER_SIZE = 8         # Hidden layer neurons
DEFAULT_BATCH_SIZE = 100          # Training batch size
ADAM_LEARNING_RATE = 0.01         # Default learning rate
```

### System Requirements
- **Small systems** (a,b ≤ 7): 4GB RAM, <1 minute computation
- **Medium systems** (7 < a,b ≤ 13): 8GB RAM, 1-5 minutes
- **Large systems** (a,b > 13): 16GB+ RAM, 5+ minutes

---

## 🔄 Checkpoint System & Crash Recovery

### Data Generation Checkpoints
- **File**: `training_checkpoint.json` (project root)
- **Saves every**: 100 samples (automatic)
- **Resume**: Automatically loads on restart
- **Command**: `python NN_Dirac_point.py --generate-data` (same command to resume)

### Network Training Checkpoints
- **Files**: `checkpoint_epoch_XXX.npz` (e.g., `checkpoint_epoch_050.npz`)
- **Saves every**: 10 epochs (configurable)
- **Resume**: Automatically loads latest checkpoint
- **Command**: `python NN_Dirac_point.py` (same command to resume)

**Interruption Handling**:
- Press Ctrl+C to stop gracefully (checkpoint saves)
- Resume with same command - progress restored automatically
- No work lost - all progress saved incrementally

---

## ⚙️ Parallel Processing Configuration

### Current Setup (Updated 2025-11-07 - Fixed after system freeze)
- **Total CPU cores**: 16 logical processors
- **Reserved for user**: 8 cores (INCREASED from 4 after testing showed freeze)
- **Hard limit**: 4 parallel processes maximum (prevents freeze)
- **Actual processes used**: 4 = min(16 - 8, 4)
- **Process priority**: BELOW_NORMAL (workers don't compete with user apps)
- **Configuration file**: [constants.py](constants.py) (lines 113-116)

### Adjusting Core Usage
Edit `constants.py` to change limits:
```python
RESERVED_CORES: int = 8  # Cores reserved for user work (currently 8)
MAX_PARALLEL_PROCESSES: int = 4  # HARD LIMIT - reduce to 2-3 if still freezing
```

**⚠ IMPORTANT**: Do NOT increase MAX_PARALLEL_PROCESSES above 4! Each worker processes 12 weight combinations internally (CPU-intensive). More than 4 workers can freeze your system.

---

## 🐛 Troubleshooting

### Data Generation Issues

**Problem**: "No training data generated"
- ✓ Check `Training_data/` directory exists
- ✓ Check disk space available
- ✓ Review `training_output.log` for error details

**Problem**: Process killed or crashed
- ✓ Checkpoint saved automatically in `training_checkpoint.json`
- ✓ Resume with: `python NN_Dirac_point.py --generate-data`
- ✓ Check log for last processed parameters

**Problem**: Computer too slow or unresponsive
- ✓ Increase `RESERVED_CORES` in `constants.py` (from 4 to 6 or 8)
- ✓ This leaves more CPU cores for your work
- ✓ Generation will take longer but system stays responsive

### Network Training Issues

**Problem**: "No training data found"
- ✓ Run data generation first: `python NN_Dirac_point.py --generate-data`
- ✓ Check `Training_data/dirac_training_data.csv` exists and has data

**Problem**: Training crashes or stops
- ✓ Check for checkpoint files: `checkpoint_epoch_XXX.npz`
- ✓ Resume automatically: `python NN_Dirac_point.py`
- ✓ Review `training_output.log` for error messages

**Problem**: Loss not decreasing
- ✓ Verify data quality (new thresholds: gap<0.001, R²<0.05)
- ✓ Check learning rate (default: 0.0001 in constants.py)
- ✓ Review validation loss trends in logs

### General Issues

**Problem**: Import errors or missing modules
- ✓ Ensure all dependencies installed: `pip install -r requirements.txt`
- ✓ Check Python version: Python 3.12 recommended
- ✓ Verify virtual environment activated

**Problem**: Out of memory errors
- ✓ Close other applications
- ✓ Reduce batch size in training config
- ✓ For data generation: reduce number of parallel processes

---

## 🧪 Testing and Validation

### Running Tests
```bash
# Run all tests using pytest (recommended)
pytest -v

# Run all tests with coverage report
pytest --cov=. --cov-report=html

# Run specific test files
pytest Tests/test_tbg.py -v
pytest Tests/test_neural_network.py -v

# Run tests using unittest (alternative)
python -m unittest discover Tests -v
```

### Validation Methods
- **Physics Validation**: Compare neural network predictions with direct calculations
- **Cross-validation**: K-fold validation on training datasets
- **Parameter Sweep Tests**: Systematic testing across parameter ranges
- **Performance Benchmarks**: Speed and memory usage analysis

---

## 🎯 Use Cases

### Research Applications
- **Band Structure Studies**: Analyze flat band formation and topology
- **Parameter Optimization**: Find optimal TBG configurations for specific properties
- **High-throughput Screening**: Rapidly evaluate thousands of parameter combinations
- **Machine Learning Research**: Develop physics-informed neural networks

### Educational Applications  
- **Interactive Learning**: GUI-based exploration of TBG physics
- **Visualization**: 3D band structure plots and lattice animations
- **Parameter Studies**: Understand how twist angle affects electronic properties
- **Computational Physics**: Learn numerical methods for condensed matter systems

---

## 📈 Performance Analysis Framework

### Built-in Benchmarking Tools
The framework includes comprehensive performance measurement capabilities:
- **Timing Analysis**: Automatic measurement of physics calculation vs. neural network prediction times
- **Memory Profiling**: System resource usage tracking for different TBG configurations
- **Accuracy Validation**: Physics-based validation of neural network predictions
- **Scalability Testing**: Performance analysis across varying system sizes

### System Requirements & Scaling
- **Small TBG Systems** (a,b ≤ 7): 4-8 GB RAM, moderate computation time
- **Medium TBG Systems** (7 < a,b ≤ 13): 8-16 GB RAM, extended computation time
- **Large TBG Systems** (a,b > 13): 16+ GB RAM, significant computation time

### Benchmarking Methodology
- **Physics Baseline**: Direct eigenvalue computation with sparse matrix methods
- **NN Prediction**: Forward pass through trained network with preprocessing
- **Validation**: Cross-validation against physics calculations for accuracy assessment

*Detailed performance results will be published as training and validation complete.*

---

## 🛠️ Troubleshooting

### Common Issues

**1. Memory Errors**
```python
# Reduce system size
system = tbg(n=5, m=5, ...)  # Instead of n=20, m=20

# Use sparse matrices
constants.MATRIX_SIZE_SPARSE_THRESHOLD = 50
```

**2. Training Data Issues**
```bash
# Generate fresh training data
python Generate_training_data.py

# Check data directory
ls Training_data/
```

**3. GUI Display Problems**
```bash
# Install PyQt6 development version
pip install --upgrade PyQt6

# Check display settings
export QT_AUTO_SCREEN_SCALE_FACTOR=1
```

**4. Convergence Issues**
```python
# Adjust tolerances
constants.DEFAULT_E_TOLERANCE = 0.01  # Looser tolerance
constants.MAX_ITERATIONS = 500        # More iterations
```

---

## 🔬 Advanced Usage

### Custom Network Architecture
```python
from dirac_network_builder import dirac_network_builder

# Custom configuration
config = {
    'input_features': 6,
    'output_features': 3, 
    'hidden_layer_size': 16,  # Larger layers
    'num_hidden_layers': 3    # More layers
}

builder = dirac_network_builder(config)
network = builder.build_network()
```

### Physics-Based Training
```python
from dirac_network_trainer import dirac_network_trainer

# Custom training parameters  
trainer = dirac_network_trainer({
    'learning_rate': 0.001,
    'momentum': 0.9,
    'physics_weight': 0.7,
    'data_weight': 0.3
})

# Train for specific TBG system
final_loss = trainer.train_physics_based([5, 1, 1.0, 0.8, 0.5, 1.0], 
                                        iterations=200)
```

### Batch Processing
```python
import glob
from simulation_data_loader import simulation_data_analyzer

# Process multiple data files
data_files = glob.glob("Training_data/*.csv")
analyzer = simulation_data_analyzer()

for file in data_files:
    results = analyzer.analyze_file(file)
    print(f"File: {file}, Systems: {len(results)}")
```

---

## 📚 References and Citations

### Primary Reference
```bibtex
@article{malinovitch2024twisted,
  title={Twisted Bilayer Graphene in Commensurate Angles},
  author={Malinovitch, Tal},
  journal={arXiv preprint arXiv:2409.12344},
  year={2024}
}
```

### Related Work
- **TBG Theory**: Bistritzer & MacDonald, PNAS 2011
- **Neural Networks for Physics**: Carleo & Troyer, Science 2017
- **Periodic Boundary Conditions**: Thouless et al., Phys. Rev. Lett. 1982

---

## 🤝 Contributing

We welcome contributions to this project! Areas where contributions are particularly valuable:

### Code Contributions
- **Performance Optimization**: Faster algorithms, memory efficiency
- **New Features**: Additional analysis tools, visualization options
- **Testing**: More comprehensive test coverage
- **Documentation**: Code comments, tutorials, examples

### Research Contributions
- **Validation Studies**: Compare predictions with experimental data
- **Parameter Studies**: Systematic exploration of TBG parameter space
- **Method Development**: New neural network architectures, training strategies
- **Applications**: Novel use cases and scientific applications

### Code Quality Standards
This project maintains high software engineering standards:
- **100% Type Hints Coverage**: All functions and methods have comprehensive type annotations
- **Professional Docstrings**: Google-style docstrings throughout with parameter and return documentation
- **Comprehensive Error Handling**: Custom exception hierarchy with physics-aware validation
- **Professional Logging**: Structured logging with appropriate levels (no print statements)
- **Testing Framework**: pytest-based test suite with coverage reporting
- **Code Linting**: flake8, black, and mypy compatibility

### Getting Started with Contributing
1. **Fork the repository** and create a feature branch
2. **Follow PEP 8** coding standards and maintain type hint coverage
3. **Add comprehensive docstrings** for new functions and classes
4. **Add tests** for new functionality with pytest
5. **Update documentation** for any API changes
6. **Submit a pull request** with detailed description

---

## 📄 License

This project is open for academic and research use. The code is provided under the MIT License for academic purposes. Commercial use requires explicit permission.

### Academic Use
- ✅ Research and educational purposes
- ✅ Modification and distribution for research
- ✅ Citation of original work required

### Commercial Use
- ❓ Requires explicit permission
- 📧 Contact author for licensing terms

---

## ✍️ Author & Contact

**Tal Malinovitch**  
Rice University  
📧 tal.malinovitch@rice.edu  
🔗 [Personal Website](talmalinovitch.notion.site/)  
🐙 [GitHub Profile](https://github.com/Tal-Malinovitch)

### Getting Help
- **Issues**: Report bugs and request features via GitHub Issues
- **Questions**: Contact author via email for research collaborations
- **Updates**: Watch the repository for new releases and features

---

---

## 🚧 Project Status

This is an **active research project** currently in development:

### Current Phase (December 2025)
- ✅ Core physics engine complete and validated
- ✅ Training data generation pipeline operational (25,362 parameter sets)
- ✅ Input transformation bug fixed (critical for learning)
- 🔄 Neural network architecture optimization in progress
  - Testing capacity scaling: 10→20→30→40 neuron architectures
  - Investigating multi-valued prediction strategies
  - Measuring variance recovery and mode collapse behavior
- ⏳ Comprehensive benchmarking pending (awaits optimal architecture)
- ⏳ Performance validation in progress

### Known Limitations
- Multi-valued prediction remains challenging (78% variance recovery with best architecture)
- Larger networks show training instability and mode collapse
- Optimal architecture size still under investigation
- Comprehensive acceleration factor benchmarks not yet published

### Future Work
- Alternative architectures: mixture of experts, multiple output heads
- Alternative loss functions: mixture density networks, k-means style assignment
- Data augmentation strategies
- Production deployment and optimization

*Last Updated: December 2025*
*Framework Version: 2.2.0 (Active Development)*
*Documentation Status: ✅ Complete with comprehensive type hints and professional docstrings*