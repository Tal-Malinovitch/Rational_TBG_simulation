# Release Notes - v2.1.0 (December 2025)

## Overview

This is a **major update** representing a complete overhaul of the neural network system. The release includes critical bug fixes, architecture redesign for multi-valued prediction, comprehensive benchmark system improvements, and extensive capacity scaling experiments.

**⚠️ BREAKING CHANGE**: The input transformation bug fix requires complete model retraining. All models from v2.0.0 and earlier are incompatible.

---

## 🔴 Critical Bug Fixes

### 1. Input Transformation Bug (Dec 1) - **BREAKING FIX**
**Impact**: Complete model retraining required

- **Problem**: Training used raw (a,b) values [2, 333] instead of transformed (1/a, 1/b) [0.003, 0.5]
  - Created 100-200x scale mismatch between input features
  - Network ignored inputs and predicted nearly constant values (mode collapse)
  - Previous models achieved only 30% k-point error with prediction std ~0.05

- **Root Cause**:
  - `preprocess_training_data.py` stored raw parameters
  - Training loop never applied (a,b) → (1/a, 1/b) transformation
  - Benchmarks DID apply transformation → train/test mismatch

- **Fix Applied**:
  - Modified `preprocess_training_data.py` to apply transformation during preprocessing
  - Created dual fields: `parameters_transformed` (for training), `parameters_raw` (for reference)
  - Updated `dirac_network_trainer.py` to use transformed parameters
  - Regenerated `grouped_training_data.pkl` with correct transformations

- **Files**: `preprocess_training_data.py`, `dirac_network_trainer.py`

### 2. Benchmark Test Generation Bug (Nov 27)
- **Problem**: Generated systems with 20k-40k nodes that were never in training data
  - Extracted max_a from SCALED data instead of base coprime range
  - Created test cases like (a=37, b=8) - wild extrapolation, not interpolation
  - Benchmarks took 4+ hours with only 4 tests completing

- **Fix**: Complete redesign with physics-motivated test categories:
  - Weight interpolation/extrapolation using base coprime pairs [2,10]
  - Threshold extrapolation (enforces intralayer < interlayer constraint)
  - Scaling invariance tests (factors 10-20)
  - Coprime extrapolation: close (a∈[11,15]) and far (a∈[16,20])
  - Size limits: N×scale² < 500 prevents huge systems
  - Symmetry checking with `compute_sym_factor()` avoids duplicates

- **Files**: `comprehensive_benchmark_design.py`, `comprehensive_benchmark_executor.py`

### 3. Benchmark Parallelization Bug (Nov 26)
- **Problem**: Used 15 processes on 16-CPU system, violated MAX_PARALLEL_PROCESSES=8
  - CPU oversubscription caused massive context switching
  - Individual tests took 10-174 minutes (should be <5 minutes)

- **Fix**: Enforces proper process limits
  - Uses: min(cpu_count - RESERVED_CORES, MAX_PARALLEL_PROCESSES) = 8
  - Applies to both new runs and checkpoint resumption
  - Logs warning if checkpoint had invalid num_processes

- **File**: `comprehensive_benchmark_executor.py`

### 4. ADAM Optimization Return Behavior (Nov 26)
- **Problem**: ADAM raised ValueError and discarded best point when couldn't reach threshold
  - Computation time included all 500 iterations
  - Best point found (metric ~0.08) was discarded
  - Benchmarks fell back to worse NN prediction

- **Fix**: Returns best point found even if threshold not met
  - Tracks best point/metric during iterations
  - Logs warning when doesn't fully converge
  - No more ValueError on convergence failure

- **File**: `Generate_training_data.py`

### 5. JSON Serialization Bug (Nov 30)
- **Problem**: Benchmark results crashed on save with numpy array types
- **Fix**: Convert all numpy types to Python native types before serialization
- **File**: `comprehensive_benchmark_executor.py`

---

## 🧠 Neural Network Architecture - Complete Redesign

### Multi-Valued Prediction System (Dec 1-5)

**Problem**: TBG systems have 2-8 different Dirac points for the same parameters. Previous architecture predicted only ONE point.

**Solution**: Redesigned to predict TWO Dirac points simultaneously

- **Architecture Changes**:
  - Output expanded from 3 → 6 features: [k_x1, k_y1, nu1, k_x2, k_y2, nu2]
  - Both predictions use same input, different learned representations
  - k-points use wrapped coordinates (periodic boundary conditions)
  - nu converted to velocity: v = (1-nu)/nu

- **Loss Function**:
  - **Minimum distance matching**: Compare each prediction to CLOSEST target among all valid Dirac points
  - **Coulomb repulsion**: Penalty term (weight: 0.1) forces two predictions apart
  - **Weighted components**: 0.45×k₁ + 0.45×k₂ + 0.1×nu per prediction
  - Prevents mode collapse where both predictions converge to same point

- **Files**: `dirac_network_builder.py`, `dirac_network_trainer.py`

### Capacity Scaling Experiments (Dec 1-5)

Systematically tested network sizes to find optimal capacity:

| Architecture | Parameters | Samples/Param | Val Loss | Pred1 k_x std | Variance Recovery | Result |
|--------------|------------|---------------|----------|---------------|-------------------|--------|
| 10-10-10 | 356 | 71.2x | 0.094275 | 0.081 | 31% | ❌ Too small |
| **20-20-20** | **1,106** | **22.9x** | **0.092368** | **0.203** | **78%** | ✅ **BEST** |
| 30-30-30 | 2,100 | 12.1x | Testing | - | - | 🔄 In progress |
| 40-40-40 | 3,806 | 6.7x | 0.094513 | 0.060 | 23% | ❌ Mode collapse |

**Training Data Statistics**:
- 25,362 parameter sets → 49,248 Dirac points (avg 1.94 per parameter set)
- k_x distribution: std=0.261, range=[-0.500, 0.500]
- k_y distribution: std=0.236, range=[-0.500, 0.500]

**Key Findings**:
1. **20-20-20 network achieves 78% variance recovery** - best result so far
2. **Non-monotonic relationship**: Larger ≠ better
   - Too small (10-10-10): Insufficient capacity
   - Sweet spot (20-20-20): Balanced capacity and stability
   - Too large (40-40-40): Training instability, worse local minima
3. **Mode collapse**: 40-40-40 predictions had 8% variance on validation (vs 23% on training batches)

**Current Configuration**: Testing 30-30-30 as middle ground between 20 (good) and 40 (collapsed)

---

## 📊 Training Data & Preprocessing

### New Preprocessing Pipeline (Dec 1)

Created dedicated preprocessing system: `preprocess_training_data.py`

**Features**:
- Groups Dirac points by parameter set
- Applies input transformation: (a,b) → (1/a, 1/b)
- Generates `grouped_training_data.pkl`:
  - 25,362 parameter sets
  - 49,248 unique Dirac points
  - Dual parameter fields (transformed + raw)
  - Metadata: num_parameter_sets, total_dirac_points, preprocessing_info

**Data Validation**:
- Verifies transformation applied correctly
- Computes variance statistics
- Validates grouping by parameter set
- Ensures no data loss during processing

---

## 🎯 Training System Enhancements

### Enhanced Monitoring & Diagnostics

New training metrics tracked in real-time:

1. **Variance Tracking**:
   - Prediction variance across batches
   - Compares to training data variance (target: k_x std=0.261)
   - Identifies mode collapse early

2. **Hidden Layer Diagnostics**:
   - Saturation monitoring (mean ± std of activations)
   - Nu clamping statistics (tracks when nu exceeds [0,1])
   - Pre-activation bias values for first 20 batches

3. **Gradient Statistics**:
   - Per-component gradients: k_x, k_y, nu (mean ± std)
   - Total gradient magnitude
   - Identifies gradient explosion/vanishing

4. **Prediction Examples**:
   - Logs sample predictions every 10 batches
   - Shows k-separation between two predictions
   - Tracks Coulomb repulsion effectiveness

### Checkpoint & Resume System

**Automatic Checkpointing**:
- Saves every 10 epochs: `checkpoint_epoch_XXX.npz`
- Best model saved separately: `best_model.npz`
- Tracks: epoch, val_loss, training_loss, patience_counter, LR, timestamps

**Metadata Saved**:
- Training progress (epochs_without_improvement, patience_counter)
- Learning rate history
- Average epoch time
- Training/validation sample counts

**Resume Capability**:
- Automatically loads latest checkpoint on restart
- Restores complete training state
- Continues from exact epoch where stopped

**Files**: `dirac_network_trainer.py`, `dirac_network_persistence.py`

### Training Stability Features

1. **Gradient Clipping**: Hard clip at ±2.0 prevents explosion
2. **Learning Rate Reduction**: Halves when loss plateaus (min: 1e-6)
3. **Early Stopping**: Patience=50 epochs prevents overfitting
4. **Loss Explosion Detection**: Threshold=10.0, triggers LR reduction

---

## 🔬 Benchmark System Overhaul

### New Test Categories (Nov 27)

Replaced broken interpolation with physics-motivated categories:

1. **Weight Interpolation** (10 tests):
   - Base coprime pairs from training range [2,10]
   - Weight combinations NOT in training data
   - Tests network's parameter space coverage

2. **Weight Extrapolation** (10 tests):
   - Weights above training maximum
   - Base coprime pairs from [2,10]
   - Tests generalization beyond training range

3. **Threshold Extrapolation** (10 tests):
   - Thresholds above training max
   - Enforces intralayer < interlayer constraint
   - Tests physics constraint handling

4. **Scaling Invariance** (10 tests):
   - Large scale factors [10,20]
   - Base coprime pairs from [2,10]
   - Verifies network learned scale invariance property

5. **Coprime Extrapolation - Close** (5 tests):
   - New coprime pairs: a∈[11,15]
   - Tests near-neighbor extrapolation

6. **Coprime Extrapolation - Far** (5 tests):
   - New coprime pairs: a∈[16,20]
   - Tests far extrapolation capabilities

**Safety Features**:
- Size limit: N×scale² < 500 (prevents multi-hour tests)
- Symmetry checking: Uses `compute_sym_factor()` to avoid duplicate conjugate pairs
- Validation: All tests verified against training data structure

### Adaptive ADAM Optimization (Nov 26)

**Feature**: Switches to gap-only optimization when gap is small

- **Gap threshold**: 0.01
- **Weights**: Switches from [0.6, 0.3, 0.1] → [1.0, 0.0, 0.0]
- **Rationale**: When gap already tiny, focus on minimizing it further
- **Constants**: `GAP_THRESHOLD_FOR_REFINEMENT=0.01`, `GAP_ONLY_WEIGHTS=[1.0,0,0]`

**Note**: Analysis showed NN predictions already have gap < 0.01 at iteration 0, so this mostly remains inactive. Real performance issue was CPU oversubscription.

**File**: `Generate_training_data.py`

### Benchmark Analysis & Reporting

**New Tools**:
- `comprehensive_benchmark_analyzer.py`: Statistical analysis of results
- `comprehensive_benchmark_reporter.py`: Formatted output generation

**Enhanced Tracking**:
- Detailed predictions: NN k-point, NN velocity, physics k-point, physics velocity
- Category-wise performance breakdown
- Acceleration factor measurement
- Convergence statistics

---

## 📈 Data Analysis Tools

### Training Data Quality Analysis

**New File**: `analyze_training_data_quality.py`

**Features**:
- Validates preprocessing pipeline
- Computes variance statistics for all features
- Compares transformed vs raw parameter distributions
- Verifies no data corruption during grouping
- Generates quality report

### Accuracy Comparison Tools

**Enhanced**: `dirac_network_accuracy.py`

**New Methods**:
1. `load_all_training_targets_for_params()`: Loads ALL Dirac points for same parameters
2. `find_closest_training_target()`: Finds closest target to NN prediction
3. `compare_prediction_accuracy_against_training()`: Proper accuracy assessment

**Purpose**: Previous benchmark compared NN to ONE arbitrary target. New tools compare to CLOSEST target among all valid solutions.

---

## 📝 Documentation

### New Documentation

1. **claude.md**: Comprehensive investigation history
   - All bug discoveries and fixes
   - Capacity scaling experiment results
   - Training dynamics observations
   - Decision rationale documentation

2. **README.md Updates**:
   - Current architecture description (multi-valued prediction)
   - Training results and variance metrics
   - Project status section (active development)
   - Known limitations and challenges
   - Future work directions

### Code Quality

**Standards Maintained**:
- 100% type hints coverage
- Professional docstrings (Google style)
- Comprehensive error handling
- Professional logging (no print statements)
- Constants compliance (no magic numbers)

**Removed Hardcoded Values**:
- `num_iterations=5` → `BENCHMARK_NUM_ITERATIONS`
- Added physics-motivated constants: `GAP_THRESHOLD_FOR_REFINEMENT`, `GAP_ONLY_WEIGHTS`

---

## 📦 Files Modified (30+ files)

### Core Neural Network
- `dirac_network_builder.py` - Multi-valued architecture, capacity configs
- `dirac_network_trainer.py` - Min-distance loss, Coulomb repulsion, diagnostics
- `dirac_network_persistence.py` - Enhanced checkpoint metadata
- `neural_network_base.py` - Core network support

### Data Processing
- `preprocess_training_data.py` - **NEW** - Preprocessing pipeline
- `Generate_training_data.py` - ADAM optimization improvements
- `simulation_data_loader.py` - Enhanced data loading
- `data_structures_for_training_data.py` - Data container updates

### Benchmark System
- `comprehensive_benchmark_design.py` - New test categories
- `comprehensive_benchmark_executor.py` - Parallelization fixes
- `comprehensive_benchmark_analyzer.py` - **NEW** - Results analysis
- `comprehensive_benchmark_reporter.py` - **NEW** - Report generation
- `run_comprehensive_benchmark.py` - Updated runner

### Analysis & Accuracy
- `dirac_network_accuracy.py` - Closest target comparison
- `analyze_training_data_quality.py` - **NEW** - Data validation
- `dirac_network_benchmark.py` - Benchmark tools
- `dirac_network_report.py` - Reporting utilities

### Configuration & Utilities
- `constants.py` - New constants, benchmark configs
- `NN_Dirac_point.py` - Main orchestrator updates
- `graph.py` - Graph theory utilities
- `plotting.py` - Visualization updates
- `band_comp_and_plot.py` - Band structure tools

### GUI & Visualization
- `GuiFile.py` - Main GUI updates
- `gui_data_analysis.py` - Data analysis GUI
- `analysis_plot_widgets.py` - Plot widgets
- `sim_data_analysis.py` - Simulation analysis backend

---

## ⚠️ Known Issues & Limitations

### Multi-Valued Prediction Challenges
- **Best variance recovery**: 78% with 20-20-20 network (Pred1 k_x std=0.203 vs 0.261 target)
- **Not capturing all modes**: Network learns to explore k-space but doesn't fully recover all Dirac points
- **Mode collapse risk**: Larger networks (40-40-40) collapse to predicting nearly constant values

### Training Stability
- **Non-monotonic capacity scaling**: Larger networks perform worse
- **40-40-40 network**: Complete training collapse (predictions → 0, gradient explosion)
- **Overfitting**: 40-40-40 showed 23% variance on training, only 8% on validation

### Pending Work
- **Optimal architecture**: Still testing 30-30-30 as middle ground
- **Comprehensive benchmarks**: Awaiting optimal architecture before full benchmark run
- **Acceleration factors**: Not yet measured with current architecture

### Potential Solutions Under Investigation
1. **Alternative architectures**: Mixture of experts, multiple output heads, conditional networks
2. **Alternative loss functions**: Mixture density networks, k-means assignment
3. **Data augmentation**: Generate more training samples
4. **Regularization**: Additional techniques to prevent mode collapse

---

## 🚀 Migration Guide

### For Users of v2.0.0

**⚠️ BREAKING CHANGES**:

1. **All previous models are invalid** due to input transformation bug
   - Must completely retrain from scratch
   - Cannot load checkpoints from v2.0.0 or earlier

2. **Training data must be regenerated**:
   ```bash
   python preprocess_training_data.py
   ```
   - Creates new `grouped_training_data.pkl`
   - Applies correct input transformation

3. **Network architecture changed**:
   - Old: 3 outputs [k_x, k_y, nu]
   - New: 6 outputs [k_x1, k_y1, nu1, k_x2, k_y2, nu2]
   - Must rebuild network with new config

4. **Benchmark test suite regeneration**:
   ```bash
   python comprehensive_benchmark_design.py
   ```
   - Creates new physics-motivated test categories
   - Old test suite is incompatible

### Configuration Updates

**Update constants.py if you modified**:
- `BENCHMARK_NUM_ITERATIONS` (was hardcoded `5`)
- `GAP_THRESHOLD_FOR_REFINEMENT` (new: `0.01`)
- `GAP_ONLY_WEIGHTS` (new: `[1.0, 0.0, 0.0]`)

### Fresh Installation Recommended

Due to extensive changes, recommend clean installation:

```bash
# Backup your old data
cp -r "Training_data" "Training_data_backup_v2.0.0"

# Pull latest code
git pull origin main

# Reinstall dependencies (if needed)
pip install -r requirements.txt

# Regenerate training data
python preprocess_training_data.py

# Start training with new architecture
python NN_Dirac_point.py
```

---

## 🎯 Future Work

### Short Term
- Complete 30-30-30 capacity testing
- Determine optimal architecture size
- Run comprehensive benchmarks with best architecture
- Measure acceleration factors

### Medium Term
- Investigate alternative loss functions (mixture density networks)
- Test alternative architectures (mixture of experts)
- Implement data augmentation strategies
- Optimize training hyperparameters

### Long Term
- Production deployment optimization
- Integration with experimental validation data
- Extension to other 2D materials
- Real-time prediction service

---

## 🙏 Acknowledgments

This release represents months of systematic investigation, debugging, and experimentation. Special thanks to the physics community for insights into multi-valued function learning challenges.

---

## 📚 References

**Primary Paper**:
```bibtex
@article{malinovitch2024twisted,
  title={Twisted Bilayer Graphene in Commensurate Angles},
  author={Malinovitch, Tal},
  journal={arXiv preprint arXiv:2409.12344},
  year={2024}
}
```


---

**Release Date**: December 5, 2025
**Version**: 2.1.0
**Previous Version**: 2.0.0 (September 21, 2024)
**Status**: Active Development - Research Grade

---

*For questions, issues, or contributions, please visit the GitHub repository or contact the author.*
