# Periodic Graph Framework for Twisted Bilayer Graphene

This repository provides a Python-based simulation environment for constructing, analyzing, and visualizing 2d periodic lattices,
specifically aimed at twisted bilayer graphene (TBG) at commensurate angles.
It visualizes the full graph- and able to extract the periodic unit cell.
It can construct the Floquet Laplacian, given by adding a phase factor given by e^{ika}
where a is the periodic shift, and k is the chosen moemntum. 
This feature is then utilized to plot the lower bands stucture. 

---

## 🔬 Scientific Context

This project implements the description of TBG as given in the paper 

Malinovitch, Tal. "Twisted Bilayer Graphene in Commensurate Angles." arXiv preprint arXiv:2409.12344 (2024).
📄 [arxiv.org/abs/2409.12344](https://arxiv.org/abs/2409.12344)

The system supports constructing bilayer graphene structures with rational twist angles, using reciprocal lattice vectors to calculate Bloch phases
 and generate Hermitian Laplacians suitable for band structure analysis, with focus on the Dirac points at the K points.
 

---

## 📁 Project Structure

- `Lattice.py` — Core logic for graph creation, periodic embedding, TBG construction, and Laplacian computation.
- `GuiFile.py`: This file contains the primary graphical user interface (GUI) application built with PyQt6. It provides an interactive environment for users to define and visualize Twisted Bilayer Graphene (TBG) structures, adjust various physical parameters (such as twist angle, layer distance, and hopping strengths), and view the resulting lattice geometry and electronic band structure. It integrates the core computational logic from `Lattice.py` and presents it through an intuitive graphical interface.
---

## ✅ Features

- Construct 2D hexagonal lattices with user-defined radius
- Create twisted bilayer configurations in commensurate angles 
- Compute momentum-dependent Laplacians with periodic boundary conditions
- Plot unit cells and highlight periodic edge pairs
- Plot the band structure according to the user inputs
- Fully documented classes for reusability and extension

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- Git (for cloning the repository)

### Step-by-step Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/tbg-simulation.git
   cd tbg-simulation
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv tbg_env
   
   # On Windows:
   tbg_env\\Scripts\\activate
   
   # On macOS/Linux:
   source tbg_env/bin/activate
   ```

3. **Install required packages:**
   ```bash
   pip install numpy matplotlib scipy PyQt6
   ```

   Or install from requirements file:
   ```bash
   pip install -r requirements.txt
   ```
---

### Quick Start

**Option 1: GUI Application**
```bash
python GuiFile.py
```

**Option 2: Programmatic Usage**
```python
import numpy as np
from Lattice import TBG, compute_twist_constants
import matplotlib.pyplot as plt

# Create a TBG system with twist parameters a=5, b=1
tbg = TBG(maxsize_n=10, maxsize_m=10, a=5, b=1, 
          interlayer_dist_threshold=1.0, unit_cell_radius_factor=3)

# Plot the structure
fig, ax = plt.subplots(figsize=(10, 10))
tbg.plot(ax, plot_color_top='blue', plot_color_bottom='red', plot_color_full='green')
plt.title(f'TBG with a={tbg.a}, b={tbg.b}')
plt.show()

# Create periodic version for band structure
periodic_graph = tbg.full_graph.create_periodic_copy(tbg.lattice_vectors, (1/3, 1/3))

# Plot band structure
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
periodic_graph.plot_band_structure(ax, num_of_points=20, min_band=1, max_band=5,
                                 inter_graph_weight=1.0, intra_graph_weight=1.0,
                                 K_flag=True)
plt.show()
### Troubleshooting

**Common Issues:**

1. **PyQt6 installation fails:**
   ```bash
   # Try installing PyQt6 separately first
   pip install --upgrade pip
   pip install PyQt6
   ```

2. **Import errors:**
   - Ensure you're in the correct directory
   - Check that all dependencies are installed: `pip list`

3. **Memory issues with large systems:**
   - Reduce `maxsize_n` and `maxsize_m` parameters
   - Lower `num_of_points` for band structure calculations
   
```
## 📌 How to Cite

If you use this code in academic work, please cite the associated arXiv paper:

```
@article{malinovitch2024twisted,
  title={Twisted Bilayer Graphene in Commensurate Angles},
  author={Malinovitch, Tal},
  journal={arXiv preprint arXiv:2409.12344},
  year={2024}
}
```

---

## 📄 License

This project is open for academic and research use. Licensing for broader use can be negotiated upon request.

---
## Contributing

I welcome contributions to this project! If you'd like to contribute, please follow these guidelines:

1.  **Bug Reports**: If you find a bug, please open an issue on the [project's GitHub repository](YOUR_GITHUB_REPO_LINK_HERE) describing the problem, steps to reproduce it, and expected behavior.
2.  **Feature Requests**: For new features or enhancements, please open an issue to discuss your ideas before starting work.
3.  **Code Contributions**:
    * Fork the repository.
    * Create a new branch (`git checkout -b feature/your-feature-name` or `bugfix/issue-description`).
    * Make your changes, adhering to [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code style.
    * Write clear, concise commit messages.
    * Ensure all existing tests pass and add new tests for your changes if applicable.
    * Submit a pull request to the `main` branch, providing a clear description of your changes.


## ✍️ Author

**Tal Malinovitch**  
Rice University  
tal.malinovitch@rice.edu
