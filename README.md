# State-Space Models
A modular, production-ready implementation of State-Space Models featuring multiple implementations including Learnable SSM, S4, and S5.

## Overview

This repository contains efficient and well-structured implementations of modern state-space models for sequence modeling tasks. Each implementation is modular, documented, and ready for production use.

## Repository Structure

```
state-space-models/
├── learnable-ssm/          # Learnable State-Space Model
├── s4/                     # Structured State Space (S4)
├── s5/                     # S5 Model
└── README.md
```

## Modules

### 1. Learnable-SSM (Learnable State-Space Model)

A modular implementation of a learnable state-space model using Kalman filtering with learnable parameters.

#### Files
- **`model.py`** — Core LearnableSSM model with Kalman filter implementation
- **`main.py`** — End-to-end pipeline with data generation and training
- **`dataset.py`** — Data simulation utilities for state-space models
- **`train.py`** — Training loop and optimization
- **`losses.py`** — Loss functions (NLL-based losses)

#### Key Features
- **Kalman Filter**: Implements the discrete-time Kalman filter for state estimation
- **Learnable Parameters**: State transition matrix (A), observation matrix (C), process noise (Q), and observation noise (R)
- **Uncertainty Quantification**: Tracks and returns estimated covariances
- **Numerical Stability**: Includes regularization for matrix inversion

#### Quick Start
```python
from learnable_ssm.model import LearnableSSM
import torch

# Initialize model
state_dim = 2
obs_dim = 1
model = LearnableSSM(state_dim, obs_dim)

# Run forward pass (Kalman filter)
observations = torch.randn(100, obs_dim)
estimated_states, covariances, innovations, innovation_covs = model(observations)
```

#### Architecture
The LearnableSSM model learns:
- **A matrix** (state transition) — initialized as eye matrix with specific structure
- **Q matrix** (process noise covariance) — diagonal, positive definite
- **C matrix** (observation) — maps states to observations
- **R matrix** (observation noise covariance) — diagonal, positive definite
- **Initial state** (x₀) and covariance (P₀)

---

### 2. S4 (Structured State Space)

A production-ready implementation of Structured State Space (S4) models for efficient sequence modeling, combining state-space efficiency with deep learning expressiveness.

#### Files
- **`layers.py`** — Core S4 layer implementations with frequency-domain processing
- **`model.py`** — Complete S4 model architecture
- **`main.py`** — Main entry point for experiments
- **`train.py`** — Training utilities and loop
- **`inference.py`** — Inference and evaluation functions
- **`dataset.py`** — Data loading and preprocessing
- **`README.md`** — Detailed S4 documentation

#### Key Features
- **Efficient Sequence Processing**: O(N log N) complexity using FFT
- **Stable Discretization**: Uses log-space parameterization for numerical stability
- **Frequency-Domain Kernels**: Converts state-space to convolution kernels
- **Complex State Representation**: HiPPO-inspired diagonal plus low-rank (DPLR) structure
- **Production-Ready**: Residual connections, layer normalization, and feed-forward networks

#### Quick Start
```python
from s4.model import S4Model
from s4.dataset import get_dataloader

# Create model
model = S4Model(d_model=256, n_layers=4)

# Load data
train_loader = get_dataloader('train', batch_size=32)

# Train
for batch in train_loader:
    outputs = model(batch)
```

#### Architecture Highlights
- **S4 Layers**: Efficient state-space computations with frequency-domain processing
- **Residual Connections**: Improved gradient flow
- **Layer Normalization**: Stable training
- **Feed-forward Networks**: Additional expressiveness
- **Long-range Dependencies**: Efficiently captures long-range interactions

#### Performance
- Handles long sequences efficiently
- Competitive performance on benchmark tasks
- Scales to production workloads

---

### 3. S5 (Simplified Structured State Space)

A simplified implementation of structured state-space models focusing on layer composition and flexibility.

#### Files
- **`layers.py`** — S5 layer implementation with state-space dynamics
- **`model.py`** — Complete S5 model architecture
- **`main.py`** — Main entry point with example usage
- **`train.py`** — Training utilities
- **`README.md`** — S5 documentation

#### Key Features
- **Modular Layer Design**: Flexible composition of S5 layers
- **Simplified Architecture**: Easier to understand and extend
- **State Management**: Efficient handling of hidden states
- **Parameter Efficiency**: Optimized for memory and computation

#### Quick Start
```python
from s5.model import S5Model
import torch

# Initialize model
model = S5Model(
    num_layers=4,
    state_dim=256,
    input_dim=64,
    hidden_dim=256,
    output_dim=10
)

# Forward pass
x = torch.randn(32, 100, 64)  # (batch, seq_len, input_dim)
output = model(x)  # (batch, 100, output_dim)
```

#### Architecture
```
Input → Input Projection → S5 Layers → Output Projection → Output
         (Linear)          (Stacked)      (Linear)
```

---

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- NumPy
- Matplotlib (for visualization)

### Setup
```bash
# Clone the repository
git clone https://github.com/krishnashashanth-sks/state-space-models.git
cd state-space-models

# Install dependencies
pip install torch numpy matplotlib
```

## Usage Examples

### Training a Learnable SSM
```bash
cd learnable-ssm
python main.py
```

### Training S4
```bash
cd s4
python main.py --mode train --epochs 10 --batch_size 32
```

### S4 Inference
```bash
cd s4
python main.py --mode inference --checkpoint model.pt
```

### Using S5 in Your Code
```python
from s5.model import S5Model

model = S5Model(num_layers=4, state_dim=256, input_dim=64, hidden_dim=256, output_dim=10)
# Configure and train your model
```

## Comparison of Implementations

| Feature | Learnable-SSM | S4 | S5 |
|---------|---------------|-----|-----|
| **Complexity** | O(N × d²) | O(N log N) | O(N × d²) |
| **Use Case** | Kalman filtering, uncertainty | Long sequences | Flexible, modular |
| **Stability** | Excellent | Excellent | Good |
| **Ease of Use** | Simple | Intermediate | Simple |
| **Production Ready** | ✓ | ✓ | ✓ |

## References

- **S4 Paper**: [Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396)
- **State-Space Models**: Classical approach to sequential modeling
- **Kalman Filtering**: Optimal linear state estimation

## Performance Benchmarks

Each module is designed with performance in mind:
- **Learnable-SSM**: Ideal for small-to-medium state spaces with uncertainty quantification
- **S4**: Optimal for very long sequences (10k+ timesteps)
- **S5**: Balanced approach for typical sequence modeling tasks

## Development

### Adding New Features
1. Create a new directory for your model: `mkdir my-model`
2. Implement core layers in `layers.py` or `model.py`
3. Add dataset utilities in `dataset.py`
4. Include training logic in `train.py`
5. Document in a `README.md`

### Running Tests
```bash
# Add your test files and run:
python -m pytest tests/
```

## Contributing

Contributions are welcome! Please ensure:
- Code is well-documented
- Tests pass
- Performance benchmarks are included for new models


## Contact

For questions or suggestions, please open an issue on GitHub or contact the repository maintainer.
