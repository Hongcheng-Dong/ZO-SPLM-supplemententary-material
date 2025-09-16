# ZO-SPLM Supplementary Material

## Zeroth-Order Structured Probabilistic Learning Method - Complete Mathematical Proofs and Implementation

This repository contains comprehensive supplementary material with detailed mathematical proofs for the Zeroth-Order Structured Probabilistic Learning Method (ZO-SPLM), including theoretical foundations, convergence analysis, and practical implementation.

### Contents

📄 **[Mathematical Proofs](proofs.md)** - Core theoretical results including:
- Convergence analysis with detailed proofs
- Sample complexity bounds (O(d/ε²))
- High probability convergence guarantees  
- Structural consistency properties

📚 **[Extended Appendix](appendix.md)** - Comprehensive technical details:
- Extended mathematical derivations
- Algorithm implementation details
- Experimental validation framework
- Performance comparison with related methods

💻 **[Python Implementation](zo_splm_implementation.py)** - Complete working code:
- ZO-SPLM optimizer class with theoretical guarantees
- Test functions (quadratic, Rosenbrock, Rastrigin)
- Convergence experiments and complexity analysis
- Visualization tools for performance evaluation

📖 **[Bibliography](bibliography.md)** - Extensive references covering:
- Theoretical foundations of zeroth-order optimization
- Recent advances in derivative-free methods
- Applications to machine learning and black-box optimization

### Key Theoretical Contributions

1. **Convergence Guarantee**: Proved convergence to stationary points under standard smoothness assumptions
2. **Sample Complexity**: Established O(d/ε²) sample complexity bound for ε-stationary points
3. **High Probability Results**: Derived concentration inequalities for finite-sample guarantees
4. **Structural Properties**: Showed preservation of optimization landscape structure

### Mathematical Framework

The ZO-SPLM algorithm uses coordinate-wise finite differences for gradient estimation:

```
ĝᵢ(x) = [f(x + μeᵢ) - f(x - μeᵢ)]/(2μ)
```

With update rule:
```
x_{t+1} = x_t - α_t ĝ_t
```

### Key Results

- **Convergence Rate**: O(log T / T) for smooth convex functions
- **Function Evaluations**: 2d per iteration (optimal for coordinate methods)
- **Noise Robustness**: Handles observation noise with σ² variance
- **Dimension Dependence**: Scales as O(d) rather than O(d²) of sphere methods

### Usage

```python
from zo_splm_implementation import ZOSPLMOptimizer, ZOSPLMConfig

# Configure optimizer
config = ZOSPLMConfig(mu=1e-2, alpha_init=1e-1, max_iterations=1000)
optimizer = ZOSPLMOptimizer(config)

# Optimize function
x_final, info = optimizer.optimize(objective_function, initial_point)
```

### Experimental Validation

Run the implementation to reproduce theoretical results:

```bash
python zo_splm_implementation.py
```

This generates convergence plots and complexity analysis confirming the theoretical predictions.

### Citation

If you use this supplementary material, please cite the main paper and reference this repository for the detailed proofs and implementation.

---

*This supplementary material provides rigorous mathematical foundations for ZO-SPLM with complete proofs, practical implementation, and experimental validation of all theoretical claims.*
