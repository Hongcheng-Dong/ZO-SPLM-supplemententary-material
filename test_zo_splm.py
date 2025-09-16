#!/usr/bin/env python3
"""
Simple test of ZO-SPLM implementation to verify it works.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from zo_splm_implementation import ZOSPLMOptimizer, ZOSPLMConfig, TestFunctions

def simple_test():
    """Run a simple test to verify the implementation works."""
    print("Testing ZO-SPLM implementation...")
    
    # Simple quadratic function: f(x) = ||x||^2
    def simple_quadratic(x):
        return np.sum(x**2)
    
    # Configuration
    config = ZOSPLMConfig(
        mu=1e-2,
        alpha_init=1e-1,
        max_iterations=100,
        tolerance=1e-4,
        noise_variance=0.0
    )
    
    # Initial point
    x0 = np.array([1.0, 2.0])
    
    # Run optimization
    optimizer = ZOSPLMOptimizer(config)
    x_final, info = optimizer.optimize(simple_quadratic, x0)
    
    print(f"Initial point: {x0}")
    print(f"Final point: {x_final}")
    print(f"Final function value: {simple_quadratic(x_final):.6f}")
    print(f"Converged: {info['converged']}")
    print(f"Iterations: {info['total_iterations']}")
    print(f"Function evaluations: {info['total_function_evaluations']}")
    
    # Check if we're close to the optimum (should be close to [0, 0])
    distance_to_optimum = np.linalg.norm(x_final)
    print(f"Distance to optimum: {distance_to_optimum:.6f}")
    
    if distance_to_optimum < 0.1:
        print("✅ Test PASSED: Found solution close to optimum")
        return True
    else:
        print("❌ Test FAILED: Solution not close enough to optimum")
        return False

if __name__ == "__main__":
    success = simple_test()
    sys.exit(0 if success else 1)