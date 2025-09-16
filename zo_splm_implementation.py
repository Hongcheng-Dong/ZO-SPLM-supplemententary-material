"""
ZO-SPLM Implementation: Zeroth-Order Structured Probabilistic Learning Method

This module provides a Python implementation of the ZO-SPLM algorithm
with theoretical guarantees as described in the supplementary material.

Author: Supplementary Material for ZO-SPLM
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class ZOSPLMConfig:
    """Configuration for ZO-SPLM algorithm."""
    mu: float = 1e-3  # Smoothing parameter
    alpha_init: float = 1e-2  # Initial step size
    max_iterations: int = 1000
    tolerance: float = 1e-6
    adaptive_mu: bool = True
    noise_variance: float = 0.0
    

class ZOSPLMOptimizer:
    """
    Zeroth-Order Structured Probabilistic Learning Method Optimizer.
    
    Implements the ZO-SPLM algorithm with theoretical convergence guarantees.
    """
    
    def __init__(self, config: ZOSPLMConfig):
        self.config = config
        self.iteration_count = 0
        self.function_evaluations = 0
        self.convergence_history = []
        
    def finite_difference_gradient(self, f: Callable, x: np.ndarray) -> np.ndarray:
        """
        Compute finite difference gradient estimate.
        
        Args:
            f: Objective function
            x: Current point
            
        Returns:
            Gradient estimate using coordinate-wise finite differences
        """
        d = len(x)
        grad_estimate = np.zeros(d)
        
        for i in range(d):
            e_i = np.zeros(d)
            e_i[i] = 1.0
            
            # Add noise if specified
            noise_plus = np.random.normal(0, self.config.noise_variance) if self.config.noise_variance > 0 else 0
            noise_minus = np.random.normal(0, self.config.noise_variance) if self.config.noise_variance > 0 else 0
            
            f_plus = f(x + self.config.mu * e_i) + noise_plus
            f_minus = f(x - self.config.mu * e_i) + noise_minus
            
            grad_estimate[i] = (f_plus - f_minus) / (2 * self.config.mu)
            self.function_evaluations += 2
            
        return grad_estimate
    
    def adaptive_step_size(self, iteration: int) -> float:
        """
        Compute adaptive step size based on theoretical analysis.
        
        Args:
            iteration: Current iteration number
            
        Returns:
            Step size for current iteration
        """
        return self.config.alpha_init / np.sqrt(iteration + 1)
    
    def optimize(self, f: Callable, x0: np.ndarray, 
                 true_gradient: Optional[Callable] = None) -> Tuple[np.ndarray, dict]:
        """
        Run ZO-SPLM optimization algorithm.
        
        Args:
            f: Objective function to minimize
            x0: Initial point
            true_gradient: Optional true gradient for comparison
            
        Returns:
            Tuple of (final_point, optimization_info)
        """
        x = x0.copy()
        self.iteration_count = 0
        self.function_evaluations = 0
        self.convergence_history = []
        
        for t in range(self.config.max_iterations):
            # Compute gradient estimate
            grad_estimate = self.finite_difference_gradient(f, x)
            
            # Adaptive step size
            alpha_t = self.adaptive_step_size(t)
            
            # Update step
            x_new = x - alpha_t * grad_estimate
            
            # Track convergence
            grad_norm = np.linalg.norm(grad_estimate)
            self.convergence_history.append({
                'iteration': t,
                'function_value': f(x),
                'gradient_norm': grad_norm,
                'step_size': alpha_t,
                'function_evaluations': self.function_evaluations
            })
            
            # Check convergence
            if grad_norm < self.config.tolerance:
                print(f"Converged at iteration {t} with gradient norm {grad_norm:.2e}")
                break
                
            x = x_new
            self.iteration_count = t + 1
            
        optimization_info = {
            'converged': grad_norm < self.config.tolerance,
            'final_gradient_norm': grad_norm,
            'total_iterations': self.iteration_count,
            'total_function_evaluations': self.function_evaluations,
            'convergence_history': self.convergence_history
        }
        
        return x, optimization_info


class TestFunctions:
    """Collection of test functions for validating ZO-SPLM."""
    
    @staticmethod
    def quadratic(A: np.ndarray, b: np.ndarray, c: float = 0.0):
        """
        Quadratic function: f(x) = 0.5 * x^T A x + b^T x + c
        
        Args:
            A: Positive definite matrix
            b: Linear coefficient vector
            c: Constant term
        """
        def f(x):
            return 0.5 * np.dot(x, A @ x) + np.dot(b, x) + c
        
        def grad(x):
            return A @ x + b
            
        return f, grad
    
    @staticmethod
    def rosenbrock(a: float = 1.0, b: float = 100.0):
        """
        Rosenbrock function: f(x) = (a - x[0])^2 + b(x[1] - x[0]^2)^2
        
        Args:
            a, b: Rosenbrock parameters
        """
        def f(x):
            return (a - x[0])**2 + b * (x[1] - x[0]**2)**2
        
        def grad(x):
            return np.array([
                -2*(a - x[0]) - 4*b*x[0]*(x[1] - x[0]**2),
                2*b*(x[1] - x[0]**2)
            ])
            
        return f, grad
    
    @staticmethod
    def rastrigin(A: float = 10.0):
        """
        Rastrigin function: f(x) = A*n + sum(x_i^2 - A*cos(2*pi*x_i))
        
        Args:
            A: Rastrigin parameter
        """
        def f(x):
            n = len(x)
            return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))
        
        def grad(x):
            return 2 * x + 2 * A * np.pi * np.sin(2 * np.pi * x)
            
        return f, grad


def run_convergence_experiment():
    """Run convergence experiment to validate theoretical results."""
    
    # Test on quadratic function
    d = 10
    A = np.eye(d) + 0.1 * np.random.randn(d, d)
    A = A.T @ A  # Ensure positive definite
    b = np.random.randn(d)
    
    f, grad = TestFunctions.quadratic(A, b)
    
    # Configuration
    config = ZOSPLMConfig(
        mu=1e-2,
        alpha_init=1e-1,
        max_iterations=500,
        tolerance=1e-8,
        noise_variance=0.01
    )
    
    # Initial point
    x0 = np.random.randn(d)
    
    # Run optimization
    optimizer = ZOSPLMOptimizer(config)
    x_final, info = optimizer.optimize(f, x0, grad)
    
    print(f"Optimization Results:")
    print(f"Converged: {info['converged']}")
    print(f"Final gradient norm: {info['final_gradient_norm']:.2e}")
    print(f"Total iterations: {info['total_iterations']}")
    print(f"Total function evaluations: {info['total_function_evaluations']}")
    
    # Plot convergence
    history = info['convergence_history']
    iterations = [h['iteration'] for h in history]
    grad_norms = [h['gradient_norm'] for h in history]
    func_vals = [h['function_value'] for h in history]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.semilogy(iterations, grad_norms, 'b-', label='Gradient Norm')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Gradient Norm')
    ax1.set_title('Convergence: Gradient Norm')
    ax1.grid(True)
    ax1.legend()
    
    ax2.plot(iterations, func_vals, 'r-', label='Function Value')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Function Value')
    ax2.set_title('Convergence: Function Value')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('convergence_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return x_final, info


def complexity_analysis():
    """Analyze sample complexity vs. dimension."""
    
    dimensions = [5, 10, 20, 50]
    target_accuracy = 1e-4
    results = []
    
    for d in dimensions:
        print(f"Testing dimension d = {d}")
        
        # Create test function
        A = np.eye(d)
        b = np.zeros(d)
        f, grad = TestFunctions.quadratic(A, b)
        
        config = ZOSPLMConfig(
            mu=1e-2,
            alpha_init=1e-1,
            max_iterations=1000,
            tolerance=target_accuracy,
            noise_variance=0.001
        )
        
        x0 = np.random.randn(d)
        optimizer = ZOSPLMOptimizer(config)
        x_final, info = optimizer.optimize(f, x0)
        
        results.append({
            'dimension': d,
            'function_evaluations': info['total_function_evaluations'],
            'iterations': info['total_iterations'],
            'converged': info['converged']
        })
    
    # Plot complexity results
    dims = [r['dimension'] for r in results]
    evals = [r['function_evaluations'] for r in results]
    
    plt.figure(figsize=(8, 6))
    plt.loglog(dims, evals, 'bo-', label='ZO-SPLM')
    plt.loglog(dims, [d/target_accuracy**2 for d in dims], 'r--', label='Theoretical O(d/ε²)')
    plt.xlabel('Dimension')
    plt.ylabel('Function Evaluations')
    plt.title('Sample Complexity Analysis')
    plt.legend()
    plt.grid(True)
    plt.savefig('complexity_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return results


if __name__ == "__main__":
    print("ZO-SPLM: Zeroth-Order Structured Probabilistic Learning Method")
    print("=" * 60)
    
    print("\n1. Running convergence experiment...")
    x_final, info = run_convergence_experiment()
    
    print("\n2. Running complexity analysis...")
    complexity_results = complexity_analysis()
    
    print("\nExperimental validation completed!")
    print("Plots saved as 'convergence_analysis.png' and 'complexity_analysis.png'")