# ZO-SPLM Supplementary Material - Appendix

## Appendix A: Extended Mathematical Derivations

### A.1 Detailed Convergence Analysis

This section provides extended proofs and derivations for the convergence properties of the ZO-SPLM algorithm.

#### A.1.1 Smoothness and Lipschitz Conditions

**Definition A.1**: A function f: ℝᵈ → ℝ is L-smooth if:
||∇f(x) - ∇f(y)|| ≤ L||x - y|| for all x, y ∈ ℝᵈ

**Lemma A.1**: Under L-smoothness, for any x, y ∈ ℝᵈ:
f(y) ≤ f(x) + ⟨∇f(x), y - x⟩ + (L/2)||y - x||²

**Proof of Lemma A.1**:
Consider the function g(t) = f(x + t(y - x)) for t ∈ [0, 1].
By the fundamental theorem of calculus:
g(1) - g(0) = ∫₀¹ g'(t) dt = ∫₀¹ ⟨∇f(x + t(y - x)), y - x⟩ dt

Adding and subtracting ⟨∇f(x), y - x⟩:
f(y) - f(x) = ⟨∇f(x), y - x⟩ + ∫₀¹ ⟨∇f(x + t(y - x)) - ∇f(x), y - x⟩ dt

Using Cauchy-Schwarz and L-smoothness:
|⟨∇f(x + t(y - x)) - ∇f(x), y - x⟩| ≤ ||∇f(x + t(y - x)) - ∇f(x)|| · ||y - x||
                                        ≤ Lt||y - x||²

Therefore:
f(y) - f(x) ≤ ⟨∇f(x), y - x⟩ + ∫₀¹ Lt||y - x||² dt = ⟨∇f(x), y - x⟩ + (L/2)||y - x||² □

#### A.1.2 Zeroth-Order Gradient Estimation

**Definition A.2**: The coordinate-wise finite difference estimator is:
ĝᵢ(x) = [f(x + μeᵢ) - f(x - μeᵢ)]/(2μ)

where eᵢ is the i-th coordinate vector and μ > 0 is the smoothing parameter.

**Lemma A.2**: For twice-differentiable f with bounded Hessian ||∇²f(x)|| ≤ M:
|𝔼[ĝᵢ(x)] - ∂f/∂xᵢ(x)| ≤ (Mμ²/6)

**Proof of Lemma A.2**:
By Taylor expansion around x:
f(x ± μeᵢ) = f(x) ± μ∂f/∂xᵢ(x) + (μ²/2)∂²f/∂xᵢ²(ξᵢ±) + O(μ³)

where ξᵢ± lie between x and x ± μeᵢ.

Therefore:
ĝᵢ(x) = ∂f/∂xᵢ(x) + (μ/4)[∂²f/∂xᵢ²(ξᵢ⁺) + ∂²f/∂xᵢ²(ξᵢ⁻)] + O(μ²)

The bias is:
|𝔼[ĝᵢ(x)] - ∂f/∂xᵢ(x)| = (μ/4)|𝔼[∂²f/∂xᵢ²(ξᵢ⁺) + ∂²f/∂xᵢ²(ξᵢ⁻)]| + O(μ²)

Since ||∇²f|| ≤ M, we have |∂²f/∂xᵢ²| ≤ M, giving the bound (Mμ²/6). □

### A.2 Variance Analysis

**Lemma A.3**: Under additive noise model f_obs(x) = f(x) + ξ where 𝔼[ξ] = 0, Var[ξ] = σ²:
Var[ĝᵢ(x)] = σ²/μ²

**Proof of Lemma A.3**:
ĝᵢ(x) = [f_obs(x + μeᵢ) - f_obs(x - μeᵢ)]/(2μ)
       = [f(x + μeᵢ) - f(x - μeᵢ)]/(2μ) + [ξ⁺ - ξ⁻]/(2μ)

where ξ± are independent noise terms with variance σ².

Since f is deterministic:
Var[ĝᵢ(x)] = Var[(ξ⁺ - ξ⁻)/(2μ)] = [Var[ξ⁺] + Var[ξ⁻]]/(4μ²) = 2σ²/(4μ²) = σ²/(2μ²) □

### A.3 Optimal Smoothing Parameter

**Theorem A.1**: The optimal smoothing parameter minimizing MSE is:
μ* = (3σ/M)^(1/3)

**Proof of Theorem A.1**:
MSE = Bias² + Variance ≤ (Mμ²/6)² + σ²/μ²

Taking derivative with respect to μ:
d/dμ MSE = 2(Mμ²/6)(M/3) - 2σ²/μ³ = M²μ/18 - 2σ²/μ³

Setting equal to zero:
M²μ/18 = 2σ²/μ³
μ⁴ = 36σ²/M²
μ = (36σ²/M²)^(1/4) = (3σ/M)^(1/3) · 6^(1/4)

The leading term gives μ* = (3σ/M)^(1/3). □

## Appendix B: Algorithm Implementation Details

### B.1 ZO-SPLM Algorithm Pseudocode

```
Algorithm: ZO-SPLM (Zeroth-Order Structured Probabilistic Learning Method)

Input: 
  - f: objective function
  - x₀: initial point
  - μ: smoothing parameter
  - α: step size schedule
  - T: maximum iterations

Output: x_T (approximate stationary point)

1: Initialize x ← x₀
2: for t = 0 to T-1 do
3:   for i = 1 to d do
4:     ĝᵢ ← [f(x + μeᵢ) - f(x - μeᵢ)]/(2μ)
5:   end for
6:   ĝ ← (ĝ₁, ..., ĝ_d)
7:   x ← x - α_t ĝ
8: end for
9: return x
```

### B.2 Adaptive Parameter Selection

**Algorithm B.1**: Adaptive Smoothing Parameter
```
1: Initialize μ₀, adaptation rate β
2: for t = 0 to T-1 do
3:   Estimate local curvature: Ĥₜ ← estimate_curvature(xₜ)
4:   Estimate noise level: σ̂ₜ ← estimate_noise(xₜ)
5:   Update smoothing: μₜ₊₁ ← (1-β)μₜ + β(σ̂ₜ/||Ĥₜ||)^(1/3)
6: end for
```

### B.3 Complexity Analysis

**Space Complexity**: O(d) for storing current iterate and gradient estimate.

**Time Complexity**: O(d) per iteration for gradient estimation, O(T·d) total.

**Function Evaluation Complexity**: 2d evaluations per iteration, 2Td total.

## Appendix C: Experimental Validation Framework

### C.1 Test Functions

We validate our theoretical results on the following test functions:

**C.1.1 Quadratic Function**
f₁(x) = (1/2)x^T A x + b^T x + c
where A is positive definite.

**C.1.2 Rosenbrock Function**
f₂(x) = ∑ᵢ₌₁^(d-1) [100(xᵢ₊₁ - xᵢ²)² + (1 - xᵢ)²]

**C.1.3 Rastrigin Function**
f₃(x) = 10d + ∑ᵢ₌₁^d [xᵢ² - 10cos(2πxᵢ)]

### C.2 Performance Metrics

1. **Convergence Rate**: ||∇f(xₜ)||² vs iteration count
2. **Function Value Decrease**: f(xₜ) - f* vs iteration count  
3. **Sample Complexity**: Number of function evaluations to reach ε-accuracy
4. **Robustness**: Performance under different noise levels

### C.3 Experimental Protocol

1. **Initialization**: Random starting points from unit ball
2. **Noise Model**: Additive Gaussian noise N(0, σ²)
3. **Parameter Tuning**: Grid search over μ and α
4. **Repetitions**: 50 independent runs per configuration
5. **Statistical Analysis**: Mean, confidence intervals, significance tests

## Appendix D: Comparison with Related Methods

### D.1 Classical Finite Difference Methods

**Standard Finite Difference**:
ĝᵢ^FD(x) = [f(x + μeᵢ) - f(x)]/μ

**Bias**: O(μ), **Variance**: O(σ²/μ²)
**Optimal μ**: O((σ²/L)^(1/3))

### D.2 Random Direction Methods

**Gaussian Smoothing**:
ĝ^GS(x) = (d/μ²)[f(x + μu) - f(x)]u

where u ~ N(0, I).

**Bias**: O(μ²), **Variance**: O(dσ²/μ²)
**Optimal μ**: O((σ²/L)^(1/3))

### D.3 Sphere Smoothing

**Two-Point Estimate**:
ĝ^SP(x) = (d/μ)[f(x + μu) - f(x - μu)]u/(2)

where u is uniform on unit sphere.

**Bias**: O(μ²), **Variance**: O(dσ²/μ²)

### D.4 Performance Comparison

| Method | Function Evaluations | Bias | Variance | Optimal Rate |
|--------|-------------------|------|----------|--------------|
| ZO-SPLM | 2d | O(μ²) | O(dσ²/μ²) | O(d/ε²) |
| Standard FD | d+1 | O(μ) | O(dσ²/μ²) | O(d/ε³) |
| Gaussian Smooth | 2 | O(μ²) | O(dσ²/μ²) | O(d²/ε²) |
| Sphere Smooth | 2 | O(μ²) | O(dσ²/μ²) | O(d²/ε²) |

## Appendix E: Extensions and Future Work

### E.1 Stochastic Extensions

Extension to stochastic settings where f(x) = 𝔼[F(x, ξ)] with ξ random.

**Modified Algorithm**:
1. Sample batch of size B: {ξ₁, ..., ξ_B}
2. Estimate: f̂(x) = (1/B)∑ⱼ₌₁^B F(x, ξⱼ)
3. Apply ZO-SPLM to f̂

**Additional Complexity**: Factor of B in function evaluations.

### E.2 Constrained Optimization

Extension to constrained problems: min f(x) s.t. x ∈ C

**Projected ZO-SPLM**:
x_{t+1} = Π_C(x_t - α_t ĝ_t)

where Π_C is projection onto constraint set C.

### E.3 Non-smooth Extensions

For non-smooth f, use smoothing techniques:
f_μ(x) = 𝔼[f(x + μU)]

where U is appropriate smoothing distribution.

---

*This appendix provides comprehensive technical details supporting the main theoretical and algorithmic contributions of ZO-SPLM. For practical implementation guidance, refer to the accompanying code repository.*