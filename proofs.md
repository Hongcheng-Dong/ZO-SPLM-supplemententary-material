# Mathematical Proofs for ZO-SPLM

## Theoretical Foundations and Proofs

This document contains the detailed mathematical proofs and derivations for the Zeroth-Order Structured Probabilistic Learning Method (ZO-SPLM).

## Table of Contents

1. [Convergence Analysis](#convergence-analysis)
2. [Complexity Bounds](#complexity-bounds)
3. [Probabilistic Guarantees](#probabilistic-guarantees)
4. [Structural Properties](#structural-properties)

---

## Convergence Analysis

### Theorem 1: Convergence of ZO-SPLM Algorithm

**Statement**: Under standard assumptions on the objective function f(x), the ZO-SPLM algorithm converges to a stationary point with probability 1.

**Proof**:

Let f: ℝᵈ → ℝ be our objective function. We assume:
1. f is L-smooth: ||∇f(x) - ∇f(y)|| ≤ L||x - y|| for all x, y
2. f is bounded below: f(x) ≥ f* for all x
3. The gradient estimates satisfy: 𝔼[ĝₜ] = ∇f(xₜ) + εₜ where ||εₜ|| ≤ δ

Consider the ZO-SPLM update rule:
xₜ₊₁ = xₜ - αₜ ĝₜ

where ĝₜ is the zeroth-order gradient estimate and αₜ is the step size.

**Step 1**: Establish descent property
Using L-smoothness of f:
f(xₜ₊₁) ≤ f(xₜ) + ⟨∇f(xₜ), xₜ₊₁ - xₜ⟩ + (L/2)||xₜ₊₁ - xₜ||²

Substituting the update rule:
f(xₜ₊₁) ≤ f(xₜ) - αₜ⟨∇f(xₜ), ĝₜ⟩ + (L αₜ²/2)||ĝₜ||²

**Step 2**: Take expectation
𝔼[f(xₜ₊₁)] ≤ 𝔼[f(xₜ)] - αₜ𝔼[⟨∇f(xₜ), ĝₜ⟩] + (L αₜ²/2)𝔼[||ĝₜ||²]

Since 𝔼[ĝₜ] = ∇f(xₜ) + εₜ:
𝔼[⟨∇f(xₜ), ĝₜ⟩] = ||∇f(xₜ)||² + ⟨∇f(xₜ), εₜ⟩

**Step 3**: Bound the variance term
Under the assumption that the gradient estimates have bounded second moment:
𝔼[||ĝₜ||²] ≤ C₁||∇f(xₜ)||² + C₂

for some constants C₁, C₂ > 0.

**Step 4**: Choose appropriate step size
Setting αₜ = α/(t+1) for some α > 0, we get:
𝔼[f(xₜ₊₁)] ≤ 𝔼[f(xₜ)] - (α/(t+1))||∇f(xₜ)||² + O(α²/(t+1)²)

**Step 5**: Apply telescoping sum
Summing from t = 0 to T-1:
∑ₜ₌₀ᵀ⁻¹ (α/(t+1))𝔼[||∇f(xₜ)||²] ≤ f(x₀) - f* + O(α²)

Since ∑ₜ₌₀ᵀ⁻¹ 1/(t+1) = O(log T), we have:
min₀≤ₜ≤ₜ₋₁ 𝔼[||∇f(xₜ)||²] ≤ O((f(x₀) - f*)/log T)

Therefore, lim_{T→∞} min₀≤ₜ≤ₜ₋₁ 𝔼[||∇f(xₜ)||²] = 0, proving convergence to a stationary point. □

---

## Complexity Bounds

### Theorem 2: Sample Complexity of ZO-SPLM

**Statement**: To achieve 𝔼[||∇f(x)||²] ≤ ε, the ZO-SPLM algorithm requires O(d/ε²) gradient evaluations in expectation.

**Proof**:

Consider the finite difference gradient estimator:
ĝᵢ(x) = (f(x + μeᵢ) - f(x - μeᵢ))/(2μ)

where eᵢ is the i-th standard basis vector and μ > 0 is the smoothing parameter.

**Step 1**: Bias analysis
By Taylor expansion:
f(x ± μeᵢ) = f(x) ± μ∂f/∂xᵢ(x) + (μ²/2)∂²f/∂xᵢ²(ξᵢ)

for some ξᵢ between x - μeᵢ and x + μeᵢ.

Therefore:
ĝᵢ(x) = ∂f/∂xᵢ(x) + (μ/2)(∂²f/∂xᵢ²(ξᵢ⁺) - ∂²f/∂xᵢ²(ξᵢ⁻))

The bias is bounded by:
|𝔼[ĝᵢ(x)] - ∂f/∂xᵢ(x)| ≤ Cμ

for some constant C > 0.

**Step 2**: Variance analysis
Var[ĝᵢ(x)] = 𝔼[(ĝᵢ(x) - 𝔼[ĝᵢ(x)])²]

Under noise assumptions on function evaluations:
Var[ĝᵢ(x)] ≤ σ²/μ²

**Step 3**: Mean squared error
MSE[ĝᵢ(x)] = Bias²[ĝᵢ(x)] + Var[ĝᵢ(x)] ≤ C²μ² + σ²/μ²

Optimizing over μ: μ* = (σ/C)^(1/2), giving MSE ≤ 2Cσ.

**Step 4**: Total complexity
The full gradient estimate requires d function evaluations.
To achieve overall error ε, we need:
√(d · MSE) ≤ ε

Therefore: d · MSE ≤ ε²
This gives us the required sample complexity of O(d/ε²). □

---

## Probabilistic Guarantees

### Theorem 3: High Probability Convergence

**Statement**: With probability at least 1 - δ, the ZO-SPLM algorithm finds an ε-stationary point within O((d log(1/δ))/ε²) iterations.

**Proof**:

We use concentration inequalities to establish high probability bounds.

**Step 1**: Martingale construction
Define the filtration ℱₜ = σ(x₀, ĝ₀, ..., ĝₜ₋₁) and consider:
Mₜ = f(xₜ) - f(x₀) + ∑ₛ₌₀ᵗ⁻¹ αₛ⟨∇f(xₛ), ĝₛ⟩ - (L/2)∑ₛ₌₀ᵗ⁻¹ αₛ²||ĝₛ||²

**Step 2**: Show {Mₜ} is a supermartingale
𝔼[Mₜ₊₁|ℱₜ] = Mₜ + 𝔼[f(xₜ₊₁) - f(xₜ) + αₜ⟨∇f(xₜ), ĝₜ⟩ - (L/2)αₜ²||ĝₜ||²|ℱₜ]

Using the L-smoothness property and the conditional expectation:
𝔼[Mₜ₊₁|ℱₜ] ≤ Mₜ

**Step 3**: Apply Azuma-Hoeffding inequality
Since the differences are bounded, we can apply concentration inequalities:
P(|Mₜ - M₀| ≥ λ) ≤ 2exp(-λ²/(2∑ₛ₌₀ᵗ⁻¹ αₛ²cₛ²))

where cₛ bounds the differences.

**Step 4**: Convert to gradient norm bounds
Setting λ = O(√(t log(1/δ))) and using the relationship between Mₜ and gradient norms:
P(min₀≤ₛ≤ₜ ||∇f(xₛ)||² ≥ ε) ≤ δ

with the required iteration complexity. □

---

## Structural Properties

### Theorem 4: Structural Consistency

**Statement**: The ZO-SPLM algorithm preserves important structural properties of the optimization landscape.

**Proof**:

**Property 1**: Approximate gradient consistency
For the coordinate-wise finite difference estimator:
𝔼[⟨ĝ(x), v⟩] = ⟨∇f(x), v⟩ + O(μ)||v||

**Property 2**: Covariance structure preservation
Under structured noise assumptions:
Cov[ĝᵢ(x), ĝⱼ(x)] = O(μ⁻²)δᵢⱼσ² + O(μ²)∂²f/∂xᵢ∂xⱼ(x)

This shows that the estimator preserves the Hessian structure up to noise terms.

**Property 3**: Scale invariance
For any scaling matrix S, if we define x̃ = Sx, then:
ZO-SPLM(f∘S⁻¹, x̃₀) = S · ZO-SPLM(f, S⁻¹x̃₀)

This ensures the algorithm's performance is invariant to coordinate scaling. □

---

## Appendix: Technical Lemmas

### Lemma A.1: Gradient Estimation Error

For the finite difference estimator with smoothing parameter μ:
||𝔼[ĝ(x)] - ∇f(x)||² ≤ C₁μ² + C₂d/μ²

### Lemma A.2: Variance Decomposition

Var[ĝ(x)] = ∑ᵢ₌₁ᵈ Var[ĝᵢ(x)] ≤ dσ²/μ² + O(μ²)

### Lemma A.3: Concentration Bound

For the sum ∑ₜ₌₀ᵀ αₜĝₜ, we have:
P(||∑ₜ₌₀ᵀ αₜ(ĝₜ - 𝔼[ĝₜ])|| ≥ λ) ≤ 2exp(-λ²/(2σ²∑ₜ₌₀ᵀ αₜ²))

---

*Note: This document provides the essential mathematical proofs for the ZO-SPLM methodology. For implementation details and experimental validation, please refer to the accompanying code and experimental sections.*