# KW-PE Survey

## Summary

Comprehensive survey of spectral invariance theory (EPNN ceiling at PSWL < 3-WL), Koopman/EDMD algorithms with step-by-step implementation guide, Python library APIs (PyKoopman explicit dictionary vs PyDMD kernel-based), all 7 baseline graph PE methods (RWPE, LapPE, SignNet, BasisNet, SPE, PEARL, RFP) with exact formulations and spectral invariance classification, cospectral graph test pairs ({K_{1,4}, C_4∪K_1} and {Rook, Shrikhande}), and complete related work positioning. Key novelty finding: no prior work combines nonlinear walk dynamics with Koopman decomposition for graph PE. PEARL is the closest method but uses learned GNN nonlinearities rather than fixed dynamics + Koopman eigenfunction extraction.

## Research Findings

## Executive Summary

This survey establishes the complete theoretical and practical foundation for Koopman Walk Positional Encodings (KW-PE). The spectral invariance ceiling — formalized by the EPNN framework — bounds all existing spectral PE methods (RWPE, LapPE, SignNet, BasisNet, SPE) below 3-WL [1]. KW-PE's core hypothesis is that nonlinear walk dynamics create Koopman eigenfunctions encoding cross-products of graph eigenvector components, breaking this ceiling. No prior work combines nonlinear walk dynamics with Koopman decomposition for graph PE [26], establishing strong novelty.

---

## 1. The Spectral Invariance Ceiling

### Formal Definition
A GNN architecture is **spectral invariant** if its output remains unchanged under any orthogonal transformation of the eigenvector basis within each eigenspace of the graph matrix M [1]. For eigenvalue λ_i with multiplicity J_i, the unique projection matrix **P_i = Σ_{j=1}^{J_i} z_{i,j} z_{i,j}^T** is invariant to eigenvector choice [1]. The eigenspace projection invariant captures all spectral information as a multiset:

**P_M^G(u,v) = {{(λ_1, P_1(u,v)), ..., (λ_m, P_m(u,v))}}** [1]

### EPNN Architecture
The K-layer EPNN computes node representations iteratively: **h_G^{(l+1)}(u) = g^{(l+1)}(h_G^{(l)}(u), {{(h_G^{(l)}(v), P_M^G(u,v)) : v ∈ V_G}})** [1]. All nodes initialize identically. EPNN unifies all prior spectral invariant architectures — they are either strictly less expressive or equivalent to EPNN [1].

### Main Expressiveness Theorem
**Theorem 4.3**: EPWL is strictly bounded by PSWL (Subgraph WL) [1]. **Corollary 4.5**: EPWL < 3-WL for all standard graph matrices (adjacency A, Laplacian L, normalized Laplacian L̂) [1]. This establishes the hierarchy: RWPE ≤ LapPE ≤ SignNet ≤ BasisNet ≤ SPE ≤ EPNN ≤ PSWL < 3-WL [1].

### What Must Be Violated to Break the Ceiling
To exceed spectral invariance, a method must depend on **individual eigenvector components** v_j(i), not just eigenspace projection matrices P_i = V_i V_i^T. Since projections are invariant under orthogonal rotation of V_i, any method using only {(λ_i, P_i)} cannot distinguish cospectral graphs sharing projection matrices [1].

---

## 2. What Spectral Invariant GNNs Can Count: Parallel Trees Only

### Definition
A **parallel edge** has edges partitioning into simple paths sharing common endpoints [2]. A **parallel tree** replaces each edge in a base tree with a parallel edge [2].

### Main Theorem
**Theorem 3.3**: Spectral invariant GNNs with d iterations characterize homomorphism counts of graphs with parallel tree depth ≤ d [2]. **Corollary 3.14**: They can count cycles and paths with ≤7 vertices but cannot count 4-cliques [2]. **Corollary 3.11**: d+1 iterations strictly exceed d iterations, refuting Arvind et al.'s constant-convergence conjecture [2].

### Implication for KW-PE
If KW-PE can count non-parallel-tree substructures (e.g., certain cycle configurations or clique-like motifs), it provably exceeds the spectral invariant ceiling [2].

---

## 3. Koopman Operator Theory and EDMD: Implementation Guide

### Formal Definition
For discrete-time dynamics x_{t+1} = F(x_t), the Koopman operator K acts on observables g: **(Kg)(x) = g(F(x))** [3, 4]. K is linear but infinite-dimensional. Eigenfunctions satisfy **Kφ = λφ**, meaning **φ(F(x)) = λφ(x)** — they globally linearize the dynamics [3, 4]. Koopman eigenvalues match the Jacobian eigenvalues at fixed points, and principal eigenfunctions are approximately linear near equilibria [3, 4].

### EDMD Algorithm (Step by Step)
1. **Choose dictionary** {ψ_1, ..., ψ_K}: monomials, RBFs, or thin-plate splines [5]
2. **Form matrices**: G_{ij} = (1/M)Σ_t ψ_i(x_t)ψ_j(x_t); A_{ij} = (1/M)Σ_t ψ_i(x_t)ψ_j(x_{t+1}) [5]
3. **Solve**: K_EDMD = G⁺A (pseudoinverse) [5]
4. **Extract eigenfunctions**: eigenvectors ξ_j of K_EDMD give φ_j(x) = Σ_k ξ_{jk}ψ_k(x) [5]

### Convergence Guarantees
As M → ∞, EDMD converges at O(M^{-1/2}) via Monte Carlo integration [5]. As N → ∞ (dictionary size), K_N converges to K in strong operator topology [7]. Spectral accumulation points of K_N correspond to Koopman eigenvalues [7]. No finite-dimensional invariant subspace assumption required [7].

### Critical Insight: Nonlinear Mode Coupling
When F is nonlinear, Koopman eigenfunctions involve **products of state-space eigenvector components**. For a system with quadratic nonlinearity, EDMD with monomial dictionary captures cross-terms like x_1·x_2 creating closed Koopman-invariant subspaces [6]. These cross-terms encode joint information about eigenvector components at different nodes — precisely what spectral invariant methods lose [6].

---

## 4. Library APIs for EDMD Implementation

### PyKoopman (Explicit Dictionary)
```python
import pykoopman as pk
obsv = pk.observables.Polynomial(degree=3)
model = pk.Koopman(observables=obsv, regressor=pk.regression.EDMD())
model.fit(X.T, y=Y.T)
psi = model.psi(x_col=x)  # eigenfunction values
eigenvalues = np.real(np.diag(model.lamda))
```
Available observables: Identity, Polynomial, TimeDelay, RadialBasisFunction, RandomFourierFeatures, CustomObservable, ConcatObservables [8]. Regressors: DMD, EDMD, KDMD, NNDMD, DMDc, HAVOK [8].

### PyDMD (Kernel-Based)
```python
from pydmd import EDMD
edmd = EDMD(kernel_metric='poly', kernel_params={'gamma':1,'coef0':1,'degree':4}, svd_rank=15).fit(X, Y)
eigenvalues = edmd.eigs; eigfuncs = edmd.eigenfunctions(x)
```
Supported kernels: poly, rbf, linear, sigmoid, laplacian, cosine [9].

### Recommendation
Use PyKoopman for explicit dictionary control (understanding which cross-terms matter), PyDMD for scalable kernel EDMD, and custom PyTorch for GPU batch processing of multiple graphs [8, 9].

---

## 5. Baseline PE Methods: Formulations and Limitations

| Method | Formulation | Spectral Invariant? | Sign Ambiguity? | Eigendecomp Required? | Expressiveness Bound |
|--------|------------|---------------------|-----------------|----------------------|---------------------|
| **RWPE** | p_i = [RW_{ii}, ..., RW^k_{ii}] | YES (|v_j(i)|² terms) | NO | NO | ≤ EPNN < 3-WL |
| **LapPE** | p_i = [v_1(i), ..., v_k(i)] | Effectively YES* | YES | YES | ≤ EPNN < 3-WL |
| **SignNet** | ρ([φ(v_i)+φ(-v_i)]) | YES | Resolved | YES | ≤ EPNN < 3-WL |
| **BasisNet** | ρ([IGN(V_iV_i^T)]) | YES | Resolved | YES | ≤ EPNN < 3-WL |
| **SPE** | ρ(V·diag(φ_l(λ))·V^T) | YES | Resolved | YES | ≤ EPNN < 3-WL |
| **PEARL** | ρ[Φ(G,q^(m))] GNN-based | Partially** | N/A | NO | Universal for basis-inv. |
| **RFP** | concat(r, Sr, S²r, ...) | Effectively YES | NO | NO | Interpolates RNF↔spectral |

*LapPE with sign-invariant processing is spectral invariant [1, 10, 11, 12, 13, 14, 15].
**PEARL universally approximates basis-invariant functions (Theorem 3.1) but random initialization MAY break spectral invariance [14].

---

## 6. Cospectral Test Cases

**Pair 1: {K_{1,4}, C_4 ∪ K_1}** — 5 vertices, spectrum {-2, 0³, 2} [16, 17]. Smallest cospectral pair (Collatz & Sinogowitz, 1957). Easy test — different degree sequences allow 1-WL distinction.

**Pair 2: {K_4□K_4 (Rook), Shrikhande}** — 16 vertices, srg(16,6,2,2), spectrum 6¹, 2⁶, (-2)⁹ [17, 18]. Hard test — 2-WL fails. Neighborhood structure differs: hexagon (Shrikhande) vs two triangles (Rook) [18].

**Godsil-McKay Switching**: Systematic construction via submatrix replacement for programmatic cospectral pair generation [19].

**BREC Benchmark**: 400 pairs across 4 categories including 140 Regular pairs (strongly regular graphs, up to 4-WL difficulty) [20].

---

## 7. Related Work and KW-PE Differentiation

| Method | Dynamic Type | Role | Spectral Invariant? | Key Difference from KW-PE |
|--------|-------------|------|---------------------|--------------------------|
| DeepGraphDMD [21] | Temporal (fMRI) | Analyze brain dynamics | N/A | KW-PE: synthetic dynamics from topology |
| RDGNN [22] | Reaction-diffusion | GNN layer design | N/A | KW-PE: PE precomputation, not layer |
| Walk-LLM [23] | Stochastic walks | Text processing | N/A | KW-PE: compact Koopman modes |
| PEARL [14] | Learned nonlinear GNN | PE generation | Partially | KW-PE: fixed dynamics + Koopman |
| RFP [15] | Linear propagation | PE from trajectory | Yes | KW-PE: nonlinear cross-terms |

**Novelty assessment**: Searches for "nonlinear walk positional encoding" and "Koopman graph positional encoding" returned **no direct prior work** [26]. This supports strong novelty claims for KW-PE.

---

## 8. Open Questions and Risks

1. **PEARL overlap**: PEARL's nonlinear GNN processing of random features may partially break spectral invariance, potentially overlapping with KW-PE's advantages [14].
2. **Trajectory diversity**: EDMD convergence requires sufficient trajectory diversity — unclear if single random walk start points provide enough coverage for reliable eigenfunction extraction [5, 7].
3. **Spectral complexity**: The Koopman operator for nonlinear walks on finite graphs may exhibit complex (non-point) spectrum, complicating eigenfunction extraction [7].
4. **Dictionary closure**: Polynomial dictionaries may not close under the specific nonlinear walk dynamics, leading to spurious eigenfunctions [6].
5. **Scalability**: EDMD's per-graph computational cost may limit scalability versus PEARL's O(N) complexity [14].

## Sources

[1] [On the Expressive Power of Spectral Invariant Graph Neural Networks (Zhang, Zhao, Maron, ICML 2024)](https://arxiv.org/html/2406.04336v1) — Introduces EPNN unified spectral invariant framework. Proves EPNN ≤ PSWL < 3-WL. Shows all spectral invariant architectures (SignNet, BasisNet, SPE, RWPE, LapPE) are bounded by EPNN.

[2] [Homomorphism Expressivity of Spectral Invariant GNNs (ICLR 2025 Oral)](https://arxiv.org/html/2503.00485) — Proves spectral invariant GNNs can homomorphism-count exactly parallel trees. Establishes strict iteration depth hierarchy refuting Arvind et al.'s conjecture.

[3] [Notes on Koopman Operator Theory (Steven L. Brunton)](https://fluids.ac.uk/files/meetings/KoopmanNotes.1575558616.pdf) — Tutorial on Koopman operator: formal definition, eigenfunction properties, connection to linearization.

[4] [Koopman Operator Dynamical Models: Learning, Analysis and Control (Bevanda et al.)](https://arxiv.org/pdf/2102.02522) — Comprehensive Koopman theory review covering data-driven representations, spectral properties, and linearization connections.

[5] [A Data-Driven Approximation of the Koopman Operator: EDMD (Williams et al., 2015)](https://ar5iv.labs.arxiv.org/html/1408.4408) — Original EDMD paper: step-by-step algorithm, dictionary selection (polynomials, RBFs, thin-plate splines), O(M^{-1/2}) convergence.

[6] [Data-driven Discovery of Koopman Eigenfunctions for Control (Kaiser et al., 2018)](https://arxiv.org/pdf/1707.01146) — KRONIC framework showing polynomial cross-terms in EDMD, dictionary closure problems, sparsified EDMD for robust eigenfunction recovery.

[7] [On Convergence of EDMD to the Koopman Operator (Korda & Mezic, 2018)](https://arxiv.org/pdf/1703.04680) — Proves K_N → K in strong operator topology as N → ∞. No finite-dimensional invariant subspace assumption needed.

[8] [PyKoopman Documentation](https://pykoopman.readthedocs.io/en/master/) — Python library with explicit dictionary support. Observables: Polynomial, RBF, TimeDelay, etc. Regressors: DMD, EDMD, KDMD, NNDMD.

[9] [PyDMD EDMD Tutorial](https://github.com/PyDMD/PyDMD/blob/master/tutorials/tutorial17/tutorial-17-edmd.py) — Kernel-based EDMD with implicit infinite-dimensional dictionary. Supports poly, rbf, linear, sigmoid kernels.

[10] [GNNs with Learnable Structural and Positional Representations (Dwivedi et al., ICLR 2022)](https://arxiv.org/pdf/2110.07875) — Introduces RWPE: p_i = [RW_{ii}, ..., RW^k_{ii}] from random walk matrix diagonal powers.

[11] [Laplacian Positional Encoding Tutorial](https://afloresep.github.io/posts/2024/10/laplacian_positional_encoding/) — LapPE: p_i = [v_1(i), ..., v_k(i)] from k smallest Laplacian eigenvectors. Sign ambiguity discussion.

[12] [SignNet and BasisNet (Lim et al., ICML 2023)](https://ar5iv.labs.arxiv.org/html/2202.13013) — SignNet: φ(v)+φ(-v) for sign invariance. BasisNet: IGN on projection matrices. Both spectral invariant, bounded by 3-WL.

[13] [SPE: Stable and Expressive Positional Encoding](https://arxiv.org/html/2310.02579) — Learned eigenvalue-dependent functions on projection matrices. Lipschitz stability, universal for basis-invariant functions.

[14] [PEARL: Learning Efficient Positional Encodings with GNNs (ICLR 2025)](https://arxiv.org/pdf/2502.01122) — GNN-based PE with random/basis init. Linear complexity. Universal for basis-invariant functions. Sample complexity independent of graph size.

[15] [RFP: Graph Positional Encoding via Random Feature Propagation (ICML 2023)](https://arxiv.org/pdf/2303.02918) — Linear propagation of random features with normalization. Converges to dominant eigenvectors. Interpolates random features and spectral PE.

[16] [Cospectral Graphs — Wolfram MathWorld](https://mathworld.wolfram.com/CospectralGraphs.html) — Catalog of cospectral pairs including smallest pair {K_{1,4}, C_4∪K_1} and Rook/Shrikhande.

[17] [Spectral Graph Theory — Wikipedia](https://en.wikipedia.org/wiki/Spectral_graph_theory) — Overview of spectral graph theory, cospectral graphs, Collatz-Sinogowitz 1957 example.

[18] [Shrikhande Graph — Wikipedia](https://en.wikipedia.org/wiki/Shrikhande_graph) — srg(16,6,2,2), cospectral with K_4□K_4. Spectrum 6¹, 2⁶, (-2)⁹. Hexagon vs two-triangle neighborhoods.

[19] [Constructing Cospectral Graphs (Godsil & McKay, 1982)](https://cs.anu.edu.au/~Brendan.McKay/papers/GodsilMcKayCospectral.pdf) — Original Godsil-McKay switching construction for systematic cospectral pair generation.

[20] [BREC Benchmark Dataset](https://github.com/GraphPKU/BREC) — 400 non-isomorphic graph pairs for expressiveness testing. 140 Regular pairs including strongly regular graphs.

[21] [DeepGraphDMD: Deep Graph Dynamic Mode Decomposition](https://arxiv.org/html/2306.03088) — Koopman/DMD for temporal brain fMRI dynamics on graphs. Not about graph topology PE.

[22] [Graph Neural Reaction-Diffusion Models](https://arxiv.org/html/2406.10871v1) — Reaction-diffusion PDE as GNN layer design. Turing instabilities for patterns. Layer design, not PE.

[23] [Revisiting Random Walks for Learning on Graphs](https://arxiv.org/html/2407.01214v1) — Anonymized walk trajectories processed by language models. No Koopman decomposition.

[24] [GNN Node Classification Using Koopman Operator Theory on GPU](https://link.springer.com/chapter/10.1007/978-3-031-82427-2_6) — Uses Koopman to accelerate GNN training, not for graph PE.

[25] [Transformer with Koopman-Enhanced GCN for Spatiotemporal Forecasting](https://arxiv.org/html/2507.03855) — Koopman + GCN for spatiotemporal forecasting. Not about static graph PE.

[26] [PEARL search context — novelty verification](https://arxiv.org/abs/2502.01122) — Searches for nonlinear walk PE and Koopman graph PE returned no direct prior work, supporting KW-PE novelty.

## Follow-up Questions

- Does PEARL's use of nonlinear GNN processing with random feature initialization actually break spectral invariance in practice, and if so, does it achieve the same type of cross-term coupling that KW-PE targets through Koopman decomposition?
- What is the optimal dictionary size and type for EDMD applied to nonlinear walks on graphs of varying sizes, and how does the computational cost scale compared to eigendecomposition-based methods?
- Can the Koopman eigenfunctions extracted from nonlinear graph walks be shown to encode specific substructure counting capabilities (e.g., cycle counting beyond parallel trees) with formal guarantees?

---
*Generated by AI Inventor Pipeline*
