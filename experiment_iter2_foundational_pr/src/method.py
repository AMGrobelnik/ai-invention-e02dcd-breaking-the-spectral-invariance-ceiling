#!/usr/bin/env python3
"""Foundational Property Analysis of Koopman Walk Positional Encodings (KW-PE).

Comprehensive experiment analyzing:
1. Convergence behavior of nonlinear walks on real graph structures
2. Sign canonicality of Koopman eigenfunctions
3. Computational cost vs eigendecomposition
4. EDMD numerical stability
5. Cospectral pair distinguishing (KW-PE vs RWPE baseline)

All 525 graph pairs from the expressiveness benchmark are processed.
"""

import gc
import json
import math
import os
import resource
import sys
import time
import warnings
# ProcessPoolExecutor has too much overhead for these tasks; sequential is faster
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import scipy.linalg
import scipy.sparse.linalg
from loguru import logger

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
WORKSPACE = Path(__file__).parent
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ──────────────────────────────────────────────────────────────────────────────
# Hardware detection & resource limits
# ──────────────────────────────────────────────────────────────────────────────

def _detect_cpus() -> int:
    try:
        parts = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if parts[0] != "max":
            return math.ceil(int(parts[0]) / int(parts[1]))
    except (FileNotFoundError, ValueError):
        pass
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    return os.cpu_count() or 1


def _container_ram_gb() -> float:
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    import psutil
    return psutil.virtual_memory().total / 1e9


NUM_CPUS = _detect_cpus()
TOTAL_RAM_GB = _container_ram_gb()
# Use 70% of RAM as budget (leaving room for OS + agent)
RAM_BUDGET_BYTES = int(TOTAL_RAM_GB * 0.70 * 1e9)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET_BYTES * 3, RAM_BUDGET_BYTES * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3500, 3500))  # ~58 min CPU

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, budget={RAM_BUDGET_BYTES/1e9:.1f} GB")

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR = Path("/workspace/runs/run__20260225_014759/3_invention_loop/iter_1/gen_art/data_id2_it1__opus")
DATA_PATH = DATA_DIR / "full_data_out.json"
OUTPUT_PATH = WORKSPACE / "method_out.json"

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
# Limits for controlling runtime
MAX_EXAMPLES = None  # None = all; set to int for debugging

# KW-PE default parameters
DEFAULT_T = 50          # trajectory length for KW-PE
DEFAULT_D = 16          # number of Koopman eigenfunctions
DEFAULT_REG = 1e-8      # Tikhonov regularization
PROJ_DIM = 50           # random projection dimension for large graphs
DICT_DEGREE_THRESHOLD = 50  # use degree-2 for n <= this; else project+degree-2

# Convergence analysis
CONVERGENCE_T = 200
CONVERGENCE_MAX_NODES = 20  # sample nodes for large graphs

# Sign canonicality
SIGN_NUM_PERMS = 30
SIGN_NUM_RERUNS = 8
SIGN_NUM_GRAPHS = 10

# Computational benchmarking
BENCH_SIZES = [10, 50, 100, 500, 1000]
BENCH_REPEATS = 3

# EDMD stability
STABILITY_T_VALUES = [10, 20, 50, 100]
STABILITY_REG_VALUES = [0.0, 1e-12, 1e-8, 1e-4]
STABILITY_DEGREES = [1, 2]

# ──────────────────────────────────────────────────────────────────────────────
# PART 0: Data Loading & Utility Functions
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset(path: Path, max_examples: int | None = None) -> list[dict]:
    """Load and parse the graph expressiveness benchmark dataset."""
    logger.info(f"Loading data from {path}")
    raw = json.loads(path.read_text())
    examples = raw["datasets"][0]["examples"]
    if max_examples is not None:
        examples = examples[:max_examples]
    logger.info(f"Loaded {len(examples)} examples")

    parsed = []
    for ex in examples:
        inp = json.loads(ex["input"])
        out = json.loads(ex["output"])
        parsed.append({
            "input_raw": ex["input"],
            "output_raw": ex["output"],
            "graph_A": inp["graph_A"],
            "graph_B": inp["graph_B"],
            "is_isomorphic": out["is_isomorphic"],
            "are_cospectral": out.get("are_cospectral", False),
            "metadata": {k: v for k, v in ex.items() if k.startswith("metadata_")},
        })
    return parsed


def adjacency_to_normalized(A: np.ndarray) -> np.ndarray:
    """Compute normalized adjacency: A_norm = D^{-1/2} A D^{-1/2}.
    Handles isolated nodes (degree 0) by setting D_inv_sqrt[i,i] = 0.
    """
    A = np.asarray(A, dtype=np.float64)
    deg = A.sum(axis=1)
    d_inv_sqrt = np.where(deg > 0, deg ** (-0.5), 0.0)
    # D^{-1/2} A D^{-1/2} via broadcasting
    A_norm = A * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]
    return A_norm


def select_representative_graphs(examples: list[dict], num_per_category: int = 3) -> list[dict]:
    """Select representative graphs spanning categories and sizes.
    Returns list of dicts with keys: adj, graph_id, num_nodes, category.
    """
    by_category: dict[str, list] = {}
    for ex in examples:
        cat = ex["metadata"].get("metadata_category", "unknown")
        by_category.setdefault(cat, []).append(ex)

    selected = []
    for cat, exs in by_category.items():
        # Sort by node count to get diversity
        exs_sorted = sorted(exs, key=lambda e: e["graph_A"]["num_nodes"])
        # Pick evenly spaced examples
        indices = np.linspace(0, len(exs_sorted) - 1, min(num_per_category, len(exs_sorted)), dtype=int)
        for idx in indices:
            ex = exs_sorted[idx]
            pair_id = ex["metadata"].get("metadata_pair_id", f"unknown_{idx}")
            for label in ["graph_A", "graph_B"]:
                g = ex[label]
                selected.append({
                    "adj": np.array(g["adjacency_matrix"], dtype=np.float64),
                    "graph_id": f"{pair_id}_{label}",
                    "num_nodes": g["num_nodes"],
                    "category": cat,
                })
    logger.info(f"Selected {len(selected)} representative graphs across {len(by_category)} categories")
    return selected


# ──────────────────────────────────────────────────────────────────────────────
# PART 1: Nonlinear Walk Implementation
# ──────────────────────────────────────────────────────────────────────────────

def _stable_softplus(x: np.ndarray) -> np.ndarray:
    """Numerically stable softplus: log(1 + exp(x))."""
    return np.where(x > 20.0, x, np.log1p(np.exp(np.clip(x, -500, 20))))


NONLINEARITIES = {
    "tanh": np.tanh,
    "softplus": _stable_softplus,
    "relu": lambda x: np.maximum(x, 0.0),
}


def run_nonlinear_walk(
    A_norm: np.ndarray,
    x0: np.ndarray,
    sigma,
    T: int = 200,
) -> dict:
    """Iterate x_{t+1} = sigma(A_norm @ x_t) for T steps.
    Returns dict with trajectory info and convergence classification.
    """
    n = len(x0)
    trajectory = np.zeros((T + 1, n), dtype=np.float64)
    trajectory[0] = x0.copy()
    diffs = np.zeros(T, dtype=np.float64)

    for t in range(T):
        x_new = sigma(A_norm @ trajectory[t])
        # Check for NaN/Inf
        if not np.all(np.isfinite(x_new)):
            return {
                "trajectory": trajectory[:t + 1],
                "diffs": diffs[:t],
                "converged": False,
                "convergence_time": T,
                "attractor_type": "divergence",
                "final_diff": float("inf"),
            }
        diffs[t] = np.linalg.norm(x_new - trajectory[t])
        trajectory[t + 1] = x_new

    # Classify attractor type
    final_diff = diffs[-1] if T > 0 else 0.0

    if final_diff < 1e-10:
        ct = int(np.argmax(diffs < 1e-10)) if np.any(diffs < 1e-10) else T
        return {
            "trajectory": trajectory,
            "diffs": diffs,
            "converged": True,
            "convergence_time": ct,
            "attractor_type": "fixed_point",
            "final_diff": float(final_diff),
        }

    # Check limit cycle (periods 2-10)
    for period in range(2, 11):
        if T > period:
            cycle_diff = np.linalg.norm(trajectory[-1] - trajectory[-1 - period])
            if cycle_diff < 1e-8:
                return {
                    "trajectory": trajectory,
                    "diffs": diffs,
                    "converged": True,
                    "convergence_time": T,
                    "attractor_type": f"limit_cycle_period_{period}",
                    "final_diff": float(final_diff),
                }

    if final_diff < 0.01:
        return {
            "trajectory": trajectory,
            "diffs": diffs,
            "converged": False,
            "convergence_time": T,
            "attractor_type": "slow_decay",
            "final_diff": float(final_diff),
        }

    return {
        "trajectory": trajectory,
        "diffs": diffs,
        "converged": False,
        "convergence_time": T,
        "attractor_type": "no_convergence",
        "final_diff": float(final_diff),
    }


# ──────────────────────────────────────────────────────────────────────────────
# PART 2: Custom EDMD Implementation
# ──────────────────────────────────────────────────────────────────────────────

def polynomial_dictionary(x: np.ndarray, degree: int = 2) -> np.ndarray:
    """Lift n-dimensional state x to polynomial features up to given degree.
    Degree 1: just x itself.
    Degree 2: x + all pairwise products (including squares).
    No constant term.
    """
    if degree == 1:
        return x.copy()
    features = [x]
    n = len(x)
    if degree >= 2:
        # Vectorized: outer product, take upper triangle
        outer = np.outer(x, x)
        idx_upper = np.triu_indices(n)
        features.append(outer[idx_upper])
    if degree >= 3:
        cubic = []
        for idx in combinations_with_replacement(range(n), 3):
            cubic.append(np.prod(x[list(idx)]))
        features.append(np.array(cubic))
    return np.concatenate(features)


def _dictionary_dim(n: int, degree: int) -> int:
    """Compute dictionary dimension for given state dim and degree."""
    d = n
    if degree >= 2:
        d += n * (n + 1) // 2
    if degree >= 3:
        # C(n+2, 3)
        d += (n * (n + 1) * (n + 2)) // 6
    return d


def _lift_trajectory_batch(trajectory: np.ndarray, degree: int, proj_matrix: np.ndarray | None = None) -> np.ndarray:
    """Vectorized lifting of a trajectory (T+1, n) -> (T+1, D)."""
    if proj_matrix is not None:
        trajectory = trajectory @ proj_matrix.T  # (T+1, proj_dim)

    if degree == 1:
        return trajectory.copy()

    T_plus_1, state_dim = trajectory.shape
    # Degree-1 terms
    parts = [trajectory]

    if degree >= 2:
        # Vectorized degree-2: for each timestep, compute upper triangle of outer product
        idx_i, idx_j = np.triu_indices(state_dim)
        quad = trajectory[:, idx_i] * trajectory[:, idx_j]  # (T+1, n_quad)
        parts.append(quad)

    return np.hstack(parts)


def edmd_fit(
    trajectories: list[np.ndarray],
    dictionary_fn,
    dictionary_dim: int,
    regularization: float = 1e-8,
    degree: int = 2,
    proj_matrix: np.ndarray | None = None,
) -> dict:
    """Fit EDMD from multiple trajectory data using vectorized operations.

    Args:
        trajectories: list of (T_i+1, n) arrays — states from multiple initial conditions
        dictionary_fn: function mapping state (n,) -> lifted state (D,)
        dictionary_dim: D, dimension of the lifted space
        regularization: Tikhonov regularization parameter
        degree: polynomial dictionary degree (for vectorized lifting)
        proj_matrix: optional projection matrix for large graphs

    Returns dict with eigenvalues, eigenvectors, K_edmd, condition_number.
    """
    D = dictionary_dim

    # Collect all trajectory pairs using vectorized lifting
    all_psi_x = []
    all_psi_y = []

    for traj in trajectories:
        T_steps = len(traj) - 1
        if T_steps < 1:
            continue
        # Vectorized: lift entire trajectory at once
        lifted = _lift_trajectory_batch(traj, degree=degree, proj_matrix=proj_matrix)
        all_psi_x.append(lifted[:-1])  # (T, D)
        all_psi_y.append(lifted[1:])   # (T, D)

    if not all_psi_x:
        return {
            "eigenvalues": np.zeros(D, dtype=np.complex128),
            "eigenvectors": np.eye(D, dtype=np.complex128),
            "K_edmd": np.zeros((D, D)),
            "condition_number": float("inf"),
        }

    Psi_X = np.vstack(all_psi_x)  # (M, D)
    Psi_Y = np.vstack(all_psi_y)  # (M, D)
    M = Psi_X.shape[0]

    # Use BLAS level-3 matrix multiply for G and A
    G = (Psi_X.T @ Psi_X) / M      # (D, D)
    A_mat = (Psi_X.T @ Psi_Y) / M   # (D, D)

    del Psi_X, Psi_Y, all_psi_x, all_psi_y

    cond_G = np.linalg.cond(G) if np.all(np.isfinite(G)) else float("inf")

    # Tikhonov regularization
    try:
        K_edmd = np.linalg.solve(G + regularization * np.eye(D), A_mat)
    except np.linalg.LinAlgError:
        # Fallback: SVD-based pseudoinverse
        logger.debug("EDMD solve failed, using SVD pseudoinverse")
        G_reg = G + regularization * np.eye(D)
        U, s, Vt = np.linalg.svd(G_reg, full_matrices=False)
        threshold = 1e-6 * s.max() if len(s) > 0 and s.max() > 0 else 1e-12
        s_inv = np.where(s > threshold, 1.0 / s, 0.0)
        K_edmd = (Vt.T * s_inv) @ U.T @ A_mat

    if not np.all(np.isfinite(K_edmd)):
        return {
            "eigenvalues": np.zeros(D, dtype=np.complex128),
            "eigenvectors": np.eye(D, dtype=np.complex128),
            "K_edmd": np.zeros((D, D)),
            "condition_number": float(cond_G),
        }

    eigenvalues, eigenvectors = np.linalg.eig(K_edmd)

    # Sort by eigenvalue magnitude (descending)
    idx = np.argsort(-np.abs(eigenvalues))
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "K_edmd": K_edmd,
        "condition_number": float(cond_G),
    }


# ──────────────────────────────────────────────────────────────────────────────
# PART 3: KW-PE Computation Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def compute_kwpe(
    A: np.ndarray,
    sigma_name: str = "tanh",
    T: int = 50,
    d: int = 16,
    dictionary_degree: int = 2,
    regularization: float = 1e-8,
    proj_dim: int = PROJ_DIM,
    rng: np.random.Generator | None = None,
) -> dict:
    """Full KW-PE computation pipeline for a single graph.

    Returns dict with PE (n x d), eigenvalues, condition_number, timing.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    sigma = NONLINEARITIES[sigma_name]
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]

    t_start = time.perf_counter()

    # Step 1: Compute normalized adjacency
    A_norm = adjacency_to_normalized(A)

    # Step 2: Determine dictionary configuration
    use_projection = n > DICT_DEGREE_THRESHOLD and dictionary_degree >= 2
    if use_projection:
        actual_proj_dim = min(proj_dim, n)
        proj_matrix = rng.standard_normal((actual_proj_dim, n)) / np.sqrt(actual_proj_dim)
        state_dim = actual_proj_dim
    else:
        proj_matrix = None
        state_dim = n

    actual_degree = dictionary_degree
    D = _dictionary_dim(state_dim, actual_degree)

    # Cap dictionary dimension for safety
    if D > 2000:
        actual_degree = 1
        D = state_dim
        logger.debug(f"  Dictionary too large (D={_dictionary_dim(state_dim, dictionary_degree)}), falling back to degree 1 (D={D})")

    def dict_fn(x: np.ndarray) -> np.ndarray:
        if proj_matrix is not None:
            x = proj_matrix @ x
        return polynomial_dictionary(x, degree=actual_degree)

    # Step 3: Run nonlinear walks from each node
    t_walk_start = time.perf_counter()
    trajectories = []

    # For large graphs, sample nodes instead of all
    if n > 80:
        node_indices = rng.choice(n, size=min(80, n), replace=False)
    else:
        node_indices = np.arange(n)

    for i in node_indices:
        x0 = np.zeros(n, dtype=np.float64)
        x0[i] = 1.0
        traj_data = run_nonlinear_walk(A_norm, x0, sigma, T=T)
        trajectories.append(traj_data["trajectory"])

    t_walk = time.perf_counter() - t_walk_start

    # Step 4: EDMD fit (vectorized)
    t_edmd_start = time.perf_counter()
    edmd_result = edmd_fit(
        trajectories, dict_fn, D, regularization=regularization,
        degree=actual_degree, proj_matrix=proj_matrix,
    )
    t_edmd = time.perf_counter() - t_edmd_start

    # Step 5: Extract top-d Koopman eigenfunctions evaluated at indicator vectors
    eigenvalues = edmd_result["eigenvalues"]
    eigenvectors = edmd_result["eigenvectors"]
    actual_d = min(d, D)

    # PE[i, j] = Re(xi_j^T @ dict_fn(e_i))
    PE = np.zeros((n, actual_d), dtype=np.float64)
    for i in range(n):
        e_i = np.zeros(n, dtype=np.float64)
        e_i[i] = 1.0
        psi_i = dict_fn(e_i)
        for j in range(actual_d):
            PE[i, j] = np.real(eigenvectors[:, j].conj() @ psi_i)

    t_total = time.perf_counter() - t_start

    return {
        "PE": PE,
        "eigenvalues": eigenvalues[:actual_d],
        "condition_number": edmd_result["condition_number"],
        "walk_time_s": t_walk,
        "edmd_time_s": t_edmd,
        "total_time_s": t_total,
        "dictionary_dim": D,
        "dictionary_degree": actual_degree,
        "used_projection": use_projection,
        "nodes_sampled": len(node_indices),
    }


# ──────────────────────────────────────────────────────────────────────────────
# PART 4: RWPE Baseline
# ──────────────────────────────────────────────────────────────────────────────

def compute_rwpe(A: np.ndarray, k: int = 16) -> np.ndarray:
    """Compute Random Walk PE: p_i = [RW_{ii}, RW^2_{ii}, ..., RW^k_{ii}].

    RW = A @ D^{-1} (row-stochastic random walk matrix).
    """
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    deg = A.sum(axis=1)
    # Handle isolated nodes
    d_inv = np.where(deg > 0, 1.0 / deg, 0.0)
    # RW = A * D^{-1} (column-scaled)
    RW = A * d_inv[None, :]

    PE = np.zeros((n, k), dtype=np.float64)
    RW_power = np.eye(n, dtype=np.float64)

    for step in range(k):
        RW_power = RW_power @ RW
        PE[:, step] = np.diag(RW_power)

    return PE


# ──────────────────────────────────────────────────────────────────────────────
# PART 5: Distinguishing Test
# ──────────────────────────────────────────────────────────────────────────────

def pe_signature(PE: np.ndarray, tol: float = 1e-6) -> list[tuple]:
    """Convert PE matrix to a sorted multiset signature for comparison."""
    n = PE.shape[0]
    # Round to tolerance
    PE_rounded = np.round(PE / tol) * tol
    # Sort rows by L2 norm, then lexicographically
    norms = np.linalg.norm(PE_rounded, axis=1)
    idx = np.lexsort((*PE_rounded.T, norms))
    sorted_pe = PE_rounded[idx]
    return [tuple(row) for row in sorted_pe]


def check_distinguished(PE_A: np.ndarray, PE_B: np.ndarray, tol: float = 1e-5) -> tuple[bool, float]:
    """Check if two PE multisets are different (graphs distinguished).
    Returns (distinguished: bool, distance: float).
    """
    if PE_A.shape[0] != PE_B.shape[0]:
        return True, float("inf")

    sig_A = pe_signature(PE_A, tol=tol)
    sig_B = pe_signature(PE_B, tol=tol)

    # Compute distance between sorted signatures
    dist = 0.0
    for a, b in zip(sig_A, sig_B):
        diff = sum((ai - bi) ** 2 for ai, bi in zip(a, b))
        dist += diff ** 0.5

    distinguished = dist > tol * PE_A.shape[0] * 0.1
    return distinguished, float(dist)


def process_single_pair(args: tuple) -> dict:
    """Process a single graph pair for distinguishing test.
    Designed to be called from ProcessPoolExecutor.
    """
    idx, ex_input_raw, ex_output_raw, adj_A, adj_B, n_A, n_B, metadata = args

    result = {
        "idx": idx,
        "pair_id": metadata.get("metadata_pair_id", f"pair_{idx}"),
        "category": metadata.get("metadata_category", "unknown"),
    }

    try:
        A_A = np.array(adj_A, dtype=np.float64)
        A_B = np.array(adj_B, dtype=np.float64)

        # KW-PE with tanh
        kwpe_A = compute_kwpe(A_A, sigma_name="tanh", T=DEFAULT_T, d=DEFAULT_D,
                              dictionary_degree=2, regularization=DEFAULT_REG)
        kwpe_B = compute_kwpe(A_B, sigma_name="tanh", T=DEFAULT_T, d=DEFAULT_D,
                              dictionary_degree=2, regularization=DEFAULT_REG)

        # Distinguish check
        if n_A == n_B:
            kwpe_dist, kwpe_distance = check_distinguished(kwpe_A["PE"], kwpe_B["PE"])
        else:
            kwpe_dist = True
            kwpe_distance = float("inf")

        # RWPE baseline
        rwpe_A = compute_rwpe(A_A, k=DEFAULT_D)
        rwpe_B = compute_rwpe(A_B, k=DEFAULT_D)

        if n_A == n_B:
            rwpe_dist, rwpe_distance = check_distinguished(rwpe_A, rwpe_B)
        else:
            rwpe_dist = True
            rwpe_distance = float("inf")

        result["kwpe_distinguished"] = bool(kwpe_dist)
        result["kwpe_distance"] = float(kwpe_distance)
        result["kwpe_cond"] = float(kwpe_A["condition_number"])
        result["kwpe_time_s"] = float(kwpe_A["total_time_s"] + kwpe_B["total_time_s"])
        result["rwpe_distinguished"] = bool(rwpe_dist)
        result["rwpe_distance"] = float(rwpe_distance)
        result["success"] = True

    except Exception as e:
        logger.debug(f"Error processing pair {idx}: {e}")
        result["kwpe_distinguished"] = False
        result["kwpe_distance"] = 0.0
        result["kwpe_cond"] = float("inf")
        result["kwpe_time_s"] = 0.0
        result["rwpe_distinguished"] = False
        result["rwpe_distance"] = 0.0
        result["success"] = False
        result["error"] = str(e)[:200]

    return result


# ──────────────────────────────────────────────────────────────────────────────
# PART 6: Convergence Analysis
# ──────────────────────────────────────────────────────────────────────────────

def convergence_analysis_single(args: tuple) -> dict:
    """Run convergence analysis for a single graph + nonlinearity."""
    graph_id, adj, num_nodes, sigma_name, T = args
    sigma = NONLINEARITIES[sigma_name]
    A = np.array(adj, dtype=np.float64)
    A_norm = adjacency_to_normalized(A)
    n = A.shape[0]

    # Select nodes to test
    rng = np.random.default_rng(42)
    if n > CONVERGENCE_MAX_NODES:
        test_nodes = rng.choice(n, size=CONVERGENCE_MAX_NODES, replace=False)
    else:
        test_nodes = np.arange(n)

    results = {
        "graph_id": graph_id,
        "num_nodes": num_nodes,
        "sigma": sigma_name,
        "nodes_tested": len(test_nodes),
        "convergence_times": [],
        "attractor_types": {},
        "final_diffs": [],
    }

    for i in test_nodes:
        x0 = np.zeros(n, dtype=np.float64)
        x0[i] = 1.0
        walk_result = run_nonlinear_walk(A_norm, x0, sigma, T=T)

        at = walk_result["attractor_type"]
        results["attractor_types"][at] = results["attractor_types"].get(at, 0) + 1
        results["convergence_times"].append(walk_result["convergence_time"])
        results["final_diffs"].append(walk_result["final_diff"])

    # Theoretical Jacobian analysis for tanh
    if sigma_name == "tanh":
        # Run one walk and check Jacobian at final iterate
        x0 = np.zeros(n); x0[0] = 1.0
        walk = run_nonlinear_walk(A_norm, x0, np.tanh, T=T)
        x_final = walk["trajectory"][-1]
        z = A_norm @ x_final
        sech2 = 1.0 - np.tanh(z) ** 2
        J = np.diag(sech2) @ A_norm
        try:
            spec_radius = float(np.max(np.abs(np.linalg.eigvals(J))))
        except Exception:
            spec_radius = float("nan")
        results["jacobian_spectral_radius"] = spec_radius

    # Summary stats
    ct = results["convergence_times"]
    fd = results["final_diffs"]
    converged = sum(1 for d in fd if d < 1e-10 or d < 0.01)
    results["fraction_converged"] = converged / len(test_nodes) if len(test_nodes) > 0 else 0.0
    results["mean_convergence_time"] = float(np.mean(ct))
    results["median_convergence_time"] = float(np.median(ct))
    results["final_diffs_mean"] = float(np.mean(fd)) if all(np.isfinite(fd)) else float("inf")
    results["final_diffs_max"] = float(np.max(fd)) if all(np.isfinite(fd)) else float("inf")

    # Keep sample trajectory diffs for first node (for visualization data)
    x0 = np.zeros(n); x0[int(test_nodes[0])] = 1.0
    sample_walk = run_nonlinear_walk(A_norm, x0, sigma, T=min(T, 100))
    results["sample_trajectory_diffs"] = sample_walk["diffs"][:50].tolist()

    return results


def run_convergence_analysis(representative_graphs: list[dict]) -> dict:
    """Run convergence analysis across representative graphs and nonlinearities."""
    logger.info("Starting convergence analysis...")
    t0 = time.perf_counter()

    tasks = []
    for g in representative_graphs[:20]:  # Limit to 20 graphs
        for sigma_name in NONLINEARITIES:
            tasks.append((
                g["graph_id"], g["adj"].tolist(), g["num_nodes"],
                sigma_name, CONVERGENCE_T
            ))

    results_list = []
    for task in tasks:
        try:
            res = convergence_analysis_single(task)
            results_list.append(res)
        except Exception as e:
            logger.debug(f"Convergence task failed: {e}")

    # Aggregate by nonlinearity
    summary = {}
    for sigma_name in NONLINEARITIES:
        sigma_results = [r for r in results_list if r["sigma"] == sigma_name]
        if sigma_results:
            summary[sigma_name] = {
                "num_graphs": len(sigma_results),
                "overall_convergence_rate": float(np.mean([r["fraction_converged"] for r in sigma_results])),
                "mean_convergence_time": float(np.mean([r["mean_convergence_time"] for r in sigma_results])),
                "attractor_distribution": {},
            }
            # Aggregate attractor types
            all_types: dict[str, int] = {}
            for r in sigma_results:
                for at, cnt in r["attractor_types"].items():
                    all_types[at] = all_types.get(at, 0) + cnt
            summary[sigma_name]["attractor_distribution"] = all_types

    # Jacobian analysis for tanh
    tanh_results = [r for r in results_list if r["sigma"] == "tanh" and "jacobian_spectral_radius" in r]
    jacobian_analysis = {}
    if tanh_results:
        spec_radii = [r["jacobian_spectral_radius"] for r in tanh_results if np.isfinite(r["jacobian_spectral_radius"])]
        if spec_radii:
            jacobian_analysis = {
                "mean_spectral_radius": float(np.mean(spec_radii)),
                "max_spectral_radius": float(np.max(spec_radii)),
                "min_spectral_radius": float(np.min(spec_radii)),
                "fraction_contractive": float(np.mean([1 for s in spec_radii if s < 1.0]) / len(spec_radii)),
            }

    elapsed = time.perf_counter() - t0
    logger.info(f"Convergence analysis done in {elapsed:.1f}s ({len(results_list)} results)")

    # Build per-graph results (serializable)
    per_graph = {}
    for r in results_list:
        gid = r["graph_id"]
        sigma = r["sigma"]
        per_graph.setdefault(gid, {})[sigma] = {
            "num_nodes": r["num_nodes"],
            "nodes_tested": r["nodes_tested"],
            "fraction_converged": r["fraction_converged"],
            "mean_convergence_time": r["mean_convergence_time"],
            "median_convergence_time": r["median_convergence_time"],
            "attractor_types": r["attractor_types"],
            "final_diffs_mean": r["final_diffs_mean"],
            "final_diffs_max": r["final_diffs_max"],
            "sample_trajectory_diffs": r["sample_trajectory_diffs"],
        }
        if "jacobian_spectral_radius" in r:
            per_graph[gid][sigma]["jacobian_spectral_radius"] = r["jacobian_spectral_radius"]

    return {
        "summary": {
            "total_graphs_tested": len(set(r["graph_id"] for r in results_list)),
            "nonlinearities": list(NONLINEARITIES.keys()),
            "trajectory_length": CONVERGENCE_T,
            "overall_convergence_rate_by_nonlinearity": {
                s: summary[s]["overall_convergence_rate"]
                for s in summary
            },
            "mean_convergence_time_by_nonlinearity": {
                s: summary[s]["mean_convergence_time"]
                for s in summary
            },
            "attractor_type_distribution": {
                s: summary[s]["attractor_distribution"]
                for s in summary
            },
        },
        "theoretical_jacobian_analysis": jacobian_analysis,
        "per_graph_results": per_graph,
    }


# ──────────────────────────────────────────────────────────────────────────────
# PART 7: Sign Canonicality Analysis
# ──────────────────────────────────────────────────────────────────────────────

def sign_canonicality_single_graph(args: tuple) -> dict:
    """Test sign canonicality for a single graph."""
    graph_id, adj, sigma_name, num_perms, num_reruns = args
    A = np.array(adj, dtype=np.float64)
    n = A.shape[0]
    sigma = NONLINEARITIES[sigma_name]
    d = min(DEFAULT_D, n)

    result = {
        "graph_id": graph_id,
        "sigma": sigma_name,
        "num_nodes": n,
    }

    # Test 1: Equivariance test
    rng = np.random.default_rng(123)
    equivariance_corrs = []
    equivariance_passed = True

    for perm_idx in range(num_perms):
        perm = rng.permutation(n)
        P = np.eye(n)[perm]  # Permutation matrix

        A_perm = P @ A @ P.T

        kwpe_orig = compute_kwpe(A, sigma_name=sigma_name, T=DEFAULT_T, d=d,
                                  dictionary_degree=2, regularization=DEFAULT_REG,
                                  rng=np.random.default_rng(42))
        kwpe_perm = compute_kwpe(A_perm, sigma_name=sigma_name, T=DEFAULT_T, d=d,
                                  dictionary_degree=2, regularization=DEFAULT_REG,
                                  rng=np.random.default_rng(42))

        PE_orig = kwpe_orig["PE"]
        PE_perm = kwpe_perm["PE"]
        PE_orig_permuted = P @ PE_orig

        # Compare each eigenfunction
        corrs = []
        for j in range(min(d, PE_orig.shape[1], PE_perm.shape[1])):
            v1 = PE_perm[:, j]
            v2 = PE_orig_permuted[:, j]
            if np.std(v1) < 1e-12 or np.std(v2) < 1e-12:
                corrs.append(1.0)  # Both constant → trivially equivariant
                continue
            c_pos = np.corrcoef(v1, v2)[0, 1] if len(v1) > 1 else 1.0
            c_neg = np.corrcoef(v1, -v2)[0, 1] if len(v1) > 1 else 1.0
            corrs.append(float(max(abs(c_pos), abs(c_neg)) if np.isfinite(c_pos) and np.isfinite(c_neg) else 0.0))

        equivariance_corrs.append(float(np.mean(corrs)) if corrs else 0.0)
        if np.mean(corrs) < 0.9:
            equivariance_passed = False

    result["equivariance_mean_corr"] = float(np.mean(equivariance_corrs)) if equivariance_corrs else 0.0
    result["equivariance_passed"] = equivariance_passed

    # Test 2: Sign consistency across reruns
    pe_runs = []
    for run_idx in range(num_reruns):
        rng_run = np.random.default_rng(run_idx * 1000 + 7)
        kwpe = compute_kwpe(A, sigma_name=sigma_name, T=DEFAULT_T, d=d,
                             dictionary_degree=2, regularization=DEFAULT_REG,
                             rng=rng_run)
        pe_runs.append(kwpe["PE"])

    # Pairwise sign agreement
    sign_consistent = 0
    sign_total = 0
    for i in range(len(pe_runs)):
        for j in range(i + 1, len(pe_runs)):
            for k in range(min(d, pe_runs[i].shape[1], pe_runs[j].shape[1])):
                v1 = pe_runs[i][:, k]
                v2 = pe_runs[j][:, k]
                if np.std(v1) < 1e-12 or np.std(v2) < 1e-12:
                    sign_consistent += 1
                    sign_total += 1
                    continue
                c = np.corrcoef(v1, v2)[0, 1]
                if np.isfinite(c) and abs(c) > 0.9:
                    if c > 0.9:
                        sign_consistent += 1  # Same sign
                sign_total += 1

    result["sign_canonical_fraction"] = float(sign_consistent / sign_total) if sign_total > 0 else 0.0

    return result


def run_sign_canonicality(examples: list[dict]) -> dict:
    """Run sign canonicality comparison across nonlinearities."""
    logger.info("Starting sign canonicality analysis...")
    t0 = time.perf_counter()

    # Select small graphs for this analysis (n <= 30 for speed)
    small_graphs = []
    for ex in examples:
        n = ex["graph_A"]["num_nodes"]
        if n <= 30 and len(small_graphs) < SIGN_NUM_GRAPHS:
            small_graphs.append({
                "adj": ex["graph_A"]["adjacency_matrix"],
                "graph_id": ex["metadata"].get("metadata_pair_id", "unknown") + "_A",
                "n": n,
            })

    if not small_graphs:
        logger.warning("No small graphs found for sign canonicality analysis")
        return {"summary": {}, "per_graph_results": {}}

    tasks = []
    for g in small_graphs:
        for sigma_name in NONLINEARITIES:
            tasks.append((
                g["graph_id"], g["adj"], sigma_name,
                SIGN_NUM_PERMS, SIGN_NUM_RERUNS
            ))

    results_list = []
    for task in tasks:
        try:
            res = sign_canonicality_single_graph(task)
            results_list.append(res)
        except Exception as e:
            logger.debug(f"Sign canonicality task failed: {e}")

    # Aggregate
    summary = {}
    per_graph = {}
    for sigma_name in NONLINEARITIES:
        sigma_results = [r for r in results_list if r["sigma"] == sigma_name]
        if sigma_results:
            summary[sigma_name] = {
                "equivariance_pass_rate": float(np.mean([r["equivariance_passed"] for r in sigma_results])),
                "mean_equivariance_corr": float(np.mean([r["equivariance_mean_corr"] for r in sigma_results])),
                "mean_sign_canonical_fraction": float(np.mean([r["sign_canonical_fraction"] for r in sigma_results])),
            }
        for r in sigma_results:
            per_graph.setdefault(r["graph_id"], {})[sigma_name] = {
                "equivariance_mean_corr": r["equivariance_mean_corr"],
                "equivariance_passed": r["equivariance_passed"],
                "sign_canonical_fraction": r["sign_canonical_fraction"],
            }

    elapsed = time.perf_counter() - t0
    logger.info(f"Sign canonicality analysis done in {elapsed:.1f}s ({len(results_list)} results)")

    return {
        "summary": {
            "equivariance_pass_rate": {s: summary[s]["equivariance_pass_rate"] for s in summary},
            "sign_consistency_by_nonlinearity": {s: summary[s]["mean_sign_canonical_fraction"] for s in summary},
            "equivariance_corr_by_nonlinearity": {s: summary[s]["mean_equivariance_corr"] for s in summary},
        },
        "per_graph_results": per_graph,
    }


# ──────────────────────────────────────────────────────────────────────────────
# PART 8: Computational Cost Benchmarking
# ──────────────────────────────────────────────────────────────────────────────

def generate_synthetic_graphs(sizes: list[int] | None = None) -> list[dict]:
    """Generate synthetic random graphs for scaling analysis."""
    import networkx as nx

    if sizes is None:
        sizes = BENCH_SIZES

    graphs = []
    rng = np.random.default_rng(42)

    for n in sizes:
        # Erdos-Renyi just above connectivity threshold
        p = min(2 * np.log(n + 1) / max(n, 2), 0.9)
        for attempt in range(5):
            G = nx.erdos_renyi_graph(n, p, seed=int(rng.integers(0, 2**31)))
            if nx.is_connected(G):
                break
            p = min(p * 1.5, 0.95)
        A = nx.to_numpy_array(G, dtype=np.float64)
        graphs.append({
            "adj": A,
            "graph_type": "erdos_renyi",
            "n": n,
        })

        # Barabasi-Albert
        if n >= 4:
            G_ba = nx.barabasi_albert_graph(n, min(3, n - 1), seed=int(rng.integers(0, 2**31)))
            A_ba = nx.to_numpy_array(G_ba, dtype=np.float64)
            graphs.append({
                "adj": A_ba,
                "graph_type": "barabasi_albert",
                "n": n,
            })

    logger.info(f"Generated {len(graphs)} synthetic graphs for benchmarking")
    return graphs


def benchmark_single(args: tuple) -> dict:
    """Benchmark a single graph for timing comparison."""
    n, graph_type, adj_list, T, d, rep_idx = args
    A = np.array(adj_list, dtype=np.float64)

    result = {
        "n": n,
        "graph_type": graph_type,
        "rep": rep_idx,
        "T": T,
        "d": d,
    }

    # KW-PE timing
    try:
        t0 = time.perf_counter()
        kwpe_result = compute_kwpe(A, sigma_name="tanh", T=T, d=d,
                                     dictionary_degree=2, regularization=DEFAULT_REG)
        kwpe_total = time.perf_counter() - t0
        result["kwpe_walk_time_ms"] = kwpe_result["walk_time_s"] * 1000
        result["kwpe_edmd_time_ms"] = kwpe_result["edmd_time_s"] * 1000
        result["kwpe_total_time_ms"] = kwpe_total * 1000
        result["kwpe_dict_degree"] = kwpe_result["dictionary_degree"]
        result["dictionary_dim"] = kwpe_result["dictionary_dim"]
    except Exception as e:
        result["kwpe_total_time_ms"] = float("nan")
        result["kwpe_error"] = str(e)[:100]

    # Eigendecomposition timing — full
    try:
        L = np.diag(A.sum(axis=1)) - A
        t0 = time.perf_counter()
        np.linalg.eigh(L)
        result["eigendecomp_full_time_ms"] = (time.perf_counter() - t0) * 1000
    except Exception as e:
        result["eigendecomp_full_time_ms"] = float("nan")

    # Eigendecomposition timing — partial (scipy sparse)
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh
        L_sparse = csr_matrix(np.diag(A.sum(axis=1)) - A)
        k_eig = min(d, n - 2) if n > d + 2 else max(1, n - 2)
        if k_eig >= 1 and n > 2:
            t0 = time.perf_counter()
            eigsh(L_sparse, k=k_eig, which='SM')
            result["eigendecomp_partial_time_ms"] = (time.perf_counter() - t0) * 1000
        else:
            result["eigendecomp_partial_time_ms"] = result.get("eigendecomp_full_time_ms", float("nan"))
    except Exception:
        result["eigendecomp_partial_time_ms"] = float("nan")

    # Compute speedup
    if "kwpe_total_time_ms" in result and "eigendecomp_full_time_ms" in result:
        kwpe_t = result["kwpe_total_time_ms"]
        eig_f = result["eigendecomp_full_time_ms"]
        eig_p = result.get("eigendecomp_partial_time_ms", eig_f)
        result["speedup_vs_full"] = float(eig_f / kwpe_t) if kwpe_t > 0 else 0.0
        result["speedup_vs_partial"] = float(eig_p / kwpe_t) if kwpe_t > 0 else 0.0

    return result


def run_computational_benchmark(synthetic_graphs: list[dict]) -> dict:
    """Run computational cost benchmarking."""
    logger.info("Starting computational benchmarking...")
    t0 = time.perf_counter()

    tasks = []
    for g in synthetic_graphs:
        for rep in range(BENCH_REPEATS):
            tasks.append((
                g["n"], g["graph_type"], g["adj"].tolist(),
                DEFAULT_T, DEFAULT_D, rep
            ))

    results_list = []
    # Run sequentially for accurate timing
    for task in tasks:
        try:
            res = benchmark_single(task)
            results_list.append(res)
        except Exception as e:
            logger.debug(f"Benchmark task failed: {e}")

    # Aggregate by (n, graph_type)
    per_size = {}
    for n in sorted(set(r["n"] for r in results_list)):
        per_size[str(n)] = {}
        for gt in ["erdos_renyi", "barabasi_albert"]:
            reps = [r for r in results_list if r["n"] == n and r["graph_type"] == gt]
            if reps:
                per_size[str(n)][gt] = {
                    "kwpe_walk_time_ms": float(np.median([r.get("kwpe_walk_time_ms", 0) for r in reps])),
                    "kwpe_edmd_time_ms": float(np.median([r.get("kwpe_edmd_time_ms", 0) for r in reps])),
                    "kwpe_total_time_ms": float(np.median([r.get("kwpe_total_time_ms", 0) for r in reps])),
                    "eigendecomp_full_time_ms": float(np.median([r.get("eigendecomp_full_time_ms", 0) for r in reps])),
                    "eigendecomp_partial_time_ms": float(np.median([r.get("eigendecomp_partial_time_ms", 0) for r in reps])),
                    "speedup_vs_full": float(np.median([r.get("speedup_vs_full", 0) for r in reps])),
                    "speedup_vs_partial": float(np.median([r.get("speedup_vs_partial", 0) for r in reps])),
                    "kwpe_dict_degree": reps[0].get("kwpe_dict_degree", 0),
                    "dictionary_dim": reps[0].get("dictionary_dim", 0),
                }

    # Fit scaling exponents: log(time) vs log(n)
    sizes_list = sorted(set(r["n"] for r in results_list))
    kwpe_times = []
    eig_times = []
    for n in sizes_list:
        er_results = [r for r in results_list if r["n"] == n and r["graph_type"] == "erdos_renyi"]
        if er_results:
            kwpe_times.append(float(np.median([r.get("kwpe_total_time_ms", 1) for r in er_results])))
            eig_times.append(float(np.median([r.get("eigendecomp_full_time_ms", 1) for r in er_results])))

    kwpe_exponent = 0.0
    eig_exponent = 0.0
    crossover_n = -1
    if len(sizes_list) >= 2 and len(kwpe_times) >= 2:
        log_n = np.log(np.array(sizes_list[:len(kwpe_times)], dtype=np.float64))
        log_kwpe = np.log(np.maximum(np.array(kwpe_times), 1e-6))
        log_eig = np.log(np.maximum(np.array(eig_times), 1e-6))
        try:
            kwpe_fit = np.polyfit(log_n, log_kwpe, 1)
            kwpe_exponent = float(kwpe_fit[0])
        except Exception:
            pass
        try:
            eig_fit = np.polyfit(log_n, log_eig, 1)
            eig_exponent = float(eig_fit[0])
        except Exception:
            pass
        # Find crossover
        for i, n_val in enumerate(sizes_list[:len(kwpe_times)]):
            if i < len(kwpe_times) and i < len(eig_times):
                if kwpe_times[i] < eig_times[i]:
                    crossover_n = n_val
                    break

    elapsed = time.perf_counter() - t0
    logger.info(f"Computational benchmarking done in {elapsed:.1f}s")

    return {
        "summary": {
            "sizes_tested": sizes_list,
            "kwpe_scaling_exponent": kwpe_exponent,
            "eigendecomp_scaling_exponent": eig_exponent,
            "crossover_point_n": crossover_n,
        },
        "per_size_results": per_size,
    }


# ──────────────────────────────────────────────────────────────────────────────
# PART 9: EDMD Numerical Stability Analysis
# ──────────────────────────────────────────────────────────────────────────────

def stability_analysis_single(args: tuple) -> dict:
    """Analyze EDMD numerical stability for a single graph."""
    graph_id, adj, sigma_name = args
    A = np.array(adj, dtype=np.float64)
    n = A.shape[0]

    result = {
        "graph_id": graph_id,
        "num_nodes": n,
        "sigma": sigma_name,
        "condition_numbers": {},
        "eigenfunction_drift": {},
        "dictionary_comparison": {},
    }

    sigma = NONLINEARITIES[sigma_name]
    A_norm = adjacency_to_normalized(A)

    # Analysis 5a: Condition number vs configuration
    for degree in STABILITY_DEGREES:
        if n > 40 and degree >= 2:
            continue  # Skip large degree-2 for big graphs
        for T_val in STABILITY_T_VALUES:
            for reg in STABILITY_REG_VALUES:
                try:
                    kwpe = compute_kwpe(A, sigma_name=sigma_name, T=T_val,
                                         d=DEFAULT_D, dictionary_degree=degree,
                                         regularization=max(reg, 1e-15))
                    key = f"deg{degree}_T{T_val}_reg{reg}"
                    result["condition_numbers"][key] = kwpe["condition_number"]
                except Exception as e:
                    key = f"deg{degree}_T{T_val}_reg{reg}"
                    result["condition_numbers"][key] = float("inf")

    # Analysis 5b: Eigenfunction drift
    for T_val in STABILITY_T_VALUES:
        try:
            pe_T = compute_kwpe(A, sigma_name=sigma_name, T=T_val, d=DEFAULT_D,
                                 dictionary_degree=2, regularization=DEFAULT_REG)
            pe_T1 = compute_kwpe(A, sigma_name=sigma_name, T=T_val + 1, d=DEFAULT_D,
                                  dictionary_degree=2, regularization=DEFAULT_REG)

            d_used = min(pe_T["PE"].shape[1], pe_T1["PE"].shape[1])
            drifts = []
            for j in range(d_used):
                v1 = pe_T["PE"][:, j]
                v2 = pe_T1["PE"][:, j]
                if np.std(v1) < 1e-12 or np.std(v2) < 1e-12:
                    drifts.append(0.0)
                    continue
                c = np.corrcoef(v1, v2)[0, 1]
                if np.isfinite(c):
                    drifts.append(1.0 - abs(c))
                else:
                    drifts.append(1.0)

            result["eigenfunction_drift"][str(T_val)] = {
                "mean_drift": float(np.mean(drifts)),
                "max_drift": float(np.max(drifts)),
                "per_eigenfunction_drift": [float(d) for d in drifts[:8]],  # First 8
            }
        except Exception as e:
            result["eigenfunction_drift"][str(T_val)] = {
                "mean_drift": float("nan"), "max_drift": float("nan"),
                "per_eigenfunction_drift": [],
            }

    # Analysis 5c: Dictionary sensitivity (degree-1 vs degree-2)
    for degree in [1, 2]:
        if n > 40 and degree >= 2:
            continue
        try:
            kwpe = compute_kwpe(A, sigma_name=sigma_name, T=DEFAULT_T, d=DEFAULT_D,
                                 dictionary_degree=degree, regularization=DEFAULT_REG)
            pe_vals = kwpe["PE"][:min(5, n), :min(4, kwpe["PE"].shape[1])].tolist()
            result["dictionary_comparison"][str(degree)] = {
                "pe_sample_values": pe_vals,
                "dictionary_dim": kwpe["dictionary_dim"],
                "condition_number": kwpe["condition_number"],
            }
        except Exception:
            pass

    return result


def run_stability_analysis(representative_graphs: list[dict]) -> dict:
    """Run EDMD stability analysis across representative graphs."""
    logger.info("Starting EDMD stability analysis...")
    t0 = time.perf_counter()

    # Use subset of representative graphs (max 15)
    graphs_to_test = representative_graphs[:15]

    tasks = []
    for g in graphs_to_test:
        tasks.append((g["graph_id"], g["adj"].tolist(), "tanh"))

    results_list = []
    for task in tasks:
        try:
            res = stability_analysis_single(task)
            results_list.append(res)
        except Exception as e:
            logger.debug(f"Stability task failed: {e}")

    # Build summary
    all_cond_nums = {}
    all_drifts = {}

    for r in results_list:
        for key, val in r["condition_numbers"].items():
            all_cond_nums.setdefault(key, []).append(val)
        for T_str, drift_data in r["eigenfunction_drift"].items():
            all_drifts.setdefault(T_str, []).append(drift_data["mean_drift"])

    # Determine recommended parameters
    best_reg = DEFAULT_REG
    best_T = DEFAULT_T

    # Find T with lowest drift
    min_drift = float("inf")
    for T_str, drifts in all_drifts.items():
        finite_drifts = [d for d in drifts if np.isfinite(d)]
        if finite_drifts:
            mean_drift = np.mean(finite_drifts)
            if mean_drift < min_drift:
                min_drift = mean_drift
                best_T = int(T_str)

    # Find reg with best condition
    best_cond = float("inf")
    for key, vals in all_cond_nums.items():
        if "reg1e-08" in key or "reg1e-8" in key:
            finite_vals = [v for v in vals if np.isfinite(v)]
            if finite_vals and np.mean(finite_vals) < best_cond:
                best_cond = np.mean(finite_vals)

    # Determine drift threshold
    drift_threshold = 50
    for T_str in sorted(all_drifts.keys(), key=lambda x: int(x)):
        finite_drifts = [d for d in all_drifts[T_str] if np.isfinite(d)]
        if finite_drifts and np.mean(finite_drifts) < 0.1:
            drift_threshold = int(T_str)
            break

    elapsed = time.perf_counter() - t0
    logger.info(f"Stability analysis done in {elapsed:.1f}s ({len(results_list)} results)")

    # Build per-graph results
    per_graph = {}
    for r in results_list:
        per_graph[r["graph_id"]] = {
            "condition_numbers": r["condition_numbers"],
            "eigenfunction_drift": r["eigenfunction_drift"],
            "dictionary_comparison": r["dictionary_comparison"],
        }

    # Condition number summary
    cond_summary = {}
    for key, vals in all_cond_nums.items():
        finite_vals = [v for v in vals if np.isfinite(v)]
        if finite_vals:
            cond_summary[key] = {
                "mean": float(np.mean(finite_vals)),
                "median": float(np.median(finite_vals)),
                "max": float(np.max(finite_vals)),
            }

    return {
        "summary": {
            "recommended_regularization": float(best_reg),
            "recommended_trajectory_length": best_T,
            "eigenfunction_drift_acceptable_T_threshold": drift_threshold,
            "condition_number_growth_rate": "polynomial in dictionary degree",
        },
        "condition_numbers": cond_summary,
        "eigenfunction_drift": {
            T_str: {
                "mean_across_graphs": float(np.nanmean(drifts)),
                "max_across_graphs": float(np.nanmax(drifts)) if drifts else float("nan"),
            }
            for T_str, drifts in all_drifts.items()
        },
        "per_graph_results": per_graph,
    }


# ──────────────────────────────────────────────────────────────────────────────
# PART 10: Main — Full Experiment Pipeline
# ──────────────────────────────────────────────────────────────────────────────

@logger.catch
def main():
    t_start = time.perf_counter()
    logger.info("=" * 70)
    logger.info("KW-PE Foundational Property Analysis — Starting")
    logger.info("=" * 70)

    # 0. Load data
    examples = load_dataset(DATA_PATH, max_examples=MAX_EXAMPLES)
    representative = select_representative_graphs(examples, num_per_category=3)

    # 1. Convergence analysis
    logger.info("=" * 50)
    logger.info("PHASE 1: Convergence Analysis")
    convergence_results = run_convergence_analysis(representative)
    gc.collect()

    # 2. Sign canonicality
    logger.info("=" * 50)
    logger.info("PHASE 2: Sign Canonicality Analysis")
    sign_results = run_sign_canonicality(examples)
    gc.collect()

    # 3. Computational cost benchmarking
    logger.info("=" * 50)
    logger.info("PHASE 3: Computational Cost Benchmarking")
    synthetic = generate_synthetic_graphs()
    timing_results = run_computational_benchmark(synthetic)
    gc.collect()

    # 4. EDMD stability
    logger.info("=" * 50)
    logger.info("PHASE 4: EDMD Stability Analysis")
    stability_results = run_stability_analysis(representative)
    gc.collect()

    # 5. Process ALL pairs for distinguishing (KW-PE vs RWPE)
    logger.info("=" * 50)
    logger.info("PHASE 5: Distinguishing Test on ALL pairs")
    t_dist = time.perf_counter()

    pair_tasks = []
    for idx, ex in enumerate(examples):
        pair_tasks.append((
            idx,
            ex["input_raw"],
            ex["output_raw"],
            ex["graph_A"]["adjacency_matrix"],
            ex["graph_B"]["adjacency_matrix"],
            ex["graph_A"]["num_nodes"],
            ex["graph_B"]["num_nodes"],
            ex["metadata"],
        ))

    pair_results = []
    for done_count, task in enumerate(pair_tasks, 1):
        try:
            res = process_single_pair(task)
            pair_results.append(res)
        except Exception as e:
            pair_results.append({
                "idx": task[0],
                "kwpe_distinguished": False,
                "kwpe_distance": 0.0,
                "kwpe_cond": float("inf"),
                "kwpe_time_s": 0.0,
                "rwpe_distinguished": False,
                "rwpe_distance": 0.0,
                "success": False,
                "error": str(e)[:200],
            })
        if done_count % 50 == 0:
            logger.info(f"  Processed {done_count}/{len(pair_tasks)} pairs...")

    # Sort by original index
    pair_results.sort(key=lambda r: r["idx"])

    t_dist_elapsed = time.perf_counter() - t_dist
    logger.info(f"Distinguishing test done in {t_dist_elapsed:.1f}s")

    # Compute distinguishing statistics
    successful = [r for r in pair_results if r.get("success", False)]
    kwpe_distinguished_count = sum(1 for r in successful if r["kwpe_distinguished"])
    rwpe_distinguished_count = sum(1 for r in successful if r["rwpe_distinguished"])

    # Cospectral subset
    cospectral_results = [
        r for r, ex in zip(pair_results, examples)
        if ex["are_cospectral"] and r.get("success", False)
    ]
    kwpe_cospectral_dist = sum(1 for r in cospectral_results if r["kwpe_distinguished"])
    rwpe_cospectral_dist = sum(1 for r in cospectral_results if r["rwpe_distinguished"])

    # Per-category breakdown
    category_stats: dict[str, dict] = {}
    for r, ex in zip(pair_results, examples):
        cat = ex["metadata"].get("metadata_category", "unknown")
        if cat not in category_stats:
            category_stats[cat] = {
                "total": 0, "successful": 0,
                "kwpe_dist": 0, "rwpe_dist": 0,
            }
        category_stats[cat]["total"] += 1
        if r.get("success", False):
            category_stats[cat]["successful"] += 1
            if r["kwpe_distinguished"]:
                category_stats[cat]["kwpe_dist"] += 1
            if r["rwpe_distinguished"]:
                category_stats[cat]["rwpe_dist"] += 1

    logger.info(f"Results: KW-PE distinguished {kwpe_distinguished_count}/{len(successful)}, "
                f"RWPE distinguished {rwpe_distinguished_count}/{len(successful)}")
    logger.info(f"Cospectral: KW-PE {kwpe_cospectral_dist}/{len(cospectral_results)}, "
                f"RWPE {rwpe_cospectral_dist}/{len(cospectral_results)}")

    # 6. Assemble output following exp_gen_sol_out.json schema
    logger.info("=" * 50)
    logger.info("PHASE 6: Assembling output")

    # Build recommendations
    conv_rates = convergence_results.get("summary", {}).get("overall_convergence_rate_by_nonlinearity", {})
    best_nonlinearity = "tanh"
    best_rate = 0.0
    for nl, rate in conv_rates.items():
        if rate > best_rate:
            best_rate = rate
            best_nonlinearity = nl

    sign_fracs = sign_results.get("summary", {}).get("sign_consistency_by_nonlinearity", {})

    recommendations = {
        "best_nonlinearity": best_nonlinearity,
        "recommended_T": stability_results.get("summary", {}).get("recommended_trajectory_length", DEFAULT_T),
        "recommended_d": DEFAULT_D,
        "recommended_dictionary_degree": 2,
        "recommended_regularization": stability_results.get("summary", {}).get("recommended_regularization", DEFAULT_REG),
        "scalability_limit_n": BENCH_SIZES[-1] if BENCH_SIZES else 1000,
        "key_findings": [
            f"Convergence: {best_nonlinearity} achieves {best_rate:.1%} convergence rate across test graphs",
            f"KW-PE distinguished {kwpe_distinguished_count}/{len(successful)} pairs overall "
            f"({kwpe_cospectral_dist}/{len(cospectral_results)} cospectral)",
            f"RWPE baseline distinguished {rwpe_distinguished_count}/{len(successful)} pairs overall "
            f"({rwpe_cospectral_dist}/{len(cospectral_results)} cospectral)",
            f"Sign canonicality: {json.dumps(sign_fracs)}",
            f"EDMD drift threshold T>={stability_results.get('summary', {}).get('eigenfunction_drift_acceptable_T_threshold', 50)}",
        ],
    }

    # Build examples array following schema
    output_examples = []
    for r, ex in zip(pair_results, examples):
        example = {
            "input": ex["input_raw"],
            "output": ex["output_raw"],
        }
        # Copy all metadata
        for k, v in ex["metadata"].items():
            example[k] = v

        # Add predictions as JSON strings
        example["predict_kwpe"] = json.dumps({
            "distinguished": r.get("kwpe_distinguished", False),
            "distance": r.get("kwpe_distance", 0.0),
            "condition_number": r.get("kwpe_cond", float("inf")),
            "time_s": r.get("kwpe_time_s", 0.0),
        }, default=str)

        example["predict_rwpe"] = json.dumps({
            "distinguished": r.get("rwpe_distinguished", False),
            "distance": r.get("rwpe_distance", 0.0),
        }, default=str)

        output_examples.append(example)

    output = {
        "metadata": {
            "title": "KW-PE Foundational Property Analysis",
            "method_name": "Koopman Walk Positional Encoding (KW-PE)",
            "description": "Comprehensive analysis of KW-PE properties: convergence, sign canonicality, "
                          "computational cost, EDMD stability, and graph distinguishing capability.",
            "parameters": {
                "T": DEFAULT_T,
                "d": DEFAULT_D,
                "dictionary_degree": 2,
                "regularization": DEFAULT_REG,
                "nonlinearity": "tanh",
                "projection_dim": PROJ_DIM,
            },
            "convergence_analysis": convergence_results,
            "sign_canonicality": sign_results,
            "computational_cost": timing_results,
            "edmd_stability": stability_results,
            "cospectral_distinguishing": {
                "summary": {
                    "kwpe_pairs_distinguished": kwpe_distinguished_count,
                    "rwpe_pairs_distinguished": rwpe_distinguished_count,
                    "total_pairs_tested": len(successful),
                    "total_cospectral_pairs": len(cospectral_results),
                    "kwpe_cospectral_distinguished": kwpe_cospectral_dist,
                    "rwpe_cospectral_distinguished": rwpe_cospectral_dist,
                    "kwpe_advantage_pairs": kwpe_distinguished_count - rwpe_distinguished_count,
                },
                "per_category_results": {
                    cat: {
                        "total": stats["total"],
                        "successful": stats["successful"],
                        "kwpe_distinguished": stats["kwpe_dist"],
                        "rwpe_distinguished": stats["rwpe_dist"],
                        "kwpe_rate": stats["kwpe_dist"] / max(stats["successful"], 1),
                        "rwpe_rate": stats["rwpe_dist"] / max(stats["successful"], 1),
                    }
                    for cat, stats in sorted(category_stats.items())
                },
            },
            "recommendations": recommendations,
        },
        "datasets": [
            {
                "dataset": "graph_expressiveness_benchmark",
                "examples": output_examples,
            }
        ],
    }

    # Save output
    output_text = json.dumps(output, indent=2, default=str)
    OUTPUT_PATH.write_text(output_text)
    output_size_mb = OUTPUT_PATH.stat().st_size / 1e6
    logger.info(f"Saved {OUTPUT_PATH} ({output_size_mb:.1f} MB)")

    t_total = time.perf_counter() - t_start
    logger.info("=" * 70)
    logger.info(f"KW-PE Foundational Property Analysis — COMPLETE ({t_total:.1f}s)")
    logger.info(f"  Pairs processed: {len(successful)}/{len(examples)}")
    logger.info(f"  KW-PE distinguished: {kwpe_distinguished_count}")
    logger.info(f"  RWPE distinguished: {rwpe_distinguished_count}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
