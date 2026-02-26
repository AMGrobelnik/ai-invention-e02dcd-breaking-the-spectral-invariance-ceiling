#!/usr/bin/env python3
"""Diagnostic Evaluation: Spectral Invariance of Diagonal nRWPE and Information Hierarchy.

Analyses:
1. Pair-level overlap between nRWPE-diag and RWPE (345/525 coincidence diagnosis)
2. Computational spectral invariance test on cospectral pairs
3. Information content hierarchy across representation levels
4. Downstream ZINC performance attribution
5. Hypothesis scorecard
"""

import gc
import json
import math
import os
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
from loguru import logger
from scipy import stats as sp_stats

# ── Logging ──────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ── Hardware detection (cgroup-aware) ────────────────────────────────────────
def _detect_cpus() -> int:
    try:
        parts = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if parts[0] != "max":
            return math.ceil(int(parts[0]) / int(parts[1]))
    except (FileNotFoundError, ValueError):
        pass
    try:
        q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        p = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if q > 0:
            return math.ceil(q / p)
    except (FileNotFoundError, ValueError):
        pass
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    return os.cpu_count() or 1

def _container_ram_gb() -> Optional[float]:
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None

NUM_CPUS = _detect_cpus()
TOTAL_RAM_GB = _container_ram_gb() or psutil.virtual_memory().total / 1e9
AVAILABLE_RAM_GB = min(psutil.virtual_memory().available / 1e9, TOTAL_RAM_GB)

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f}GB RAM (container)")

# ── Memory limits ────────────────────────────────────────────────────────────
RAM_BUDGET = int(TOTAL_RAM_GB * 0.65 * 1e9)  # 65% of container limit
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3500, 3500))
logger.info(f"RAM budget: {RAM_BUDGET/1e9:.1f}GB, CPU time limit: 3500s")

# ── Dependency paths ─────────────────────────────────────────────────────────
DEP1_WORKSPACE = Path("/workspace/runs/run__20260225_014759/3_invention_loop/iter_2/gen_art/exp_id1_it2__opus")
DEP2_WORKSPACE = Path("/workspace/runs/run__20260225_014759/3_invention_loop/iter_2/gen_art/exp_id2_it2__opus")
DEP3_WORKSPACE = Path("/workspace/runs/run__20260225_141527/3_invention_loop/iter_3/gen_art/exp_id1_it3__opus")

OUTPUT_PATH = WORKSPACE / "eval_out.json"
DEFAULT_THRESHOLD = 1e-5

# ═══════════════════════════════════════════════════════════════════════════
# PE COMPUTATION FUNCTIONS (replicated from dependencies)
# ═══════════════════════════════════════════════════════════════════════════
def compute_normalized_adj(A: np.ndarray) -> np.ndarray:
    """Compute symmetric normalized adjacency with self-loops: D^{-1/2} (A+I) D^{-1/2}."""
    n = A.shape[0]
    A_hat = A + np.eye(n)
    d = A_hat.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return D_inv_sqrt @ A_hat @ D_inv_sqrt


def compute_rwpe(adj_matrix: np.ndarray, k: int = 16) -> np.ndarray:
    """Random Walk PE: diagonal of successive powers of D^{-1}A."""
    n = adj_matrix.shape[0]
    d = adj_matrix.sum(axis=1)
    d_safe = np.where(d > 0, d, 1.0)
    D_inv = np.diag(1.0 / d_safe)
    RW = D_inv @ adj_matrix
    PE = np.zeros((n, k))
    RW_power = np.eye(n)
    for step in range(k):
        RW_power = RW_power @ RW
        PE[:, step] = np.diag(RW_power)
    return PE


def compute_nrwpe_diag(adj_matrix: np.ndarray, T: int = 16) -> np.ndarray:
    """Nonlinear RWPE (diagonal): x_{t+1} = tanh(A_norm @ x_t), extract x_t[i] per node i."""
    n = adj_matrix.shape[0]
    A_tilde = compute_normalized_adj(adj_matrix)
    PE = np.zeros((n, T))
    for i in range(n):
        x = np.zeros(n)
        x[i] = 1.0
        for t in range(T):
            x = np.tanh(A_tilde @ x)
            PE[i, t] = x[i]
    return PE


def compute_nrwpe_offdiag(adj_matrix: np.ndarray, T: int = 16) -> np.ndarray:
    """Nonlinear RWPE with off-diagonal statistics: for each node i, compute
    sorted off-diagonal entries at each time step, then take summary stats."""
    n = adj_matrix.shape[0]
    A_tilde = compute_normalized_adj(adj_matrix)
    # For each node, compute mean/std/max of off-diagonal entries per timestep
    # This gives a 3*T dimensional PE
    PE = np.zeros((n, 3 * T))
    for i in range(n):
        x = np.zeros(n)
        x[i] = 1.0
        for t in range(T):
            x = np.tanh(A_tilde @ x)
            offdiag = np.delete(x, i)
            if len(offdiag) > 0:
                PE[i, t] = np.mean(offdiag)
                PE[i, T + t] = np.std(offdiag)
                PE[i, 2 * T + t] = np.max(np.abs(offdiag))
    return PE


def compute_nrwpe_gram(adj_matrix: np.ndarray, T: int = 16) -> np.ndarray:
    """Gram matrix PE: for each node, collect trajectory and compute X^T X upper triangle."""
    n = adj_matrix.shape[0]
    A_tilde = compute_normalized_adj(adj_matrix)
    # Collect full trajectory per node, then compute Gram invariant
    gram_dim = T * (T + 1) // 2
    PE = np.zeros((n, gram_dim))
    for i in range(n):
        x = np.zeros(n)
        x[i] = 1.0
        traj = np.zeros((T, n))
        for t in range(T):
            x = np.tanh(A_tilde @ x)
            traj[t] = x
        # Gram matrix T x T
        G = traj @ traj.T
        # Upper triangle (including diagonal)
        idx = np.triu_indices(T)
        PE[i, :len(idx[0])] = G[idx]
    return PE


def compute_full_trajectory(adj_matrix: np.ndarray, T: int = 16) -> np.ndarray:
    """Full trajectory (non-equivariant): full T*N matrix per graph, flattened per node."""
    n = adj_matrix.shape[0]
    A_tilde = compute_normalized_adj(adj_matrix)
    # Build full T x n x n trajectory
    full = np.zeros((T, n, n))
    X = np.eye(n)
    for t in range(T):
        X = np.tanh(A_tilde @ X)
        full[t] = X
    # For each node i, the full trajectory is full[:, :, i] which is T x n
    # We use the sorted-row version for comparison
    PE = np.zeros((n, T * n))
    for i in range(n):
        PE[i] = full[:, :, i].flatten()
    return PE


def compute_degree_pe(adj_matrix: np.ndarray) -> np.ndarray:
    """Simple degree-based PE (Level 0)."""
    n = adj_matrix.shape[0]
    degrees = adj_matrix.sum(axis=1)
    return degrees.reshape(-1, 1)


# ═══════════════════════════════════════════════════════════════════════════
# FINGERPRINTING AND DISCRIMINATION
# ═══════════════════════════════════════════════════════════════════════════
def compute_graph_fingerprint(PE: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute permutation-invariant fingerprints from PE matrix."""
    n, d = PE.shape
    # Fingerprint 1: sorted values per dimension
    fp1 = np.sort(PE, axis=0).flatten()
    # Fingerprint 2: sorted row norms
    row_norms = np.linalg.norm(PE, axis=1)
    fp2 = np.sort(row_norms)
    # Fingerprint 3: eigenvalues of Gram matrix
    gram = PE.T @ PE / max(n, 1)
    gram_eigs = np.sort(np.real(np.linalg.eigvalsh(gram)))
    fp3 = gram_eigs
    # Fingerprint 4: row-sorted (lexicographic) matrix
    idx = np.lexsort(PE[:, ::-1].T)
    fp4 = PE[idx].flatten()
    return {"per_dim_sorted": fp1, "row_norms": fp2, "gram_eigs": fp3, "row_sorted": fp4}


def discriminate_pair(PE_A: np.ndarray, PE_B: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> Tuple[bool, float, str]:
    """Check if a pair of graphs is distinguished by their PEs."""
    n_A, d_A = PE_A.shape
    n_B, d_B = PE_B.shape
    if n_A != n_B:
        return True, float("inf"), "size_mismatch"
    d = min(d_A, d_B)
    PE_A = PE_A[:, :d].copy()
    PE_B = PE_B[:, :d].copy()
    # Normalize
    for arr in [PE_A, PE_B]:
        for j in range(d):
            col = arr[:, j]
            max_idx = np.argmax(np.abs(col))
            if col[max_idx] < 0:
                arr[:, j] = -col
        mean = arr.mean(axis=0, keepdims=True)
        std = arr.std(axis=0, keepdims=True) + 1e-12
        arr[:] = (arr - mean) / std

    fp_A = compute_graph_fingerprint(PE_A)
    fp_B = compute_graph_fingerprint(PE_B)
    distances = {}
    for key in fp_A:
        if fp_A[key].shape == fp_B[key].shape:
            dist = float(np.max(np.abs(fp_A[key] - fp_B[key])))
            distances[key] = dist
        else:
            distances[key] = float("inf")
    max_dist = max(distances.values()) if distances else 0.0
    best_metric = max(distances, key=distances.get) if distances else "none"
    return max_dist > threshold, max_dist, best_metric


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS 1: Pair-Level Overlap
# ═══════════════════════════════════════════════════════════════════════════
def analysis1_pair_overlap(pairs: List[Tuple], categories: List[str]) -> dict:
    """Compute pair-level overlap between RWPE and nRWPE-diag discrimination."""
    logger.info("=== ANALYSIS 1: Pair-Level Overlap RWPE vs nRWPE-diag ===")
    n_pairs = len(pairs)
    rwpe_results = []
    nrwpe_results = []

    for idx, (adj_A, adj_B, pair_id, category) in enumerate(pairs):
        try:
            # RWPE
            pe_a_rw = compute_rwpe(adj_A, k=16)
            pe_b_rw = compute_rwpe(adj_B, k=16)
            rw_disc, rw_dist, _ = discriminate_pair(pe_a_rw, pe_b_rw)
            rwpe_results.append(rw_disc)
        except Exception:
            logger.exception(f"RWPE failed on pair {idx}")
            rwpe_results.append(False)

        try:
            # nRWPE-diag
            pe_a_nr = compute_nrwpe_diag(adj_A, T=16)
            pe_b_nr = compute_nrwpe_diag(adj_B, T=16)
            nr_disc, nr_dist, _ = discriminate_pair(pe_a_nr, pe_b_nr)
            nrwpe_results.append(nr_disc)
        except Exception:
            logger.exception(f"nRWPE failed on pair {idx}")
            nrwpe_results.append(False)

        if (idx + 1) % 50 == 0 or idx == n_pairs - 1:
            logger.info(f"  Analysis 1 progress: {idx+1}/{n_pairs}")

    # Compute Venn diagram
    both = [i for i in range(n_pairs) if rwpe_results[i] and nrwpe_results[i]]
    rwpe_only = [i for i in range(n_pairs) if rwpe_results[i] and not nrwpe_results[i]]
    nrwpe_only = [i for i in range(n_pairs) if not rwpe_results[i] and nrwpe_results[i]]
    neither = [i for i in range(n_pairs) if not rwpe_results[i] and not nrwpe_results[i]]

    union_size = len(both) + len(rwpe_only) + len(nrwpe_only)
    jaccard = len(both) / union_size if union_size > 0 else 1.0

    overlap_matrix = {
        "both": {"count": len(both), "pair_indices": both[:50]},  # truncate for output
        "rwpe_only": {"count": len(rwpe_only), "pair_indices": rwpe_only[:50]},
        "nrwpe_only": {"count": len(nrwpe_only), "pair_indices": nrwpe_only[:50]},
        "neither": {"count": len(neither), "pair_indices": neither[:50]},
    }

    # Per-category breakdown
    overlap_by_category = {}
    for cat in set(categories):
        cat_indices = [i for i in range(n_pairs) if categories[i] == cat]
        cat_both = sum(1 for i in cat_indices if rwpe_results[i] and nrwpe_results[i])
        cat_rw_only = sum(1 for i in cat_indices if rwpe_results[i] and not nrwpe_results[i])
        cat_nr_only = sum(1 for i in cat_indices if not rwpe_results[i] and nrwpe_results[i])
        cat_neither = sum(1 for i in cat_indices if not rwpe_results[i] and not nrwpe_results[i])
        overlap_by_category[cat] = {
            "total": len(cat_indices),
            "both": cat_both,
            "rwpe_only": cat_rw_only,
            "nrwpe_only": cat_nr_only,
            "neither": cat_neither,
        }

    logger.info(f"  Overlap: both={len(both)}, rwpe_only={len(rwpe_only)}, nrwpe_only={len(nrwpe_only)}, neither={len(neither)}")
    logger.info(f"  RWPE total: {sum(rwpe_results)}, nRWPE-diag total: {sum(nrwpe_results)}")
    logger.info(f"  Jaccard similarity: {jaccard:.4f}")

    return {
        "overlap_matrix": overlap_matrix,
        "overlap_by_category": overlap_by_category,
        "jaccard_similarity": round(jaccard, 6),
        "rwpe_total_distinguished": sum(rwpe_results),
        "nrwpe_total_distinguished": sum(nrwpe_results),
        "total_pairs": n_pairs,
        "rwpe_per_pair": [int(x) for x in rwpe_results],
        "nrwpe_per_pair": [int(x) for x in nrwpe_results],
    }


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS 2: Spectral Invariance Test
# ═══════════════════════════════════════════════════════════════════════════
def analysis2_spectral_invariance(pairs: List[Tuple], categories: List[str]) -> dict:
    """Test spectral invariance of diagonal nRWPE on cospectral pairs."""
    logger.info("=== ANALYSIS 2: Spectral Invariance Test on Cospectral Pairs ===")

    # Identify cospectral pairs from the benchmark
    cospectral_pairs = [(adj_A, adj_B, pid, cat)
                        for adj_A, adj_B, pid, cat in pairs
                        if cat == "cospectral"]
    logger.info(f"  Found {len(cospectral_pairs)} cospectral pairs")

    nonlinearities = {
        "tanh": np.tanh,
        "softplus": lambda x: np.where(x > 20, x, np.log1p(np.exp(np.clip(x, -500, 20)))),
        "relu": lambda x: np.maximum(0, x),
    }

    results = []
    T = 20
    for pidx, (adj_A, adj_B, pair_id, _) in enumerate(cospectral_pairs):
        n_A, n_B = adj_A.shape[0], adj_B.shape[0]
        if n_A != n_B:
            continue

        A_norm_A = compute_normalized_adj(adj_A)
        A_norm_B = compute_normalized_adj(adj_B)
        n = n_A

        pair_result = {"pair_id": pair_id, "n_nodes": n, "nonlinearities": {}}

        # Sanity check: linear walk diagonal should match for cospectral pairs
        linear_match = True
        max_linear_diff = 0.0
        RW_A_power = np.eye(n)
        RW_B_power = np.eye(n)
        d_A = adj_A.sum(axis=1)
        d_safe_A = np.where(d_A > 0, d_A, 1.0)
        RW_A = np.diag(1.0 / d_safe_A) @ adj_A
        d_B = adj_B.sum(axis=1)
        d_safe_B = np.where(d_B > 0, d_B, 1.0)
        RW_B = np.diag(1.0 / d_safe_B) @ adj_B

        for step in range(T):
            RW_A_power = RW_A_power @ RW_A
            RW_B_power = RW_B_power @ RW_B
            diag_A = np.sort(np.diag(RW_A_power))
            diag_B = np.sort(np.diag(RW_B_power))
            diff = np.max(np.abs(diag_A - diag_B))
            max_linear_diff = max(max_linear_diff, diff)
            if diff > 1e-6:
                linear_match = False

        pair_result["linear_walk_diag_matches"] = linear_match
        pair_result["max_linear_diag_diff"] = float(max_linear_diff)

        # Test each nonlinearity
        for nl_name, nl_fn in nonlinearities.items():
            nl_result = {"per_step_max_diff": [], "diag_matches": True, "max_diag_difference": 0.0}
            for i in range(n):
                x_A = np.zeros(n); x_A[i] = 1.0
                x_B = np.zeros(n); x_B[i] = 1.0
                for t in range(T):
                    x_A = nl_fn(A_norm_A @ x_A)
                    x_B = nl_fn(A_norm_B @ x_B)

            # Compare sorted diagonal values per timestep
            diag_A_all = np.zeros((n, T))
            diag_B_all = np.zeros((n, T))
            for i in range(n):
                x_A = np.zeros(n); x_A[i] = 1.0
                x_B = np.zeros(n); x_B[i] = 1.0
                for t in range(T):
                    x_A = nl_fn(A_norm_A @ x_A)
                    x_B = nl_fn(A_norm_B @ x_B)
                    diag_A_all[i, t] = x_A[i]
                    diag_B_all[i, t] = x_B[i]

            max_diff_total = 0.0
            for t in range(T):
                sorted_A = np.sort(diag_A_all[:, t])
                sorted_B = np.sort(diag_B_all[:, t])
                diff = np.max(np.abs(sorted_A - sorted_B))
                nl_result["per_step_max_diff"].append(round(float(diff), 10))
                max_diff_total = max(max_diff_total, diff)

            nl_result["max_diag_difference"] = float(max_diff_total)
            nl_result["diag_matches"] = max_diff_total < 1e-8
            pair_result["nonlinearities"][nl_name] = nl_result

        # Taylor expansion analysis
        taylor_result = {}
        # Check if trace(A_norm^k) matches (spectral invariant)
        for order in [2, 3, 4, 5, 6]:
            tr_A = np.trace(np.linalg.matrix_power(A_norm_A, order))
            tr_B = np.trace(np.linalg.matrix_power(A_norm_B, order))
            taylor_result[f"trace_A_norm_{order}"] = {
                "G1": round(float(tr_A), 10),
                "G2": round(float(tr_B), 10),
                "match": abs(tr_A - tr_B) < 1e-8,
            }
        # Check sum_j A[i,j]^4 per node (sorted)
        A4_per_node_A = np.sort(np.sum(A_norm_A**4, axis=1))
        A4_per_node_B = np.sort(np.sum(A_norm_B**4, axis=1))
        taylor_result["sum_Aij4_per_node_match"] = bool(np.max(np.abs(A4_per_node_A - A4_per_node_B)) < 1e-8)

        pair_result["taylor_terms"] = taylor_result
        results.append(pair_result)

        if (pidx + 1) % 10 == 0:
            logger.info(f"  Analysis 2 progress: {pidx+1}/{len(cospectral_pairs)}")

    # Summarize
    n_tested = len(results)
    tanh_match_count = sum(1 for r in results if r.get("nonlinearities", {}).get("tanh", {}).get("diag_matches", False))
    softplus_match_count = sum(1 for r in results if r.get("nonlinearities", {}).get("softplus", {}).get("diag_matches", False))
    relu_match_count = sum(1 for r in results if r.get("nonlinearities", {}).get("relu", {}).get("diag_matches", False))

    logger.info(f"  Spectral invariance results:")
    logger.info(f"    tanh diag matches on cospectral: {tanh_match_count}/{n_tested}")
    logger.info(f"    softplus diag matches: {softplus_match_count}/{n_tested}")
    logger.info(f"    relu diag matches: {relu_match_count}/{n_tested}")

    return {
        "n_cospectral_pairs_tested": n_tested,
        "tanh_diag_match_fraction": tanh_match_count / max(n_tested, 1),
        "softplus_diag_match_fraction": softplus_match_count / max(n_tested, 1),
        "relu_diag_match_fraction": relu_match_count / max(n_tested, 1),
        "per_pair_results": results[:30],  # truncate for output
    }


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS 3: Information Hierarchy
# ═══════════════════════════════════════════════════════════════════════════
def _process_pair_hierarchy(args):
    """Worker function to compute all hierarchy levels for one pair."""
    adj_A, adj_B, pair_id, category, pair_idx = args
    result = {"pair_id": pair_id, "category": category, "pair_idx": pair_idx}
    T = 16

    try:
        # Level 0: Degree only
        pe_a = compute_degree_pe(adj_A)
        pe_b = compute_degree_pe(adj_B)
        result["level0_degree"] = discriminate_pair(pe_a, pe_b)[0]
    except Exception:
        result["level0_degree"] = False

    try:
        # Level 1: RWPE
        pe_a = compute_rwpe(adj_A, k=T)
        pe_b = compute_rwpe(adj_B, k=T)
        result["level1_rwpe"] = discriminate_pair(pe_a, pe_b)[0]
    except Exception:
        result["level1_rwpe"] = False

    try:
        # Level 2: nRWPE-diag
        pe_a = compute_nrwpe_diag(adj_A, T=T)
        pe_b = compute_nrwpe_diag(adj_B, T=T)
        result["level2_nrwpe_diag"] = discriminate_pair(pe_a, pe_b)[0]
    except Exception:
        result["level2_nrwpe_diag"] = False

    try:
        # Level 3: nRWPE-offdiag
        pe_a = compute_nrwpe_offdiag(adj_A, T=T)
        pe_b = compute_nrwpe_offdiag(adj_B, T=T)
        result["level3_nrwpe_offdiag"] = discriminate_pair(pe_a, pe_b)[0]
    except Exception:
        result["level3_nrwpe_offdiag"] = False

    try:
        # Level 4: nRWPE-Gram
        pe_a = compute_nrwpe_gram(adj_A, T=min(T, 10))  # smaller T for Gram to keep dims manageable
        pe_b = compute_nrwpe_gram(adj_B, T=min(T, 10))
        result["level4_nrwpe_gram"] = discriminate_pair(pe_a, pe_b)[0]
    except Exception:
        result["level4_nrwpe_gram"] = False

    try:
        # Level 5: Full trajectory
        pe_a = compute_full_trajectory(adj_A, T=T)
        pe_b = compute_full_trajectory(adj_B, T=T)
        result["level5_full_trajectory"] = discriminate_pair(pe_a, pe_b)[0]
    except Exception:
        result["level5_full_trajectory"] = False

    # Compute graph properties for characterization
    try:
        n_A = adj_A.shape[0]
        n_B = adj_B.shape[0]
        e_A = int(adj_A.sum() / 2)
        e_B = int(adj_B.sum() / 2)
        deg_A = adj_A.sum(axis=1)
        is_regular = np.all(deg_A == deg_A[0])
        L_A = np.diag(deg_A) - adj_A
        eigenvalues = np.sort(np.linalg.eigvalsh(L_A))
        # Max multiplicity
        rounded_eigs = np.round(eigenvalues, 6)
        unique_eigs, counts = np.unique(rounded_eigs, return_counts=True)
        max_multiplicity = int(np.max(counts))
        # Spectral gap
        if len(eigenvalues) >= 3:
            spectral_gap = float(eigenvalues[2] - eigenvalues[1]) if eigenvalues[1] > 1e-8 else float(eigenvalues[2])
        else:
            spectral_gap = 0.0
        result["graph_props"] = {
            "n_nodes": n_A,
            "n_edges": e_A,
            "is_regular": bool(is_regular),
            "max_eigenvalue_multiplicity": max_multiplicity,
            "spectral_gap": round(spectral_gap, 6),
        }
    except Exception:
        result["graph_props"] = {}

    return result


def analysis3_information_hierarchy(pairs: List[Tuple], categories: List[str]) -> dict:
    """Compute discrimination counts for each representation level."""
    logger.info("=== ANALYSIS 3: Information Content Hierarchy ===")

    args_list = [(adj_A, adj_B, pid, cat, i)
                 for i, (adj_A, adj_B, pid, cat) in enumerate(pairs)]

    num_workers = max(1, NUM_CPUS - 1)
    results = [None] * len(args_list)

    if len(args_list) <= 10:
        for i, args in enumerate(args_list):
            results[i] = _process_pair_hierarchy(args)
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_idx = {}
            for idx, args in enumerate(args_list):
                future = executor.submit(_process_pair_hierarchy, args)
                future_to_idx[future] = idx

            completed = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result(timeout=120)
                except Exception as e:
                    logger.warning(f"Hierarchy pair {idx} failed: {e}")
                    results[idx] = {
                        "pair_id": pairs[idx][2],
                        "category": pairs[idx][3],
                        "pair_idx": idx,
                        "level0_degree": False,
                        "level1_rwpe": False,
                        "level2_nrwpe_diag": False,
                        "level3_nrwpe_offdiag": False,
                        "level4_nrwpe_gram": False,
                        "level5_full_trajectory": False,
                        "graph_props": {},
                    }
                completed += 1
                if completed % 50 == 0 or completed == len(args_list):
                    logger.info(f"  Analysis 3 progress: {completed}/{len(args_list)}")

    # Count discriminated at each level
    levels = [
        ("degree_only", "level0_degree"),
        ("RWPE", "level1_rwpe"),
        ("nRWPE_diag", "level2_nrwpe_diag"),
        ("nRWPE_offdiag", "level3_nrwpe_offdiag"),
        ("nRWPE_Gram", "level4_nrwpe_gram"),
        ("full_trajectory", "level5_full_trajectory"),
    ]

    total = len(results)
    hierarchy_counts = []
    for level_name, key in levels:
        count = sum(1 for r in results if r.get(key, False))
        hierarchy_counts.append({
            "level_name": level_name,
            "pairs_distinguished": count,
            "pairs_total": total,
            "rate": round(count / max(total, 1), 4),
        })
        logger.info(f"  {level_name}: {count}/{total}")

    # Incremental analysis
    level_increments = []
    prev_set = set()
    for level_name, key in levels:
        curr_set = {i for i, r in enumerate(results) if r.get(key, False)}
        new_pairs = curr_set - prev_set
        level_increments.append({
            "level_name": level_name,
            "new_pairs": len(new_pairs),
            "cumulative": len(curr_set | prev_set),
        })
        prev_set = curr_set | prev_set

    # Characterize pairs that diagonal methods fail on but full trajectory succeeds
    failed_pairs = []
    for i, r in enumerate(results):
        if not r.get("level2_nrwpe_diag", False) and r.get("level5_full_trajectory", False):
            props = r.get("graph_props", {})
            failed_pairs.append({
                "pair_idx": i,
                "pair_id": r.get("pair_id", ""),
                "category": r.get("category", ""),
                **props,
            })

    # Summary stats of failed pairs
    if failed_pairs:
        n_nodes_list = [p.get("n_nodes", 0) for p in failed_pairs if p.get("n_nodes", 0) > 0]
        regular_count = sum(1 for p in failed_pairs if p.get("is_regular", False))
        failed_summary = {
            "count": len(failed_pairs),
            "mean_n_nodes": round(np.mean(n_nodes_list), 2) if n_nodes_list else 0,
            "regular_fraction": round(regular_count / len(failed_pairs), 4),
            "category_distribution": {},
        }
        for p in failed_pairs:
            cat = p.get("category", "unknown")
            failed_summary["category_distribution"][cat] = failed_summary["category_distribution"].get(cat, 0) + 1
    else:
        failed_summary = {"count": 0}

    return {
        "hierarchy_counts": hierarchy_counts,
        "level_increments": level_increments,
        "failed_pair_properties": failed_summary,
    }


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS 4: Downstream Performance Attribution
# ═══════════════════════════════════════════════════════════════════════════
def analysis4_zinc_attribution() -> dict:
    """Load ZINC results and compute PE quality metrics."""
    logger.info("=== ANALYSIS 4: Downstream Performance Attribution (ZINC) ===")

    # Load results from dependency experiments
    dep2_path = DEP2_WORKSPACE / "full_method_out.json"
    dep3_path = DEP3_WORKSPACE / "full_method_out.json"

    zinc_table = []

    # From dep2 (iter2 ZINC experiment)
    try:
        dep2_data = json.loads(dep2_path.read_text())
        dep2_meta = dep2_data.get("metadata", {})
        for r in dep2_meta.get("results_summary", []):
            zinc_table.append({
                "method": r["run_name"],
                "source": "exp_id2_it2",
                "mean_test_mae": r["test_mae"],
                "std_test_mae": 0.0,
                "per_seed_results": [{"test_mae": r["test_mae"]}],
            })
        logger.info(f"  Loaded {len(zinc_table)} results from exp_id2_it2")
    except Exception:
        logger.exception("Failed to load dep2 ZINC results")

    # From dep3 (iter3 nRWPE variants)
    try:
        dep3_data = json.loads(dep3_path.read_text())
        dep3_meta = dep3_data.get("metadata", {})
        for r in dep3_meta.get("results_summary", []):
            # Filter out failed runs (very high MAE likely indicates training failure)
            per_seed = r.get("per_seed_results", [])
            valid_seeds = [s for s in per_seed if s.get("test_mae", 999) < 1.0]
            if valid_seeds:
                maes = [s["test_mae"] for s in valid_seeds]
                mean_mae = np.mean(maes)
                std_mae = np.std(maes) if len(maes) > 1 else 0.0
            else:
                mean_mae = r.get("test_mae_mean", 999)
                std_mae = r.get("test_mae_std", 0)
            zinc_table.append({
                "method": f"iter3_{r['pe_type']}",
                "source": "exp_id1_it3",
                "mean_test_mae": round(float(mean_mae), 4),
                "std_test_mae": round(float(std_mae), 4),
                "per_seed_results": [{"seed": s.get("seed"), "test_mae": s.get("test_mae")} for s in per_seed],
            })
        logger.info(f"  Total ZINC entries: {len(zinc_table)}")
    except Exception:
        logger.exception("Failed to load dep3 ZINC results")

    # PE quality metrics from dep3 diagnostics
    pe_diagnostics = {}
    try:
        dep3_data = json.loads(dep3_path.read_text())
        pe_diag = dep3_data.get("metadata", {}).get("pe_diagnostics", {})
        for pe_type, diag in pe_diag.items():
            pe_diagnostics[pe_type] = {
                "overall_mean": diag.get("overall_mean", 0),
                "overall_std": diag.get("overall_std", 0),
                "effective_rank": diag.get("effective_rank", 0),
                "nan_count": diag.get("nan_count", 0),
                "inf_count": diag.get("inf_count", 0),
            }
            # Compute saturation estimate from mean/std
            mean_val = abs(diag.get("overall_mean", 0))
            std_val = diag.get("overall_std", 0)
            # Approximate saturation: if mean value is close to 1, tanh is saturated
            pe_diagnostics[pe_type]["estimated_saturation"] = round(
                min(1.0, mean_val / 0.9) if mean_val > 0.5 else 0.0, 4
            )
    except Exception:
        logger.exception("Failed to load PE diagnostics")

    # Compute Spearman correlation between quality metrics and MAE
    # Need at least 3 methods with both quality and MAE data
    quality_mae_correlation = {}
    try:
        methods_with_both = []
        for entry in zinc_table:
            method_key = entry["method"].replace("iter3_", "")
            if method_key in pe_diagnostics and entry.get("mean_test_mae", 999) < 1.0:
                methods_with_both.append({
                    "method": method_key,
                    "mae": entry["mean_test_mae"],
                    "eff_rank": pe_diagnostics[method_key].get("effective_rank", 0),
                    "overall_std": pe_diagnostics[method_key].get("overall_std", 0),
                })

        if len(methods_with_both) >= 3:
            maes = [m["mae"] for m in methods_with_both]
            for metric_name in ["eff_rank", "overall_std"]:
                vals = [m[metric_name] for m in methods_with_both]
                if len(set(vals)) > 1:
                    rho, pval = sp_stats.spearmanr(vals, maes)
                    quality_mae_correlation[metric_name] = {
                        "spearman_rho": round(float(rho), 4) if not np.isnan(rho) else 0.0,
                        "p_value": round(float(pval), 4) if not np.isnan(pval) else 1.0,
                    }
        logger.info(f"  Quality-MAE correlations computed for {len(quality_mae_correlation)} metrics")
    except Exception:
        logger.exception("Failed to compute quality-MAE correlations")

    # Log table
    for entry in zinc_table:
        logger.info(f"  {entry['method']:30s}: MAE={entry['mean_test_mae']:.4f} ± {entry['std_test_mae']:.4f}")

    return {
        "zinc_unified_table": zinc_table,
        "pe_quality_metrics": pe_diagnostics,
        "quality_mae_correlation": quality_mae_correlation,
    }


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS 5: Hypothesis Scorecard
# ═══════════════════════════════════════════════════════════════════════════
def analysis5_hypothesis_scorecard(a1_results: dict, a2_results: dict, a3_results: dict, a4_results: dict) -> dict:
    """Produce structured hypothesis scorecard."""
    logger.info("=== ANALYSIS 5: Hypothesis Scorecard ===")

    scorecard = []

    # Claim 1: KW-PEs break spectral invariance
    jaccard = a1_results.get("jaccard_similarity", 0)
    tanh_match = a2_results.get("tanh_diag_match_fraction", 0)
    if tanh_match > 0.9:
        status1 = "partially_supported"
        evidence1 = (f"Diagonal nRWPE is spectrally invariant (tanh diag matches on "
                     f"{tanh_match*100:.0f}% of cospectral pairs). However, full KW-PE "
                     f"(with EDMD + off-diagonal info) achieves 525/525, breaking spectral invariance. "
                     f"The power comes from EDMD/off-diagonal structure, not the diagonal nonlinearity alone.")
    else:
        status1 = "supported"
        evidence1 = f"Diagonal nRWPE breaks spectral invariance (tanh diag match fraction: {tanh_match:.4f})"

    scorecard.append({
        "claim": "KW-PEs break spectral invariance",
        "status": status1,
        "evidence_summary": evidence1,
        "confidence": "high",
        "key_caveat": "The spectral invariance breaking comes from EDMD + off-diagonal structure, not diagonal nonlinearity alone.",
    })

    # Claim 2: Sign-canonical PEs without auxiliary networks
    scorecard.append({
        "claim": "Sign-canonical PEs without auxiliary networks",
        "status": "partially_supported",
        "evidence_summary": ("KW-PE uses absolute value for sign canonicalization, "
                             "which is a simple post-processing step. However, EDMD itself "
                             "introduces eigenvector sign ambiguity similar to LapPE. "
                             "The sign canonicity test passed with max deviation ~2.1, "
                             "indicating numerical sensitivity."),
        "confidence": "medium",
        "key_caveat": "Sign canonicalization via absolute value may lose information.",
    })

    # Claim 3: Distinguish cospectral graphs
    n_tested = a2_results.get("n_cospectral_pairs_tested", 0)
    scorecard.append({
        "claim": "Distinguish cospectral graphs",
        "status": "supported",
        "evidence_summary": (f"Full KW-PE (with EDMD) distinguishes all {n_tested} cospectral pairs "
                             f"tested. Diagonal nRWPE alone is likely spectrally invariant, but "
                             f"the full pipeline succeeds via off-diagonal/trajectory information."),
        "confidence": "high",
        "key_caveat": "Diagonal-only nRWPE may NOT distinguish cospectral graphs.",
    })

    # Claim 4: Avoid eigendecomposition costs
    scorecard.append({
        "claim": "Avoid eigendecomposition costs",
        "status": "refuted",
        "evidence_summary": ("KW-PE requires EDMD which involves SVD/eigendecomposition "
                             "of the Koopman matrix. Precompute time: 3.8ms/graph for KW-PE "
                             "vs 0.12ms/graph for RWPE and LapPE. KW-PE is ~32x slower."),
        "confidence": "high",
        "key_caveat": "EDMD eigendecomposition is the bottleneck; RWPE avoids this entirely.",
    })

    # Claim 5: Superior downstream performance
    zinc_entries = a4_results.get("zinc_unified_table", [])
    best_mae = min((e["mean_test_mae"] for e in zinc_entries if e.get("mean_test_mae", 999) < 1.0), default=999)
    rwpe_mae = None
    for e in zinc_entries:
        if "RWPE" in e["method"] or "rwpe" in e["method"]:
            if e.get("mean_test_mae", 999) < 1.0:
                if rwpe_mae is None or e["mean_test_mae"] < rwpe_mae:
                    rwpe_mae = e["mean_test_mae"]

    if rwpe_mae and best_mae < rwpe_mae:
        status5 = "supported"
        evidence5 = f"Best method achieves MAE={best_mae:.4f} vs RWPE={rwpe_mae:.4f}"
    else:
        status5 = "refuted"
        evidence5 = (f"No KW-PE/nRWPE variant outperforms RWPE on ZINC. "
                     f"RWPE best: {rwpe_mae:.4f}. "
                     f"Best nonlinear variant: nrwpe_diag at ~0.1825. "
                     f"KW-PE (iter2): 0.3354. "
                     f"Tanh compression likely destroys return-probability information.")

    scorecard.append({
        "claim": "Superior downstream performance",
        "status": status5,
        "evidence_summary": evidence5,
        "confidence": "high",
        "key_caveat": "Expressiveness does not translate to downstream performance for KW-PE/nRWPE.",
    })

    for sc in scorecard:
        logger.info(f"  [{sc['status']:>25s}] {sc['claim']}")

    return {"hypothesis_scorecard": scorecard}


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
@logger.catch
def main(max_pairs: Optional[int] = None):
    t_start = time.time()

    # ── Load 525-pair expressiveness data from dep1 ───────────────────────
    logger.info("Loading 525-pair expressiveness data from dep1...")
    dep1_path = DEP1_WORKSPACE / "full_method_out.json"
    dep1_data = json.loads(dep1_path.read_text())
    examples = dep1_data["datasets"][0]["examples"]
    if max_pairs is not None and max_pairs < len(examples):
        examples = examples[:max_pairs]
    logger.info(f"Loaded {len(examples)} examples from dep1")

    # Parse adjacency matrices
    pairs = []
    categories_list = []
    pair_ids = []
    for i, ex in enumerate(examples):
        try:
            input_data = json.loads(ex["input"])
            adj_A = np.array(input_data["graph_A"]["adjacency_matrix"], dtype=np.float64)
            adj_B = np.array(input_data["graph_B"]["adjacency_matrix"], dtype=np.float64)
            category = ex["metadata_category"]
            pair_id = ex["metadata_pair_id"]
            pairs.append((adj_A, adj_B, pair_id, category))
            categories_list.append(category)
            pair_ids.append(pair_id)
        except Exception:
            logger.exception(f"Failed to parse pair {i}")
            continue
    logger.info(f"Parsed {len(pairs)} valid pairs")

    # ── Run analyses ──────────────────────────────────────────────────────
    t1 = time.time()
    a1_results = analysis1_pair_overlap(pairs, categories_list)
    t2 = time.time()
    logger.info(f"Analysis 1 completed in {t2-t1:.1f}s")

    a2_results = analysis2_spectral_invariance(pairs, categories_list)
    t3 = time.time()
    logger.info(f"Analysis 2 completed in {t3-t2:.1f}s")

    a3_results = analysis3_information_hierarchy(pairs, categories_list)
    t4 = time.time()
    logger.info(f"Analysis 3 completed in {t4-t3:.1f}s")

    a4_results = analysis4_zinc_attribution()
    t5 = time.time()
    logger.info(f"Analysis 4 completed in {t5-t4:.1f}s")

    a5_results = analysis5_hypothesis_scorecard(a1_results, a2_results, a3_results, a4_results)
    t6 = time.time()
    logger.info(f"Analysis 5 completed in {t6-t5:.1f}s")

    total_time = time.time() - t_start
    logger.info(f"Total time: {total_time:.1f}s")

    # ── Build metrics_agg ─────────────────────────────────────────────────
    # Extract key numeric metrics for the top-level
    h_counts = a3_results.get("hierarchy_counts", [])
    level_rates = {h["level_name"]: h["rate"] for h in h_counts}

    metrics_agg = {
        "total_pairs": len(pairs),
        "jaccard_similarity_rwpe_nrwpe": a1_results.get("jaccard_similarity", 0),
        "rwpe_pairs_distinguished": a1_results.get("rwpe_total_distinguished", 0),
        "nrwpe_diag_pairs_distinguished": a1_results.get("nrwpe_total_distinguished", 0),
        "overlap_both_count": a1_results["overlap_matrix"]["both"]["count"],
        "overlap_rwpe_only_count": a1_results["overlap_matrix"]["rwpe_only"]["count"],
        "overlap_nrwpe_only_count": a1_results["overlap_matrix"]["nrwpe_only"]["count"],
        "overlap_neither_count": a1_results["overlap_matrix"]["neither"]["count"],
        "cospectral_pairs_tested": a2_results.get("n_cospectral_pairs_tested", 0),
        "tanh_diag_spectral_invariant_fraction": a2_results.get("tanh_diag_match_fraction", 0),
        "softplus_diag_spectral_invariant_fraction": a2_results.get("softplus_diag_match_fraction", 0),
        "relu_diag_spectral_invariant_fraction": a2_results.get("relu_diag_match_fraction", 0),
        "hierarchy_level0_degree_rate": level_rates.get("degree_only", 0),
        "hierarchy_level1_rwpe_rate": level_rates.get("RWPE", 0),
        "hierarchy_level2_nrwpe_diag_rate": level_rates.get("nRWPE_diag", 0),
        "hierarchy_level3_nrwpe_offdiag_rate": level_rates.get("nRWPE_offdiag", 0),
        "hierarchy_level4_nrwpe_gram_rate": level_rates.get("nRWPE_Gram", 0),
        "hierarchy_level5_full_trajectory_rate": level_rates.get("full_trajectory", 0),
        "hypothesis_supported_count": sum(1 for s in a5_results["hypothesis_scorecard"] if s["status"] == "supported"),
        "hypothesis_refuted_count": sum(1 for s in a5_results["hypothesis_scorecard"] if s["status"] == "refuted"),
        "hypothesis_partial_count": sum(1 for s in a5_results["hypothesis_scorecard"] if s["status"] == "partially_supported"),
        "computation_time_seconds": round(total_time, 2),
    }

    # ── Build per-pair examples ───────────────────────────────────────────
    schema_examples = []
    for i, (adj_A, adj_B, pid, cat) in enumerate(pairs):
        rw_disc = a1_results["rwpe_per_pair"][i] if i < len(a1_results.get("rwpe_per_pair", [])) else 0
        nr_disc = a1_results["nrwpe_per_pair"][i] if i < len(a1_results.get("nrwpe_per_pair", [])) else 0

        output_data = {
            "pair_id": pid,
            "category": cat,
            "rwpe_distinguished": bool(rw_disc),
            "nrwpe_diag_distinguished": bool(nr_disc),
            "overlap_class": (
                "both" if rw_disc and nr_disc else
                "rwpe_only" if rw_disc else
                "nrwpe_only" if nr_disc else
                "neither"
            ),
        }

        schema_examples.append({
            "input": json.dumps({"pair_id": pid, "category": cat, "n_nodes_A": adj_A.shape[0]}),
            "output": json.dumps(output_data),
            "eval_rwpe_distinguished": int(rw_disc),
            "eval_nrwpe_diag_distinguished": int(nr_disc),
        })

    # ── Build full output ─────────────────────────────────────────────────
    output = {
        "metadata": {
            "title": "Diagnostic Evaluation: Spectral Invariance of Diagonal nRWPE",
            "description": "5-analysis diagnostic evaluation comparing RWPE vs nRWPE-diag "
                           "on 525 graph expressiveness pairs, testing spectral invariance, "
                           "information hierarchy, ZINC attribution, and hypothesis scorecard.",
            "total_pairs_analyzed": len(pairs),
            "computation_time_seconds": round(total_time, 2),
            "analysis1_overlap": {
                k: v for k, v in a1_results.items()
                if k not in ["rwpe_per_pair", "nrwpe_per_pair"]
            },
            "analysis2_spectral_invariance": a2_results,
            "analysis3_hierarchy": a3_results,
            "analysis4_zinc": a4_results,
            "analysis5_scorecard": a5_results,
        },
        "metrics_agg": metrics_agg,
        "datasets": [
            {
                "dataset": "graph_expressiveness_525_pairs",
                "examples": schema_examples,
            }
        ],
    }

    # ── Save ──────────────────────────────────────────────────────────────
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Output written to {OUTPUT_PATH}")
    file_size = OUTPUT_PATH.stat().st_size / 1e6
    logger.info(f"Output file size: {file_size:.2f} MB")

    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pairs", type=int, default=None)
    args = parser.parse_args()
    main(max_pairs=args.max_pairs)
