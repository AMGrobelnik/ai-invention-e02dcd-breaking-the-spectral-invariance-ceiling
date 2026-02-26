#!/usr/bin/env python3
"""Nonlinear Walk Gram Matrix Equivariant Features for Graph Discrimination.

Tests whether equivariant features from the nonlinear walk Gram matrix G_NL
(especially its eigenvalues) distinguish more graph pairs than RWPE-diagonal
and nRWPE-diagonal on the 525-pair benchmark.

Methods:
  - G_NL eigenvalue fingerprint (tanh, softplus, relu, mild, linear)
  - G_NL node statistics fingerprint
  - G_NL diagonal fingerprint
  - G_NL hybrid fingerprint (diagonal + row sums)
  - RWPE baseline (diagonal of random walk powers)
  - nRWPE-diag baseline (diagonal of nonlinear walk self-similarity)
  - Equivariance verification
  - Hard pair diagnostics
  - Trajectory length ablation
"""

import json
import math
import os
import resource
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from loguru import logger

# ── Logging ──────────────────────────────────────────────────────────────────
WORKSPACE = Path("/workspace/runs/run__20260225_141527/3_invention_loop/iter_4/gen_art/exp_id1_it4__opus")
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ── Hardware Detection ───────────────────────────────────────────────────────
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

def _container_ram_gb() -> float | None:
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None

NUM_CPUS = _detect_cpus()
TOTAL_RAM_GB = _container_ram_gb() or 29.0
RAM_BUDGET = int(TOTAL_RAM_GB * 0.80 * 1e9)  # 80% of container limit
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3500, 3500))

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM, budget={RAM_BUDGET/1e9:.1f} GB")

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path("/workspace/runs/run__20260225_014759/3_invention_loop/iter_1/gen_art/data_id2_it1__opus")
FULL_DATA = DATA_DIR / "full_data_out.json"
MINI_DATA = DATA_DIR / "mini_data_out.json"
OUTPUT_FILE = WORKSPACE / "method_out.json"

# ── Configuration ────────────────────────────────────────────────────────────
T_DEFAULT = 30       # Default trajectory length
RWPE_STEPS = 20      # Number of random walk steps for RWPE baseline
DISC_THRESHOLD = 1e-6  # L2 distance threshold for discrimination
NONLINEARITIES = ["tanh", "softplus", "relu", "mild", "linear"]
ABLATION_T_VALUES = [5, 10, 15, 20, 30, 50]

# ── Max examples (set by scaling phases, default = all) ──────────────────────
MAX_EXAMPLES = int(os.environ.get("MAX_EXAMPLES", "0"))  # 0 = all

# ── Nonlinearity functions ───────────────────────────────────────────────────
def apply_nonlinearity(X: np.ndarray, name: str) -> np.ndarray:
    """Apply pointwise nonlinearity to matrix X."""
    if name == "tanh":
        return np.tanh(X)
    elif name == "softplus":
        # numerically stable softplus
        return np.where(X > 20, X, np.log1p(np.exp(np.clip(X, -500, 20))))
    elif name == "relu":
        return np.maximum(X, 0.0)
    elif name == "mild":
        # x * tanh(x) -- mild nonlinearity
        return X * np.tanh(X)
    elif name == "linear":
        return X.copy()
    else:
        raise ValueError(f"Unknown nonlinearity: {name}")


# ── Core Graph Computations ──────────────────────────────────────────────────
def normalized_adjacency(A: np.ndarray) -> np.ndarray:
    """Compute symmetric normalized adjacency: D^{-1/2} A D^{-1/2}."""
    deg = A.sum(axis=1)
    deg = np.where(deg == 0, 1.0, deg)  # handle isolated nodes
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    # D^{-1/2} A D^{-1/2} = diag(d_inv_sqrt) @ A @ diag(d_inv_sqrt)
    return A * np.outer(d_inv_sqrt, d_inv_sqrt)


def compute_gram_matrix(A_norm: np.ndarray, nonlinearity: str, T: int) -> np.ndarray:
    """Compute nonlinear walk Gram matrix G = sum_{t=0}^{T} X_t^T X_t.

    X_0 = I_n, X_t = sigma(A_norm @ X_{t-1}).
    G is n x n, positive semidefinite.
    """
    n = A_norm.shape[0]
    X = np.eye(n, dtype=np.float64)  # X_0 = I
    G = np.eye(n, dtype=np.float64)  # G starts with X_0^T @ X_0 = I
    for _ in range(T):
        X = apply_nonlinearity(A_norm @ X, nonlinearity)
        G += X.T @ X
    return G


def extract_eigenvalue_fingerprint(G: np.ndarray) -> np.ndarray:
    """Strategy A: Sorted eigenvalues of G."""
    eigs = np.linalg.eigvalsh(G)
    return np.sort(eigs)


def extract_node_statistics(G: np.ndarray) -> np.ndarray:
    """Strategy B: Per-row statistics of G, sorted lexicographically.

    Returns n x 19 matrix (sorted rows).
    """
    n = G.shape[0]
    diag = np.diag(G)
    stats_list = []
    for i in range(n):
        row = G[i]
        offdiag = np.concatenate([row[:i], row[i+1:]])
        d = diag[i]
        mean_off = np.mean(offdiag) if len(offdiag) > 0 else 0.0
        std_off = np.std(offdiag) if len(offdiag) > 0 else 0.0
        # Skewness and kurtosis
        if len(offdiag) > 2 and std_off > 1e-15:
            skew = float(np.mean(((offdiag - mean_off) / std_off) ** 3))
            kurt = float(np.mean(((offdiag - mean_off) / std_off) ** 4))
        else:
            skew = 0.0
            kurt = 0.0
        max_off = np.max(offdiag) if len(offdiag) > 0 else 0.0
        min_off = np.min(offdiag) if len(offdiag) > 0 else 0.0
        med_off = np.median(offdiag) if len(offdiag) > 0 else 0.0
        q25 = np.percentile(offdiag, 25) if len(offdiag) > 0 else 0.0
        q75 = np.percentile(offdiag, 75) if len(offdiag) > 0 else 0.0
        rowsum = np.sum(row)
        l2norm = np.linalg.norm(row)
        sqsum = np.sum(row ** 2)
        l1_off = np.sum(np.abs(offdiag)) if len(offdiag) > 0 else 0.0
        # Top 5 sorted off-diagonal (pad with 0 if < 5)
        sorted_off = np.sort(offdiag)[::-1]
        top5 = np.zeros(5)
        top5[:min(5, len(sorted_off))] = sorted_off[:min(5, len(sorted_off))]
        stats = np.array([d, mean_off, std_off, skew, kurt, max_off, min_off,
                          med_off, q25, q75, rowsum, l2norm, sqsum, l1_off] + list(top5))
        stats_list.append(stats)
    stats_arr = np.array(stats_list)
    # Sort rows lexicographically
    idx = np.lexsort(stats_arr.T[::-1])
    return stats_arr[idx]


def extract_diagonal_fingerprint(G: np.ndarray) -> np.ndarray:
    """Strategy C: Sorted diagonal of G."""
    return np.sort(np.diag(G))


def extract_hybrid_fingerprint(G: np.ndarray) -> np.ndarray:
    """Strategy D: concat(sorted diagonal, sorted row sums)."""
    diag_sorted = np.sort(np.diag(G))
    rowsums_sorted = np.sort(G.sum(axis=1))
    return np.concatenate([diag_sorted, rowsums_sorted])


# ── Baselines ────────────────────────────────────────────────────────────────
def compute_rwpe(A: np.ndarray, steps: int = 20) -> np.ndarray:
    """RWPE baseline: diag(RW^k) for k=1..steps, rows sorted.

    RW = A @ D^{-1} (random walk transition matrix).
    Returns n x steps matrix (sorted rows).
    """
    n = A.shape[0]
    deg = A.sum(axis=1)
    deg = np.where(deg == 0, 1.0, deg)
    D_inv = np.diag(1.0 / deg)
    RW = A @ D_inv  # random walk matrix
    RW_power = np.eye(n, dtype=np.float64)
    features = np.zeros((n, steps), dtype=np.float64)
    for k in range(steps):
        RW_power = RW_power @ RW
        features[:, k] = np.diag(RW_power)
    # Sort rows lexicographically
    idx = np.lexsort(features.T[::-1])
    return features[idx]


def compute_nrwpe_diag(A_norm: np.ndarray, T: int = 30) -> np.ndarray:
    """nRWPE baseline: diag(X_t @ X_t^T) for tanh walk, rows sorted.

    Returns n x T matrix (sorted rows).
    """
    n = A_norm.shape[0]
    X = np.eye(n, dtype=np.float64)
    features = np.zeros((n, T), dtype=np.float64)
    for t in range(T):
        X = np.tanh(A_norm @ X)
        # diag(X @ X^T) = sum of squares of each row of X
        features[:, t] = np.sum(X ** 2, axis=1)
    idx = np.lexsort(features.T[::-1])
    return features[idx]


# ── Discrimination Test ──────────────────────────────────────────────────────
def is_distinguished(fp_A: np.ndarray, fp_B: np.ndarray, threshold: float = DISC_THRESHOLD) -> tuple:
    """Check if two fingerprints are distinguishable.

    Returns (is_distinguished: bool, distance: float).
    """
    if fp_A.shape != fp_B.shape:
        return True, float("inf")
    dist = float(np.linalg.norm(fp_A - fp_B))
    return dist > threshold, dist


# ── Equivariance Verification ────────────────────────────────────────────────
def verify_equivariance(A: np.ndarray, nonlinearity: str = "tanh",
                        T: int = 30, n_perms: int = 10, tol: float = 1e-8) -> dict:
    """Verify that G transforms equivariantly under permutation: G(PAP^T) = P G(A) P^T."""
    n = A.shape[0]
    A_norm = normalized_adjacency(A)
    G_orig = compute_gram_matrix(A_norm, nonlinearity, T)
    eigs_orig = np.sort(np.linalg.eigvalsh(G_orig))

    results = {"passed": True, "max_gram_error": 0.0, "max_eig_error": 0.0, "n_perms": n_perms}
    rng = np.random.RandomState(42)

    for _ in range(n_perms):
        perm = rng.permutation(n)
        P = np.eye(n)[perm]
        A_perm = P @ A @ P.T
        A_norm_perm = normalized_adjacency(A_perm)
        G_perm = compute_gram_matrix(A_norm_perm, nonlinearity, T)
        # Expected: P @ G_orig @ P^T
        G_expected = P @ G_orig @ P.T
        gram_err = float(np.linalg.norm(G_perm - G_expected))
        eigs_perm = np.sort(np.linalg.eigvalsh(G_perm))
        eig_err = float(np.linalg.norm(eigs_perm - eigs_orig))

        results["max_gram_error"] = max(results["max_gram_error"], gram_err)
        results["max_eig_error"] = max(results["max_eig_error"], eig_err)
        if gram_err > tol or eig_err > tol:
            results["passed"] = False

    return results


# ── Process a single pair for all methods ────────────────────────────────────
def process_single_pair(pair_data: dict, T: int = T_DEFAULT) -> dict:
    """Process a single graph pair through all methods.

    Returns dict with method → {distinguished, distance} for each method.
    """
    inp = json.loads(pair_data["input"])
    A_a = np.array(inp["graph_A"]["adjacency_matrix"], dtype=np.float64)
    A_b = np.array(inp["graph_B"]["adjacency_matrix"], dtype=np.float64)

    A_norm_a = normalized_adjacency(A_a)
    A_norm_b = normalized_adjacency(A_b)

    results = {}

    # ── Baselines ──
    # RWPE
    rwpe_a = compute_rwpe(A_a, RWPE_STEPS)
    rwpe_b = compute_rwpe(A_b, RWPE_STEPS)
    d, dist = is_distinguished(rwpe_a, rwpe_b)
    results["RWPE_diag"] = {"distinguished": d, "distance": dist}

    # nRWPE-diag
    nrwpe_a = compute_nrwpe_diag(A_norm_a, T)
    nrwpe_b = compute_nrwpe_diag(A_norm_b, T)
    d, dist = is_distinguished(nrwpe_a, nrwpe_b)
    results["nRWPE_diag"] = {"distinguished": d, "distance": dist}

    # ── Our methods: G_NL with different nonlinearities and feature strategies ──
    for nl in NONLINEARITIES:
        G_a = compute_gram_matrix(A_norm_a, nl, T)
        G_b = compute_gram_matrix(A_norm_b, nl, T)

        # A) Eigenvalue fingerprint
        eig_a = extract_eigenvalue_fingerprint(G_a)
        eig_b = extract_eigenvalue_fingerprint(G_b)
        d, dist = is_distinguished(eig_a, eig_b)
        results[f"G_NL_eig_{nl}"] = {"distinguished": d, "distance": dist}

        # B) Node statistics fingerprint (only for tanh to save compute)
        if nl == "tanh":
            ns_a = extract_node_statistics(G_a)
            ns_b = extract_node_statistics(G_b)
            d, dist = is_distinguished(ns_a, ns_b)
            results[f"G_NL_nodestats_{nl}"] = {"distinguished": d, "distance": dist}

        # C) Diagonal fingerprint
        diag_a = extract_diagonal_fingerprint(G_a)
        diag_b = extract_diagonal_fingerprint(G_b)
        d, dist = is_distinguished(diag_a, diag_b)
        results[f"G_NL_diag_{nl}"] = {"distinguished": d, "distance": dist}

        # D) Hybrid fingerprint
        hyb_a = extract_hybrid_fingerprint(G_a)
        hyb_b = extract_hybrid_fingerprint(G_b)
        d, dist = is_distinguished(hyb_a, hyb_b)
        results[f"G_NL_hybrid_{nl}"] = {"distinguished": d, "distance": dist}

    return results


def process_pair_wrapper(args):
    """Wrapper for parallel execution."""
    idx, pair_data, T = args
    try:
        result = process_single_pair(pair_data, T)
        return idx, result, None
    except Exception as e:
        return idx, None, str(e)


# ── Trajectory Length Ablation ───────────────────────────────────────────────
def ablation_single_pair(pair_data: dict, T_val: int) -> dict:
    """Run G_NL_eig_tanh at a specific T value for ablation."""
    inp = json.loads(pair_data["input"])
    A_a = np.array(inp["graph_A"]["adjacency_matrix"], dtype=np.float64)
    A_b = np.array(inp["graph_B"]["adjacency_matrix"], dtype=np.float64)
    A_norm_a = normalized_adjacency(A_a)
    A_norm_b = normalized_adjacency(A_b)
    G_a = compute_gram_matrix(A_norm_a, "tanh", T_val)
    G_b = compute_gram_matrix(A_norm_b, "tanh", T_val)
    eig_a = extract_eigenvalue_fingerprint(G_a)
    eig_b = extract_eigenvalue_fingerprint(G_b)
    d, dist = is_distinguished(eig_a, eig_b)
    return {"distinguished": d, "distance": dist}


def ablation_wrapper(args):
    idx, pair_data, T_val = args
    try:
        r = ablation_single_pair(pair_data, T_val)
        return idx, T_val, r, None
    except Exception as e:
        return idx, T_val, None, str(e)


# ── Main ─────────────────────────────────────────────────────────────────────
@logger.catch
def main():
    t_start = time.time()

    # ── Load data ──
    data_path = FULL_DATA
    logger.info(f"Loading data from {data_path}")
    raw = json.loads(data_path.read_text())
    examples = raw["datasets"][0]["examples"]
    total = len(examples)
    logger.info(f"Loaded {total} examples")

    if MAX_EXAMPLES > 0:
        examples = examples[:MAX_EXAMPLES]
        logger.info(f"Limiting to first {len(examples)} examples")

    n_examples = len(examples)

    # ── Phase 1: Process all pairs with all methods ──
    logger.info(f"Processing {n_examples} pairs with {len(NONLINEARITIES)} nonlinearities...")
    workers = max(1, NUM_CPUS)
    logger.info(f"Using {workers} workers")

    all_results = [None] * n_examples
    tasks = [(i, examples[i], T_DEFAULT) for i in range(n_examples)]

    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_pair_wrapper, t): t[0] for t in tasks}
        for future in as_completed(futures):
            idx, result, err = future.result()
            if err:
                logger.warning(f"Pair {idx} failed: {err}")
                all_results[idx] = {"error": err}
            else:
                all_results[idx] = result
            done += 1
            if done % 50 == 0 or done == n_examples:
                elapsed = time.time() - t_start
                logger.info(f"  [{done}/{n_examples}] elapsed={elapsed:.1f}s")

    elapsed_phase1 = time.time() - t_start
    logger.info(f"Phase 1 complete in {elapsed_phase1:.1f}s")

    # ── Phase 2: Equivariance Verification ──
    logger.info("Phase 2: Equivariance verification on 20 graphs...")
    equivariance_results = []
    # Pick first 20 unique graphs from the examples
    n_verify = min(20, n_examples)
    for i in range(n_verify):
        try:
            inp = json.loads(examples[i]["input"])
            A = np.array(inp["graph_A"]["adjacency_matrix"], dtype=np.float64)
            ev = verify_equivariance(A, nonlinearity="tanh", T=T_DEFAULT, n_perms=10)
            equivariance_results.append({
                "pair_idx": i, "pair_id": examples[i].get("metadata_pair_id", ""),
                "n_nodes": A.shape[0], **ev
            })
        except Exception:
            logger.exception(f"Equivariance check failed for pair {i}")

    all_passed = all(r["passed"] for r in equivariance_results)
    logger.info(f"Equivariance: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    for r in equivariance_results:
        logger.debug(f"  pair={r['pair_idx']} n={r['n_nodes']} passed={r['passed']} "
                      f"gram_err={r['max_gram_error']:.2e} eig_err={r['max_eig_error']:.2e}")

    # ── Phase 3: Aggregate results per category and overall ──
    logger.info("Phase 3: Aggregating results...")

    # Collect all method names
    method_names = set()
    for r in all_results:
        if r and "error" not in r:
            method_names.update(r.keys())
    method_names = sorted(method_names)
    logger.info(f"Methods: {method_names}")

    # Per-category aggregation
    categories = {}
    for i, ex in enumerate(examples):
        cat = ex.get("metadata_category", "unknown")
        if cat not in categories:
            categories[cat] = {"indices": [], "total": 0}
        categories[cat]["indices"].append(i)
        categories[cat]["total"] += 1

    per_category = {}
    for cat, info in sorted(categories.items()):
        per_category[cat] = {"total": info["total"]}
        for m in method_names:
            count = 0
            dists = []
            for idx in info["indices"]:
                r = all_results[idx]
                if r and "error" not in r and m in r:
                    if r[m]["distinguished"]:
                        count += 1
                    d = r[m]["distance"]
                    if d != float("inf"):
                        dists.append(d)
            per_category[cat][m] = {
                "distinguished": count,
                "rate": count / info["total"] if info["total"] > 0 else 0.0,
                "median_distance": float(np.median(dists)) if dists else 0.0,
                "min_distance": float(np.min(dists)) if dists else 0.0,
                "max_distance": float(np.max(dists)) if dists else 0.0,
            }

    # Overall aggregation
    overall = {"total": n_examples}
    for m in method_names:
        count = 0
        dists = []
        for r in all_results:
            if r and "error" not in r and m in r:
                if r[m]["distinguished"]:
                    count += 1
                d = r[m]["distance"]
                if d != float("inf"):
                    dists.append(d)
        overall[m] = {
            "distinguished": count,
            "rate": count / n_examples if n_examples > 0 else 0.0,
            "median_distance": float(np.median(dists)) if dists else 0.0,
        }

    logger.info("=== OVERALL RESULTS ===")
    for m in method_names:
        o = overall[m]
        logger.info(f"  {m:30s}: {o['distinguished']:>3d}/{n_examples} = {o['rate']:.4f}")

    logger.info("=== PER-CATEGORY RESULTS ===")
    for cat in sorted(per_category.keys()):
        logger.info(f"  --- {cat} ({per_category[cat]['total']} pairs) ---")
        for m in method_names:
            mc = per_category[cat][m]
            logger.info(f"    {m:30s}: {mc['distinguished']:>3d}/{per_category[cat]['total']} = {mc['rate']:.4f}")

    # ── Phase 4: Hard pair diagnostics ──
    logger.info("Phase 4: Hard pair diagnostics...")
    hard_pairs = []

    # Find pairs where RWPE fails but our method succeeds (or interesting cases)
    target_cats = ["CSL", "CFI", "Distance Regular", "cospectral", "Extension"]
    for target_cat in target_cats:
        for i, ex in enumerate(examples):
            cat = ex.get("metadata_category", "")
            subcat = ex.get("metadata_subcategory", "")
            if cat.lower() == target_cat.lower() or subcat.lower() == target_cat.lower():
                r = all_results[i]
                if r and "error" not in r:
                    rwpe_disc = r.get("RWPE_diag", {}).get("distinguished", True)
                    gnl_disc = r.get("G_NL_eig_tanh", {}).get("distinguished", False)
                    if not rwpe_disc or gnl_disc:
                        # Compute linear Gram eigenvalues for comparison
                        inp = json.loads(ex["input"])
                        A_a = np.array(inp["graph_A"]["adjacency_matrix"], dtype=np.float64)
                        A_b = np.array(inp["graph_B"]["adjacency_matrix"], dtype=np.float64)
                        An_a = normalized_adjacency(A_a)
                        An_b = normalized_adjacency(A_b)

                        G_lin_a = compute_gram_matrix(An_a, "linear", T_DEFAULT)
                        G_lin_b = compute_gram_matrix(An_b, "linear", T_DEFAULT)
                        G_nl_a = compute_gram_matrix(An_a, "tanh", T_DEFAULT)
                        G_nl_b = compute_gram_matrix(An_b, "tanh", T_DEFAULT)

                        eig_lin_a = extract_eigenvalue_fingerprint(G_lin_a)
                        eig_lin_b = extract_eigenvalue_fingerprint(G_lin_b)
                        eig_nl_a = extract_eigenvalue_fingerprint(G_nl_a)
                        eig_nl_b = extract_eigenvalue_fingerprint(G_nl_b)

                        lin_dist = float(np.linalg.norm(eig_lin_a - eig_lin_b)) if eig_lin_a.shape == eig_lin_b.shape else float("inf")
                        nl_dist = float(np.linalg.norm(eig_nl_a - eig_nl_b)) if eig_nl_a.shape == eig_nl_b.shape else float("inf")

                        # Which eigenvalue indices differ most
                        if eig_nl_a.shape == eig_nl_b.shape:
                            diff = np.abs(eig_nl_a - eig_nl_b)
                            top_idx = np.argsort(diff)[::-1][:5]
                            top_diffs = [(int(j), float(diff[j])) for j in top_idx]
                        else:
                            top_diffs = []

                        hard_pairs.append({
                            "pair_idx": i,
                            "pair_id": ex.get("metadata_pair_id", ""),
                            "category": cat,
                            "subcategory": subcat,
                            "rwpe_distinguished": rwpe_disc,
                            "gnl_eig_tanh_distinguished": gnl_disc,
                            "linear_gram_eig_distance": lin_dist,
                            "nonlinear_gram_eig_distance": nl_dist,
                            "top_eigenvalue_diffs": top_diffs,
                        })
                        if len(hard_pairs) >= 15:  # enough diagnostics
                            break
            if len(hard_pairs) >= 15:
                break

    logger.info(f"Collected {len(hard_pairs)} hard pair diagnostics")

    # ── Phase 5: Trajectory Length Ablation ──
    logger.info("Phase 5: Trajectory length ablation for G_NL_eig_tanh...")
    ablation_results = {}

    for T_val in ABLATION_T_VALUES:
        logger.info(f"  T={T_val}...")
        abl_tasks = [(i, examples[i], T_val) for i in range(n_examples)]
        abl_data = [None] * n_examples

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(ablation_wrapper, t): t[0] for t in abl_tasks}
            for future in as_completed(futures):
                idx, tv, res, err = future.result()
                if err:
                    abl_data[idx] = {"distinguished": False, "distance": 0.0}
                else:
                    abl_data[idx] = res

        # Aggregate
        total_disc = sum(1 for r in abl_data if r and r["distinguished"])
        # Per category
        abl_per_cat = {}
        for cat, info in categories.items():
            cat_disc = sum(1 for idx in info["indices"] if abl_data[idx] and abl_data[idx]["distinguished"])
            abl_per_cat[cat] = {"distinguished": cat_disc, "total": info["total"],
                                "rate": cat_disc / info["total"] if info["total"] > 0 else 0.0}
        ablation_results[str(T_val)] = {
            "overall_distinguished": total_disc,
            "overall_rate": total_disc / n_examples if n_examples > 0 else 0.0,
            "per_category": abl_per_cat,
        }
        logger.info(f"    T={T_val}: {total_disc}/{n_examples} = {total_disc/n_examples:.4f}")

    elapsed_total = time.time() - t_start
    logger.info(f"Total runtime: {elapsed_total:.1f}s")

    # ── Build output in exp_gen_sol_out.json schema ──
    logger.info("Building output JSON...")

    # Build per-example output with predictions
    output_examples = []
    for i, ex in enumerate(examples):
        r = all_results[i]
        out_ex = {
            "input": ex["input"],
            "output": ex["output"],
        }
        # Copy all metadata_* fields
        for k, v in ex.items():
            if k.startswith("metadata_"):
                out_ex[k] = v

        # Add predictions from each method
        if r and "error" not in r:
            for m in method_names:
                if m in r:
                    out_ex[f"predict_{m}"] = json.dumps({
                        "distinguished": r[m]["distinguished"],
                        "distance": r[m]["distance"] if r[m]["distance"] != float("inf") else "inf",
                    })
        else:
            for m in method_names:
                out_ex[f"predict_{m}"] = json.dumps({"distinguished": False, "distance": 0.0, "error": r.get("error", "unknown") if r else "missing"})

        output_examples.append(out_ex)

    output = {
        "metadata": {
            "title": "Nonlinear Walk Gram Matrix Equivariant Features for Graph Discrimination",
            "description": "Tests whether equivariant features from the nonlinear walk Gram matrix G_NL distinguish graph pairs better than RWPE and nRWPE baselines.",
            "method_name": "G_NL (Nonlinear Walk Gram Matrix)",
            "baselines": ["RWPE_diag", "nRWPE_diag"],
            "nonlinearities": NONLINEARITIES,
            "trajectory_length": T_DEFAULT,
            "rwpe_steps": RWPE_STEPS,
            "discrimination_threshold": DISC_THRESHOLD,
            "num_examples": n_examples,
            "runtime_seconds": round(elapsed_total, 2),
            "overall_results": overall,
            "per_category_results": per_category,
            "equivariance_verification": {
                "all_passed": all_passed,
                "n_graphs_tested": len(equivariance_results),
                "n_perms_per_graph": 10,
                "details": equivariance_results,
            },
            "hard_pair_diagnostics": hard_pairs,
            "trajectory_length_ablation": ablation_results,
        },
        "datasets": [
            {
                "dataset": "graph_expressiveness_benchmark",
                "examples": output_examples,
            }
        ]
    }

    # Write output
    OUTPUT_FILE.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Output written to {OUTPUT_FILE}")
    logger.info(f"Output size: {OUTPUT_FILE.stat().st_size / 1e6:.2f} MB")

    # ── Summary ──
    logger.info("=== SUMMARY ===")
    best_method = max(method_names, key=lambda m: overall[m]["distinguished"])
    logger.info(f"Best method: {best_method} ({overall[best_method]['distinguished']}/{n_examples})")
    logger.info(f"RWPE baseline: {overall.get('RWPE_diag', {}).get('distinguished', 0)}/{n_examples}")
    logger.info(f"nRWPE baseline: {overall.get('nRWPE_diag', {}).get('distinguished', 0)}/{n_examples}")


if __name__ == "__main__":
    main()
