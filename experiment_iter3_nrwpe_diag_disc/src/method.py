#!/usr/bin/env python3
"""
nRWPE-diag Discrimination Testing on 525-Pair Graph Expressiveness Benchmark.

Implements and evaluates 7 positional encoding methods:
1. RWPE_diag_K20 - Standard RWPE (diagonal of A_tilde^k)
2. linear_walk_diag_T20 - Linear walk (sanity check, should match RWPE)
3. nRWPE_diag_tanh_T20 - Nonlinear walk with tanh
4. nRWPE_diag_softplus_T20 - Nonlinear walk with softplus
5. nRWPE_diag_relu_T20 - Nonlinear walk with ReLU
6. nRWPE_offdiag_tanh_T20 - Nonlinear walk with off-diagonal aggregation
7. nRWPE_diag_tanh_T50 - Nonlinear walk with tanh, T=50

Measures pair discrimination rates, equivariance, convergence, and more.
"""

import json
import math
import os
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import psutil
from loguru import logger

# ─── Logging ───────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ─── Hardware detection ───────────────────────────────────────────────────────
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
TOTAL_RAM_GB = _container_ram_gb() or psutil.virtual_memory().total / 1e9

# Set memory limit (14GB budget — graphs are tiny, this is very conservative)
RAM_BUDGET = int(14 * 1024**3)
_avail = psutil.virtual_memory().available
assert RAM_BUDGET < _avail, f"Budget {RAM_BUDGET/1e9:.1f}GB > available {_avail/1e9:.1f}GB"
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f}GB RAM, budget {RAM_BUDGET/1e9:.1f}GB")

# ─── Constants ─────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent
DATA_PATH = WORKSPACE / "full_data_out.json"
MINI_DATA_PATH = WORKSPACE / "mini_data_out.json"
OUTPUT_PATH = WORKSPACE / "method_out.json"
THRESHOLD = 1e-6
THRESHOLD_STRICT = 1e-8
THRESHOLD_LOOSE = 1e-4
NUM_WORKERS = max(1, NUM_CPUS - 1)

# ─── Nonlinearity functions ───────────────────────────────────────────────────
def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(np.clip(x, -50, 50)))

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

def identity(x: np.ndarray) -> np.ndarray:
    return x

# ─── Core PE computation functions ─────────────────────────────────────────────
def build_normalized_adjacency(adj_matrix: list) -> np.ndarray:
    """Compute row-stochastic random walk matrix A_tilde = D^{-1} A.
    Isolated nodes (degree 0) get a zero row.
    """
    A = np.array(adj_matrix, dtype=np.float64)
    D = A.sum(axis=1)
    D_inv = np.zeros_like(D)
    mask = D > 0
    D_inv[mask] = 1.0 / D[mask]
    A_tilde = np.diag(D_inv) @ A
    return A_tilde


def compute_rwpe_diag(A_tilde: np.ndarray, K: int = 20) -> np.ndarray:
    """Standard RWPE: diagonal of A_tilde^k for k=1..K.
    Returns (n, K) matrix where row i = [A_tilde^1_{ii}, ..., A_tilde^K_{ii}].
    """
    n = A_tilde.shape[0]
    pe = np.zeros((n, K))
    M = A_tilde.copy()
    for k in range(K):
        pe[:, k] = np.diag(M)
        if k < K - 1:
            M = M @ A_tilde
    return pe


def compute_nrwpe_diag(A_tilde: np.ndarray, sigma, T: int = 20) -> np.ndarray:
    """Nonlinear walk PE (diagonal only).
    X_0 = I_n; X_{t+1} = sigma(A_tilde @ X_t); pe[i,t] = X_t[i,i].
    """
    n = A_tilde.shape[0]
    pe = np.zeros((n, T))
    X = np.eye(n)
    for t in range(T):
        X = sigma(A_tilde @ X)
        pe[:, t] = np.diag(X)
    return pe


def compute_nrwpe_offdiag(A_tilde: np.ndarray, sigma, T: int = 20, hops: list = None) -> np.ndarray:
    """nRWPE with off-diagonal aggregation (1-hop and 2-hop neighbor mean values).
    Returns (n, T * (1 + len(hops))) matrix.
    """
    if hops is None:
        hops = [1, 2]
    n = A_tilde.shape[0]
    A = (np.abs(A_tilde) > 1e-15).astype(float)  # Binary adjacency from A_tilde

    # Compute k-hop masks
    A1 = A.copy()
    A2 = (A @ A > 0).astype(float) - A1 - np.eye(n)
    A2 = np.maximum(A2, 0)

    # Normalize for mean aggregation
    d1 = A1.sum(axis=1, keepdims=True)
    d1[d1 == 0] = 1
    A1_norm = A1 / d1
    d2 = A2.sum(axis=1, keepdims=True)
    d2[d2 == 0] = 1
    A2_norm = A2 / d2

    n_features = 1 + len(hops)
    pe = np.zeros((n, T * n_features))
    X = np.eye(n)
    for t in range(T):
        X = sigma(A_tilde @ X)
        pe[:, t * n_features] = np.diag(X)
        if 1 in hops:
            pe[:, t * n_features + 1] = np.diag(A1_norm @ X)
        if 2 in hops:
            pe[:, t * n_features + 2] = np.diag(A2_norm @ X)
    return pe


# ─── Pair comparison ──────────────────────────────────────────────────────────
def compare_pair(pe_A: np.ndarray, pe_B: np.ndarray) -> tuple:
    """Compare PE multisets via lexicographic sort + L2/Linf distance."""
    if pe_A.shape != pe_B.shape:
        return float('inf'), float('inf')

    # Lexicographic sort by rows
    idx_A = np.lexsort(pe_A.T[::-1])
    idx_B = np.lexsort(pe_B.T[::-1])
    sorted_A = pe_A[idx_A]
    sorted_B = pe_B[idx_B]

    l2_dist = float(np.linalg.norm(sorted_A - sorted_B))
    linf_dist = float(np.max(np.abs(sorted_A - sorted_B)))
    return l2_dist, linf_dist


# ─── Data loading ─────────────────────────────────────────────────────────────
def load_data(path: Path, max_examples: int | None = None) -> list:
    """Load graph pair data from JSON file.
    Returns list of dicts with keys:
      adj_A, adj_B, category, pair_id, are_cospectral, num_nodes, metadata
    """
    logger.info(f"Loading data from {path}")
    raw = json.loads(path.read_text())
    examples = raw["datasets"][0]["examples"]
    if max_examples is not None:
        examples = examples[:max_examples]
    logger.info(f"Loaded {len(examples)} examples")

    pairs = []
    for ex in examples:
        input_data = json.loads(ex["input"])
        output_data = json.loads(ex["output"])
        pairs.append({
            "adj_A": input_data["graph_A"]["adjacency_matrix"],
            "adj_B": input_data["graph_B"]["adjacency_matrix"],
            "num_nodes_A": input_data["graph_A"]["num_nodes"],
            "num_nodes_B": input_data["graph_B"]["num_nodes"],
            "category": ex["metadata_category"],
            "pair_id": ex["metadata_pair_id"],
            "are_cospectral": output_data.get("are_cospectral", False),
            "row_index": ex.get("metadata_row_index", 0),
        })
    return pairs


# ─── Method definitions ──────────────────────────────────────────────────────
def get_methods() -> dict:
    """Return dict of method_name -> (compute_fn, description)."""
    return {
        "RWPE_diag_K20": (
            lambda A_tilde: compute_rwpe_diag(A_tilde, K=20),
            "Standard RWPE: diagonal of A_tilde^k, k=1..20"
        ),
        "linear_walk_diag_T20": (
            lambda A_tilde: compute_nrwpe_diag(A_tilde, identity, T=20),
            "Linear walk (identity nonlinearity) - sanity check against RWPE"
        ),
        "nRWPE_diag_tanh_T20": (
            lambda A_tilde: compute_nrwpe_diag(A_tilde, tanh, T=20),
            "Nonlinear walk with tanh, T=20"
        ),
        "nRWPE_diag_softplus_T20": (
            lambda A_tilde: compute_nrwpe_diag(A_tilde, softplus, T=20),
            "Nonlinear walk with softplus, T=20"
        ),
        "nRWPE_diag_relu_T20": (
            lambda A_tilde: compute_nrwpe_diag(A_tilde, relu, T=20),
            "Nonlinear walk with ReLU, T=20"
        ),
        "nRWPE_offdiag_tanh_T20": (
            lambda A_tilde: compute_nrwpe_offdiag(A_tilde, tanh, T=20, hops=[1, 2]),
            "Nonlinear walk with tanh + off-diagonal (1/2-hop), T=20"
        ),
        "nRWPE_diag_tanh_T50": (
            lambda A_tilde: compute_nrwpe_diag(A_tilde, tanh, T=50),
            "Nonlinear walk with tanh, T=50 (convergence test)"
        ),
    }


# ─── Process a single pair for one method ─────────────────────────────────────
def process_pair_all_methods(pair: dict) -> dict:
    """Compute all methods for a single pair. Returns per-method distances."""
    A_tilde_A = build_normalized_adjacency(pair["adj_A"])
    A_tilde_B = build_normalized_adjacency(pair["adj_B"])

    methods = get_methods()
    result = {
        "pair_id": pair["pair_id"],
        "category": pair["category"],
        "are_cospectral": pair["are_cospectral"],
        "num_nodes": pair["num_nodes_A"],
        "distances": {},
        "linf_distances": {},
        "distinguished": {},
    }

    for method_name, (method_fn, _desc) in methods.items():
        try:
            pe_A = method_fn(A_tilde_A)
            pe_B = method_fn(A_tilde_B)
            l2_dist, linf_dist = compare_pair(pe_A, pe_B)
            result["distances"][method_name] = l2_dist
            result["linf_distances"][method_name] = linf_dist
            result["distinguished"][method_name] = l2_dist > THRESHOLD
        except Exception as e:
            logger.exception(f"Error on pair {pair['pair_id']} method {method_name}")
            result["distances"][method_name] = float('nan')
            result["linf_distances"][method_name] = float('nan')
            result["distinguished"][method_name] = False

    return result


# ─── Aggregate results ────────────────────────────────────────────────────────
def aggregate_results(pair_results: list) -> dict:
    """Aggregate per-pair results into per-method, per-category statistics."""
    methods = list(get_methods().keys())
    categories = sorted(set(r["category"] for r in pair_results))

    per_method = {}
    for method_name in methods:
        cat_stats = {}
        total_dist = 0
        total_count = len(pair_results)

        for cat in categories:
            cat_pairs = [r for r in pair_results if r["category"] == cat]
            distances = [r["distances"][method_name] for r in cat_pairs
                         if not np.isnan(r["distances"][method_name])]
            dist_flags = [r["distinguished"][method_name] for r in cat_pairs]
            n_distinguished = sum(dist_flags)
            total_dist += n_distinguished

            dist_distances = [d for d, f in zip(distances, dist_flags) if f]
            undist_distances = [d for d, f in zip(distances, dist_flags) if not f]

            cat_stats[cat] = {
                "distinguished": n_distinguished,
                "total": len(cat_pairs),
                "rate": n_distinguished / max(1, len(cat_pairs)),
                "margin_median": float(np.median(distances)) if distances else 0.0,
                "margin_min": float(np.min(distances)) if distances else 0.0,
                "margin_max": float(np.max(distances)) if distances else 0.0,
                "margin_mean": float(np.mean(distances)) if distances else 0.0,
                "dist_margin_median": float(np.median(dist_distances)) if dist_distances else 0.0,
                "undist_margin_max": float(np.max(undist_distances)) if undist_distances else 0.0,
            }

        per_method[method_name] = {
            "overall": {
                "distinguished": total_dist,
                "total": total_count,
                "rate": total_dist / max(1, total_count),
            },
            "per_category": cat_stats,
        }

    return per_method


def compute_head_to_head(pair_results: list) -> dict:
    """Compute head-to-head comparisons between methods."""
    comparisons = {
        "nRWPE_tanh_vs_RWPE": ("nRWPE_diag_tanh_T20", "RWPE_diag_K20"),
        "nRWPE_tanh_vs_linear_walk": ("nRWPE_diag_tanh_T20", "linear_walk_diag_T20"),
        "nRWPE_offdiag_vs_diag": ("nRWPE_offdiag_tanh_T20", "nRWPE_diag_tanh_T20"),
        "tanh_vs_softplus": ("nRWPE_diag_tanh_T20", "nRWPE_diag_softplus_T20"),
        "tanh_vs_relu": ("nRWPE_diag_tanh_T20", "nRWPE_diag_relu_T20"),
    }

    h2h = {}
    for comp_name, (m1, m2) in comparisons.items():
        both = 0
        m1_only = 0
        m2_only = 0
        neither = 0
        for r in pair_results:
            d1 = r["distinguished"].get(m1, False)
            d2 = r["distinguished"].get(m2, False)
            if d1 and d2:
                both += 1
            elif d1 and not d2:
                m1_only += 1
            elif not d1 and d2:
                m2_only += 1
            else:
                neither += 1
        h2h[comp_name] = {
            "both_distinguish": both,
            f"{m1}_only": m1_only,
            f"{m2}_only": m2_only,
            "neither": neither,
        }
    return h2h


# ─── Equivariance verification ───────────────────────────────────────────────
def test_equivariance(pairs: list, n_graphs: int = 20, n_perms: int = 50) -> dict:
    """Verify equivariance of nRWPE_diag_tanh under node permutations."""
    rng = np.random.RandomState(42)

    # Select graphs from different categories
    by_cat = defaultdict(list)
    for p in pairs:
        by_cat[p["category"]].append(p)

    selected = []
    cats = sorted(by_cat.keys())
    per_cat = max(1, n_graphs // len(cats))
    for cat in cats:
        avail = by_cat[cat]
        chosen = avail[:per_cat] if len(avail) >= per_cat else avail
        selected.extend(chosen)
    selected = selected[:n_graphs]

    errors_per_graph = []
    for pair in selected:
        adj = np.array(pair["adj_A"], dtype=np.float64)
        n = adj.shape[0]
        A_tilde = build_normalized_adjacency(pair["adj_A"])
        pe_orig = compute_nrwpe_diag(A_tilde, tanh, T=20)

        max_err = 0.0
        for _ in range(n_perms):
            perm = rng.permutation(n)
            # Permute adjacency: A_perm = P @ A @ P^T
            P = np.zeros((n, n))
            for i, pi in enumerate(perm):
                P[pi, i] = 1.0
            adj_perm = P @ adj @ P.T
            A_tilde_perm = build_normalized_adjacency(adj_perm.tolist())
            pe_perm = compute_nrwpe_diag(A_tilde_perm, tanh, T=20)

            # Undo permutation: pe_perm[perm[i], :] should equal pe_orig[i, :]
            # So pe_orig[i, :] = pe_perm[perm[i], :], hence pe_unperm = pe_perm[perm]
            pe_unperm = pe_perm[perm]
            err = float(np.max(np.abs(pe_unperm - pe_orig)))
            max_err = max(max_err, err)

        errors_per_graph.append(max_err)

    overall_max = max(errors_per_graph) if errors_per_graph else 0.0
    all_pass = overall_max < 1e-10  # Loose tolerance for numerical precision

    return {
        "graphs_tested": len(selected),
        "permutations_per_graph": n_perms,
        "max_error": overall_max,
        "all_pass": all_pass,
        "errors_per_graph": errors_per_graph,
    }


# ─── Convergence analysis ────────────────────────────────────────────────────
def analyze_convergence(pair_results: list) -> dict:
    """Compare T=20 vs T=50 discrimination."""
    t20_dists = []
    t50_dists = []
    t20_flags = []
    t50_flags = []

    for r in pair_results:
        d20 = r["distances"].get("nRWPE_diag_tanh_T20", float('nan'))
        d50 = r["distances"].get("nRWPE_diag_tanh_T50", float('nan'))
        if not np.isnan(d20) and not np.isnan(d50):
            t20_dists.append(d20)
            t50_dists.append(d50)
            t20_flags.append(r["distinguished"].get("nRWPE_diag_tanh_T20", False))
            t50_flags.append(r["distinguished"].get("nRWPE_diag_tanh_T50", False))

    t20_rate = sum(t20_flags) / max(1, len(t20_flags))
    t50_rate = sum(t50_flags) / max(1, len(t50_flags))

    # Correlation
    if len(t20_dists) > 1:
        corr = float(np.corrcoef(t20_dists, t50_dists)[0, 1])
    else:
        corr = 0.0

    # Count discrimination changes
    changes = sum(1 for a, b in zip(t20_flags, t50_flags) if a != b)

    return {
        "T20_rate": t20_rate,
        "T50_rate": t50_rate,
        "T20_vs_T50_rate_difference": t50_rate - t20_rate,
        "T20_vs_T50_distance_correlation": corr if not np.isnan(corr) else 0.0,
        "T20_T50_max_discrimination_change": changes,
    }


# ─── Cospectral analysis ─────────────────────────────────────────────────────
def analyze_cospectral(pair_results: list) -> dict:
    """Analyze cospectral pairs specifically."""
    cospectral = [r for r in pair_results if r["are_cospectral"]]

    nrwpe_beats_rwpe = []
    both_fail = []
    for r in cospectral:
        rwpe_dist = r["distinguished"].get("RWPE_diag_K20", False)
        nrwpe_dist = r["distinguished"].get("nRWPE_diag_tanh_T20", False)
        if nrwpe_dist and not rwpe_dist:
            nrwpe_beats_rwpe.append(r["pair_id"])
        if not rwpe_dist and not nrwpe_dist:
            both_fail.append(r["pair_id"])

    # Strongly regular results
    sr = [r for r in pair_results if r["category"] == "strongly_regular"]
    sr_results = {}
    for r in sr:
        sr_results[r["pair_id"]] = {
            method: r["distinguished"].get(method, False)
            for method in get_methods().keys()
        }

    return {
        "total_cospectral_pairs": len(cospectral),
        "pairs_where_nRWPE_beats_RWPE": nrwpe_beats_rwpe,
        "count_nRWPE_beats_RWPE": len(nrwpe_beats_rwpe),
        "pairs_where_both_fail": both_fail,
        "count_both_fail": len(both_fail),
        "strongly_regular_results": sr_results,
    }


# ─── Threshold sensitivity ───────────────────────────────────────────────────
def analyze_threshold_sensitivity(pair_results: list) -> dict:
    """Report results at multiple thresholds to check robustness."""
    methods = list(get_methods().keys())
    thresholds = [1e-8, 1e-6, 1e-4]
    sensitivity = {}

    for thresh in thresholds:
        thresh_key = f"threshold_{thresh:.0e}"
        sensitivity[thresh_key] = {}
        for method_name in methods:
            n_dist = sum(
                1 for r in pair_results
                if r["distances"].get(method_name, 0) > thresh
                and not np.isnan(r["distances"].get(method_name, float('nan')))
            )
            sensitivity[thresh_key][method_name] = {
                "distinguished": n_dist,
                "rate": n_dist / max(1, len(pair_results)),
            }

    # Count fragile pairs (distance between 1e-8 and 1e-4)
    fragile = {}
    for method_name in methods:
        fragile_count = sum(
            1 for r in pair_results
            if 1e-8 < r["distances"].get(method_name, 0) <= 1e-4
            and not np.isnan(r["distances"].get(method_name, float('nan')))
        )
        fragile[method_name] = fragile_count

    sensitivity["fragile_zone_counts"] = fragile
    return sensitivity


# ─── Sanity checks ────────────────────────────────────────────────────────────
def run_sanity_checks(pairs: list) -> dict:
    """Run sanity checks on mini data: RWPE vs linear walk agreement, etc."""
    checks = {"rwpe_vs_linear_walk": {}, "row_stochastic": True, "nonlinearity_effect": {}}

    for i, pair in enumerate(pairs[:5]):  # Check first 5 pairs
        A_tilde_A = build_normalized_adjacency(pair["adj_A"])
        A_tilde_B = build_normalized_adjacency(pair["adj_B"])

        # Check row-stochastic
        for A_tilde, label in [(A_tilde_A, "A"), (A_tilde_B, "B")]:
            row_sums = A_tilde.sum(axis=1)
            for ri, rs in enumerate(row_sums):
                if rs > 0 and abs(rs - 1.0) > 1e-10:
                    logger.warning(f"Pair {pair['pair_id']} graph {label} row {ri} sum = {rs}")
                    checks["row_stochastic"] = False

        # RWPE vs linear walk
        pe_rwpe_A = compute_rwpe_diag(A_tilde_A, K=20)
        pe_linear_A = compute_nrwpe_diag(A_tilde_A, identity, T=20)
        max_diff = float(np.max(np.abs(pe_rwpe_A - pe_linear_A)))
        checks["rwpe_vs_linear_walk"][pair["pair_id"]] = max_diff
        if max_diff > 1e-10:
            logger.warning(f"RWPE vs linear walk mismatch on {pair['pair_id']}: {max_diff:.2e}")

        # Nonlinearity effect
        pe_tanh_A = compute_nrwpe_diag(A_tilde_A, tanh, T=20)
        diff_tanh_linear = float(np.max(np.abs(pe_tanh_A - pe_linear_A)))
        checks["nonlinearity_effect"][pair["pair_id"]] = diff_tanh_linear

    return checks


# ─── Format output for schema ────────────────────────────────────────────────
def format_output_for_schema(
    pairs: list,
    pair_results: list,
    per_method: dict,
    head_to_head: dict,
    equivariance: dict,
    convergence: dict,
    cospectral_analysis: dict,
    sanity_checks: dict,
    threshold_sensitivity: dict,
    timing: dict,
) -> dict:
    """Format all results into exp_gen_sol_out.json schema."""
    methods = list(get_methods().keys())
    method_descs = {name: desc for name, (_, desc) in get_methods().items()}

    # Build per-example outputs with predict_ fields
    examples = []
    for pair, result in zip(pairs, pair_results):
        input_str = json.dumps({
            "pair_id": result["pair_id"],
            "category": result["category"],
            "num_nodes": result["num_nodes"],
            "are_cospectral": result["are_cospectral"],
        })
        output_str = json.dumps({
            "distances": result["distances"],
            "linf_distances": result["linf_distances"],
            "distinguished": result["distinguished"],
        })

        example = {
            "input": input_str,
            "output": output_str,
            "metadata_pair_id": result["pair_id"],
            "metadata_category": result["category"],
            "metadata_are_cospectral": result["are_cospectral"],
            "metadata_num_nodes": result["num_nodes"],
        }

        # Add predict_ fields for each method
        for method_name in methods:
            dist = result["distinguished"].get(method_name, False)
            l2 = result["distances"].get(method_name, 0.0)
            example[f"predict_{method_name}"] = json.dumps({
                "distinguished": dist,
                "l2_distance": l2,
            })

        examples.append(example)

    # Build summary metadata
    categories = sorted(set(r["category"] for r in pair_results))
    cat_counts = {cat: sum(1 for r in pair_results if r["category"] == cat) for cat in categories}

    metadata = {
        "experiment": "nRWPE-diag Discrimination Testing",
        "total_pairs": len(pair_results),
        "methods_tested": len(methods),
        "method_names": methods,
        "method_descriptions": method_descs,
        "categories": cat_counts,
        "threshold": THRESHOLD,
        "per_method_results": per_method,
        "head_to_head": head_to_head,
        "equivariance_test": equivariance,
        "convergence_analysis": convergence,
        "cospectral_analysis": cospectral_analysis,
        "sanity_checks": sanity_checks,
        "threshold_sensitivity": threshold_sensitivity,
        "timing": timing,
    }

    return {
        "metadata": metadata,
        "datasets": [
            {
                "dataset": "graph_expressiveness_benchmark",
                "examples": examples,
            }
        ],
    }


# ─── Main ─────────────────────────────────────────────────────────────────────
@logger.catch
def main(data_path: Path = None, max_examples: int | None = None):
    t_start = time.time()

    if data_path is None:
        data_path = DATA_PATH

    # Load data
    pairs = load_data(data_path, max_examples=max_examples)
    n_pairs = len(pairs)

    # Discover categories
    categories = sorted(set(p["category"] for p in pairs))
    cat_counts = {cat: sum(1 for p in pairs if p["category"] == cat) for cat in categories}
    logger.info(f"Categories: {cat_counts}")

    # Step 1: Sanity checks on first few pairs
    logger.info("Running sanity checks...")
    sanity_checks = run_sanity_checks(pairs)
    logger.info(f"Sanity checks: RWPE vs linear walk max diffs = {sanity_checks['rwpe_vs_linear_walk']}")
    logger.info(f"Sanity checks: row-stochastic = {sanity_checks['row_stochastic']}")
    logger.info(f"Sanity checks: nonlinearity effect = {sanity_checks['nonlinearity_effect']}")

    # Step 2: Process all pairs with all methods
    logger.info(f"Processing {n_pairs} pairs with {len(get_methods())} methods...")
    t_process_start = time.time()

    # Use ProcessPoolExecutor for CPU-bound work
    if n_pairs >= 20 and NUM_WORKERS > 1:
        logger.info(f"Using {NUM_WORKERS} workers for parallel processing")
        pair_results = [None] * n_pairs
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {
                executor.submit(process_pair_all_methods, pair): idx
                for idx, pair in enumerate(pairs)
            }
            done_count = 0
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    pair_results[idx] = future.result()
                except Exception:
                    logger.exception(f"Failed on pair index {idx}")
                    pair_results[idx] = {
                        "pair_id": pairs[idx]["pair_id"],
                        "category": pairs[idx]["category"],
                        "are_cospectral": pairs[idx]["are_cospectral"],
                        "num_nodes": pairs[idx]["num_nodes_A"],
                        "distances": {},
                        "linf_distances": {},
                        "distinguished": {},
                    }
                done_count += 1
                if done_count % 100 == 0:
                    logger.info(f"  Processed {done_count}/{n_pairs} pairs")
    else:
        pair_results = []
        for i, pair in enumerate(pairs):
            result = process_pair_all_methods(pair)
            pair_results.append(result)
            if (i + 1) % 50 == 0:
                logger.info(f"  Processed {i+1}/{n_pairs} pairs")

    t_process_end = time.time()
    process_time = t_process_end - t_process_start
    logger.info(f"Processing completed in {process_time:.2f}s ({process_time/max(1,n_pairs)*1000:.1f}ms/pair)")

    # Check for NaN/Inf in results
    nan_count = 0
    for r in pair_results:
        for method_name, dist in r["distances"].items():
            if np.isnan(dist) or np.isinf(dist):
                nan_count += 1
    if nan_count > 0:
        logger.warning(f"Found {nan_count} NaN/Inf distance values!")
    else:
        logger.info("No NaN/Inf values found in distances")

    # Step 3: Aggregate results
    logger.info("Aggregating results...")
    per_method = aggregate_results(pair_results)

    # Log per-method summary
    for method_name, stats in per_method.items():
        ov = stats["overall"]
        logger.info(f"  {method_name}: {ov['distinguished']}/{ov['total']} = {ov['rate']:.4f}")

    # Step 4: Head-to-head comparisons
    logger.info("Computing head-to-head comparisons...")
    head_to_head = compute_head_to_head(pair_results)
    for comp_name, comp in head_to_head.items():
        logger.info(f"  {comp_name}: {comp}")

    # Step 5: Equivariance verification
    logger.info("Running equivariance verification...")
    t_equiv_start = time.time()
    equivariance = test_equivariance(pairs, n_graphs=min(20, n_pairs), n_perms=50)
    t_equiv_end = time.time()
    logger.info(f"Equivariance test: max_error={equivariance['max_error']:.2e}, "
                f"all_pass={equivariance['all_pass']} ({t_equiv_end-t_equiv_start:.1f}s)")

    # Step 6: Convergence analysis
    logger.info("Analyzing convergence (T=20 vs T=50)...")
    convergence = analyze_convergence(pair_results)
    logger.info(f"Convergence: T20 rate={convergence['T20_rate']:.4f}, "
                f"T50 rate={convergence['T50_rate']:.4f}, "
                f"diff={convergence['T20_vs_T50_rate_difference']:.4f}")

    # Step 7: Cospectral analysis
    logger.info("Analyzing cospectral pairs...")
    cospectral_analysis = analyze_cospectral(pair_results)
    logger.info(f"Cospectral: nRWPE beats RWPE on {cospectral_analysis['count_nRWPE_beats_RWPE']} pairs, "
                f"both fail on {cospectral_analysis['count_both_fail']} pairs")

    # Step 8: Threshold sensitivity
    logger.info("Analyzing threshold sensitivity...")
    threshold_sensitivity = analyze_threshold_sensitivity(pair_results)

    # Timing
    t_end = time.time()
    timing = {
        "total_seconds": t_end - t_start,
        "processing_seconds": process_time,
        "equivariance_seconds": t_equiv_end - t_equiv_start,
        "pairs_per_second": n_pairs / max(0.001, process_time),
    }
    logger.info(f"Total time: {timing['total_seconds']:.2f}s")

    # Step 9: Format and save output
    logger.info("Formatting output...")
    output = format_output_for_schema(
        pairs=pairs,
        pair_results=pair_results,
        per_method=per_method,
        head_to_head=head_to_head,
        equivariance=equivariance,
        convergence=convergence,
        cospectral_analysis=cospectral_analysis,
        sanity_checks=sanity_checks,
        threshold_sensitivity=threshold_sensitivity,
        timing=timing,
    )

    output_path = OUTPUT_PATH
    output_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Saved output to {output_path}")

    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None, help="Path to data file")
    parser.add_argument("--max-examples", type=int, default=None, help="Max examples to process")
    args = parser.parse_args()

    data_path = Path(args.data) if args.data else None
    main(data_path=data_path, max_examples=args.max_examples)
