#!/usr/bin/env python3
"""Compute positional encodings (PEs) for ZINC-12k graphs.

PE Types:
1. rwpe_16: Standard RWPE diagonal of (D^-1 A)^t for t=1..16
2. nrwpe_diag_softplus_16: Nonlinear walk with softplus, diagonal
3. nrwpe_diag_tanh_16: Nonlinear walk with tanh, diagonal
4. nrwpe_offdiag_16: Off-diagonal statistics of nonlinear walk matrix
5. nrwpe_combined_16: 8-dim diag softplus + 8-dim offdiag softplus
6. no_pe: Zero vectors (baseline with no PE)

Output: precomputed_pes.pkl with all PE arrays for all graphs.
"""

import os
# Set thread limits BEFORE numpy import
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import json
import math
import pickle
import sys
import time
import resource
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from loguru import logger

# ── Logging setup ──
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/compute_pes.log", rotation="30 MB", level="DEBUG")

# ── Hardware detection ──
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

# Set memory limit (use ~60% of container RAM for PE computation)
RAM_BUDGET = int(TOTAL_RAM_GB * 0.6 * 1e9)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f}GB RAM, budget={RAM_BUDGET/1e9:.1f}GB")

# ── Constants ──
PE_DIM = 16
OFFDIAG_STEPS = [1, 3, 7, 15]
DATA_DIR = Path("/workspace/runs/run__20260225_014759/3_invention_loop/iter_1/gen_art/data_id3_it1__opus")
SCRIPT_DIR = Path(__file__).resolve().parent


def stable_softplus(x: np.ndarray) -> np.ndarray:
    """Numerically stable softplus: log(1 + exp(x))."""
    return np.where(x > 20, x, np.log1p(np.exp(np.clip(x, -50, 20))))


def build_norm_adj(edge_index: list, n: int) -> np.ndarray:
    """Build symmetric normalized adjacency: D^{-1/2}(A+I)D^{-1/2}."""
    A = np.zeros((n, n), dtype=np.float64)
    src, dst = edge_index[0], edge_index[1]
    for s, d in zip(src, dst):
        A[s, d] = 1.0
    # Add self-loops
    A_hat = A + np.eye(n)
    # Degree
    deg = A_hat.sum(axis=1)
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    return D_inv_sqrt @ A_hat @ D_inv_sqrt


def compute_rwpe(edge_index: list, n: int, k: int = 16) -> np.ndarray:
    """Standard RWPE: diagonal of (D^{-1}A)^t for t=1..k."""
    A = np.zeros((n, n), dtype=np.float64)
    src, dst = edge_index[0], edge_index[1]
    for s, d in zip(src, dst):
        A[s, d] = 1.0
    deg = A.sum(axis=1)
    D_inv = np.diag(np.where(deg > 0, 1.0 / deg, 0.0))
    RW = D_inv @ A
    pe = np.zeros((n, k), dtype=np.float32)
    RW_power = RW.copy()
    for t in range(k):
        pe[:, t] = np.diag(RW_power).astype(np.float32)
        if t < k - 1:
            RW_power = RW_power @ RW
    return pe


def compute_nrwpe_diag(edge_index: list, n: int, k: int = 16,
                       nonlinearity: str = "softplus") -> np.ndarray:
    """Nonlinear RWPE (diagonal): σ(Ã·X) iterated k times."""
    A_norm = build_norm_adj(edge_index, n)
    X = np.eye(n, dtype=np.float64)
    pe = np.zeros((n, k), dtype=np.float32)
    sigma = stable_softplus if nonlinearity == "softplus" else np.tanh
    for t in range(k):
        X = sigma(A_norm @ X)
        # Renormalize if values get too large (softplus doesn't saturate)
        max_val = np.abs(X).max()
        if max_val > 50:
            X = X / max_val * 10.0
        pe[:, t] = np.diag(X).astype(np.float32)
    return pe


def compute_nrwpe_offdiag(edge_index: list, n: int,
                          steps: list = None,
                          nonlinearity: str = "softplus",
                          stats_per_step: int = 4) -> np.ndarray:
    """Off-diagonal nRWPE: statistics of off-diagonal entries at selected steps.

    For each node i, at each selected step t, compute:
    [mean, std, max, third_largest] of the off-diagonal entries of column i.

    Returns shape (n, len(steps) * stats_per_step).
    """
    if steps is None:
        steps = OFFDIAG_STEPS
    A_norm = build_norm_adj(edge_index, n)
    X = np.eye(n, dtype=np.float64)
    sigma = stable_softplus if nonlinearity == "softplus" else np.tanh

    total_dim = len(steps) * stats_per_step
    pe = np.zeros((n, total_dim), dtype=np.float32)
    step_counter = 0

    for t in range(1, max(steps) + 1):
        X = sigma(A_norm @ X)
        # Renormalize if values get too large
        max_val = np.abs(X).max()
        if max_val > 50:
            X = X / max_val * 10.0

        if t in steps:
            for i in range(n):
                col = X[:, i]
                mask = np.arange(n) != i
                offdiag = col[mask]
                if len(offdiag) == 0:
                    step_counter += 1
                    continue
                sorted_desc = np.sort(offdiag)[::-1]
                feat_idx = step_counter * stats_per_step
                pe[i, feat_idx + 0] = np.mean(offdiag)
                pe[i, feat_idx + 1] = np.std(offdiag)
                pe[i, feat_idx + 2] = sorted_desc[0]  # max
                pe[i, feat_idx + 3] = sorted_desc[min(2, len(sorted_desc) - 1)]  # 3rd largest
            step_counter += 1

    return pe


def compute_nrwpe_offdiag_compact(edge_index: list, n: int,
                                  steps: list = None,
                                  stats: list = None,
                                  nonlinearity: str = "softplus") -> np.ndarray:
    """Compact off-diagonal nRWPE: [mean, max] at 4 steps → 8-dim."""
    if steps is None:
        steps = [1, 3, 7, 15]
    if stats is None:
        stats = ["mean", "max"]

    A_norm = build_norm_adj(edge_index, n)
    X = np.eye(n, dtype=np.float64)
    sigma = stable_softplus if nonlinearity == "softplus" else np.tanh

    total_dim = len(steps) * len(stats)
    pe = np.zeros((n, total_dim), dtype=np.float32)
    step_counter = 0

    for t in range(1, max(steps) + 1):
        X = sigma(A_norm @ X)
        max_val = np.abs(X).max()
        if max_val > 50:
            X = X / max_val * 10.0

        if t in steps:
            for i in range(n):
                col = X[:, i]
                mask = np.arange(n) != i
                offdiag = col[mask]
                if len(offdiag) == 0:
                    step_counter += 1
                    continue
                feat_idx = step_counter * len(stats)
                for si, stat in enumerate(stats):
                    if stat == "mean":
                        pe[i, feat_idx + si] = np.mean(offdiag)
                    elif stat == "max":
                        pe[i, feat_idx + si] = np.max(offdiag)
            step_counter += 1

    return pe


def compute_nrwpe_combined(edge_index: list, n: int) -> np.ndarray:
    """Combined nRWPE: 8-dim diag softplus + 8-dim offdiag compact."""
    diag_pe = compute_nrwpe_diag(edge_index, n, k=8, nonlinearity="softplus")
    offdiag_pe = compute_nrwpe_offdiag_compact(
        edge_index, n, steps=[1, 3, 7, 15], stats=["mean", "max"],
        nonlinearity="softplus"
    )
    return np.hstack([diag_pe, offdiag_pe])  # shape (n, 16)


def compute_all_pes_for_graph(args: tuple) -> dict:
    """Compute all PE types for a single graph. Used in parallel."""
    idx, edge_index, n = args
    try:
        result = {
            "idx": idx,
            "rwpe_16": compute_rwpe(edge_index, n, k=16),
            "nrwpe_diag_softplus_16": compute_nrwpe_diag(edge_index, n, k=16, nonlinearity="softplus"),
            "nrwpe_diag_tanh_16": compute_nrwpe_diag(edge_index, n, k=16, nonlinearity="tanh"),
            "nrwpe_offdiag_16": compute_nrwpe_offdiag(edge_index, n, steps=OFFDIAG_STEPS, nonlinearity="softplus"),
            "nrwpe_combined_16": compute_nrwpe_combined(edge_index, n),
            "no_pe": np.zeros((n, 16), dtype=np.float32),
        }
        return result
    except Exception as e:
        logger.error(f"Failed on graph {idx}: {e}")
        return {"idx": idx, "error": str(e)}


def normalize_pes(pe_dict: dict, train_indices: list, pe_types: list) -> dict:
    """Z-score normalize PEs using training set statistics, clip to [-5, 5]."""
    for pe_type in pe_types:
        if pe_type == "no_pe":
            continue
        # Gather training PEs
        train_pes = np.concatenate([pe_dict[pe_type][i] for i in train_indices], axis=0)
        mean = train_pes.mean(axis=0)
        std = train_pes.std(axis=0) + 1e-8
        logger.debug(f"  {pe_type}: mean={mean[:4]}..., std={std[:4]}...")
        # Normalize all graphs
        for i in range(len(pe_dict[pe_type])):
            pe_dict[pe_type][i] = np.clip(
                (pe_dict[pe_type][i] - mean) / std, -5, 5
            ).astype(np.float32)
    return pe_dict


@logger.catch
def main(max_graphs: int = None):
    """Main PE computation pipeline."""
    start_time = time.time()

    # Load data
    data_path = DATA_DIR / "full_data_out.json"
    logger.info(f"Loading data from {data_path}")
    raw = json.loads(data_path.read_text())
    examples = raw["datasets"][0]["examples"]
    if max_graphs is not None:
        examples = examples[:max_graphs]
    logger.info(f"Loaded {len(examples)} graphs")

    # Parse graphs
    graphs = []
    train_indices = []
    for i, ex in enumerate(examples):
        inp = json.loads(ex["input"])
        graphs.append((i, inp["edge_index"], inp["num_nodes"]))
        if ex["metadata_fold"] == "train":
            train_indices.append(i)
    logger.info(f"Parsed {len(graphs)} graphs, {len(train_indices)} train")

    # Compute PEs in parallel
    num_workers = max(1, NUM_CPUS - 1)
    logger.info(f"Computing PEs with {num_workers} workers...")

    pe_types = ["rwpe_16", "nrwpe_diag_softplus_16", "nrwpe_diag_tanh_16",
                "nrwpe_offdiag_16", "nrwpe_combined_16", "no_pe"]

    # Initialize result containers
    pe_dict = {pt: [None] * len(graphs) for pt in pe_types}
    errors = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(compute_all_pes_for_graph, g): g[0]
                   for g in graphs}
        done_count = 0
        for future in as_completed(futures):
            result = future.result()
            idx = result["idx"]
            if "error" in result:
                errors.append((idx, result["error"]))
            else:
                for pt in pe_types:
                    pe_dict[pt][idx] = result[pt]
            done_count += 1
            if done_count % 2000 == 0 or done_count == len(graphs):
                elapsed = time.time() - start_time
                logger.info(f"  Computed {done_count}/{len(graphs)} graphs ({elapsed:.1f}s)")

    if errors:
        logger.warning(f"{len(errors)} graphs had errors: {errors[:5]}")

    # Validate PEs
    logger.info("Validating PEs...")
    for pt in pe_types:
        nan_count = 0
        inf_count = 0
        zero_count = 0
        total_nodes = 0
        for i, pe in enumerate(pe_dict[pt]):
            if pe is None:
                logger.error(f"PE {pt} graph {i} is None!")
                continue
            nan_count += np.isnan(pe).sum()
            inf_count += np.isinf(pe).sum()
            if np.allclose(pe, 0):
                zero_count += 1
            total_nodes += pe.shape[0]
        all_pe = np.concatenate([p for p in pe_dict[pt] if p is not None], axis=0)
        logger.info(f"  {pt}: nodes={total_nodes}, nan={nan_count}, inf={inf_count}, "
                     f"all_zero_graphs={zero_count}/{len(graphs)}, "
                     f"var={all_pe.var():.6f}, range=[{all_pe.min():.4f}, {all_pe.max():.4f}]")

    # Normalize
    logger.info("Normalizing PEs (z-score on train set)...")
    pe_dict = normalize_pes(pe_dict, train_indices, pe_types)

    # Post-normalization stats
    for pt in pe_types:
        if pt == "no_pe":
            continue
        all_pe = np.concatenate(pe_dict[pt], axis=0)
        logger.info(f"  After norm {pt}: mean={all_pe.mean():.4f}, std={all_pe.std():.4f}, "
                     f"range=[{all_pe.min():.4f}, {all_pe.max():.4f}]")

    # Save (gzip-compressed to stay under 100MB file size limit)
    import gzip as _gzip
    output_path = SCRIPT_DIR / "precomputed_pes.pkl.gz"
    save_data = {
        "pe_dict": pe_dict,
        "pe_types": pe_types,
        "num_graphs": len(graphs),
        "train_indices": train_indices,
    }
    with _gzip.open(output_path, "wb") as f:
        pickle.dump(save_data, f)
    logger.info(f"Saved PEs to {output_path} ({output_path.stat().st_size / 1e6:.1f}MB)")

    elapsed = time.time() - start_time
    logger.info(f"PE computation complete in {elapsed:.1f}s")
    return pe_dict


if __name__ == "__main__":
    max_g = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(max_graphs=max_g)
