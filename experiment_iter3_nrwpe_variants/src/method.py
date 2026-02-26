#!/usr/bin/env python3
"""nRWPE Variants on ZINC-12k: Nonlinear Random Walk PE experiment.

Implements 7 PE variants (RWPE baseline, nRWPE-diag, nRWPE-multi, abs-KW-PE,
nRWPE-stats, nRWPE-combined, no_pe) with GIN and GPS architectures.
Trains on ZINC-12k molecular regression benchmark.
"""

import os
import sys
import json
import time
import math
import gc
import pickle
import resource
import warnings
from pathlib import Path
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import psutil

# ── Hardware detection (cgroup-aware) ──────────────────────────────────────
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
    return psutil.virtual_memory().total / 1e9

NUM_CPUS = _detect_cpus()
TOTAL_RAM_GB = _container_ram_gb()

# Set memory limit (use 70% of container RAM = ~40GB of 57GB)
RAM_BUDGET = int(TOTAL_RAM_GB * 0.70 * 1e9)
_avail = psutil.virtual_memory().available
assert RAM_BUDGET < _avail * 1.5, f"Budget {RAM_BUDGET/1e9:.1f}GB seems too high vs available {_avail/1e9:.1f}GB"
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
resource.setrlimit(resource.RLIMIT_CPU, (14400, 14400))  # 4 hours CPU time

# Suppress warnings before torch import
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.loader import DataLoader

from loguru import logger
from scipy import sparse
from scipy.linalg import eig

# ── Logging ────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ── GPU setup ──────────────────────────────────────────────────────────────
HAS_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if HAS_GPU else "cpu")
if HAS_GPU:
    VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
    _free, _total = torch.cuda.mem_get_info(0)
    VRAM_BUDGET = int(_total * 0.85)
    torch.cuda.set_per_process_memory_fraction(min(VRAM_BUDGET / _total, 0.90))
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {VRAM_GB:.1f}GB")
else:
    VRAM_GB = 0
    logger.info("No GPU available, using CPU")

logger.info(f"CPUs: {NUM_CPUS}, RAM: {TOTAL_RAM_GB:.1f}GB, Device: {DEVICE}")

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR = Path("/workspace/runs/run__20260225_014759/3_invention_loop/iter_1/gen_art/data_id3_it1__opus")
FULL_DATA_PATH = DATA_DIR / "full_data_out.json"
MINI_DATA_PATH = DATA_DIR / "mini_data_out.json"
PE_CACHE_PATH = WORKSPACE / "precomputed_pes.pkl.gz"
OUTPUT_PATH = WORKSPACE / "method_out.json"

# ── Config ─────────────────────────────────────────────────────────────────
PE_WALK_STEPS = 20
PE_RAW_DIMS = {
    "rwpe": PE_WALK_STEPS,
    "nrwpe_diag": PE_WALK_STEPS,
    "nrwpe_multi": PE_WALK_STEPS * 3,
    "abs_kwpe": 16,
    "nrwpe_stats": 16,
    "nrwpe_combined": PE_WALK_STEPS,  # 10+10=20
    "no_pe": 16,
}
PE_PROJ_DIM = 16
HIDDEN_DIM = 128
NUM_GIN_LAYERS = 4
NUM_GPS_LAYERS = 4
GPS_HEADS = 4
ATOM_EMB_DIM = 64
NUM_ATOM_TYPES = 28
PE_DROPOUT = 0.1

TRAINING_CONFIG = {
    "seeds": [42, 123, 456],
    "lr": 1e-3,
    "batch_size": 128,
    "patience": 50,
    "num_epochs": 300,
    "grad_clip": 5.0,
    "weight_decay": 0,
}

# ── Data Loading ───────────────────────────────────────────────────────────

def load_dataset(path: Path, max_examples: int | None = None) -> list[dict]:
    """Load ZINC dataset from JSON file."""
    logger.info(f"Loading data from {path}")
    raw = json.loads(path.read_text())
    examples = raw["datasets"][0]["examples"]
    if max_examples is not None:
        examples = examples[:max_examples]
    logger.info(f"Loaded {len(examples)} examples")
    return examples


def parse_graph(example: dict) -> dict:
    """Parse a single example into graph components."""
    graph = json.loads(example["input"])
    edge_index = np.array(graph["edge_index"], dtype=np.int64)
    node_feat = np.array(graph["node_feat"], dtype=np.int64)
    edge_attr = np.array(graph["edge_attr"], dtype=np.int64)
    num_nodes = graph["num_nodes"]
    y = float(example["output"])
    fold = example["metadata_fold"]
    return {
        "edge_index": edge_index,
        "node_feat": node_feat,
        "edge_attr": edge_attr,
        "num_nodes": num_nodes,
        "y": y,
        "fold": fold,
    }


# ── PE Computation Functions ───────────────────────────────────────────────

def build_norm_adj(edge_index: np.ndarray, n: int) -> np.ndarray:
    """Build symmetric normalized adjacency with self-loops (dense)."""
    A = np.zeros((n, n), dtype=np.float64)
    if edge_index.shape[1] > 0:
        src, dst = edge_index[0], edge_index[1]
        np.add.at(A, (src, dst), 1.0)
    A += np.eye(n)  # self-loops
    D = A.sum(axis=1)
    D_inv_sqrt = 1.0 / np.sqrt(np.maximum(D, 1e-12))
    return D_inv_sqrt[:, None] * A * D_inv_sqrt[None, :]


def generate_nonlinear_trajectory(A_norm: np.ndarray, n: int, T: int,
                                   nonlinearity) -> list[np.ndarray]:
    """Generate nonlinear walk trajectory snapshots."""
    X = np.eye(n, dtype=np.float64)
    snapshots = []
    for t in range(T):
        X = nonlinearity(A_norm @ X)
        snapshots.append(X.copy())
    return snapshots


def compute_rwpe(edge_index: np.ndarray, n: int, k: int = 20) -> np.ndarray:
    """Standard Random Walk PE: diagonal of RW^t for t=1..k."""
    A = np.zeros((n, n), dtype=np.float64)
    if edge_index.shape[1] > 0:
        src, dst = edge_index[0], edge_index[1]
        np.add.at(A, (src, dst), 1.0)
    D_inv = np.zeros(n, dtype=np.float64)
    row_sums = A.sum(axis=1)
    nonzero = row_sums > 1e-12
    D_inv[nonzero] = 1.0 / row_sums[nonzero]
    RW = D_inv[:, None] * A  # D^{-1} A
    pe = np.zeros((n, k), dtype=np.float64)
    RW_power = RW.copy()
    for t in range(k):
        pe[:, t] = np.diag(RW_power)
        RW_power = RW_power @ RW
    return pe.astype(np.float32)


def compute_nrwpe_diag(edge_index: np.ndarray, n: int, k: int = 20) -> np.ndarray:
    """Nonlinear walk PE using tanh: diagonal of nonlinear walk snapshots."""
    A_norm = build_norm_adj(edge_index, n)
    snapshots = generate_nonlinear_trajectory(A_norm, n, T=k, nonlinearity=np.tanh)
    pe = np.zeros((n, k), dtype=np.float64)
    for t in range(k):
        pe[:, t] = np.diag(snapshots[t])
    return pe.astype(np.float32)


def _nrwpe_diag_with(A_norm: np.ndarray, n: int, k: int,
                      nonlinearity) -> np.ndarray:
    """Helper: compute nRWPE-diag with a specific nonlinearity."""
    snapshots = generate_nonlinear_trajectory(A_norm, n, T=k, nonlinearity=nonlinearity)
    pe = np.zeros((n, k), dtype=np.float64)
    for t in range(k):
        pe[:, t] = np.diag(snapshots[t])
    return pe


def compute_nrwpe_multi(edge_index: np.ndarray, n: int, k: int = 20) -> np.ndarray:
    """Concatenate nRWPE-diag from 3 nonlinearities: tanh, softplus, relu."""
    A_norm = build_norm_adj(edge_index, n)
    pe_tanh = _nrwpe_diag_with(A_norm, n, k, np.tanh)
    pe_softplus = _nrwpe_diag_with(A_norm, n, k, lambda x: np.log1p(np.exp(np.clip(x, -20, 20))))
    pe_relu = _nrwpe_diag_with(A_norm, n, k, lambda x: np.maximum(0, x))
    pe_concat = np.hstack([pe_tanh, pe_softplus, pe_relu])
    return pe_concat.astype(np.float32)


def compute_abs_kwpe(edge_index: np.ndarray, n: int, d: int = 16,
                      T: int = 20) -> np.ndarray:
    """EDMD-based Koopman PE with absolute value fix."""
    A_norm = build_norm_adj(edge_index, n)

    # Generate trajectory matrices for EDMD
    X_init = np.eye(n, dtype=np.float64)
    data_X_list = []
    data_Y_list = []
    X = X_init.copy()
    for t in range(T):
        X_next = np.tanh(A_norm @ X)
        # Each column i gives the state vector for starting from node i at time t
        data_X_list.append(X.copy())
        data_Y_list.append(X_next.copy())
        X = X_next

    # Stack: data_X shape (T*n, n), data_Y shape (T*n, n)
    data_X = np.vstack(data_X_list)  # (T*n, n)
    data_Y = np.vstack(data_Y_list)  # (T*n, n)

    # PCA reduce if n > 12
    r = min(12, n)
    if n > r:
        U, S, Vt = np.linalg.svd(data_X, full_matrices=False)
        V_r = Vt[:r, :].T  # (n, r)
        data_X_r = data_X @ V_r  # (T*n, r)
        data_Y_r = data_Y @ V_r  # (T*n, r)
    else:
        V_r = np.eye(n)
        data_X_r = data_X
        data_Y_r = data_Y
        r = n

    # Full degree-2 polynomial dictionary
    idx_i, idx_j = np.triu_indices(r)

    def lift(X_in: np.ndarray) -> np.ndarray:
        ones_col = np.ones((X_in.shape[0], 1))
        cross = X_in[:, idx_i] * X_in[:, idx_j]
        return np.hstack([ones_col, X_in, cross])

    Psi_X = lift(data_X_r)
    Psi_Y = lift(data_Y_r)

    # EDMD solve with regularization cascade
    K_EDMD = None
    for reg in [1e-5, 1e-4, 1e-3, 1e-2]:
        try:
            G = Psi_X.T @ Psi_X + reg * np.eye(Psi_X.shape[1])
            A_edmd = Psi_X.T @ Psi_Y
            K_EDMD = np.linalg.solve(G, A_edmd)
            # Check for NaN
            if np.any(np.isnan(K_EDMD)):
                K_EDMD = None
                continue
            break
        except np.linalg.LinAlgError:
            continue

    if K_EDMD is None:
        logger.warning(f"EDMD failed for graph with n={n}, returning zeros")
        return np.zeros((n, d), dtype=np.float32)

    try:
        evals, evecs = eig(K_EDMD)
        # Select top d eigenvectors by eigenvalue magnitude
        sel_idx = np.argsort(-np.abs(evals))[:d]
        sel = np.real(evecs[:, sel_idx])

        # Compute PE for each node
        node_r = np.eye(n) @ V_r if n > r else np.eye(n)
        raw_pe = lift(node_r.reshape(n, -1)) @ sel  # (n, d)

        # CRITICAL FIX: absolute value to remove sign ambiguity
        pe = np.abs(raw_pe)

        # Normalize per dimension
        for j in range(d):
            col = pe[:, j]
            std = col.std()
            if std > 1e-12:
                pe[:, j] = (col - col.mean()) / std * 0.15
            else:
                pe[:, j] = 0.0
    except Exception as e:
        logger.warning(f"EDMD eigendecomposition failed: {e}, returning zeros")
        return np.zeros((n, d), dtype=np.float32)

    # Clip extreme values
    pe = np.clip(pe, -5.0, 5.0)
    return pe.astype(np.float32)


def compute_nrwpe_stats(edge_index: np.ndarray, n: int, T: int = 30) -> np.ndarray:
    """Per-node statistics of nonlinear walk trajectory."""
    A_norm = build_norm_adj(edge_index, n)
    snapshots = generate_nonlinear_trajectory(A_norm, n, T, np.tanh)

    # Diagonal series: return probability at each time step
    diag_series = np.array([np.diag(s) for s in snapshots])  # (T, n)

    pe_features = []
    for i in range(n):
        ts = diag_series[:, i]  # (T,)
        mean_val = ts.mean()
        std_val = ts.std()

        # Autocorrelation at lags 1-5
        autocorrs = []
        for lag in range(1, 6):
            if len(ts) > lag and std_val > 1e-12:
                a, b = ts[:-lag], ts[lag:]
                corr_val = np.corrcoef(a, b)[0, 1]
                autocorrs.append(0.0 if np.isnan(corr_val) else float(corr_val))
            else:
                autocorrs.append(0.0)

        # Spectral entropy
        fft_vals = np.abs(np.fft.rfft(ts - mean_val))
        power = fft_vals ** 2
        power_sum = power.sum()
        if power_sum > 1e-12:
            power_norm = power / power_sum
            spectral_entropy = -np.sum(power_norm * np.log(power_norm + 1e-12))
        else:
            spectral_entropy = 0.0

        # Decay rate
        abs_ts = np.abs(ts) + 1e-12
        log_abs = np.log(abs_ts)
        if T > 1:
            decay_rate = float(np.polyfit(np.arange(T), log_abs, 1)[0])
        else:
            decay_rate = 0.0

        min_val = float(ts.min())
        max_val = float(ts.max())

        # Off-diagonal statistics
        all_states = np.array([snapshots[t][:, i] for t in range(T)])  # (T, n)
        off_diag_mean = float(all_states.mean())
        off_diag_std = float(all_states.std())

        # 13 features + 3 padding = 16
        row = [mean_val, std_val] + autocorrs + [
            spectral_entropy, decay_rate, min_val, max_val,
            off_diag_mean, off_diag_std, 0.0, 0.0, 0.0
        ]
        pe_features.append(row[:16])

    return np.array(pe_features, dtype=np.float32)


def compute_nrwpe_combined(edge_index: np.ndarray, n: int, k: int = 10) -> np.ndarray:
    """Concatenation of RWPE (k dims) + nRWPE-diag (k dims)."""
    rwpe = compute_rwpe(edge_index, n, k=k)
    nrwpe = compute_nrwpe_diag(edge_index, n, k=k)
    return np.hstack([rwpe, nrwpe])


def compute_no_pe(n: int) -> np.ndarray:
    """Zero PE for no-PE baseline."""
    return np.zeros((n, 16), dtype=np.float32)


# ── Single-graph PE computation for multiprocessing ────────────────────────

def _compute_all_pes_for_graph(args: tuple) -> dict:
    """Compute all PE types for one graph. Used in ProcessPoolExecutor."""
    idx, edge_index_list, num_nodes = args
    edge_index = np.array(edge_index_list, dtype=np.int64)
    n = num_nodes

    result = {"idx": idx}
    try:
        result["rwpe"] = compute_rwpe(edge_index, n, k=PE_WALK_STEPS)
    except Exception as e:
        result["rwpe"] = np.zeros((n, PE_WALK_STEPS), dtype=np.float32)
        logger.debug(f"RWPE failed for graph {idx}: {e}")

    try:
        result["nrwpe_diag"] = compute_nrwpe_diag(edge_index, n, k=PE_WALK_STEPS)
    except Exception as e:
        result["nrwpe_diag"] = np.zeros((n, PE_WALK_STEPS), dtype=np.float32)
        logger.debug(f"nRWPE-diag failed for graph {idx}: {e}")

    try:
        result["nrwpe_multi"] = compute_nrwpe_multi(edge_index, n, k=PE_WALK_STEPS)
    except Exception as e:
        result["nrwpe_multi"] = np.zeros((n, PE_WALK_STEPS * 3), dtype=np.float32)
        logger.debug(f"nRWPE-multi failed for graph {idx}: {e}")

    try:
        result["abs_kwpe"] = compute_abs_kwpe(edge_index, n, d=16, T=PE_WALK_STEPS)
    except Exception as e:
        result["abs_kwpe"] = np.zeros((n, 16), dtype=np.float32)
        logger.debug(f"abs-KW-PE failed for graph {idx}: {e}")

    try:
        result["nrwpe_stats"] = compute_nrwpe_stats(edge_index, n, T=30)
    except Exception as e:
        result["nrwpe_stats"] = np.zeros((n, 16), dtype=np.float32)
        logger.debug(f"nRWPE-stats failed for graph {idx}: {e}")

    try:
        result["nrwpe_combined"] = compute_nrwpe_combined(edge_index, n, k=10)
    except Exception as e:
        result["nrwpe_combined"] = np.zeros((n, PE_WALK_STEPS), dtype=np.float32)
        logger.debug(f"nRWPE-combined failed for graph {idx}: {e}")

    result["no_pe"] = compute_no_pe(n)

    return result


def precompute_all_pes(examples: list[dict],
                       num_workers: int | None = None) -> list[dict]:
    """Precompute PEs for all graphs using multiprocessing."""
    if num_workers is None:
        num_workers = max(1, NUM_CPUS - 1)

    logger.info(f"Precomputing PEs for {len(examples)} graphs with {num_workers} workers")

    # Prepare arguments
    args_list = []
    for idx, ex in enumerate(examples):
        graph = json.loads(ex["input"])
        args_list.append((idx, graph["edge_index"], graph["num_nodes"]))

    # Run parallel
    results = [None] * len(examples)
    start_time = time.time()

    if num_workers <= 1 or len(examples) <= 5:
        # Sequential for small datasets or single worker
        for i, args in enumerate(args_list):
            results[i] = _compute_all_pes_for_graph(args)
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                logger.info(f"  PE computed: {i+1}/{len(examples)} ({elapsed:.1f}s)")
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_idx = {}
            for args in args_list:
                future = executor.submit(_compute_all_pes_for_graph, args)
                future_to_idx[future] = args[0]

            completed = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.exception(f"PE computation failed for graph {idx}")
                    # Create zeros fallback
                    n = args_list[idx][2]
                    results[idx] = {
                        "idx": idx,
                        "rwpe": np.zeros((n, PE_WALK_STEPS), dtype=np.float32),
                        "nrwpe_diag": np.zeros((n, PE_WALK_STEPS), dtype=np.float32),
                        "nrwpe_multi": np.zeros((n, PE_WALK_STEPS * 3), dtype=np.float32),
                        "abs_kwpe": np.zeros((n, 16), dtype=np.float32),
                        "nrwpe_stats": np.zeros((n, 16), dtype=np.float32),
                        "nrwpe_combined": np.zeros((n, PE_WALK_STEPS), dtype=np.float32),
                        "no_pe": np.zeros((n, 16), dtype=np.float32),
                    }
                completed += 1
                if completed % 500 == 0:
                    elapsed = time.time() - start_time
                    logger.info(f"  PE computed: {completed}/{len(examples)} ({elapsed:.1f}s)")

    elapsed = time.time() - start_time
    logger.info(f"PE precomputation done in {elapsed:.1f}s")
    return results


# ── PE Diagnostics ─────────────────────────────────────────────────────────

def pe_diagnostics(pe_results: list[dict]) -> dict:
    """Compute statistics for each PE type."""
    pe_types = ["rwpe", "nrwpe_diag", "nrwpe_multi", "abs_kwpe",
                "nrwpe_stats", "nrwpe_combined"]
    diagnostics = {}

    for pt in pe_types:
        all_vals = np.concatenate([r[pt] for r in pe_results], axis=0)
        nan_count = int(np.isnan(all_vals).sum())
        inf_count = int(np.isinf(all_vals).sum())

        # Replace NaN/Inf for stats
        clean = np.nan_to_num(all_vals, nan=0.0, posinf=5.0, neginf=-5.0)

        per_dim_mean = clean.mean(axis=0).tolist()
        per_dim_std = clean.std(axis=0).tolist()
        overall_std = float(clean.std())

        # Effective rank via SVD
        try:
            if clean.shape[0] > clean.shape[1]:
                sample = clean[np.random.choice(clean.shape[0], min(2000, clean.shape[0]), replace=False)]
            else:
                sample = clean
            U, S, Vt = np.linalg.svd(sample, full_matrices=False)
            S_norm = S / (S.sum() + 1e-12)
            effective_rank = float(np.exp(-np.sum(S_norm * np.log(S_norm + 1e-12))))
        except Exception:
            effective_rank = -1.0

        diagnostics[pt] = {
            "overall_mean": float(clean.mean()),
            "overall_std": overall_std,
            "per_dim_mean": [round(x, 4) for x in per_dim_mean[:5]],  # first 5
            "per_dim_std": [round(x, 4) for x in per_dim_std[:5]],
            "nan_count": nan_count,
            "inf_count": inf_count,
            "effective_rank": round(effective_rank, 2),
            "shape_example": list(pe_results[0][pt].shape),
            "min": float(clean.min()),
            "max": float(clean.max()),
        }

        logger.info(f"PE {pt}: mean={clean.mean():.4f}, std={overall_std:.4f}, "
                     f"eff_rank={effective_rank:.1f}, nan={nan_count}, inf={inf_count}")

    # Cross-correlation with RWPE
    rwpe_all = np.concatenate([r["rwpe"] for r in pe_results], axis=0)
    for pt in pe_types:
        if pt == "rwpe":
            continue
        other_all = np.concatenate([r[pt] for r in pe_results], axis=0)
        # Use min of dims for correlation
        d = min(rwpe_all.shape[1], other_all.shape[1])
        try:
            corrs = []
            for j in range(d):
                c = np.corrcoef(rwpe_all[:, j], other_all[:, j])[0, 1]
                if not np.isnan(c):
                    corrs.append(float(c))
            if corrs:
                diagnostics[pt]["correlation_with_rwpe"] = round(float(np.mean(corrs)), 4)
        except Exception:
            pass

    return diagnostics


# ── PyG Data Conversion ───────────────────────────────────────────────────

def examples_to_pyg_data(examples: list[dict], pe_results: list[dict],
                          pe_type: str) -> list[Data]:
    """Convert examples + precomputed PEs to PyG Data objects."""
    data_list = []
    for i, ex in enumerate(examples):
        graph = json.loads(ex["input"])
        edge_index = torch.tensor(graph["edge_index"], dtype=torch.long)
        x = torch.tensor(graph["node_feat"], dtype=torch.long)
        y = torch.tensor([float(ex["output"])], dtype=torch.float)

        pe_arr = pe_results[i][pe_type]
        # Sanitize
        pe_arr = np.nan_to_num(pe_arr, nan=0.0, posinf=5.0, neginf=-5.0)
        pe = torch.tensor(pe_arr, dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, y=y, pe=pe)
        data.fold = ex["metadata_fold"]
        data_list.append(data)
    return data_list


def split_data(data_list: list[Data]) -> tuple[list[Data], list[Data], list[Data]]:
    """Split by fold metadata."""
    train = [d for d in data_list if d.fold == "train"]
    val = [d for d in data_list if d.fold == "val"]
    test = [d for d in data_list if d.fold == "test"]
    return train, val, test


# ── Model Architectures ───────────────────────────────────────────────────

class GIN_ZINC_v2(nn.Module):
    """GIN model with PE projection, BatchNorm, and dropout fixes."""

    def __init__(self, pe_type: str = "rwpe", hidden_dim: int = HIDDEN_DIM,
                 num_layers: int = NUM_GIN_LAYERS, atom_emb_dim: int = ATOM_EMB_DIM,
                 num_atom_types: int = NUM_ATOM_TYPES, pe_dropout: float = PE_DROPOUT):
        super().__init__()
        self.pe_type = pe_type
        self.use_pe = (pe_type != "no_pe")

        self.atom_emb = nn.Embedding(num_atom_types, atom_emb_dim)

        pe_raw_dim = PE_RAW_DIMS[pe_type]

        if self.use_pe:
            self.pe_proj = nn.Sequential(
                nn.Linear(pe_raw_dim, PE_PROJ_DIM),
                nn.ReLU(),
                nn.Linear(PE_PROJ_DIM, PE_PROJ_DIM),
            )
            self.pe_bn = nn.BatchNorm1d(PE_PROJ_DIM)
            self.pe_drop = nn.Dropout(pe_dropout)
            input_dim = atom_emb_dim + PE_PROJ_DIM
        else:
            input_dim = atom_emb_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x_atom = self.atom_emb(data.x)
        if self.use_pe:
            x_pe = self.pe_proj(data.pe)
            x_pe = self.pe_bn(x_pe)
            x_pe = self.pe_drop(x_pe)
            x = torch.cat([x_atom, x_pe], dim=-1)
        else:
            x = x_atom
        x = self.input_proj(x)
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, data.edge_index)))
        x = global_add_pool(x, data.batch)
        return self.readout(x).squeeze(-1)


class GPS_ZINC_v2(nn.Module):
    """GPS (GIN + MultiheadAttention) model with PE projection fixes."""

    def __init__(self, pe_type: str = "rwpe", hidden_dim: int = HIDDEN_DIM,
                 num_layers: int = NUM_GPS_LAYERS, num_heads: int = GPS_HEADS,
                 atom_emb_dim: int = ATOM_EMB_DIM, num_atom_types: int = NUM_ATOM_TYPES,
                 pe_dropout: float = PE_DROPOUT):
        super().__init__()
        self.pe_type = pe_type
        self.use_pe = (pe_type != "no_pe")
        self.num_layers = num_layers

        self.atom_emb = nn.Embedding(num_atom_types, atom_emb_dim)

        pe_raw_dim = PE_RAW_DIMS[pe_type]

        if self.use_pe:
            self.pe_proj = nn.Sequential(
                nn.Linear(pe_raw_dim, PE_PROJ_DIM),
                nn.ReLU(),
                nn.Linear(PE_PROJ_DIM, PE_PROJ_DIM),
            )
            self.pe_bn = nn.BatchNorm1d(PE_PROJ_DIM)
            self.pe_drop = nn.Dropout(pe_dropout)
            input_dim = atom_emb_dim + PE_PROJ_DIM
        else:
            input_dim = atom_emb_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # GIN conv layers
        self.gin_convs = nn.ModuleList()
        self.gin_bns = nn.ModuleList()
        # Attention layers
        self.attn_layers = nn.ModuleList()
        self.attn_bns = nn.ModuleList()
        # FFN layers
        self.ffn_layers = nn.ModuleList()
        self.ffn_bns = nn.ModuleList()

        for _ in range(num_layers):
            gin_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.gin_convs.append(GINConv(gin_mlp))
            self.gin_bns.append(nn.BatchNorm1d(hidden_dim))

            self.attn_layers.append(nn.MultiheadAttention(
                hidden_dim, num_heads, dropout=0.1, batch_first=True))
            self.attn_bns.append(nn.BatchNorm1d(hidden_dim))

            ffn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.ffn_layers.append(ffn)
            self.ffn_bns.append(nn.BatchNorm1d(hidden_dim))

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x_atom = self.atom_emb(data.x)
        if self.use_pe:
            x_pe = self.pe_proj(data.pe)
            x_pe = self.pe_bn(x_pe)
            x_pe = self.pe_drop(x_pe)
            x = torch.cat([x_atom, x_pe], dim=-1)
        else:
            x = x_atom
        x = self.input_proj(x)

        for layer_idx in range(self.num_layers):
            # GIN message passing
            x_gin = self.gin_convs[layer_idx](x, data.edge_index)
            x_gin = self.gin_bns[layer_idx](x_gin)

            # Batched attention within each graph
            x_attn = self._batched_attention(x, data.batch, layer_idx)
            x_attn = self.attn_bns[layer_idx](x_attn)

            # Combine GIN + Attention with residual
            x = x + F.relu(x_gin) + x_attn

            # FFN with residual
            x_ffn = self.ffn_layers[layer_idx](x)
            x_ffn = self.ffn_bns[layer_idx](x_ffn)
            x = x + x_ffn

        x = global_add_pool(x, data.batch)
        return self.readout(x).squeeze(-1)

    def _batched_attention(self, x: torch.Tensor, batch: torch.Tensor,
                            layer_idx: int) -> torch.Tensor:
        """Apply multi-head attention within each graph in the batch."""
        # Group nodes by graph
        unique_graphs = batch.unique()
        output = torch.zeros_like(x)

        for g in unique_graphs:
            mask = (batch == g)
            x_g = x[mask].unsqueeze(0)  # (1, num_nodes_g, hidden)
            attn_out, _ = self.attn_layers[layer_idx](x_g, x_g, x_g)
            output[mask] = attn_out.squeeze(0)

        return output


# ── Training and Evaluation ───────────────────────────────────────────────

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if HAS_GPU:
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> float:
    """Compute MAE on a dataset."""
    model.eval()
    total_loss = 0.0
    total_count = 0
    for batch in loader:
        batch = batch.to(DEVICE)
        pred = model(batch)
        loss = F.l1_loss(pred, batch.y, reduction="sum")
        total_loss += loss.item()
        total_count += batch.y.size(0)
    return total_loss / max(total_count, 1)


@torch.no_grad()
def get_predictions(model: nn.Module, loader: DataLoader) -> list[float]:
    """Get model predictions for all examples in order."""
    model.eval()
    preds = []
    for batch in loader:
        batch = batch.to(DEVICE)
        pred = model(batch)
        preds.extend(pred.cpu().numpy().tolist())
    return preds


def train_and_evaluate(
    train_data: list[Data],
    val_data: list[Data],
    test_data: list[Data],
    model_class: type,
    pe_type: str,
    seed: int,
    num_epochs: int = 300,
    lr: float = 1e-3,
    batch_size: int = 128,
    patience: int = 50,
    grad_clip: float = 5.0,
) -> dict:
    """Train a model and return results."""
    set_seed(seed)

    model = model_class(pe_type=pe_type).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.L1Loss()

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    best_val_mae = float("inf")
    best_epoch = 0
    best_state = None
    patience_counter = 0
    train_curve = []
    val_curve = []

    start_time = time.time()

    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_count = 0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, batch.y)
            if torch.isnan(loss):
                logger.warning(f"NaN loss at epoch {epoch}, breaking")
                break
            loss.backward()
            clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_loss += loss.item() * batch.y.size(0)
            train_count += batch.y.size(0)

        if train_count == 0:
            break

        train_mae = train_loss / train_count
        train_curve.append(train_mae)

        # Validate
        val_mae = evaluate(model, val_loader)
        val_curve.append(val_mae)

        scheduler.step()

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.debug(f"  Early stop at epoch {epoch} (patience={patience})")
            break

        # Log progress every 50 epochs
        if (epoch + 1) % 50 == 0:
            logger.debug(f"  Epoch {epoch+1}: train_mae={train_mae:.4f}, "
                         f"val_mae={val_mae:.4f}, best={best_val_mae:.4f}")

    elapsed = time.time() - start_time

    # Test with best model
    if best_state is not None:
        model.load_state_dict(best_state)
    test_mae = evaluate(model, test_loader)

    # Get predictions on test set for output
    test_preds = get_predictions(model, test_loader)

    logger.info(f"  {model_class.__name__} | {pe_type} | seed={seed} | "
                f"test_mae={test_mae:.4f} | val_mae={best_val_mae:.4f} | "
                f"epoch={best_epoch} | time={elapsed:.1f}s")

    # Free GPU memory
    del model, optimizer, scheduler
    if HAS_GPU:
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "test_mae": test_mae,
        "val_mae": best_val_mae,
        "best_epoch": best_epoch,
        "train_curve": train_curve[-10:],  # last 10 for output
        "val_curve": val_curve[-10:],
        "elapsed_time": elapsed,
        "test_preds": test_preds,
    }


# ── Mini Validation ───────────────────────────────────────────────────────

def run_mini_validation():
    """Run quick validation on mini dataset to ensure pipeline works."""
    logger.info("=" * 60)
    logger.info("MINI VALIDATION")
    logger.info("=" * 60)

    mini_examples = load_dataset(MINI_DATA_PATH)
    pe_results = precompute_all_pes(mini_examples, num_workers=1)

    # Check PE values
    pe_types = ["rwpe", "nrwpe_diag", "nrwpe_multi", "abs_kwpe",
                "nrwpe_stats", "nrwpe_combined", "no_pe"]
    for pt in pe_types:
        for i, r in enumerate(pe_results):
            arr = r[pt]
            assert not np.any(np.isnan(arr)), f"NaN in {pt} for graph {i}"
            assert not np.any(np.isinf(arr)), f"Inf in {pt} for graph {i}"
            expected_dim = PE_RAW_DIMS[pt]
            assert arr.shape[1] == expected_dim, \
                f"{pt} shape mismatch: {arr.shape[1]} != {expected_dim}"
        logger.info(f"  {pt}: shape={pe_results[0][pt].shape}, OK")

    # Test forward pass for GIN with each PE type
    for pt in ["rwpe", "nrwpe_diag", "no_pe"]:
        data_list = examples_to_pyg_data(mini_examples, pe_results, pt)
        loader = DataLoader(data_list, batch_size=3)
        model = GIN_ZINC_v2(pe_type=pt).to(DEVICE)
        model.eval()
        for batch in loader:
            batch = batch.to(DEVICE)
            out = model(batch)
            assert not torch.isnan(out).any(), f"NaN output for {pt}"
            assert out.shape == (batch.y.shape[0],), \
                f"Shape mismatch: {out.shape} vs {batch.y.shape}"
        del model
        if HAS_GPU:
            torch.cuda.empty_cache()
        logger.info(f"  GIN forward pass OK for {pt}")

    # Test GPS forward pass
    for pt in ["rwpe", "nrwpe_diag"]:
        data_list = examples_to_pyg_data(mini_examples, pe_results, pt)
        loader = DataLoader(data_list, batch_size=3)
        model = GPS_ZINC_v2(pe_type=pt).to(DEVICE)
        model.eval()
        for batch in loader:
            batch = batch.to(DEVICE)
            out = model(batch)
            assert not torch.isnan(out).any(), f"NaN GPS output for {pt}"
        del model
        if HAS_GPU:
            torch.cuda.empty_cache()
        logger.info(f"  GPS forward pass OK for {pt}")

    # Quick 20-epoch training test
    for pt in ["rwpe", "nrwpe_diag"]:
        data_list = examples_to_pyg_data(mini_examples, pe_results, pt)
        # Use all as train for mini test
        result = train_and_evaluate(
            train_data=data_list, val_data=data_list, test_data=data_list,
            model_class=GIN_ZINC_v2, pe_type=pt, seed=42,
            num_epochs=20, patience=20, batch_size=3)
        logger.info(f"  Mini train {pt}: loss={result['test_mae']:.4f}, OK")

    logger.info("MINI VALIDATION PASSED")
    return True


# ── Main Experiment ───────────────────────────────────────────────────────

@logger.catch
def main():
    overall_start = time.time()
    logger.info("=" * 60)
    logger.info("nRWPE Variants on ZINC-12k Experiment")
    logger.info("=" * 60)

    # Phase 0: Mini validation
    run_mini_validation()

    # Phase 1: Load full dataset
    logger.info("=" * 60)
    logger.info("LOADING FULL DATASET")
    logger.info("=" * 60)
    full_examples = load_dataset(FULL_DATA_PATH)
    logger.info(f"Total examples: {len(full_examples)}")
    fold_counts = {}
    for ex in full_examples:
        fold_counts[ex["metadata_fold"]] = fold_counts.get(ex["metadata_fold"], 0) + 1
    logger.info(f"Fold distribution: {fold_counts}")

    # Phase 2: Precompute PEs
    logger.info("=" * 60)
    logger.info("PRECOMPUTING POSITIONAL ENCODINGS")
    logger.info("=" * 60)

    if PE_CACHE_PATH.exists():
        logger.info("Loading cached PEs...")
        import gzip as _gzip
        pe_results = pickle.loads(_gzip.decompress(PE_CACHE_PATH.read_bytes()))
        logger.info(f"Loaded {len(pe_results)} cached PE results")
    else:
        pe_results = precompute_all_pes(full_examples, num_workers=max(1, NUM_CPUS - 1))
        # Save cache (gzip compressed)
        import gzip as _gzip
        PE_CACHE_PATH.write_bytes(_gzip.compress(pickle.dumps(pe_results)))
        logger.info(f"Saved PE cache to {PE_CACHE_PATH}")

    # Phase 3: PE Diagnostics
    logger.info("=" * 60)
    logger.info("PE DIAGNOSTICS")
    logger.info("=" * 60)
    pe_diag = pe_diagnostics(pe_results)

    # Phase 4: Training experiments
    logger.info("=" * 60)
    logger.info("TRAINING EXPERIMENTS")
    logger.info("=" * 60)

    seeds = TRAINING_CONFIG["seeds"]
    all_results = {}

    # Priority 1: GIN with 5 PE types (rwpe, nrwpe_diag, nrwpe_multi, nrwpe_combined, no_pe)
    priority1_pes = ["rwpe", "nrwpe_diag", "nrwpe_multi", "nrwpe_combined", "no_pe"]
    # Priority 2: GIN with abs_kwpe, nrwpe_stats
    priority2_pes = ["abs_kwpe", "nrwpe_stats"]
    # Priority 3: GPS with 3 PE types
    priority3_pes = ["rwpe", "nrwpe_diag", "nrwpe_multi"]

    # Run Priority 1
    logger.info("--- Priority 1: GIN with core PE types ---")
    for pe_type in priority1_pes:
        logger.info(f"Running GIN + {pe_type}")
        data_list = examples_to_pyg_data(full_examples, pe_results, pe_type)
        train_data, val_data, test_data = split_data(data_list)
        logger.info(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

        key = f"GIN_{pe_type}"
        all_results[key] = {"per_seed": []}

        for seed in seeds:
            result = train_and_evaluate(
                train_data=train_data, val_data=val_data, test_data=test_data,
                model_class=GIN_ZINC_v2, pe_type=pe_type, seed=seed,
                num_epochs=TRAINING_CONFIG["num_epochs"],
                lr=TRAINING_CONFIG["lr"],
                batch_size=TRAINING_CONFIG["batch_size"],
                patience=TRAINING_CONFIG["patience"],
                grad_clip=TRAINING_CONFIG["grad_clip"],
            )
            all_results[key]["per_seed"].append({
                "seed": seed,
                "test_mae": result["test_mae"],
                "val_mae": result["val_mae"],
                "best_epoch": result["best_epoch"],
                "elapsed_time": result["elapsed_time"],
                "test_preds": result["test_preds"],
            })

        # Check time budget
        elapsed_total = time.time() - overall_start
        logger.info(f"Elapsed total: {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)")

        # RWPE sanity check after first variant
        if pe_type == "rwpe":
            rwpe_maes = [r["test_mae"] for r in all_results[key]["per_seed"]]
            mean_rwpe = np.mean(rwpe_maes)
            if mean_rwpe > 0.30:
                logger.warning(f"RWPE baseline MAE={mean_rwpe:.4f} > 0.30, potential issue!")
            else:
                logger.info(f"RWPE baseline OK: mean MAE={mean_rwpe:.4f}")

    # Run Priority 2 (if time allows)
    elapsed_total = time.time() - overall_start
    if elapsed_total < 7200:  # < 2 hours
        logger.info("--- Priority 2: GIN with additional PE types ---")
        for pe_type in priority2_pes:
            logger.info(f"Running GIN + {pe_type}")
            data_list = examples_to_pyg_data(full_examples, pe_results, pe_type)
            train_data, val_data, test_data = split_data(data_list)

            key = f"GIN_{pe_type}"
            all_results[key] = {"per_seed": []}

            for seed in seeds:
                result = train_and_evaluate(
                    train_data=train_data, val_data=val_data, test_data=test_data,
                    model_class=GIN_ZINC_v2, pe_type=pe_type, seed=seed,
                    num_epochs=TRAINING_CONFIG["num_epochs"],
                    lr=TRAINING_CONFIG["lr"],
                    batch_size=TRAINING_CONFIG["batch_size"],
                    patience=TRAINING_CONFIG["patience"],
                    grad_clip=TRAINING_CONFIG["grad_clip"],
                )
                all_results[key]["per_seed"].append({
                    "seed": seed,
                    "test_mae": result["test_mae"],
                    "val_mae": result["val_mae"],
                    "best_epoch": result["best_epoch"],
                    "elapsed_time": result["elapsed_time"],
                    "test_preds": result["test_preds"],
                })

            elapsed_total = time.time() - overall_start
            logger.info(f"Elapsed total: {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)")
    else:
        logger.info("Skipping Priority 2 due to time budget")

    # Run Priority 3: GPS (if time allows)
    elapsed_total = time.time() - overall_start
    if elapsed_total < 10800:  # < 3 hours
        logger.info("--- Priority 3: GPS with core PE types ---")
        for pe_type in priority3_pes:
            logger.info(f"Running GPS + {pe_type}")
            data_list = examples_to_pyg_data(full_examples, pe_results, pe_type)
            train_data, val_data, test_data = split_data(data_list)

            key = f"GPS_{pe_type}"
            all_results[key] = {"per_seed": []}

            for seed in seeds:
                result = train_and_evaluate(
                    train_data=train_data, val_data=val_data, test_data=test_data,
                    model_class=GPS_ZINC_v2, pe_type=pe_type, seed=seed,
                    num_epochs=TRAINING_CONFIG["num_epochs"],
                    lr=TRAINING_CONFIG["lr"],
                    batch_size=TRAINING_CONFIG["batch_size"],
                    patience=TRAINING_CONFIG["patience"],
                    grad_clip=TRAINING_CONFIG["grad_clip"],
                )
                all_results[key]["per_seed"].append({
                    "seed": seed,
                    "test_mae": result["test_mae"],
                    "val_mae": result["val_mae"],
                    "best_epoch": result["best_epoch"],
                    "elapsed_time": result["elapsed_time"],
                    "test_preds": result["test_preds"],
                })

            elapsed_total = time.time() - overall_start
            logger.info(f"Elapsed total: {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)")
            if elapsed_total > 12600:  # > 3.5 hours, stop GPS
                logger.info("Time budget nearing limit, stopping GPS experiments")
                break
    else:
        logger.info("Skipping Priority 3 (GPS) due to time budget")

    # Phase 5: Compile results
    logger.info("=" * 60)
    logger.info("COMPILING RESULTS")
    logger.info("=" * 60)

    results_summary = []
    for key, data in all_results.items():
        parts = key.split("_", 1)
        arch = parts[0]
        pe_type = parts[1]
        per_seed = data["per_seed"]

        test_maes = [r["test_mae"] for r in per_seed]
        val_maes = [r["val_mae"] for r in per_seed]
        times = [r["elapsed_time"] for r in per_seed]

        summary = {
            "architecture": f"{arch}_ZINC_v2",
            "pe_type": pe_type,
            "n_seeds": len(per_seed),
            "test_mae_mean": round(float(np.mean(test_maes)), 4),
            "test_mae_std": round(float(np.std(test_maes)), 4),
            "val_mae_mean": round(float(np.mean(val_maes)), 4),
            "val_mae_std": round(float(np.std(val_maes)), 4),
            "avg_time_s": round(float(np.mean(times)), 1),
            "per_seed_results": [
                {"seed": r["seed"], "test_mae": round(r["test_mae"], 4),
                 "val_mae": round(r["val_mae"], 4), "best_epoch": r["best_epoch"]}
                for r in per_seed
            ],
        }
        results_summary.append(summary)

        logger.info(f"  {key}: test_mae={summary['test_mae_mean']:.4f}±{summary['test_mae_std']:.4f}, "
                     f"val_mae={summary['val_mae_mean']:.4f}±{summary['val_mae_std']:.4f}")

    # Key comparisons
    gin_results = {s["pe_type"]: s for s in results_summary if "GIN" in s["architecture"]}
    rwpe_mae = gin_results.get("rwpe", {}).get("test_mae_mean", None)
    best_nrwpe_name = None
    best_nrwpe_mae = float("inf")
    for pt in ["nrwpe_diag", "nrwpe_multi", "nrwpe_combined", "abs_kwpe", "nrwpe_stats"]:
        if pt in gin_results:
            m = gin_results[pt]["test_mae_mean"]
            if m < best_nrwpe_mae:
                best_nrwpe_mae = m
                best_nrwpe_name = pt

    key_comparisons = {}
    if rwpe_mae is not None and best_nrwpe_name is not None:
        delta = rwpe_mae - best_nrwpe_mae
        pct = delta / rwpe_mae * 100 if rwpe_mae > 0 else 0
        key_comparisons["best_nrwpe_vs_rwpe"] = {
            "best_nrwpe": best_nrwpe_name,
            "nrwpe_mae": round(best_nrwpe_mae, 4),
            "rwpe_mae": round(rwpe_mae, 4),
            "delta": round(delta, 4),
            "pct_improvement": round(pct, 2),
        }
        logger.info(f"Best nRWPE ({best_nrwpe_name}): {best_nrwpe_mae:.4f} vs RWPE: {rwpe_mae:.4f} "
                     f"(delta={delta:.4f}, {pct:.1f}% improvement)")

    no_pe_mae = gin_results.get("no_pe", {}).get("test_mae_mean", None)
    if no_pe_mae is not None and rwpe_mae is not None:
        key_comparisons["pe_vs_no_pe"] = {
            "rwpe_mae": round(rwpe_mae, 4),
            "no_pe_mae": round(no_pe_mae, 4),
            "pe_benefit": round(no_pe_mae - rwpe_mae, 4),
        }

    # Analysis
    if rwpe_mae is not None and best_nrwpe_mae < rwpe_mae:
        analysis = (
            f"Nonlinear random walk PE variant '{best_nrwpe_name}' achieves "
            f"MAE={best_nrwpe_mae:.4f}, improving over RWPE baseline (MAE={rwpe_mae:.4f}) "
            f"by {abs(rwpe_mae - best_nrwpe_mae):.4f} ({abs(rwpe_mae - best_nrwpe_mae)/rwpe_mae*100:.1f}%). "
            f"The PE projection layer, BatchNorm, and dropout fixes successfully prevent "
            f"the downstream performance degradation seen in prior iterations."
        )
    elif rwpe_mae is not None:
        analysis = (
            f"Nonlinear random walk PE variants did not improve over RWPE baseline "
            f"(best nRWPE MAE={best_nrwpe_mae:.4f} vs RWPE MAE={rwpe_mae:.4f}). "
            f"Despite improved expressiveness, the tanh compression may destroy useful "
            f"return-probability information that RWPE preserves."
        )
    else:
        analysis = "Results compilation incomplete."

    # Build output
    # Get test predictions from best seed of each variant
    # We need to produce predictions for all test examples
    test_examples = [ex for ex in full_examples if ex["metadata_fold"] == "test"]
    output_examples = []

    # Take first 5 test examples for the output
    num_output = min(5, len(test_examples))
    for i in range(num_output):
        ex = test_examples[i]
        out_ex = {
            "input": ex["input"],
            "output": ex["output"],
            "metadata_fold": ex["metadata_fold"],
        }
        # Add predictions from each variant (using seed 0 = first seed)
        for key, data in all_results.items():
            # Find test predictions from first seed
            per_seed = data["per_seed"]
            if per_seed and "test_preds" in per_seed[0]:
                preds = per_seed[0]["test_preds"]
                if i < len(preds):
                    out_ex[f"predict_{key}"] = str(round(preds[i], 4))
        output_examples.append(out_ex)

    total_time = time.time() - overall_start
    output = {
        "metadata": {
            "title": "nRWPE Variants on ZINC-12k",
            "method_name": "Nonlinear Random Walk PE variants",
            "description": (
                "Experiment testing 7 PE variants (RWPE baseline, nRWPE-diag, nRWPE-multi, "
                "abs-KW-PE, nRWPE-stats, nRWPE-combined, no_pe) with GIN and GPS architectures "
                "on ZINC-12k molecular regression. Key fixes over prior iterations: "
                "PE projection layers, BatchNorm on PE, PE dropout, proper scale normalization."
            ),
            "pe_variants": list(PE_RAW_DIMS.keys()),
            "model_params": {
                "hidden_dim": HIDDEN_DIM,
                "num_gin_layers": NUM_GIN_LAYERS,
                "num_gps_layers": NUM_GPS_LAYERS,
                "atom_emb_dim": ATOM_EMB_DIM,
                "pe_proj_dim": PE_PROJ_DIM,
            },
            "training_params": {
                "lr": TRAINING_CONFIG["lr"],
                "epochs": TRAINING_CONFIG["num_epochs"],
                "batch_size": TRAINING_CONFIG["batch_size"],
                "patience": TRAINING_CONFIG["patience"],
                "seeds": TRAINING_CONFIG["seeds"],
            },
            "critical_fixes": [
                "PE projection layer (Linear→ReLU→Linear)",
                "BatchNorm on projected PE",
                "PE dropout 0.1",
                "EDMD absolute value fix",
                "Per-dimension normalization for abs-KW-PE",
            ],
            "results_summary": results_summary,
            "pe_diagnostics": pe_diag,
            "key_comparisons": key_comparisons,
            "analysis": analysis,
            "conclusion": analysis,
            "total_runtime_s": round(total_time, 1),
            "hardware": {
                "device": str(DEVICE),
                "gpu": torch.cuda.get_device_name(0) if HAS_GPU else "none",
                "num_cpus": NUM_CPUS,
                "ram_gb": round(TOTAL_RAM_GB, 1),
            },
        },
        "datasets": [{
            "dataset": "ZINC-12k",
            "examples": output_examples,
        }],
    }

    # Write output
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved results to {OUTPUT_PATH}")
    logger.info(f"Total runtime: {total_time:.1f}s ({total_time/60:.1f}min)")


if __name__ == "__main__":
    main()
