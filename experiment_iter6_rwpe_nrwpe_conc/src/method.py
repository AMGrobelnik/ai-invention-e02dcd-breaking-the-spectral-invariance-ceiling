#!/usr/bin/env python3
"""
RWPE + nRWPE Concatenation on ZINC-12k with GINEConv.

Tests whether concatenating RWPE and nRWPE-diag-tanh positional encodings
improves ZINC-12k regression MAE beyond the GINEConv+RWPE-16 baseline.

PE configurations (9 total):
  rwpe_16:       RWPE(k=16)                → 16-dim
  nrwpe_tanh_16: nRWPE(k=16)              → 16-dim
  concat_16:     RWPE(k=8) || nRWPE(k=8)  → 16-dim
  concat_24:     RWPE(k=16) || nRWPE(k=8) → 24-dim
  concat_32:     RWPE(k=16) || nRWPE(k=16)→ 32-dim
  mild_16:       Mild-nRWPE(k=16)          → 16-dim
  rwpe_24:       RWPE(k=24) (dim control)  → 24-dim
  rwpe_32:       RWPE(k=32) (dim control)  → 32-dim
  no_pe:         zeros                     → 16-dim
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import json
import gc
import math
import pickle
import resource
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import psutil
import scipy.sparse as sp
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool
from torch_geometric.loader import DataLoader

warnings.filterwarnings("ignore")

# ============================================================
# LOGGING
# ============================================================
WORKSPACE = Path(__file__).parent
LOGS_DIR = WORKSPACE / "logs"
LOGS_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOGS_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ============================================================
# HARDWARE DETECTION
# ============================================================
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
HAS_GPU = torch.cuda.is_available()
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9 if HAS_GPU else 0
DEVICE = torch.device("cuda" if HAS_GPU else "cpu")
TOTAL_RAM_GB = _container_ram_gb() or psutil.virtual_memory().total / 1e9
AVAILABLE_RAM_GB = min(psutil.virtual_memory().available / 1e9, TOTAL_RAM_GB)

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f}GB RAM, GPU={HAS_GPU} ({VRAM_GB:.1f}GB VRAM)")
logger.info(f"Device: {DEVICE}")

# Memory limits
RAM_BUDGET = int(min(AVAILABLE_RAM_GB * 0.8, TOTAL_RAM_GB * 0.85) * 1e9)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

if HAS_GPU:
    _free, _total = torch.cuda.mem_get_info(0)
    VRAM_BUDGET = int(_total * 0.9)
    torch.cuda.set_per_process_memory_fraction(min(VRAM_BUDGET / _total, 0.95))

logger.info(f"RAM budget: {RAM_BUDGET/1e9:.1f}GB, VRAM budget: {VRAM_BUDGET/1e9:.1f}GB" if HAS_GPU else f"RAM budget: {RAM_BUDGET/1e9:.1f}GB")

# ============================================================
# DATA PATHS
# ============================================================
DATA_DIR = Path("/workspace/runs/run__20260225_014759/3_invention_loop/iter_1/gen_art/data_id3_it1__opus")
FULL_DATA_PATH = DATA_DIR / "full_data_out.json"
MINI_DATA_PATH = DATA_DIR / "mini_data_out.json"
PE_CACHE_PATH = WORKSPACE / "precomputed_pes.pkl"

# ============================================================
# HYPERPARAMETERS
# ============================================================
HIDDEN_DIM = 128
NUM_LAYERS = 4
ATOM_EMB_DIM = 64
PE_PROJ_DIM = 64
BATCH_SIZE = 128
LR = 1e-3
ETA_MIN = 1e-5
T_MAX = 150
MAX_EPOCHS = 150
PATIENCE = 30
GRAD_CLIP = 5.0
DROPOUT = 0.1
SEEDS = [42, 123, 456]

# PE configurations
PE_CONFIGS = {
    "rwpe_16":       {"rwpe_k": 16, "nrwpe_k": 0,  "mild_k": 0,  "pe_dim": 16},
    "nrwpe_tanh_16": {"rwpe_k": 0,  "nrwpe_k": 16, "mild_k": 0,  "pe_dim": 16},
    "concat_16":     {"rwpe_k": 8,  "nrwpe_k": 8,  "mild_k": 0,  "pe_dim": 16},
    "concat_24":     {"rwpe_k": 16, "nrwpe_k": 8,  "mild_k": 0,  "pe_dim": 24},
    "concat_32":     {"rwpe_k": 16, "nrwpe_k": 16, "mild_k": 0,  "pe_dim": 32},
    "mild_16":       {"rwpe_k": 0,  "nrwpe_k": 0,  "mild_k": 16, "pe_dim": 16},
    "rwpe_24":       {"rwpe_k": 24, "nrwpe_k": 0,  "mild_k": 0,  "pe_dim": 24},
    "rwpe_32":       {"rwpe_k": 32, "nrwpe_k": 0,  "mild_k": 0,  "pe_dim": 32},
    "no_pe":         {"rwpe_k": 0,  "nrwpe_k": 0,  "mild_k": 0,  "pe_dim": 16},
}

# ============================================================
# PE COMPUTATION
# ============================================================
def compute_rwpe(edge_index: list, n: int, k: int) -> np.ndarray:
    """Standard RWPE: diag of (D^{-1}A)^t for t=1..k. NO self-loops."""
    if n == 0 or k == 0:
        return np.zeros((max(n, 1), k), dtype=np.float32)

    row = np.array(edge_index[0], dtype=np.int32)
    col = np.array(edge_index[1], dtype=np.int32)
    data = np.ones(len(row), dtype=np.float32)
    A = sp.csr_matrix((data, (row, col)), shape=(n, n))

    # D^{-1}
    deg = np.array(A.sum(axis=1)).flatten()
    deg_inv = np.where(deg > 0, 1.0 / deg, 0.0).astype(np.float32)
    D_inv = sp.diags(deg_inv)

    # RW = D^{-1} A (row-stochastic)
    RW = D_inv @ A

    pe = np.zeros((n, k), dtype=np.float32)
    P = RW.copy()
    for t in range(k):
        pe[:, t] = P.diagonal()
        if t < k - 1:
            P = P @ RW
    return pe


def compute_nrwpe_tanh(edge_index: list, n: int, k: int) -> np.ndarray:
    """nRWPE: diag of iterated tanh(A_norm @ X). WITH self-loops, symmetric norm."""
    if n == 0 or k == 0:
        return np.zeros((max(n, 1), k), dtype=np.float32)

    row = np.array(edge_index[0], dtype=np.int32)
    col = np.array(edge_index[1], dtype=np.int32)
    data = np.ones(len(row), dtype=np.float32)
    A = sp.csr_matrix((data, (row, col)), shape=(n, n))

    # Add self-loops
    A = A + sp.eye(n, dtype=np.float32)

    # Symmetric normalization: D^{-1/2} A D^{-1/2}
    deg = np.array(A.sum(axis=1)).flatten()
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0).astype(np.float32)
    D_inv_sqrt = sp.diags(deg_inv_sqrt)
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt

    # Convert to dense for iterated matrix multiplication
    if n <= 100:
        A_norm_dense = A_norm.toarray()
        X = np.eye(n, dtype=np.float32)
        pe = np.zeros((n, k), dtype=np.float32)
        for t in range(k):
            X = np.tanh(A_norm_dense @ X)
            pe[:, t] = np.diag(X)
    else:
        # For larger graphs, only track diagonal elements efficiently
        # X_diag[i] = (tanh(A_norm @ X))[i,i] = tanh(sum_j A_norm[i,j] * X[j,i])
        # We need to track full X since diagonal depends on all columns
        A_norm_dense = A_norm.toarray()
        X = np.eye(n, dtype=np.float32)
        pe = np.zeros((n, k), dtype=np.float32)
        for t in range(k):
            X = np.tanh(A_norm_dense @ X)
            pe[:, t] = np.diag(X)

    return pe


def compute_mild_nrwpe(edge_index: list, n: int, k: int) -> np.ndarray:
    """Mild nonlinearity: X_{t+1} = X_t + 0.1*tanh(A_norm @ X_t)"""
    if n == 0 or k == 0:
        return np.zeros((max(n, 1), k), dtype=np.float32)

    row = np.array(edge_index[0], dtype=np.int32)
    col = np.array(edge_index[1], dtype=np.int32)
    data = np.ones(len(row), dtype=np.float32)
    A = sp.csr_matrix((data, (row, col)), shape=(n, n))

    A = A + sp.eye(n, dtype=np.float32)
    deg = np.array(A.sum(axis=1)).flatten()
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0).astype(np.float32)
    D_inv_sqrt = sp.diags(deg_inv_sqrt)
    A_norm = (D_inv_sqrt @ A @ D_inv_sqrt).toarray()

    X = np.eye(n, dtype=np.float32)
    pe = np.zeros((n, k), dtype=np.float32)
    for t in range(k):
        X = X + 0.1 * np.tanh(A_norm @ X)
        pe[:, t] = np.diag(X)

    return pe


def compute_all_pes_for_graph(args):
    """Compute all PE types for a single graph. Used by ProcessPoolExecutor."""
    idx, edge_index, n, max_rwpe_k, max_nrwpe_k, max_mild_k = args
    result = {"idx": idx}

    if max_rwpe_k > 0:
        result["rwpe"] = compute_rwpe(edge_index, n, max_rwpe_k)
    if max_nrwpe_k > 0:
        result["nrwpe"] = compute_nrwpe_tanh(edge_index, n, max_nrwpe_k)
    if max_mild_k > 0:
        result["mild"] = compute_mild_nrwpe(edge_index, n, max_mild_k)

    return result


# ============================================================
# DATA LOADING
# ============================================================
def load_data(path: Path, max_examples: int = None) -> dict:
    """Load ZINC-12k data from JSON."""
    logger.info(f"Loading data from {path}")
    raw = json.loads(path.read_text())
    examples = raw["datasets"][0]["examples"]
    if max_examples is not None:
        examples = examples[:max_examples]
    logger.info(f"Loaded {len(examples)} examples")
    return examples


def parse_graph(example: dict) -> dict:
    """Parse a single example into graph components."""
    inp = json.loads(example["input"])
    return {
        "edge_index": inp["edge_index"],
        "node_feat": inp["node_feat"],
        "edge_attr": inp["edge_attr"],
        "num_nodes": inp["num_nodes"],
        "y": float(example["output"]),
        "fold": example["metadata_fold"],
        "row_index": example.get("metadata_row_index", -1),
    }


def compute_pes_parallel(graphs: list, max_rwpe_k: int = 32, max_nrwpe_k: int = 16, max_mild_k: int = 16) -> dict:
    """Compute PEs for all graphs using multiprocessing."""
    logger.info(f"Computing PEs for {len(graphs)} graphs (rwpe_k={max_rwpe_k}, nrwpe_k={max_nrwpe_k}, mild_k={max_mild_k})")
    t0 = time.time()

    args_list = [
        (i, g["edge_index"], g["num_nodes"], max_rwpe_k, max_nrwpe_k, max_mild_k)
        for i, g in enumerate(graphs)
    ]

    n_workers = max(1, NUM_CPUS - 1)
    results = [None] * len(graphs)

    if len(graphs) < 20:
        # Sequential for small datasets
        for args in args_list:
            res = compute_all_pes_for_graph(args)
            results[res["idx"]] = res
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(compute_all_pes_for_graph, args): args[0] for args in args_list}
            done_count = 0
            for future in as_completed(futures):
                res = future.result()
                results[res["idx"]] = res
                done_count += 1
                if done_count % 2000 == 0:
                    logger.info(f"  PE computation: {done_count}/{len(graphs)}")

    elapsed = time.time() - t0
    logger.info(f"PE computation done in {elapsed:.1f}s")
    return results


def build_pe_tensor(pe_results: list, config: dict, train_stats: dict = None) -> tuple:
    """Build PE tensors for a given config, with optional z-score normalization.
    Returns (pe_tensors_list, train_stats_dict).
    """
    rwpe_k = config["rwpe_k"]
    nrwpe_k = config["nrwpe_k"]
    mild_k = config["mild_k"]
    pe_dim = config["pe_dim"]

    pe_list = []
    for res in pe_results:
        parts = []
        if rwpe_k > 0:
            rwpe = res["rwpe"][:, :rwpe_k]
            parts.append(rwpe)
        if nrwpe_k > 0:
            nrwpe = res["nrwpe"][:, :nrwpe_k]
            parts.append(nrwpe)
        if mild_k > 0:
            mild = res["mild"][:, :mild_k]
            parts.append(mild)

        if len(parts) > 0:
            pe = np.concatenate(parts, axis=1)
        else:
            # no_pe: zeros
            n = res["rwpe"].shape[0] if "rwpe" in res else (res["nrwpe"].shape[0] if "nrwpe" in res else (res["mild"].shape[0] if "mild" in res else 1))
            pe = np.zeros((n, pe_dim), dtype=np.float32)

        pe_list.append(pe)

    # Compute or apply z-score normalization
    if train_stats is None:
        # Compute from these (training) data
        all_pe = np.concatenate(pe_list, axis=0)
        mean = all_pe.mean(axis=0)
        std = all_pe.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        train_stats = {"mean": mean, "std": std}

    # Apply normalization
    for i in range(len(pe_list)):
        pe_list[i] = (pe_list[i] - train_stats["mean"]) / train_stats["std"]

    return pe_list, train_stats


def build_pyg_data(graphs: list, pe_list: list, pe_dim: int) -> list:
    """Convert graphs + PEs to PyG Data objects."""
    data_list = []
    for g, pe in zip(graphs, pe_list):
        edge_index = torch.tensor(g["edge_index"], dtype=torch.long)
        x = torch.tensor(g["node_feat"], dtype=torch.long)
        edge_attr = torch.tensor(g["edge_attr"], dtype=torch.long)
        y = torch.tensor([g["y"]], dtype=torch.float32)
        pe_tensor = torch.tensor(pe, dtype=torch.float32)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pe=pe_tensor)
        data_list.append(data)
    return data_list


# ============================================================
# MODEL
# ============================================================
class GINEConvZINC(nn.Module):
    def __init__(self, pe_dim: int = 16, hidden_dim: int = 128, num_layers: int = 4,
                 atom_emb_dim: int = 64, dropout: float = 0.1, use_pe: bool = True):
        super().__init__()
        self.use_pe = use_pe
        self.atom_emb = nn.Embedding(28, atom_emb_dim)  # atom types 0-27
        self.bond_emb = nn.Embedding(4, hidden_dim)       # bond types 1-3 (0 unused but safe)

        if use_pe:
            self.pe_proj = nn.Sequential(
                nn.Linear(pe_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, atom_emb_dim),
            )
            self.pe_bn = nn.BatchNorm1d(atom_emb_dim)
            self.input_proj = nn.Linear(atom_emb_dim + atom_emb_dim, hidden_dim)
        else:
            self.input_proj = nn.Linear(atom_emb_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            conv = GINEConv(mlp, edge_dim=hidden_dim)
            self.convs.append(conv)
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Readout: global_add_pool || global_mean_pool -> MLP
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x = self.atom_emb(data.x)  # (N, 64)

        if self.use_pe:
            pe = self.pe_bn(self.pe_proj(data.pe))  # (N, 64)
            x = self.input_proj(torch.cat([x, pe], dim=-1))  # (N, 128)
        else:
            x = self.input_proj(x)  # (N, 128)

        edge_attr = self.bond_emb(data.edge_attr)  # (E, 128)

        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, data.edge_index, edge_attr)))

        # Pooling
        x_add = global_add_pool(x, data.batch)   # (B, 128)
        x_mean = global_mean_pool(x, data.batch)  # (B, 128)
        x = torch.cat([x_add, x_mean], dim=-1)    # (B, 256)

        return self.readout(x).squeeze(-1)  # (B,)


# ============================================================
# TRAINING
# ============================================================
def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_count = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = F.l1_loss(pred, batch.y.squeeze(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        total_count += batch.num_graphs
    return total_loss / total_count


@torch.no_grad()
def evaluate(model, loader, device) -> tuple:
    """Returns (mae, per_graph_predictions)."""
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_preds = []
    all_targets = []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        targets = batch.y.squeeze(-1)
        loss = F.l1_loss(pred, targets, reduction='sum')
        total_loss += loss.item()
        total_count += batch.num_graphs
        all_preds.extend(pred.cpu().numpy().tolist())
        all_targets.extend(targets.cpu().numpy().tolist())
    mae = total_loss / total_count
    return mae, all_preds, all_targets


def run_single_config(config_name: str, config: dict, train_data: list, val_data: list,
                      test_data: list, pe_results_train: list, pe_results_val: list,
                      pe_results_test: list, seed: int, device: torch.device,
                      max_epochs: int = MAX_EPOCHS, patience: int = PATIENCE) -> dict:
    """Train and evaluate a single configuration with a single seed."""
    set_seed(seed)
    pe_dim = config["pe_dim"]
    use_pe = config_name != "no_pe"

    # Build PE tensors
    pe_train, train_stats = build_pe_tensor(pe_results_train, config)
    pe_val, _ = build_pe_tensor(pe_results_val, config, train_stats)
    pe_test, _ = build_pe_tensor(pe_results_test, config, train_stats)

    # Build PyG datasets
    train_dataset = build_pyg_data(train_data, pe_train, pe_dim)
    val_dataset = build_pyg_data(val_data, pe_val, pe_dim)
    test_dataset = build_pyg_data(test_data, pe_test, pe_dim)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # Model
    model = GINEConvZINC(
        pe_dim=pe_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
        atom_emb_dim=ATOM_EMB_DIM, dropout=DROPOUT, use_pe=use_pe
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=T_MAX, eta_min=ETA_MIN)

    best_val_mae = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_state = None

    for epoch in range(1, max_epochs + 1):
        train_mae = train_one_epoch(model, train_loader, optimizer, device)
        val_mae, _, _ = evaluate(model, val_loader, device)
        scheduler.step()

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if epoch % 25 == 0 or epoch == 1:
            logger.info(f"  [{config_name}|s{seed}] Epoch {epoch}: train={train_mae:.4f} val={val_mae:.4f} best={best_val_mae:.4f}@{best_epoch}")

        if patience_counter >= patience:
            logger.info(f"  [{config_name}|s{seed}] Early stopping at epoch {epoch}")
            break

    # Load best model and evaluate on test
    model.load_state_dict(best_state)
    model.to(device)
    test_mae, test_preds, test_targets = evaluate(model, test_loader, device)
    val_mae_final, _, _ = evaluate(model, val_loader, device)

    logger.info(f"  [{config_name}|s{seed}] DONE: test_mae={test_mae:.4f} val_mae={val_mae_final:.4f} best_epoch={best_epoch}")

    # Clean up GPU memory
    del model, optimizer, scheduler, train_loader, val_loader
    del train_dataset, val_dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "config_name": config_name,
        "seed": seed,
        "test_mae": test_mae,
        "val_mae": val_mae_final,
        "best_epoch": best_epoch,
        "test_preds": test_preds,
        "test_targets": test_targets,
        "num_params": sum(p.numel() for p in GINEConvZINC(
            pe_dim=pe_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
            atom_emb_dim=ATOM_EMB_DIM, dropout=DROPOUT, use_pe=use_pe
        ).parameters()),
    }


# ============================================================
# ANALYSIS
# ============================================================
def analyze_results(all_results: dict) -> dict:
    """Compute aggregate statistics and statistical tests."""
    analysis = {}

    for config_name, seed_results in all_results.items():
        if len(seed_results) == 0:
            continue  # Skip configs with no completed runs
        test_maes = [r["test_mae"] for r in seed_results]
        val_maes = [r["val_mae"] for r in seed_results]
        best_epochs = [r["best_epoch"] for r in seed_results]

        analysis[config_name] = {
            "test_mae_mean": float(np.mean(test_maes)),
            "test_mae_std": float(np.std(test_maes)),
            "val_mae_mean": float(np.mean(val_maes)),
            "val_mae_std": float(np.std(val_maes)),
            "best_epoch_mean": float(np.mean(best_epochs)),
            "test_maes": test_maes,
            "val_maes": val_maes,
            "best_epochs": best_epochs,
            "num_params": seed_results[0]["num_params"],
        }

    # Statistical tests: each concat config vs rwpe_16 baseline
    baseline_key = "rwpe_16"
    if baseline_key in all_results:
        baseline_maes = [r["test_mae"] for r in all_results[baseline_key]]
        for config_name in all_results:
            if config_name != baseline_key and config_name in analysis:
                other_maes = [r["test_mae"] for r in all_results[config_name]]
                if len(baseline_maes) >= 2 and len(other_maes) >= 2:
                    try:
                        # Paired t-test (per-seed comparison)
                        min_len = min(len(baseline_maes), len(other_maes))
                        t_stat, p_val = stats.ttest_rel(baseline_maes[:min_len], other_maes[:min_len])
                        analysis[config_name]["vs_rwpe16_ttest_t"] = float(t_stat)
                        analysis[config_name]["vs_rwpe16_ttest_p"] = float(p_val)
                        analysis[config_name]["vs_rwpe16_improvement"] = float(
                            np.mean(baseline_maes) - np.mean(other_maes)
                        )
                    except Exception as e:
                        logger.warning(f"T-test failed for {config_name}: {e}")

    # Per-graph win rates (seed 42)
    if baseline_key in all_results:
        baseline_s42 = [r for r in all_results[baseline_key] if r["seed"] == 42]
        if baseline_s42:
            baseline_preds = np.array(baseline_s42[0]["test_preds"])
            baseline_targets = np.array(baseline_s42[0]["test_targets"])
            baseline_errors = np.abs(baseline_preds - baseline_targets)

            for config_name in all_results:
                if config_name != baseline_key:
                    other_s42 = [r for r in all_results[config_name] if r["seed"] == 42]
                    if other_s42:
                        other_preds = np.array(other_s42[0]["test_preds"])
                        other_errors = np.abs(other_preds - baseline_targets)
                        wins = int(np.sum(other_errors < baseline_errors))
                        ties = int(np.sum(other_errors == baseline_errors))
                        losses = int(np.sum(other_errors > baseline_errors))
                        n_test = len(baseline_errors)
                        analysis[config_name]["win_rate_vs_rwpe16"] = {
                            "wins": wins, "ties": ties, "losses": losses,
                            "win_pct": float(wins / n_test * 100),
                        }

    return analysis


# ============================================================
# OUTPUT FORMATTING
# ============================================================
def format_output(examples: list, all_results: dict, analysis: dict, pe_times: dict) -> dict:
    """Format results into exp_gen_sol_out.json schema."""
    # Only include test-fold examples
    test_examples = [ex for ex in examples if ex["metadata_fold"] == "test"]

    output_examples = []
    for i, ex in enumerate(test_examples):
        entry = {
            "input": ex["input"],
            "output": ex["output"],
            "metadata_fold": ex["metadata_fold"],
            "metadata_task_type": ex["metadata_task_type"],
            "metadata_row_index": ex["metadata_row_index"],
            "metadata_num_nodes": ex["metadata_num_nodes"],
            "metadata_num_edges": ex["metadata_num_edges"],
        }

        # Add predictions from each config (seed 42 primary)
        for config_name, seed_results in all_results.items():
            s42_results = [r for r in seed_results if r["seed"] == 42]
            if s42_results and i < len(s42_results[0]["test_preds"]):
                entry[f"predict_{config_name}"] = str(s42_results[0]["test_preds"][i])

        output_examples.append(entry)

    metadata = {
        "method_name": "RWPE_nRWPE_Concatenation_GINEConv",
        "description": "Test RWPE+nRWPE concatenation vs RWPE-only baseline on ZINC-12k with GINEConv",
        "architecture": "GINEConv (4 layers, 128 hidden, add+mean pooling)",
        "training": {
            "optimizer": "Adam", "lr": LR, "scheduler": "CosineAnnealingLR",
            "T_max": T_MAX, "eta_min": ETA_MIN, "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "grad_clip": GRAD_CLIP,
        },
        "seeds": SEEDS,
        "pe_configs": {k: v for k, v in PE_CONFIGS.items()},
        "analysis": analysis,
        "pe_computation_times": pe_times,
    }

    return {
        "metadata": metadata,
        "datasets": [
            {
                "dataset": "ZINC-12k",
                "examples": output_examples,
            }
        ]
    }


# ============================================================
# MAIN
# ============================================================
@logger.catch
def main():
    t_start = time.time()
    logger.info("=" * 60)
    logger.info("RWPE + nRWPE Concatenation on ZINC-12k with GINEConv")
    logger.info("=" * 60)

    # ---- Step 1: Load data ----
    logger.info("Step 1: Loading data...")
    examples = load_data(FULL_DATA_PATH)

    graphs = []
    for i, ex in enumerate(examples):
        try:
            g = parse_graph(ex)
            graphs.append(g)
        except Exception:
            logger.exception(f"Failed to parse graph {i}")
            continue

    logger.info(f"Parsed {len(graphs)} graphs")

    # Split by fold
    train_graphs = [g for g in graphs if g["fold"] == "train"]
    val_graphs = [g for g in graphs if g["fold"] == "val"]
    test_graphs = [g for g in graphs if g["fold"] == "test"]
    logger.info(f"Split: train={len(train_graphs)}, val={len(val_graphs)}, test={len(test_graphs)}")

    # ---- Step 2: Compute PEs ----
    logger.info("Step 2: Computing positional encodings...")
    pe_times = {}

    t0 = time.time()
    pe_results_train = compute_pes_parallel(train_graphs, max_rwpe_k=32, max_nrwpe_k=16, max_mild_k=16)
    pe_times["train"] = time.time() - t0

    t0 = time.time()
    pe_results_val = compute_pes_parallel(val_graphs, max_rwpe_k=32, max_nrwpe_k=16, max_mild_k=16)
    pe_times["val"] = time.time() - t0

    t0 = time.time()
    pe_results_test = compute_pes_parallel(test_graphs, max_rwpe_k=32, max_nrwpe_k=16, max_mild_k=16)
    pe_times["test"] = time.time() - t0

    logger.info(f"PE computation times: train={pe_times['train']:.1f}s val={pe_times['val']:.1f}s test={pe_times['test']:.1f}s")

    # ---- Step 3: Sanity checks ----
    logger.info("Step 3: PE sanity checks...")
    sample_res = pe_results_train[0]
    logger.info(f"  RWPE shape: {sample_res['rwpe'].shape}, range: [{sample_res['rwpe'].min():.4f}, {sample_res['rwpe'].max():.4f}]")
    logger.info(f"  nRWPE shape: {sample_res['nrwpe'].shape}, range: [{sample_res['nrwpe'].min():.4f}, {sample_res['nrwpe'].max():.4f}]")
    logger.info(f"  Mild shape: {sample_res['mild'].shape}, range: [{sample_res['mild'].min():.4f}, {sample_res['mild'].max():.4f}]")

    # Verify RWPE in [0,1] and nRWPE in [-1,1]
    all_rwpe_vals = np.concatenate([r["rwpe"].flatten() for r in pe_results_train[:100]])
    all_nrwpe_vals = np.concatenate([r["nrwpe"].flatten() for r in pe_results_train[:100]])
    logger.info(f"  RWPE stats (100 graphs): min={all_rwpe_vals.min():.4f}, max={all_rwpe_vals.max():.4f}")
    logger.info(f"  nRWPE stats (100 graphs): min={all_nrwpe_vals.min():.4f}, max={all_nrwpe_vals.max():.4f}")

    assert not np.any(np.isnan(all_rwpe_vals)), "NaN in RWPE!"
    assert not np.any(np.isnan(all_nrwpe_vals)), "NaN in nRWPE!"

    # ---- Step 4: Smoke test ----
    logger.info("Step 4: Smoke test (small subset, 10 epochs)...")
    smoke_train = train_graphs[:100]
    smoke_val = val_graphs[:20]
    smoke_test = test_graphs[:20]
    smoke_pe_train = pe_results_train[:100]
    smoke_pe_val = pe_results_val[:20]
    smoke_pe_test = pe_results_test[:20]

    smoke_result = run_single_config(
        "concat_16", PE_CONFIGS["concat_16"],
        smoke_train, smoke_val, smoke_test,
        smoke_pe_train, smoke_pe_val, smoke_pe_test,
        seed=42, device=DEVICE, max_epochs=10, patience=100
    )
    logger.info(f"Smoke test done: test_mae={smoke_result['test_mae']:.4f}")
    del smoke_result
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Step 5: Full training ----
    logger.info("Step 5: Full training")
    results_cache_path = WORKSPACE / "results_cache.pkl"
    all_results = {}

    # Try to load cached results from previous interrupted run
    if results_cache_path.exists():
        try:
            all_results = pickle.loads(results_cache_path.read_bytes())
            logger.info(f"Loaded cached results: {', '.join(f'{k}({len(v)})' for k, v in all_results.items() if len(v) > 0)}")
        except Exception:
            logger.warning("Failed to load cached results, starting fresh")
            all_results = {}

    # Priority order from the plan
    priority_configs = [
        "rwpe_16",       # Baseline
        "concat_32",     # Largest concat
        "concat_24",     # Mid concat
        "concat_16",     # Equal-dim concat
        "nrwpe_tanh_16", # nRWPE standalone
        "no_pe",         # No-PE baseline
        "mild_16",       # Alternative nonlinearity
    ]

    TIME_BUDGET = 3000  # seconds (~50 min, leaving time for analysis and output)

    for config_name in priority_configs:
        # Skip configs already completed with enough seeds
        if config_name in all_results and len(all_results[config_name]) >= 3:
            logger.info(f"Skipping {config_name} (already have {len(all_results[config_name])} seed results cached)")
            continue

        config = PE_CONFIGS[config_name]
        if config_name not in all_results:
            all_results[config_name] = []
        logger.info(f"\n{'='*40}")
        logger.info(f"Training config: {config_name} (pe_dim={config['pe_dim']})")

        # Check time budget
        elapsed = time.time() - t_start
        remaining = TIME_BUDGET - elapsed
        if remaining < 300:
            logger.warning(f"Time budget nearly exhausted ({remaining:.0f}s remaining). Stopping.")
            break

        # Use fewer seeds for low-priority configs if time is tight
        seeds_to_use = SEEDS
        if remaining < 1500 and config_name in ["mild_16", "no_pe", "nrwpe_tanh_16"]:
            seeds_to_use = SEEDS[:2]
            logger.info(f"  Reduced to {len(seeds_to_use)} seeds due to time budget")
        if remaining < 800:
            seeds_to_use = SEEDS[:1]
            logger.info(f"  Only 1 seed due to very tight time budget ({remaining:.0f}s remaining)")

        # Skip seeds already completed
        completed_seeds = {r["seed"] for r in all_results[config_name]}

        for seed in seeds_to_use:
            if seed in completed_seeds:
                logger.info(f"  Skipping seed {seed} (already cached)")
                continue

            # Check time within seed loop too
            elapsed = time.time() - t_start
            remaining = TIME_BUDGET - elapsed
            if remaining < 200:
                logger.warning(f"Time budget nearly exhausted ({remaining:.0f}s). Breaking seed loop.")
                break

            t_run = time.time()
            try:
                result = run_single_config(
                    config_name, config,
                    train_graphs, val_graphs, test_graphs,
                    pe_results_train, pe_results_val, pe_results_test,
                    seed=seed, device=DEVICE
                )
                all_results[config_name].append(result)
                run_time = time.time() - t_run
                logger.info(f"  Run time: {run_time:.1f}s")

                # Save results incrementally
                results_cache_path.write_bytes(pickle.dumps(all_results))
                logger.info(f"  Results cached to {results_cache_path}")
            except torch.cuda.OutOfMemoryError:
                logger.warning(f"OOM for {config_name} seed={seed}. Trying smaller batch...")
                torch.cuda.empty_cache()
                gc.collect()
                try:
                    result = run_single_config(
                        config_name, config,
                        train_graphs, val_graphs, test_graphs,
                        pe_results_train, pe_results_val, pe_results_test,
                        seed=seed, device=DEVICE
                    )
                    all_results[config_name].append(result)
                    results_cache_path.write_bytes(pickle.dumps(all_results))
                except Exception:
                    logger.exception(f"Failed on retry for {config_name} seed={seed}")
            except Exception:
                logger.exception(f"Failed for {config_name} seed={seed}")

    # ---- Phase C: Dimension controls (conditional) ----
    elapsed = time.time() - t_start
    remaining = TIME_BUDGET - elapsed

    # Check if any concat beats rwpe_16
    concat_beats_rwpe = False
    if "rwpe_16" in all_results and len(all_results["rwpe_16"]) > 0:
        rwpe_mean = np.mean([r["test_mae"] for r in all_results["rwpe_16"]])
        for cn in ["concat_16", "concat_24", "concat_32"]:
            if cn in all_results and len(all_results[cn]) > 0:
                cn_mean = np.mean([r["test_mae"] for r in all_results[cn]])
                if cn_mean < rwpe_mean:
                    concat_beats_rwpe = True
                    break

    if concat_beats_rwpe and remaining > 600:
        logger.info("\nPhase C: Dimension controls (concat beats rwpe)")
        for config_name in ["rwpe_24", "rwpe_32"]:
            if config_name in all_results and len(all_results[config_name]) >= 2:
                logger.info(f"Skipping {config_name} (already cached)")
                continue
            elapsed = time.time() - t_start
            remaining = TIME_BUDGET - elapsed
            if remaining < 300:
                logger.warning("Time budget exhausted for dimension controls")
                break

            config = PE_CONFIGS[config_name]
            if config_name not in all_results:
                all_results[config_name] = []
            logger.info(f"Training dim control: {config_name}")

            seeds_to_use = SEEDS if remaining > 900 else SEEDS[:2]
            completed_seeds = {r["seed"] for r in all_results.get(config_name, [])}
            for seed in seeds_to_use:
                if seed in completed_seeds:
                    continue
                try:
                    result = run_single_config(
                        config_name, config,
                        train_graphs, val_graphs, test_graphs,
                        pe_results_train, pe_results_val, pe_results_test,
                        seed=seed, device=DEVICE
                    )
                    all_results[config_name].append(result)
                    results_cache_path.write_bytes(pickle.dumps(all_results))
                except Exception:
                    logger.exception(f"Failed for {config_name} seed={seed}")
    else:
        if not concat_beats_rwpe:
            logger.info("Phase C skipped: no concat config beats rwpe_16")
        else:
            logger.info(f"Phase C skipped: insufficient time ({remaining:.0f}s remaining)")

    # Save final cache
    results_cache_path.write_bytes(pickle.dumps(all_results))

    # Remove empty configs
    all_results = {k: v for k, v in all_results.items() if len(v) > 0}
    logger.info(f"Completed configs: {', '.join(f'{k}({len(v)} seeds)' for k, v in all_results.items())}")

    # ---- Step 6: Analysis ----
    logger.info("\nStep 6: Analysis")
    analysis = analyze_results(all_results)

    # Print summary table
    logger.info("\n" + "=" * 80)
    logger.info(f"{'Config':<18} {'Test MAE':>12} {'Val MAE':>12} {'Best Ep':>10} {'Params':>10}")
    logger.info("-" * 80)
    for config_name in priority_configs + ["rwpe_24", "rwpe_32"]:
        if config_name in analysis:
            a = analysis[config_name]
            logger.info(
                f"{config_name:<18} "
                f"{a['test_mae_mean']:.4f}±{a['test_mae_std']:.4f}  "
                f"{a['val_mae_mean']:.4f}±{a['val_mae_std']:.4f}  "
                f"{a['best_epoch_mean']:>8.1f}  "
                f"{a['num_params']:>10d}"
            )
    logger.info("=" * 80)

    # Print statistical tests
    if "rwpe_16" in analysis:
        logger.info("\nStatistical tests vs rwpe_16:")
        for config_name in analysis:
            if "vs_rwpe16_ttest_p" in analysis[config_name]:
                a = analysis[config_name]
                sig = "***" if a["vs_rwpe16_ttest_p"] < 0.01 else "**" if a["vs_rwpe16_ttest_p"] < 0.05 else "*" if a["vs_rwpe16_ttest_p"] < 0.1 else "ns"
                logger.info(
                    f"  {config_name:<18} improvement={a['vs_rwpe16_improvement']:+.4f} "
                    f"t={a['vs_rwpe16_ttest_t']:.3f} p={a['vs_rwpe16_ttest_p']:.4f} {sig}"
                )

    # ---- Step 7: Output ----
    logger.info("\nStep 7: Saving output...")
    output = format_output(examples, all_results, analysis, pe_times)

    output_path = WORKSPACE / "method_out.json"
    output_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved output to {output_path}")

    total_time = time.time() - t_start
    logger.info(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f}min)")
    logger.info("Done!")


if __name__ == "__main__":
    main()
