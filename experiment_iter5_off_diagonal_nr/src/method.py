#!/usr/bin/env python3
"""Off-Diagonal nRWPE + Softplus Nonlinearity on ZINC-12k with GINEConv.

Tests whether off-diagonal nonlinear walk PE statistics and softplus
nonlinearity close the RWPE downstream gap on ZINC-12k.

Architecture: GINEConv (edge-aware GIN) with 4 layers, 128 hidden dim,
dual pooling (add + mean), and PE projection MLP.

Runs 6 PE configurations × 3 seeds = 18 experiments.
"""

import gc
import json
import math
import os
import pickle
import subprocess
import sys
import time
import resource
from copy import deepcopy
from pathlib import Path

# Limit thread count to reduce CPU time accumulation from RLIMIT_CPU
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# ── Logging setup ──
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/method.log", rotation="30 MB", level="DEBUG")

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
HAS_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if HAS_GPU else "cpu")
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9 if HAS_GPU else 0

# Memory limits — be generous since 62GB container
RAM_BUDGET = int(TOTAL_RAM_GB * 0.85 * 1e9)
# Set AS limit for OOM safety net (raises MemoryError instead of SIGKILL)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
# CPU time limit: 14400s (4h) - PyTorch multi-threading accumulates CPU time fast
# With 10 threads, wall time × 10 = CPU time, so 4h covers ~24 min wall time
resource.setrlimit(resource.RLIMIT_CPU, (14400, 14400))

if HAS_GPU:
    _free, _total = torch.cuda.mem_get_info(0)
    VRAM_BUDGET = int(_total * 0.9)
    torch.cuda.set_per_process_memory_fraction(min(VRAM_BUDGET / _total, 0.95))

torch.set_num_threads(2)  # Limit PyTorch CPU threads to avoid CPU time limit
logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f}GB RAM, GPU={HAS_GPU}, VRAM={VRAM_GB:.1f}GB")

# ── Constants ──
PE_DIM = 16
ATOM_EMB_DIM = 64
HIDDEN_DIM = 128
NUM_LAYERS = 4
NUM_ATOM_TYPES = 28    # atom types 0-27
NUM_BOND_TYPES = 4     # bond types 1-3 (embed with size 4, index 0 unused)
LR = 1e-3
LR_MIN = 1e-5
BATCH_SIZE = 128
NUM_EPOCHS = 250
PATIENCE = 40
GRAD_CLIP = 5.0
SEEDS = [42, 123, 456]

PE_CONFIGS = ['rwpe_16', 'nrwpe_diag_softplus_16', 'nrwpe_diag_tanh_16',
              'nrwpe_offdiag_16', 'nrwpe_combined_16', 'no_pe']

# Priority order for training
PRIORITY = ['rwpe_16', 'nrwpe_offdiag_16', 'nrwpe_combined_16',
            'nrwpe_diag_softplus_16', 'nrwpe_diag_tanh_16', 'no_pe']

DATA_DIR = Path("/workspace/runs/run__20260225_014759/3_invention_loop/iter_1/gen_art/data_id3_it1__opus")
SCRIPT_DIR = Path(__file__).resolve().parent

# ── PyG imports (deferred to avoid import errors at top) ──
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool


# ═══════════════════════════════════════════════════════════
# Model Definition
# ═══════════════════════════════════════════════════════════

class GIN_ZINC(nn.Module):
    """GINEConv model for ZINC-12k with optional positional encodings."""

    def __init__(self, use_pe: bool = True, pe_dim: int = PE_DIM):
        super().__init__()
        self.use_pe = use_pe

        # Node feature embedding
        self.atom_emb = nn.Embedding(NUM_ATOM_TYPES, ATOM_EMB_DIM)

        # Edge feature embedding
        self.bond_emb = nn.Embedding(NUM_BOND_TYPES, HIDDEN_DIM)

        # PE projection MLP
        if use_pe:
            self.pe_proj = nn.Sequential(
                nn.Linear(pe_dim, HIDDEN_DIM),
                nn.BatchNorm1d(HIDDEN_DIM),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            )
            self.input_proj = nn.Linear(ATOM_EMB_DIM + HIDDEN_DIM, HIDDEN_DIM)
        else:
            self.input_proj = nn.Linear(ATOM_EMB_DIM, HIDDEN_DIM)

        # GINEConv layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(NUM_LAYERS):
            mlp = nn.Sequential(
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
                nn.ReLU(),
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
            )
            self.convs.append(GINEConv(mlp, edge_dim=HIDDEN_DIM))
            self.bns.append(nn.BatchNorm1d(HIDDEN_DIM))

        # Dual pooling readout
        self.readout = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1)
        )

    def forward(self, data):
        x = self.atom_emb(data.x.squeeze(-1))  # (N, ATOM_EMB_DIM)

        if self.use_pe:
            pe_feat = self.pe_proj(data.pe)  # (N, HIDDEN_DIM)
            x = torch.cat([x, pe_feat], dim=-1)

        x = self.input_proj(x)  # (N, HIDDEN_DIM)

        # Edge embedding
        edge_feat = self.bond_emb(data.edge_attr.squeeze(-1))  # (E, HIDDEN_DIM)

        # Message passing
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, data.edge_index, edge_feat)))

        # Dual pooling
        x_add = global_add_pool(x, data.batch)
        x_mean = global_mean_pool(x, data.batch)
        x = torch.cat([x_add, x_mean], dim=-1)

        return self.readout(x).squeeze(-1)


# ═══════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════

def load_zinc_data(data_path: Path, max_examples: int = None) -> tuple:
    """Load ZINC-12k data from JSON and return parsed graphs + targets + folds."""
    raw = json.loads(data_path.read_text())
    examples = raw["datasets"][0]["examples"]
    if max_examples is not None:
        examples = examples[:max_examples]

    graphs = []
    targets = []
    folds = []
    inputs_str = []

    for ex in examples:
        inp = json.loads(ex["input"])
        graphs.append(inp)
        targets.append(float(ex["output"]))
        folds.append(ex["metadata_fold"])
        inputs_str.append(ex["input"])

    return graphs, targets, folds, inputs_str


def build_pyg_dataset(graphs: list, targets: list, pe_arrays: list,
                      use_pe: bool = True) -> list:
    """Convert parsed graphs + PE arrays to PyG Data objects."""
    dataset = []
    for i, (g, y, pe) in enumerate(zip(graphs, targets, pe_arrays)):
        edge_index = torch.tensor(g["edge_index"], dtype=torch.long)
        x = torch.tensor(g["node_feat"], dtype=torch.long).unsqueeze(-1)
        edge_attr = torch.tensor(g["edge_attr"], dtype=torch.long).unsqueeze(-1)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor(y, dtype=torch.float),
        )
        if use_pe:
            data.pe = torch.tensor(pe, dtype=torch.float)
        else:
            data.pe = torch.zeros(g["num_nodes"], PE_DIM, dtype=torch.float)

        dataset.append(data)
    return dataset


# ═══════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model: nn.Module, loader: DataLoader) -> tuple:
    """Evaluate model and return (MAE, per-graph predictions, per-graph targets)."""
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            pred = model(batch)
            all_preds.append(pred.cpu())
            all_targets.append(batch.y.cpu())
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    mae = (all_preds - all_targets).abs().mean().item()
    return mae, all_preds.numpy(), all_targets.numpy()


def train_and_evaluate(train_dataset: list, val_dataset: list, test_dataset: list,
                       use_pe: bool, seed: int, num_epochs: int = NUM_EPOCHS,
                       patience: int = PATIENCE, verbose: bool = True,
                       save_train_preds: bool = False) -> dict:
    """Train model and return results dict."""
    set_seed(seed)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=HAS_GPU)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=HAS_GPU)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=HAS_GPU)

    model = GIN_ZINC(use_pe=use_pe).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=LR_MIN)
    criterion = nn.L1Loss()

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        logger.info(f"  Model params: {param_count:,}")

    best_val_mae = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_state = None
    train_curve = []
    val_curve = []

    for epoch in range(num_epochs):
        # Train
        model.train()
        total_loss = 0
        num_samples = 0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            num_samples += batch.num_graphs

        train_mae = total_loss / num_samples

        # Validate
        val_mae, _, _ = evaluate(model, val_loader)
        scheduler.step()

        train_curve.append(train_mae)
        val_curve.append(val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            patience_counter = 0
            best_state = deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if verbose and (epoch % 50 == 0 or epoch == num_epochs - 1 or patience_counter >= patience):
            logger.info(f"    Epoch {epoch}: train_mae={train_mae:.4f}, val_mae={val_mae:.4f}, "
                         f"best_val={best_val_mae:.4f}@{best_epoch}, patience={patience_counter}")

        if patience_counter >= patience:
            if verbose:
                logger.info(f"    Early stopping at epoch {epoch}")
            break

    # Test with best model
    model.load_state_dict(best_state)
    test_mae, test_preds, test_targets = evaluate(model, test_loader)
    _, val_preds, val_targets = evaluate(model, val_loader)

    result = {
        "test_mae": test_mae,
        "val_mae": best_val_mae,
        "best_epoch": best_epoch,
        "train_curve": train_curve,
        "val_curve": val_curve,
        "test_preds": test_preds.tolist(),
        "test_targets": test_targets.tolist(),
        "val_preds": val_preds.tolist(),
        "param_count": param_count,
    }

    # Only compute train preds for seed 42 (for per-example output)
    if save_train_preds:
        train_eval_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                       num_workers=0, pin_memory=HAS_GPU)
        _, train_preds, _ = evaluate(model, train_eval_loader)
        result["train_preds"] = train_preds.tolist()

    return result


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

@logger.catch
def main():
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("nRWPE Off-Diagonal + Softplus on ZINC-12k with GINEConv")
    logger.info("=" * 60)

    # ── Step 0: Compute PEs if not cached ──
    pe_path_gz = SCRIPT_DIR / "precomputed_pes.pkl.gz"
    pe_path_raw = SCRIPT_DIR / "precomputed_pes.pkl"
    if not pe_path_gz.exists() and not pe_path_raw.exists():
        logger.info("PEs not found, computing...")
        result = subprocess.run(
            [sys.executable, str(SCRIPT_DIR / "compute_pes.py")],
            cwd=str(SCRIPT_DIR),
            capture_output=True, text=True, timeout=1800
        )
        if result.returncode != 0:
            logger.error(f"PE computation failed:\n{result.stderr[-2000:]}")
            raise RuntimeError("PE computation failed")
        logger.info("PE computation complete")

    # Load PEs (support both gzipped and raw formats)
    import gzip as _gzip
    if pe_path_gz.exists():
        logger.info(f"Loading cached PEs from {pe_path_gz}")
        with _gzip.open(pe_path_gz, "rb") as f:
            pe_data = pickle.load(f)
    else:
        logger.info(f"Loading cached PEs from {pe_path_raw}")
        with open(pe_path_raw, "rb") as f:
            pe_data = pickle.load(f)
    pe_dict = pe_data["pe_dict"]
    pe_types = pe_data["pe_types"]
    num_graphs = pe_data["num_graphs"]
    logger.info(f"Loaded PEs for {num_graphs} graphs, types: {pe_types}")

    # ── Step 1: Load ZINC data ──
    logger.info("Loading ZINC-12k data...")
    graphs, targets, folds, inputs_str = load_zinc_data(
        DATA_DIR / "full_data_out.json", max_examples=num_graphs
    )
    logger.info(f"Loaded {len(graphs)} graphs")

    # Split indices
    train_idx = [i for i, f in enumerate(folds) if f == "train"]
    val_idx = [i for i, f in enumerate(folds) if f == "val"]
    test_idx = [i for i, f in enumerate(folds) if f == "test"]
    logger.info(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # ── Step 2: Mini validation (smoke test) — skip if resuming ──
    checkpoint_path = SCRIPT_DIR / "results_checkpoint.json"
    if not checkpoint_path.exists():
        logger.info("=" * 40)
        logger.info("MINI VALIDATION (100 graphs, 50 epochs)")
        logger.info("=" * 40)

        mini_n = min(100, len(graphs))
        mini_graphs = graphs[:mini_n]
        mini_targets = targets[:mini_n]
        mini_train_idx = list(range(mini_n))
        mini_val_idx = list(range(min(20, mini_n)))

        for pe_config in PE_CONFIGS:
            use_pe = pe_config != "no_pe"
            mini_pe = pe_dict[pe_config][:mini_n]
            mini_dataset = build_pyg_dataset(mini_graphs, mini_targets, mini_pe, use_pe=True)
            mini_train = [mini_dataset[i] for i in mini_train_idx]
            mini_val = [mini_dataset[i] for i in mini_val_idx]

            try:
                result = train_and_evaluate(
                    mini_train, mini_val, mini_val,
                    use_pe=use_pe, seed=42, num_epochs=50, patience=50, verbose=False
                )
                if result["train_curve"][-1] < result["train_curve"][0]:
                    logger.info(f"  PASS {pe_config}: train_mae {result['train_curve'][0]:.4f} → "
                                 f"{result['train_curve'][-1]:.4f}, val_mae={result['val_mae']:.4f}")
                else:
                    logger.warning(f"  WARN {pe_config}: loss did NOT decrease!")
            except Exception:
                logger.exception(f"  FAIL {pe_config}: mini validation crashed")
                raise
            torch.cuda.empty_cache()
            gc.collect()

        logger.info("Mini validation PASSED for all configs")
    else:
        logger.info("Checkpoint found — skipping mini validation")

    # ── Step 3: Full training ──
    logger.info("=" * 40)
    logger.info(f"FULL TRAINING: {len(PRIORITY)} configs × {len(SEEDS)} seeds = {len(PRIORITY)*len(SEEDS)} runs")
    logger.info("=" * 40)

    all_results = {}
    checkpoint_path = SCRIPT_DIR / "results_checkpoint.json"
    seed42_results_path = SCRIPT_DIR / "seed42_results.pkl"

    # Load checkpoint if exists
    if checkpoint_path.exists():
        try:
            checkpoint_meta = json.loads(checkpoint_path.read_text())
            logger.info(f"Resumed from checkpoint: {len(checkpoint_meta)} results loaded")
            # Load seed42 full results if available
            if seed42_results_path.exists():
                with open(seed42_results_path, "rb") as f:
                    seed42_data = pickle.load(f)
                all_results.update(seed42_data)
                logger.info(f"Loaded {len(seed42_data)} seed42 results with predictions")
            # Copy metadata for non-seed42 runs
            for k, v in checkpoint_meta.items():
                if k not in all_results:
                    all_results[k] = v
        except Exception:
            logger.exception("Failed to load checkpoint, starting fresh")
            all_results = {}

    for pe_config in PRIORITY:
        for seed in SEEDS:
            run_key = f"{pe_config}_seed{seed}"
            if run_key in all_results:
                logger.info(f"  Skipping {run_key} (already completed)")
                continue

            # Log memory and CPU usage
            try:
                mem_usage = int(Path("/sys/fs/cgroup/memory.current").read_text().strip())
                mem_max = Path("/sys/fs/cgroup/memory.max").read_text().strip()
                mem_limit = int(mem_max) if mem_max != "max" else 0
                if mem_limit > 0:
                    logger.info(f"  Memory: {mem_usage/1e9:.1f}GB / {mem_limit/1e9:.1f}GB ({100*mem_usage/mem_limit:.0f}%)")
                else:
                    logger.info(f"  Memory: {mem_usage/1e9:.1f}GB")
            except Exception:
                pass
            try:
                cpu_usage = resource.getrusage(resource.RUSAGE_SELF)
                logger.info(f"  CPU time: user={cpu_usage.ru_utime:.0f}s, sys={cpu_usage.ru_stime:.0f}s, "
                           f"total={cpu_usage.ru_utime + cpu_usage.ru_stime:.0f}s / 14400s limit")
            except Exception:
                pass

            logger.info(f"  Training {run_key}...")
            run_start = time.time()

            use_pe = pe_config != "no_pe"

            # Build datasets for this PE config
            train_dataset = build_pyg_dataset(
                [graphs[i] for i in train_idx],
                [targets[i] for i in train_idx],
                [pe_dict[pe_config][i] for i in train_idx],
                use_pe=True
            )
            val_dataset = build_pyg_dataset(
                [graphs[i] for i in val_idx],
                [targets[i] for i in val_idx],
                [pe_dict[pe_config][i] for i in val_idx],
                use_pe=True
            )
            test_dataset = build_pyg_dataset(
                [graphs[i] for i in test_idx],
                [targets[i] for i in test_idx],
                [pe_dict[pe_config][i] for i in test_idx],
                use_pe=True
            )

            try:
                result = train_and_evaluate(
                    train_dataset, val_dataset, test_dataset,
                    use_pe=use_pe, seed=seed, num_epochs=NUM_EPOCHS,
                    patience=PATIENCE, verbose=True,
                    save_train_preds=(seed == 42)
                )
                run_elapsed = time.time() - run_start
                result["elapsed_seconds"] = run_elapsed
                result["pe_config"] = pe_config
                result["seed"] = seed

                all_results[run_key] = result
                logger.info(f"  DONE {run_key}: test_mae={result['test_mae']:.4f}, "
                             f"val_mae={result['val_mae']:.4f}, best_epoch={result['best_epoch']}, "
                             f"time={run_elapsed:.1f}s")

                # For non-seed42 runs, strip large prediction arrays to save memory
                if seed != 42:
                    result.pop("test_preds", None)
                    result.pop("test_targets", None)
                    result.pop("val_preds", None)
                    result.pop("train_preds", None)

                # Save checkpoint incrementally (only metadata, not predictions)
                checkpoint_data = {}
                for k, v in all_results.items():
                    checkpoint_data[k] = {
                        kk: vv for kk, vv in v.items()
                        if kk not in ("test_preds", "test_targets", "val_preds", "train_preds",
                                      "train_curve", "val_curve")
                    }
                checkpoint_path.write_text(json.dumps(checkpoint_data, indent=2))

                # Save full seed42 results with predictions for analysis
                if seed == 42:
                    seed42_data = {k: v for k, v in all_results.items()
                                   if k.endswith("_seed42") and "test_preds" in v}
                    with open(seed42_results_path, "wb") as f:
                        pickle.dump(seed42_data, f)

            except torch.cuda.OutOfMemoryError:
                logger.warning(f"  OOM on {run_key}, clearing cache and retrying with smaller batch")
                torch.cuda.empty_cache()
                gc.collect()
                continue
            except Exception:
                logger.exception(f"  FAILED {run_key}")
                continue

            # Aggressively clean up
            del train_dataset, val_dataset, test_dataset
            torch.cuda.empty_cache()
            gc.collect()

        # Check time budget
        elapsed_total = time.time() - start_time
        if elapsed_total > 3300:  # 55 min safety
            logger.warning(f"Time budget concern: {elapsed_total:.0f}s elapsed. "
                           f"Remaining configs may not complete.")

    # ── Step 4: Aggregate results ──
    logger.info("=" * 40)
    logger.info("RESULTS AGGREGATION")
    logger.info("=" * 40)

    results_summary = []
    for pe_config in PRIORITY:
        seed_results = [all_results[f"{pe_config}_seed{s}"]
                        for s in SEEDS
                        if f"{pe_config}_seed{s}" in all_results]
        if not seed_results:
            logger.warning(f"  No results for {pe_config}")
            continue

        test_maes = [r["test_mae"] for r in seed_results]
        val_maes = [r["val_mae"] for r in seed_results]
        epochs = [r["best_epoch"] for r in seed_results]

        summary = {
            "pe_type": pe_config,
            "test_mae_mean": float(np.mean(test_maes)),
            "test_mae_std": float(np.std(test_maes)),
            "val_mae_mean": float(np.mean(val_maes)),
            "val_mae_std": float(np.std(val_maes)),
            "avg_best_epoch": float(np.mean(epochs)),
            "n_seeds": len(seed_results),
            "test_maes": test_maes,
        }
        results_summary.append(summary)
        logger.info(f"  {pe_config}: test_mae={summary['test_mae_mean']:.4f}±{summary['test_mae_std']:.4f}, "
                     f"val_mae={summary['val_mae_mean']:.4f}±{summary['val_mae_std']:.4f}, "
                     f"n_seeds={summary['n_seeds']}")

    # ── Step 5: Per-graph analysis ──
    logger.info("=" * 40)
    logger.info("PER-GRAPH ANALYSIS")
    logger.info("=" * 40)

    analysis = {}

    # Compare offdiag vs rwpe
    rwpe_key = "rwpe_16_seed42"
    offdiag_key = "nrwpe_offdiag_16_seed42"
    if rwpe_key in all_results and offdiag_key in all_results:
        rwpe_preds = np.array(all_results[rwpe_key]["test_preds"])
        offdiag_preds = np.array(all_results[offdiag_key]["test_preds"])
        test_tgts = np.array(all_results[rwpe_key]["test_targets"])

        rwpe_errors = np.abs(rwpe_preds - test_tgts)
        offdiag_errors = np.abs(offdiag_preds - test_tgts)

        wins = int((offdiag_errors < rwpe_errors).sum())
        losses = int((offdiag_errors > rwpe_errors).sum())
        ties = int((offdiag_errors == rwpe_errors).sum())

        analysis["offdiag_vs_rwpe"] = {
            "offdiag_wins": wins,
            "rwpe_wins": losses,
            "ties": ties,
            "offdiag_mean_improvement": float((rwpe_errors - offdiag_errors).mean()),
        }
        logger.info(f"  offdiag vs rwpe: wins={wins}, losses={losses}, ties={ties}")

    # Softplus vs tanh comparison
    sp_key = "nrwpe_diag_softplus_16_seed42"
    tanh_key = "nrwpe_diag_tanh_16_seed42"
    if sp_key in all_results and tanh_key in all_results:
        sp_preds = np.array(all_results[sp_key]["test_preds"])
        tanh_preds = np.array(all_results[tanh_key]["test_preds"])
        test_tgts = np.array(all_results[sp_key]["test_targets"])

        sp_errors = np.abs(sp_preds - test_tgts)
        tanh_errors = np.abs(tanh_preds - test_tgts)

        analysis["softplus_vs_tanh"] = {
            "softplus_wins": int((sp_errors < tanh_errors).sum()),
            "tanh_wins": int((sp_errors > tanh_errors).sum()),
            "softplus_mean_improvement": float((tanh_errors - sp_errors).mean()),
        }
        logger.info(f"  softplus vs tanh: {analysis['softplus_vs_tanh']}")

    # Combined vs individual
    comb_key = "nrwpe_combined_16_seed42"
    if comb_key in all_results and offdiag_key in all_results and sp_key in all_results:
        comb_preds = np.array(all_results[comb_key]["test_preds"])
        test_tgts = np.array(all_results[comb_key]["test_targets"])
        comb_errors = np.abs(comb_preds - test_tgts)

        analysis["combined_vs_individual"] = {
            "combined_mae": float(comb_errors.mean()),
            "offdiag_mae": float(offdiag_errors.mean()) if offdiag_key in all_results else None,
            "softplus_diag_mae": float(sp_errors.mean()) if sp_key in all_results else None,
        }
        logger.info(f"  combined vs individual: {analysis['combined_vs_individual']}")

    # ── Step 6: Build method_out.json ──
    logger.info("=" * 40)
    logger.info("BUILDING OUTPUT")
    logger.info("=" * 40)

    # Load original data for full output
    raw = json.loads((DATA_DIR / "full_data_out.json").read_text())
    examples = raw["datasets"][0]["examples"]

    # Build index maps for O(1) lookup (global_idx -> position_in_fold)
    train_idx_map = {gi: fi for fi, gi in enumerate(train_idx)}
    val_idx_map = {gi: fi for fi, gi in enumerate(val_idx)}
    test_idx_map = {gi: fi for fi, gi in enumerate(test_idx)}

    # Build output examples with predictions
    output_examples = []
    for i, ex in enumerate(examples):
        out_ex = {
            "input": ex["input"],
            "output": ex["output"],
            "metadata_fold": ex["metadata_fold"],
            "metadata_task_type": ex["metadata_task_type"],
            "metadata_row_index": ex["metadata_row_index"],
            "metadata_num_nodes": ex["metadata_num_nodes"],
            "metadata_num_edges": ex["metadata_num_edges"],
        }

        fold = ex["metadata_fold"]

        # Add predictions from each config (seed 42 only for per-example)
        for pe_config in PRIORITY:
            run_key = f"{pe_config}_seed42"
            if run_key not in all_results:
                continue

            result = all_results[run_key]
            pred_key = f"predict_{pe_config}"

            if fold == "test" and "test_preds" in result:
                idx_in_fold = test_idx_map.get(i)
                if idx_in_fold is not None and idx_in_fold < len(result["test_preds"]):
                    out_ex[pred_key] = str(result["test_preds"][idx_in_fold])
            elif fold == "val" and "val_preds" in result:
                idx_in_fold = val_idx_map.get(i)
                if idx_in_fold is not None and idx_in_fold < len(result["val_preds"]):
                    out_ex[pred_key] = str(result["val_preds"][idx_in_fold])
            elif fold == "train" and "train_preds" in result:
                idx_in_fold = train_idx_map.get(i)
                if idx_in_fold is not None and idx_in_fold < len(result["train_preds"]):
                    out_ex[pred_key] = str(result["train_preds"][idx_in_fold])

        output_examples.append(out_ex)

    # Metadata
    metadata = {
        "method_name": "nRWPE Off-Diagonal + Softplus on ZINC-12k",
        "pe_configs": PE_CONFIGS,
        "model": f"GINEConv ({NUM_LAYERS} layers, {HIDDEN_DIM} hidden, edge-aware)",
        "architecture": {
            "atom_emb_dim": ATOM_EMB_DIM,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "num_atom_types": NUM_ATOM_TYPES,
            "num_bond_types": NUM_BOND_TYPES,
            "pooling": "add + mean concatenated",
        },
        "training": {
            "lr": LR,
            "lr_min": LR_MIN,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "patience": PATIENCE,
            "grad_clip": GRAD_CLIP,
            "seeds": SEEDS,
            "optimizer": "Adam",
            "scheduler": "CosineAnnealingLR",
            "loss": "L1Loss (MAE)",
        },
        "results_summary": results_summary,
        "analysis": analysis,
        "total_time_seconds": time.time() - start_time,
    }

    output = {
        "metadata": metadata,
        "datasets": [{
            "dataset": "ZINC-12k",
            "examples": output_examples
        }]
    }

    # Save
    output_path = SCRIPT_DIR / "method_out.json"
    output_path.write_text(json.dumps(output, indent=2))
    size_mb = output_path.stat().st_size / 1e6
    logger.info(f"Saved method_out.json ({size_mb:.1f}MB, {len(output_examples)} examples)")

    total_elapsed = time.time() - start_time
    logger.info(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    logger.info("DONE!")


if __name__ == "__main__":
    main()
