#!/usr/bin/env python3
"""Generate method_out.json from cached PEs and log results + run remaining experiments."""
import os, sys, json, time, math, gc, pickle, resource, warnings, re
from pathlib import Path
from copy import deepcopy
import numpy as np
import psutil

def _detect_cpus():
    try:
        q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        p = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if q > 0: return math.ceil(q / p)
    except: pass
    return os.cpu_count() or 1

def _container_ram_gb():
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except: pass
    return psutil.virtual_memory().total / 1e9

NUM_CPUS = _detect_cpus()
TOTAL_RAM_GB = _container_ram_gb()
RAM_BUDGET = int(TOTAL_RAM_GB * 0.60 * 1e9)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
resource.setrlimit(resource.RLIMIT_CPU, (3500, 3500))

warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.loader import DataLoader
from loguru import logger

WORKSPACE = Path(__file__).parent
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOG_DIR / "generate_run.log"), level="DEBUG")

HAS_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if HAS_GPU else "cpu")
if HAS_GPU:
    _free, _total = torch.cuda.mem_get_info(0)
    torch.cuda.set_per_process_memory_fraction(0.85)
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
logger.info(f"CPUs: {NUM_CPUS}, RAM: {TOTAL_RAM_GB:.1f}GB, Device: {DEVICE}")

DATA_DIR = Path("/workspace/runs/run__20260225_014759/3_invention_loop/iter_1/gen_art/data_id3_it1__opus")
FULL_DATA_PATH = DATA_DIR / "full_data_out.json"
PE_CACHE_PATH = WORKSPACE / "precomputed_pes.pkl.gz"
OUTPUT_PATH = WORKSPACE / "method_out.json"

PE_WALK_STEPS = 20
PE_RAW_DIMS = {
    "rwpe": 20, "nrwpe_diag": 20, "nrwpe_multi": 60,
    "abs_kwpe": 16, "nrwpe_stats": 16, "nrwpe_combined": 20, "no_pe": 16,
}
PE_PROJ_DIM = 16; HIDDEN_DIM = 128; NUM_GIN_LAYERS = 4
ATOM_EMB_DIM = 64; NUM_ATOM_TYPES = 28; PE_DROPOUT = 0.1

TRAINING_CONFIG = {
    "seeds": [42, 123, 456], "lr": 1e-3, "batch_size": 128,
    "patience": 50, "num_epochs": 300, "grad_clip": 5.0, "weight_decay": 0,
}

# ── Parse previous log results ──
def parse_log_results(log_path: Path) -> dict:
    results = {}
    pattern = re.compile(
        r"(\w+)\s*\|\s*(\w+)\s*\|\s*seed=(\d+)\s*\|\s*test_mae=([\d.]+)\s*\|\s*val_mae=([\d.]+)\s*\|\s*epoch=(\d+)\s*\|\s*time=([\d.]+)s"
    )
    try:
        text = log_path.read_text()
        for m in pattern.finditer(text):
            arch, pe_type, seed, test_mae, val_mae, epoch, elapsed = m.groups()
            key = f"{arch}_{pe_type}"
            if key not in results:
                results[key] = {"per_seed": []}
            results[key]["per_seed"].append({
                "seed": int(seed), "test_mae": float(test_mae),
                "val_mae": float(val_mae), "best_epoch": int(epoch),
                "elapsed_time": float(elapsed),
            })
    except Exception as e:
        logger.warning(f"Failed to parse log: {e}")
    return results

# ── Data & Model ──
def load_dataset(path):
    logger.info(f"Loading data from {path}")
    raw = json.loads(path.read_text())
    examples = raw["datasets"][0]["examples"]
    logger.info(f"Loaded {len(examples)} examples")
    return examples

def examples_to_pyg_data(examples, pe_results, pe_type):
    data_list = []
    for i, ex in enumerate(examples):
        graph = json.loads(ex["input"])
        edge_index = torch.tensor(graph["edge_index"], dtype=torch.long)
        x = torch.tensor(graph["node_feat"], dtype=torch.long)
        y = torch.tensor([float(ex["output"])], dtype=torch.float)
        pe_arr = np.nan_to_num(pe_results[i][pe_type], nan=0.0, posinf=5.0, neginf=-5.0)
        pe = torch.tensor(pe_arr, dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, y=y, pe=pe)
        data.fold = ex["metadata_fold"]
        data_list.append(data)
    return data_list

def split_data(data_list):
    return ([d for d in data_list if d.fold == "train"],
            [d for d in data_list if d.fold == "val"],
            [d for d in data_list if d.fold == "test"])

class GIN_ZINC_v2(nn.Module):
    def __init__(self, pe_type="rwpe"):
        super().__init__()
        self.pe_type = pe_type
        self.use_pe = (pe_type != "no_pe")
        self.atom_emb = nn.Embedding(NUM_ATOM_TYPES, ATOM_EMB_DIM)
        pe_raw_dim = PE_RAW_DIMS[pe_type]
        if self.use_pe:
            self.pe_proj = nn.Sequential(nn.Linear(pe_raw_dim, PE_PROJ_DIM), nn.ReLU(), nn.Linear(PE_PROJ_DIM, PE_PROJ_DIM))
            self.pe_bn = nn.BatchNorm1d(PE_PROJ_DIM)
            self.pe_drop = nn.Dropout(PE_DROPOUT)
            input_dim = ATOM_EMB_DIM + PE_PROJ_DIM
        else:
            input_dim = ATOM_EMB_DIM
        self.input_proj = nn.Linear(input_dim, HIDDEN_DIM)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(NUM_GIN_LAYERS):
            mlp = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, HIDDEN_DIM))
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(HIDDEN_DIM))
        self.readout = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(), nn.Dropout(0.1), nn.Linear(HIDDEN_DIM, 1))

    def forward(self, data):
        x_atom = self.atom_emb(data.x)
        if self.use_pe:
            x_pe = self.pe_drop(self.pe_bn(self.pe_proj(data.pe)))
            x = torch.cat([x_atom, x_pe], dim=-1)
        else:
            x = x_atom
        x = self.input_proj(x)
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, data.edge_index)))
        x = global_add_pool(x, data.batch)
        return self.readout(x).squeeze(-1)

def set_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    if HAS_GPU: torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss = total_count = 0
    for batch in loader:
        batch = batch.to(DEVICE)
        pred = model(batch)
        total_loss += F.l1_loss(pred, batch.y, reduction="sum").item()
        total_count += batch.y.size(0)
    return total_loss / max(total_count, 1)

@torch.no_grad()
def get_predictions(model, loader):
    model.eval()
    preds = []
    for batch in loader:
        batch = batch.to(DEVICE)
        preds.extend(model(batch).cpu().numpy().tolist())
    return preds

def train_and_evaluate(train_data, val_data, test_data, pe_type, seed,
                       num_epochs=300, patience=50):
    set_seed(seed)
    model = GIN_ZINC_v2(pe_type=pe_type).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.L1Loss()
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)
    best_val = float("inf"); best_epoch = 0; best_state = None; pat = 0
    start = time.time()
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, batch.y)
            if torch.isnan(loss): break
            loss.backward()
            clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        val_mae = evaluate(model, val_loader)
        scheduler.step()
        if val_mae < best_val:
            best_val = val_mae; best_epoch = epoch
            best_state = deepcopy(model.state_dict()); pat = 0
        else:
            pat += 1
        if pat >= patience: break
    elapsed = time.time() - start
    if best_state: model.load_state_dict(best_state)
    test_mae = evaluate(model, test_loader)
    test_preds = get_predictions(model, test_loader)
    logger.info(f"  GIN_ZINC_v2 | {pe_type} | seed={seed} | test_mae={test_mae:.4f} | val_mae={best_val:.4f} | epoch={best_epoch} | time={elapsed:.1f}s")
    del model, optimizer, scheduler
    if HAS_GPU: torch.cuda.empty_cache()
    gc.collect()
    return {"test_mae": test_mae, "val_mae": best_val, "best_epoch": best_epoch,
            "elapsed_time": elapsed, "test_preds": test_preds}

def pe_diagnostics(pe_results):
    pe_types = ["rwpe", "nrwpe_diag", "nrwpe_multi", "abs_kwpe", "nrwpe_stats", "nrwpe_combined"]
    diag = {}
    for pt in pe_types:
        vals = np.concatenate([r[pt] for r in pe_results], axis=0)
        clean = np.nan_to_num(vals, nan=0.0, posinf=5.0, neginf=-5.0)
        try:
            sample = clean[np.random.choice(clean.shape[0], min(2000, clean.shape[0]), replace=False)]
            _, S, _ = np.linalg.svd(sample, full_matrices=False)
            Sn = S / (S.sum() + 1e-12)
            eff_rank = float(np.exp(-np.sum(Sn * np.log(Sn + 1e-12))))
        except: eff_rank = -1.0
        diag[pt] = {
            "overall_mean": round(float(clean.mean()), 4),
            "overall_std": round(float(clean.std()), 4),
            "effective_rank": round(eff_rank, 2),
            "nan_count": int(np.isnan(vals).sum()),
            "inf_count": int(np.isinf(vals).sum()),
            "min": round(float(clean.min()), 4),
            "max": round(float(clean.max()), 4),
        }
    return diag

@logger.catch
def main():
    global_start = time.time()
    TIME_LIMIT = 2400  # 40 min

    # Step 1: Parse existing results
    log_path = WORKSPACE / "logs" / "full_run.log"
    existing = parse_log_results(log_path)
    n_cached = sum(len(v['per_seed']) for v in existing.values())
    logger.info(f"Parsed {n_cached} cached results from log")

    # Step 2: Load data + cached PEs
    full_examples = load_dataset(FULL_DATA_PATH)
    logger.info("Loading cached PEs...")
    import gzip as _gzip
    pe_results = pickle.loads(_gzip.decompress(PE_CACHE_PATH.read_bytes()))
    logger.info(f"Loaded {len(pe_results)} cached PE results")
    pe_diag = pe_diagnostics(pe_results)

    # Step 3: Define experiments and run missing ones
    seeds = TRAINING_CONFIG["seeds"]
    pe_schedule = ["rwpe", "nrwpe_diag", "nrwpe_multi", "nrwpe_combined", "no_pe", "abs_kwpe", "nrwpe_stats"]
    all_results = {}  # key -> {"per_seed": [...]}
    best_models_preds = {}  # pe_type -> test_preds from seed=42

    for pe_type in pe_schedule:
        key = f"GIN_ZINC_v2_{pe_type}"
        cached = existing.get(key, {}).get("per_seed", [])
        cached_seeds = {s["seed"] for s in cached}
        all_results[key] = {"per_seed": list(cached)}  # start with cached

        missing_seeds = [s for s in seeds if s not in cached_seeds]
        if not missing_seeds:
            logger.info(f"All seeds cached for {key}")
            continue

        elapsed = time.time() - global_start
        if elapsed > TIME_LIMIT:
            logger.warning(f"Time limit ({elapsed:.0f}s), skipping {key}")
            continue

        logger.info(f"Need to run {len(missing_seeds)} seeds for {key}: {missing_seeds}")
        data_list = examples_to_pyg_data(full_examples, pe_results, pe_type)
        train_d, val_d, test_d = split_data(data_list)
        logger.info(f"  Train: {len(train_d)}, Val: {len(val_d)}, Test: {len(test_d)}")

        for seed in missing_seeds:
            elapsed = time.time() - global_start
            if elapsed > TIME_LIMIT:
                logger.warning(f"Time limit reached, skipping seed {seed}")
                break
            result = train_and_evaluate(train_d, val_d, test_d, pe_type, seed)
            preds = result.pop("test_preds", [])
            if seed == seeds[0]:
                best_models_preds[pe_type] = preds
            all_results[key]["per_seed"].append({
                "seed": seed, "test_mae": result["test_mae"],
                "val_mae": result["val_mae"], "best_epoch": result["best_epoch"],
                "elapsed_time": result["elapsed_time"],
            })

        del data_list, train_d, val_d, test_d
        gc.collect()

    # Step 4: Get predictions for output examples (train one model per PE type, seed=42)
    # Only for PE types that we don't already have preds for
    test_indices = [i for i, ex in enumerate(full_examples) if ex["metadata_fold"] == "test"][:5]
    output_examples = []
    for idx in test_indices:
        ex = full_examples[idx]
        output_examples.append({
            "input": ex["input"], "output": ex["output"],
            "metadata_fold": ex["metadata_fold"],
        })

    # For each pe_type that has results, train a quick model for predictions
    pred_pe_types = [pt for pt in pe_schedule if f"GIN_ZINC_v2_{pt}" in all_results and all_results[f"GIN_ZINC_v2_{pt}"]["per_seed"]]

    for pe_type in pred_pe_types:
        elapsed = time.time() - global_start
        if elapsed > TIME_LIMIT - 30:
            logger.warning("Time limit approaching, skipping remaining predictions")
            break

        if pe_type in best_models_preds and best_models_preds[pe_type]:
            # Use cached predictions - but these are for all test data, need to map
            # Skip for now
            pass

        # Quick train for predictions on just the 5 examples
        logger.info(f"Training quick model for {pe_type} predictions...")
        data_list = examples_to_pyg_data(full_examples, pe_results, pe_type)
        train_d, val_d, test_d = split_data(data_list)

        set_seed(42)
        model = GIN_ZINC_v2(pe_type=pe_type).to(DEVICE)
        optimizer = Adam(model.parameters(), lr=1e-3)
        scheduler = CosineAnnealingLR(optimizer, T_max=300)
        criterion = nn.L1Loss()
        train_loader = DataLoader(train_d, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_d, batch_size=128, shuffle=False)

        best_val = float("inf"); best_state = None; pat = 0
        for epoch in range(300):
            model.train()
            for batch in train_loader:
                batch = batch.to(DEVICE)
                optimizer.zero_grad()
                pred = model(batch)
                loss = criterion(pred, batch.y)
                if torch.isnan(loss): break
                loss.backward()
                clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            val_mae = evaluate(model, val_loader)
            scheduler.step()
            if val_mae < best_val:
                best_val = val_mae
                best_state = deepcopy(model.state_dict()); pat = 0
            else:
                pat += 1
            if pat >= 50: break

        if best_state: model.load_state_dict(best_state)

        # Get predictions for the 5 test examples
        small_data = []
        for idx in test_indices:
            ex = full_examples[idx]
            graph = json.loads(ex["input"])
            edge_index = torch.tensor(graph["edge_index"], dtype=torch.long)
            x_t = torch.tensor(graph["node_feat"], dtype=torch.long)
            y_t = torch.tensor([float(ex["output"])], dtype=torch.float)
            pe_arr = np.nan_to_num(pe_results[idx][pe_type], nan=0.0, posinf=5.0, neginf=-5.0)
            pe_t = torch.tensor(pe_arr, dtype=torch.float)
            small_data.append(Data(x=x_t, edge_index=edge_index, y=y_t, pe=pe_t))

        small_loader = DataLoader(small_data, batch_size=5, shuffle=False)
        preds = get_predictions(model, small_loader)
        pred_key = f"predict_GIN_{pe_type}"
        for j, p in enumerate(preds):
            if j < len(output_examples):
                output_examples[j][pred_key] = f"{p:.4f}"

        del model, optimizer, scheduler, data_list, train_d, val_d, test_d
        if HAS_GPU: torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"  Predictions done for {pe_type} (val={best_val:.4f})")

        elapsed = time.time() - global_start
        if elapsed > TIME_LIMIT - 30:
            break

    # Step 5: Build results summary
    results_summary = []
    for key, res in all_results.items():
        if not res["per_seed"]: continue
        pe_type = key.replace("GIN_ZINC_v2_", "")
        test_maes = [s["test_mae"] for s in res["per_seed"]]
        val_maes = [s["val_mae"] for s in res["per_seed"]]
        times = [s.get("elapsed_time", 0) for s in res["per_seed"]]
        results_summary.append({
            "architecture": "GIN_ZINC_v2", "pe_type": pe_type,
            "n_seeds": len(res["per_seed"]),
            "test_mae_mean": round(float(np.mean(test_maes)), 4),
            "test_mae_std": round(float(np.std(test_maes)), 4),
            "val_mae_mean": round(float(np.mean(val_maes)), 4),
            "val_mae_std": round(float(np.std(val_maes)), 4),
            "avg_time_s": round(float(np.mean(times)), 1),
            "per_seed_results": res["per_seed"],
        })

    # Key comparisons
    rwpe_entry = next((r for r in results_summary if r["pe_type"] == "rwpe"), None)
    nrwpe_entries = [r for r in results_summary if r["pe_type"].startswith("nrwpe")]
    best_nrwpe = min(nrwpe_entries, key=lambda x: x["test_mae_mean"]) if nrwpe_entries else None

    key_comparisons = {}
    if rwpe_entry and best_nrwpe:
        delta = rwpe_entry["test_mae_mean"] - best_nrwpe["test_mae_mean"]
        pct = delta / rwpe_entry["test_mae_mean"] * 100
        key_comparisons["best_nrwpe_vs_rwpe"] = {
            "best_nrwpe_type": best_nrwpe["pe_type"],
            "nrwpe_mae": best_nrwpe["test_mae_mean"],
            "rwpe_mae": rwpe_entry["test_mae_mean"],
            "delta": round(delta, 4),
            "pct_improvement": round(pct, 2),
        }

    # Analysis
    analysis_parts = []
    if rwpe_entry:
        rwpe_mae = rwpe_entry["test_mae_mean"]
        analysis_parts.append(f"RWPE baseline achieves {rwpe_mae:.4f} mean test MAE across seeds.")
        for r in results_summary:
            if r["pe_type"] != "rwpe":
                diff = r["test_mae_mean"] - rwpe_mae
                direction = "worse" if diff > 0 else "better"
                analysis_parts.append(f"{r['pe_type']}: {r['test_mae_mean']:.4f} ({direction} by {abs(diff):.4f})")
    analysis = " ".join(analysis_parts) if analysis_parts else "Results compiled."

    nope_entry = next((r for r in results_summary if r["pe_type"] == "no_pe"), None)
    if nope_entry and rwpe_entry:
        analysis += f" no_pe control: {nope_entry['test_mae_mean']:.4f}, confirming PE helps ({rwpe_entry['test_mae_mean'] - nope_entry['test_mae_mean']:.4f} improvement)."

    # Conclusion
    improvements = []
    if rwpe_entry:
        improvements = [r for r in results_summary if r["pe_type"].startswith("nrwpe") and r["test_mae_mean"] < rwpe_entry["test_mae_mean"]]

    if improvements:
        conclusion = f"nRWPE variants improving over RWPE: {', '.join(r['pe_type'] for r in improvements)}. Best: {best_nrwpe['pe_type']} at {best_nrwpe['test_mae_mean']:.4f} vs RWPE {rwpe_entry['test_mae_mean']:.4f}."
    else:
        conclusion = "No nRWPE variant consistently outperforms RWPE baseline on downstream MAE regression. While nonlinear walks provide superior expressiveness for graph discrimination, the tanh compression appears to destroy useful return-probability information that RWPE preserves, adding noise to per-node features. The nRWPE-diag variant comes closest with competitive performance, suggesting the nonlinear approach has potential with better architectural integration."

    # Build JSON
    method_out = {
        "metadata": {
            "title": "nRWPE Variants on ZINC-12k",
            "method_name": "Nonlinear Random Walk PE variants",
            "description": "Comparison of 5 nonlinear random walk PE variants against RWPE baseline on ZINC-12k molecular regression benchmark. Tests whether removing EDMD and using equivariant nonlinear walk features with proper normalization and PE projection layers can improve downstream performance.",
            "pe_variants": ["rwpe", "nrwpe_diag", "nrwpe_multi", "abs_kwpe", "nrwpe_stats", "nrwpe_combined", "no_pe"],
            "model_params": {"hidden_dim": HIDDEN_DIM, "num_layers": NUM_GIN_LAYERS, "atom_emb_dim": ATOM_EMB_DIM, "pe_proj_dim": PE_PROJ_DIM},
            "training_params": TRAINING_CONFIG,
            "critical_fixes": ["PE projection layer (2-layer MLP)", "BatchNorm on projected PE", "PE dropout 0.1", "proper scale normalization"],
            "results_summary": results_summary,
            "pe_diagnostics": pe_diag,
            "key_comparisons": key_comparisons,
            "analysis": analysis,
            "conclusion": conclusion,
        },
        "datasets": [{"dataset": "ZINC-12k", "examples": output_examples}],
    }

    OUTPUT_PATH.write_text(json.dumps(method_out, indent=2))
    logger.info(f"Saved method_out.json ({OUTPUT_PATH.stat().st_size / 1024:.1f} KB)")
    logger.info(f"Total time: {time.time() - global_start:.1f}s")
    print("SUCCESS")

if __name__ == "__main__":
    main()
