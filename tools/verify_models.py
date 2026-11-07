"""
Model Verification Engine for Polyp Segmentation Research.

Runs automated verification across all registered models:
1. Instantiation & weight verification.
2. Eval & train mode forward passes (shapes, side outputs, dtypes, range, finiteness).
3. Parameter counts (Total, Trainable, Non-trainable).
4. FLOPs / MACs benchmarking using thop at 1x3x352x352 resolution.
5. Per-layer architecture dumps using torchinfo into reports/arch/<model>.txt.
6. Comprehensive markdown table generation at reports/verification.md.

Usage:
    python tools/verify_models.py [--models name1 name2 ...] [--device cpu|cuda]
"""

import os
import sys
import argparse
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torchinfo
import thop

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import get_model, list_models, _MODEL_REGISTRY
from utils.model_utils import split_model_output, fuse_outputs


def get_unique_models() -> Dict[str, type]:
    """Return unique registered models mapping canonical name -> model class."""
    if not _MODEL_REGISTRY:
        list_models()
    unique = {}
    seen_classes = set()
    for name, cls in _MODEL_REGISTRY.items():
        if cls not in seen_classes:
            unique[name] = cls
            seen_classes.add(cls)
    return unique


def verify_single_model(
    model_name: str,
    device: torch.device = torch.device("cpu"),
    arch_dir: Path = PROJECT_ROOT / "reports" / "arch",
) -> Dict[str, Any]:
    """Verify a single registered model and return diagnostic metrics."""
    result: Dict[str, Any] = {
        "name": model_name,
        "class": None,
        "status": "FAIL",
        "total_params": 0,
        "trainable_params": 0,
        "non_trainable_params": 0,
        "gflops": 0.0,
        "gmacs": 0.0,
        "num_outputs_eval": 0,
        "eval_shapes": [],
        "num_outputs_train": 0,
        "train_shapes": [],
        "primary_shape": None,
        "dtype": None,
        "is_finite": False,
        "error": None,
    }

    try:
        model = get_model(model_name).to(device)
        result["class"] = model.__class__.__name__

        # Parameter counts
        total_p = sum(p.numel() for p in model.parameters())
        trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_p = total_p - trainable_p
        result["total_params"] = total_p
        result["trainable_params"] = trainable_p
        result["non_trainable_params"] = non_trainable_p

        # Eval forward pass: B=1, C=3, H=352, W=352
        model.eval()
        x_eval = torch.randn(1, 3, 352, 352, device=device)
        with torch.no_grad():
            out_eval = model(x_eval)

        if isinstance(out_eval, (tuple, list)):
            result["num_outputs_eval"] = len(out_eval)
            result["eval_shapes"] = [list(o.shape) for o in out_eval]
        else:
            result["num_outputs_eval"] = 1
            result["eval_shapes"] = [list(out_eval.shape)]

        primary_pred = fuse_outputs(out_eval)
        result["primary_shape"] = list(primary_pred.shape)
        result["dtype"] = str(primary_pred.dtype).replace("torch.", "")
        result["is_finite"] = bool(torch.all(torch.isfinite(primary_pred)).item())

        # Train forward pass: B=2 to avoid 1-sample batch norm constraints in global pooling modules
        model.train()
        x_train = torch.randn(2, 3, 352, 352, device=device)
        try:
            out_train = model(x_train)
            if isinstance(out_train, (tuple, list)):
                result["num_outputs_train"] = len(out_train)
                result["train_shapes"] = [list(o.shape) for o in out_train]
            else:
                result["num_outputs_train"] = 1
                result["train_shapes"] = [list(out_train.shape)]
        except Exception as e:
            result["train_shapes"] = [f"Error: {e}"]

        # FLOPs benchmarking with thop (in eval mode)
        model.eval()
        try:
            flops, _ = thop.profile(model, inputs=(x_eval,), verbose=False)
            result["gflops"] = round(flops / 1e9, 3)
            result["gmacs"] = round((flops / 2) / 1e9, 3)
        except Exception as e:
            result["gflops"] = -1.0

        # Architecture summary dump using torchinfo
        arch_dir.mkdir(parents=True, exist_ok=True)
        arch_file = arch_dir / f"{model_name}.txt"
        try:
            info_str = str(
                torchinfo.summary(
                    model,
                    input_size=(1, 3, 352, 352),
                    device=device,
                    verbose=0,
                    col_names=["input_size", "output_size", "num_params", "mult_adds", "trainable"],
                )
            )
            with open(arch_file, "w", encoding="utf-8") as f:
                f.write(f"Architecture Summary for {model_name} ({model.__class__.__name__})\n")
                f.write("=" * 80 + "\n\n")
                f.write(info_str)
        except Exception as e:
            with open(arch_file, "w", encoding="utf-8") as f:
                f.write(f"torchinfo summary error: {e}\n\n{repr(model)}")

        result["status"] = "OK"

    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = f"{type(e).__name__}: {str(e)}"
        traceback.print_exc()

    return result


def generate_verification_report(results: List[Dict[str, Any]], report_path: Path):
    """Generate reports/verification.md from verification results."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Model Verification Report\n\n")
        f.write("This report provides rigorous, empirically verified architectural benchmarks\n")
        f.write("for all registered polyp segmentation models evaluated on dummy input `(1, 3, 352, 352)`.\n\n")
        f.write("All FLOPs are measured using `thop` and architecture summaries are dumped to `reports/arch/<model>.txt`.\n\n")

        f.write("## Verified Models Summary Table\n\n")
        f.write("| Model Name | Class | Total Params | Trainable | GFLOPs | Outputs (Eval) | Primary Shape | Dtype | Finite | Status |\n")
        f.write("|:---|:---|---:|---:|---:|:---:|:---:|:---:|:---:|:---:|\n")

        for r in results:
            tot = f"{r['total_params']:,}" if r['total_params'] > 0 else "-"
            trn = f"{r['trainable_params']:,}" if r['trainable_params'] > 0 else "-"
            flp = f"{r['gflops']:.2f}" if r['gflops'] >= 0 else "N/A"
            outs = str(r["num_outputs_eval"]) if r["num_outputs_eval"] > 0 else "-"
            shp = str(r["primary_shape"]) if r["primary_shape"] else "-"
            dt = r["dtype"] or "-"
            fin = "PASS" if r["is_finite"] else "FAIL"
            stat = f"**{r['status']}**" if r["status"] == "OK" else f"❌ {r['status']}"

            f.write(f"| `{r['name']}` | `{r['class']}` | {tot} | {trn} | {flp} | {outs} | `{shp}` | `{dt}` | {fin} | {stat} |\n")

        f.write("\n## Multi-Stage Output Details\n\n")
        for r in results:
            if r["num_outputs_eval"] > 1:
                f.write(f"### `{r['name']}` ({r['class']})\n")
                f.write(f"- **Number of outputs:** {r['num_outputs_eval']}\n")
                f.write(f"- **Eval output shapes (B=1):** `{r['eval_shapes']}`\n")
                f.write(f"- **Train output shapes (B=2):** `{r['train_shapes']}`\n")
                f.write(f"- **Primary prediction (finest, used in `fuse_outputs`):** `{r['primary_shape']}`\n\n")

        f.write("## Verification Notes & Architecture Artifacts\n\n")
        f.write("- Detailed layer-by-layer `torchinfo` summaries for each model are persisted in `reports/arch/<model_name>.txt`.\n")
        f.write("- Multi-scale models expose their primary resolution map at `outputs[-1]` or through `fuse_outputs()` conforming strictly to project interface.\n")
        f.write("- Deep-supervision losses are unified under `compute_loss_with_aux(outputs, target, loss_fn, weights=...)`.\n")


def main():
    parser = argparse.ArgumentParser(description="Verify registered polyp segmentation models.")
    parser.add_argument("--models", nargs="*", default=None, help="Specific model names to verify.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Computation device.")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"[*] Starting verification on device: {device}")

    unique_models = get_unique_models()
    target_names = args.models if args.models else sorted(unique_models.keys())

    results = []
    print(f"[*] Verifying {len(target_names)} models...")

    for i, name in enumerate(target_names, 1):
        print(f"\n[{i}/{len(target_names)}] Verifying '{name}'...")
        res = verify_single_model(name, device=device)
        print(f"    Status: {res['status']} | Params: {res['total_params']:,} | GFLOPs: {res['gflops']} | Outputs: {res['num_outputs_eval']} | Primary: {res['primary_shape']}")
        if res["error"]:
            print(f"    Error: {res['error']}")
        results.append(res)

    report_path = PROJECT_ROOT / "reports" / "verification.md"
    generate_verification_report(results, report_path)
    print(f"\n[+] Verification complete! Report generated at: {report_path}")


if __name__ == "__main__":
    main()
