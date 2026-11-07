"""
Integrated Pipeline Runner for Polyp Segmentation Research.

Single-command entrypoint to:
1. Train model (train.py) with multi-scale deep supervision and OOM auto-recovery.
2. Evaluate across all 5 benchmark test sets and profile Edge AI metrics.
3. Generate publication-ready figures & Grad-CAM heatmaps (visualize.py).
4. Package weights, figures, CSVs, and LaTeX tables into a single .zip file.

Usage:
    python run.py --model pranet --seed 0
    python run.py --model polyp_pvt --seed 0 --epochs 100
"""

import argparse
import os
import sys
import subprocess
import shutil
import zipfile


def run_command(cmd, desc):
    """Run command with live stdout streaming."""
    print(f"\n[Pipeline] >>> {desc}...")
    print(f"[Pipeline] Command: {' '.join(cmd)}")
    process = subprocess.run(cmd)
    return process.returncode


def main():
    parser = argparse.ArgumentParser(description="End-to-End Polyp Segmentation Pipeline")
    parser.add_argument('--model', type=str, required=True, help='Model name to train and evaluate')
    parser.add_argument('--loss', type=str, default='structure_loss', help='Loss function')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--epochs', type=int, default=100, help='Total training epochs')
    parser.add_argument('--data_root', type=str, default='./data', help='Dataset root')
    parser.add_argument('--zip_dir', type=str, default=None, help='Target directory for output zip file')
    args = parser.parse_args()

    # Determine default zip directory (/content on Colab, current dir otherwise)
    if args.zip_dir is None:
        args.zip_dir = '/content' if os.path.exists('/content') else '.'
    os.makedirs(args.zip_dir, exist_ok=True)

    print("=" * 70)
    print(f"🚀 [PIPELINE] STARTING MODEL: {args.model.upper()} | SEED: {args.seed} | EPOCHS: {args.epochs}")
    print("=" * 70)

    # 1. Train & Benchmark Evaluate
    train_cmd = [
        sys.executable, "-u", "train.py",
        "--model", args.model,
        "--loss", args.loss,
        "--seed", str(args.seed),
        "--epochs", str(args.epochs),
        "--data_root", args.data_root
    ]
    ret_train = run_command(train_cmd, f"Training & Benchmarking {args.model.upper()}")
    if ret_train != 0:
        print(f"⚠️ [Pipeline] Training process exited with code {ret_train}.")
        return

    weights_path = os.path.join("snapshots", args.model, "best.pth")
    if not os.path.exists(weights_path):
        print(f"\n⚠️ [Pipeline] Best checkpoint not found at '{weights_path}' (likely skipped due to OOM).")
        print(f"👉 Exiting pipeline cleanly for model '{args.model}'.")
        return

    # 2. Visualize Grad-CAM & Paper Figures
    viz_cmd = [
        sys.executable, "-u", "visualize.py",
        "--model", args.model,
        "--weights", weights_path,
        "--dataset", "all",
        "--num_samples", "5",
        "--data_root", args.data_root
    ]
    run_command(viz_cmd, f"Generating Visualizations & Grad-CAM for {args.model.upper()}")

    # 3. Package Artifacts into .zip
    zip_path = os.path.join(args.zip_dir, f"PolypSeg_{args.model}_seed{args.seed}.zip")
    print(f"\n[Pipeline] >>> Packaging full model artifacts into '{zip_path}'...")

    target_dirs = [
        os.path.join("snapshots", args.model),
        os.path.join("results", args.model),
        os.path.join("visualizations", args.model),
        os.path.join("logs", args.model)
    ]
    existing_dirs = [d for d in target_dirs if os.path.exists(d)]

    # Use system zip if available, fallback to python zipfile
    if shutil.which("zip"):
        cmd_str = f"zip -q -r {zip_path} " + " ".join(existing_dirs)
        os.system(cmd_str)
    else:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for edir in existing_dirs:
                for root, _, files in os.walk(edir):
                    for file in files:
                        filepath = os.path.join(root, file)
                        zipf.write(filepath, arcname=filepath)

    if os.path.exists(zip_path):
        size_mb = os.path.getsize(zip_path) / (1024 * 1024)
        print(f"✅ [Pipeline] Package generated successfully: {zip_path} ({size_mb:.2f} MB)")
    else:
        print(f"⚠️ [Pipeline] Could not generate zip file at {zip_path}")

    print("=" * 70)
    print(f"🎉 [PIPELINE] FINISHED MODEL: {args.model.upper()}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
