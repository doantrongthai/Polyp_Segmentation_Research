"""
Comprehensive Evaluator for Polyp Segmentation Across 5 Benchmark Datasets.

Evaluates on:
1. CVC-300 (EndoScene test subset)
2. CVC-ClinicDB
3. CVC-ColonDB
4. ETIS-LaribPolypDB
5. Kvasir-SEG

Calculates all standard benchmarks: mDice, mIoU, wFb, Sm, Em, MAE.
Outputs beautiful tables in PSQL, CSV, and LaTeX formats.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
from utils.metrics import MetricCalculator
from utils.dataloader import TestDataset

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


class Evaluator:
    """Evaluates segmentation models across standard polyp benchmarks."""

    TEST_SETS = ['CVC-300', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Kvasir']

    def __init__(self, data_root: str, model=None, device=None, testsize: int = 352):
        self.data_root = os.path.join(data_root, 'TestDataset') if not data_root.endswith('TestDataset') else data_root
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.testsize = testsize
        self.metrics = MetricCalculator()

    def evaluate_all(self, save_results_dir: str = None) -> dict:
        """
        Run inference and compute metrics on all 5 test sets.
        Optionally save binary prediction masks to save_results_dir.
        """
        assert self.model is not None, "Model must be provided for evaluation."
        self.model.eval()
        self.model.to(self.device)

        all_results = {}

        for ds_name in self.TEST_SETS:
            img_dir = os.path.join(self.data_root, ds_name, 'images')
            gt_dir = os.path.join(self.data_root, ds_name, 'masks')

            if not os.path.exists(img_dir) or not os.path.exists(gt_dir):
                print(f"[Evaluator] Skipping '{ds_name}': directory not found.")
                continue

            test_loader = TestDataset(img_dir, gt_dir, testsize=self.testsize)
            self.metrics.reset()

            save_mask_dir = None
            if save_results_dir:
                save_mask_dir = os.path.join(save_results_dir, ds_name)
                os.makedirs(save_mask_dir, exist_ok=True)

            with torch.no_grad():
                for _ in tqdm(range(test_loader.size), desc=f"Evaluating {ds_name}"):
                    img_tensor, gt_pil, name = test_loader.load_data()
                    gt_np = np.asarray(gt_pil, dtype=np.float32)
                    if gt_np.max() > 0:
                        gt_np /= gt_np.max()

                    orig_h, orig_w = gt_np.shape

                    img_tensor = img_tensor.to(self.device)
                    outputs = self.model(img_tensor)

                    # Extract fine prediction (lateral_map_2 for PraNet)
                    pred = outputs[-1] if isinstance(outputs, (tuple, list)) else outputs

                    # Upsample prediction back to original image size
                    pred = F.interpolate(pred, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
                    pred_sig = torch.sigmoid(pred).squeeze().cpu().numpy()

                    # Normalize
                    pred_norm = (pred_sig - pred_sig.min()) / (pred_sig.max() - pred_sig.min() + 1e-8)

                    # Accumulate metrics
                    self.metrics.update(pred_norm, gt_np)

                    # Save prediction mask if requested
                    if save_mask_dir:
                        out_img = (pred_norm * 255.0).astype(np.uint8)
                        Image.fromarray(out_img).save(os.path.join(save_mask_dir, name))

            res = self.metrics.get_results()
            all_results[ds_name] = res

        # Compute average across all datasets
        if all_results:
            metric_keys = list(next(iter(all_results.values())).keys())
            mean_row = {}
            for k in metric_keys:
                vals = [all_results[ds][k] for ds in all_results]
                mean_row[k] = float(np.mean(vals))
            all_results['Mean'] = mean_row

        return all_results

    def print_table(self, results: dict):
        """Format and print benchmark results."""
        if not results:
            print("[Evaluator] No results to display.")
            return

        df = pd.DataFrame(results).T
        cols = ['mDice', 'mIoU', 'wFb', 'Sm', 'Em', 'MAE']
        display_cols = [c for c in cols if c in df.columns]
        df_disp = df[display_cols]

        print("\n" + "=" * 70)
        print("POLYP SEGMENTATION BENCHMARK EVALUATION RESULTS")
        print("=" * 70)
        if HAS_TABULATE:
            print(tabulate(df_disp, headers='keys', tablefmt='psql', floatfmt=".4f"))
        else:
            print(df_disp.to_string())
        print("=" * 70 + "\n")

    def save_csv(self, results: dict, path: str):
        """Save results to CSV."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        df = pd.DataFrame(results).T
        df.to_csv(path)
        print(f"[Evaluator] Saved CSV results to '{path}'.")

    def save_latex(self, results: dict, path: str):
        """Generate and save LaTeX table."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        df = pd.DataFrame(results).T
        cols = ['mDice', 'mIoU', 'wFb', 'Sm', 'Em', 'MAE']
        display_cols = [c for c in cols if c in df.columns]
        df_disp = df[display_cols]

        latex_str = df_disp.to_latex(float_format="%.4f")
        with open(path, 'w', encoding='utf-8') as f:
            f.write(latex_str)
        print(f"[Evaluator] Saved LaTeX table to '{path}'.")
