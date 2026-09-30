"""
Visualization Utilities for Polyp Segmentation.

Generates:
1. 4-Panel diagnostic comparison: [Original Frame | Ground Truth | Prediction | Error Map]
2. Translucent mask overlay with boundary contours on RGB frames
3. Multi-dataset comparative bar charts for publication papers
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class Visualizer:
    """Handles all visual reporting for polyp segmentation models."""

    def __init__(self, save_dir: str = './visualizations'):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def plot_sample_grid(
        self,
        image: np.ndarray,
        gt: np.ndarray,
        pred: np.ndarray,
        sample_name: str,
        save_path: str = None
    ):
        """
        Render 4-panel diagnostic figure:
        [Image | Ground Truth | Prediction | Error Map]
        Error Map colors:
            Green  : True Positive (Correct detection)
            Red    : False Positive (Over-segmentation)
            Blue   : False Negative (Missed polyp region)
        """
        # Ensure image is in [0, 1]
        if image.max() > 1.0:
            image = image / 255.0

        gt_b = (gt > 0.5).astype(np.float32)
        pred_b = (pred > 0.5).astype(np.float32)

        # Create error map
        h, w = gt_b.shape[:2]
        error_map = np.zeros((h, w, 3), dtype=np.float32)
        tp = (pred_b == 1) & (gt_b == 1)
        fp = (pred_b == 1) & (gt_b == 0)
        fn = (pred_b == 0) & (gt_b == 1)

        error_map[tp] = [0.0, 0.8, 0.0]  # Green: TP
        error_map[fp] = [0.9, 0.1, 0.1]  # Red: FP
        error_map[fn] = [0.1, 0.4, 0.9]  # Blue: FN

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        axes[0].imshow(image)
        axes[0].set_title("Input Colonoscopy", fontsize=12, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(gt_b, cmap='gray')
        axes[1].set_title("Ground Truth", fontsize=12, fontweight='bold')
        axes[1].axis('off')

        axes[2].imshow(pred, cmap='magma', vmin=0, vmax=1)
        axes[2].set_title("Predicted Mask", fontsize=12, fontweight='bold')
        axes[2].axis('off')

        axes[3].imshow(error_map)
        axes[3].set_title("Diagnostic Error Map\n(G:TP, R:FP, B:FN)", fontsize=11, fontweight='bold')
        axes[3].axis('off')

        plt.suptitle(f"Sample: {sample_name}", fontsize=14, y=1.02)
        plt.tight_layout()

        target_file = save_path or os.path.join(self.save_dir, f"grid_{sample_name}.png")
        os.makedirs(os.path.dirname(os.path.abspath(target_file)), exist_ok=True)
        plt.savefig(target_file, dpi=200, bbox_inches='tight')
        plt.close(fig)

    def plot_overlay(
        self,
        image: np.ndarray,
        pred: np.ndarray,
        sample_name: str,
        alpha: float = 0.45,
        save_path: str = None
    ):
        """Overlay translucent green segmentation mask onto colonoscopy image."""
        if image.max() > 1.0:
            image = image / 255.0

        pred_b = (pred > 0.5).astype(np.float32)

        overlay = image.copy()
        # Blend green channel
        overlay[..., 1] = np.where(pred_b > 0.5, (1 - alpha) * overlay[..., 1] + alpha * 0.95, overlay[..., 1])
        overlay[..., 0] = np.where(pred_b > 0.5, (1 - alpha) * overlay[..., 0], overlay[..., 0])
        overlay[..., 2] = np.where(pred_b > 0.5, (1 - alpha) * overlay[..., 2], overlay[..., 2])

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(overlay)
        ax.set_title(f"Overlay: {sample_name}", fontsize=12, fontweight='bold')
        ax.axis('off')

        target_file = save_path or os.path.join(self.save_dir, f"overlay_{sample_name}.png")
        os.makedirs(os.path.dirname(os.path.abspath(target_file)), exist_ok=True)
        plt.savefig(target_file, dpi=200, bbox_inches='tight')
        plt.close(fig)

    def plot_metrics_bar(self, results_dict: dict, save_path: str = None):
        """Plot comparative bar chart across evaluated datasets."""
        # Filter datasets (exclude 'Mean' for bar grouping, or include separately)
        datasets = [k for k in results_dict.keys() if k != 'Mean']
        if not datasets:
            return

        metrics = ['mDice', 'mIoU', 'wFb', 'Sm', 'Em']
        x = np.arange(len(datasets))
        width = 0.15

        fig, ax = plt.subplots(figsize=(12, 6))

        for idx, m in enumerate(metrics):
            vals = [results_dict[d].get(m, 0.0) for d in datasets]
            ax.bar(x + idx * width, vals, width, label=m)

        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Cross-Dataset Performance Profile', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(datasets, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.legend(loc='lower right', fontsize=11)

        plt.tight_layout()
        target_file = save_path or os.path.join(self.save_dir, "metrics_bar_chart.png")
        os.makedirs(os.path.dirname(os.path.abspath(target_file)), exist_ok=True)
        plt.savefig(target_file, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"[Visualizer] Saved metrics bar chart to '{target_file}'.")
