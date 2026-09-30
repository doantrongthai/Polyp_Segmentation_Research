"""
CLI Visualization Script for Polyp Segmentation Models.

Generates:
1. 4-Panel diagnostic comparison [Image | GT | Prediction | Error Map]
2. Translucent green mask overlay on original colonoscopy frames
3. Cross-dataset metrics bar chart
"""

import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import pandas as pd
from models import get_model
from utils.checkpoint import Checkpointer
from utils.visualizer import Visualizer
from utils.dataloader import TestDataset
from utils.setup_dataset import setup_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate Visualizations for Polyp Segmentation")
    parser.add_argument('--model', type=str, default='pranet', help='Model name')
    parser.add_argument('--weights', type=str, required=True, help='Path to weights file (.pth)')
    parser.add_argument('--dataset', type=str, default='Kvasir', help='Test dataset to sample from')
    parser.add_argument('--num_samples', type=int, default=8, help='Number of sample images to visualize')
    parser.add_argument('--testsize', type=int, default=352, help='Input resolution')
    parser.add_argument('--data_root', type=str, default='./data', help='Dataset root path')
    parser.add_argument('--output_dir', type=str, default=None, help='Output visualization directory')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = args.output_dir or os.path.join('./visualizations', args.model)
    viz = Visualizer(save_dir=output_dir)

    # 1. Setup dataset if needed
    data_root = setup_dataset(args.data_root)

    # 2. Load model
    print(f"[Visualize] Initializing model '{args.model}' on {device}...")
    model = get_model(args.model)
    Checkpointer.load(model, args.weights, device=device)
    model.to(device)
    model.eval()

    # 3. Load test samples
    img_dir = os.path.join(data_root, 'TestDataset', args.dataset, 'images')
    gt_dir = os.path.join(data_root, 'TestDataset', args.dataset, 'masks')

    if not os.path.exists(img_dir) or not os.path.exists(gt_dir):
        print(f"[Visualize] Dataset '{args.dataset}' not found at '{img_dir}'.")
        return

    test_loader = TestDataset(img_dir, gt_dir, testsize=args.testsize)
    num_to_viz = min(args.num_samples, test_loader.size)
    print(f"[Visualize] Generating visual diagnostics for {num_to_viz} samples from '{args.dataset}'...")

    for idx in range(num_to_viz):
        img_tensor, gt_pil, name = test_loader.load_data()
        gt_np = np.asarray(gt_pil, dtype=np.float32)
        if gt_np.max() > 0:
            gt_np /= gt_np.max()

        orig_h, orig_w = gt_np.shape
        img_path = os.path.join(img_dir, name)
        if not os.path.exists(img_path):
            img_path = os.path.splitext(img_path)[0] + '.jpg'
        orig_img_pil = Image.open(img_path).convert('RGB')
        orig_img_np = np.asarray(orig_img_pil, dtype=np.float32)

        with torch.no_grad():
            img_t = img_tensor.to(device)
            outputs = model(img_t)
            pred = outputs[-1] if isinstance(outputs, (tuple, list)) else outputs
            pred = F.interpolate(pred, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
            pred_np = torch.sigmoid(pred).squeeze().cpu().numpy()
            pred_norm = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min() + 1e-8)

        stem = os.path.splitext(name)[0]
        # 4-panel diagnostic grid
        viz.plot_sample_grid(
            orig_img_np, gt_np, pred_norm, sample_name=stem,
            save_path=os.path.join(output_dir, f"diagnostic_{args.dataset}_{stem}.png")
        )
        # Translucent overlay
        viz.plot_overlay(
            orig_img_np, pred_norm, sample_name=stem,
            save_path=os.path.join(output_dir, f"overlay_{args.dataset}_{stem}.png")
        )

    # 4. If metrics CSV exists, render bar chart
    csv_path = os.path.join('./results', args.model, 'metrics.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0)
        results_dict = df.to_dict(orient='index')
        viz.plot_metrics_bar(results_dict, save_path=os.path.join(output_dir, f"{args.model}_metrics_bar.png"))

    print(f"[Visualize] All visual diagnostics saved to '{output_dir}'.")


if __name__ == '__main__':
    main()
