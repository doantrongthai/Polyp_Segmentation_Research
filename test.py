"""
Test Script for Polyp Segmentation Models.

Runs inference across all 5 benchmark test sets:
- CVC-300
- CVC-ClinicDB
- CVC-ColonDB
- ETIS-LaribPolypDB
- Kvasir

Saves predicted masks to './results/{model}/{dataset}/' and generates full benchmark tables.
"""

import argparse
import os
import torch
from models import get_model
from utils.checkpoint import Checkpointer
from utils.evaluator import Evaluator
from utils.setup_dataset import setup_dataset


def main():
    parser = argparse.ArgumentParser(description="Run Evaluation on Polyp Segmentation Models")
    parser.add_argument('--model', type=str, default='pranet', help='Model name (registered in models/)')
    parser.add_argument('--weights', type=str, required=True, help='Path to weights file (.pth)')
    parser.add_argument('--testsize', type=int, default=352, help='Input evaluation resolution')
    parser.add_argument('--data_root', type=str, default='./data', help='Path to dataset directory')
    parser.add_argument('--save_dir', type=str, default='./results', help='Directory to save output masks')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Test] Hardware platform: {device}")

    # 1. Setup dataset if not already extracted
    data_root = setup_dataset(args.data_root)

    # 2. Build model and load weights
    print(f"[Test] Building model '{args.model}'...")
    model = get_model(args.model)
    Checkpointer.load(model, args.weights, device=device)
    model.to(device)
    model.eval()

    # 3. Run evaluation across all 5 test sets
    results_dir = os.path.join(args.save_dir, args.model)
    evaluator = Evaluator(data_root=data_root, model=model, device=device, testsize=args.testsize)

    print(f"[Test] Running evaluation and saving masks to '{results_dir}'...")
    results = evaluator.evaluate_all(save_results_dir=results_dir)

    # 4. Print and save results
    evaluator.print_table(results)
    evaluator.save_csv(results, path=os.path.join(results_dir, 'metrics.csv'))
    evaluator.save_latex(results, path=os.path.join(results_dir, 'metrics.tex'))

    print("[Test] Completed successfully.")


if __name__ == '__main__':
    main()
