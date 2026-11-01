"""Measure steering error on a held-out validation split.

The README reports an average steering error in degrees. This script makes that
number reproducible: it splits drive_log.csv into train / validation frames,
loads the trained weights, and prints the mean absolute error (MAE) in degrees
on the validation frames only.

Usage (on a PC or Colab, not the Pi):
    python evaluate_val.py --data data --weights model_regression.pth

Note: if the model was trained on all frames, the validation numbers are
optimistic. For an honest number, retrain on the train split only
(same seed) and evaluate here.
"""

import argparse
import os

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

IMAGE_SIZE = (160, 120)  # same as training (torchvision order: height, width)


class RegressionCNN(nn.Module):
    """Identical to the model in CNN_regression_model.ipynb and model_new.py."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2), nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(nn.Linear(64 * 17 * 12, 100), nn.ReLU(), nn.Linear(100, 1))

    def forward(self, x):
        return self.fc(self.flatten(self.conv(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="folder with drive_log.csv and frames")
    ap.add_argument("--weights", default="model_regression.pth")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    log = pd.read_csv(os.path.join(args.data, "drive_log.csv"))
    val = log.sample(frac=args.val_ratio, random_state=args.seed)
    print(f"frames: {len(log)} total, {len(val)} validation")

    tf = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    model = RegressionCNN()
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params:,}")

    errors = []
    with torch.no_grad():
        for _, row in val.iterrows():
            img = Image.open(os.path.join(args.data, row["frame"])).convert("RGB")
            pred_deg = model(tf(img).unsqueeze(0)).item() * 180.0  # labels were angle / 180
            errors.append(abs(pred_deg - float(row["steering_angle"])))

    errors = torch.tensor(errors)
    print(f"validation MAE: {errors.mean():.2f} deg  (median {errors.median():.2f}, max {errors.max():.2f})")


if __name__ == "__main__":
    main()
