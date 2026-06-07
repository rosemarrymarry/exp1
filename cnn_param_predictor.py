from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from io_utils import imread_gray_float, ensure_dir
from noise import add_gaussian_noise_sigma255
from bitonic_filter import bitonic_filter_tiled
from metrics import psnr, ssim


def find_image_by_stem(images_dir: Path, stem: str) -> Path | None:
    for p in images_dir.iterdir():
        if p.stem == stem:
            return p
    return None


class ParamDataset(Dataset):
    """Dataset that yields (noisy_image_tensor, target_params) rows from a labels CSV."""

    def __init__(self, images_dir: Path, labels_csv: Path):
        self.images_dir = Path(images_dir)
        self.rows = []
        with open(labels_csv, newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                # expect fields: image, sigma, ksize, centile, gauss_sigma
                self.rows.append(r)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        stem = r["image"]
        sigma = float(r["sigma"])  # sigma_255 used to generate noisy input

        img_path = find_image_by_stem(self.images_dir, stem)
        if img_path is None:
            raise FileNotFoundError(f"Image with stem {stem} not found in {self.images_dir}")

        img = imread_gray_float(img_path)

        # deterministic RNG per sample
        seed = hash((stem, sigma)) & 0xFFFFFFFF
        rng = np.random.default_rng(seed=seed)
        noisy = add_gaussian_noise_sigma255(img, sigma_255=sigma, rng=rng)

        # normalize to [0,1] (imread_gray_float already float in [0,1] expected)
        x = torch.from_numpy(noisy.astype(np.float32))[None, ...]

        # Targets: ksize, centile, gauss_sigma
        ksize = float(r.get("ksize", 3))
        cent = float(r.get("centile", 1.0))
        gsig = float(r.get("gauss_sigma", 0.0))

        y = torch.tensor([ksize, cent, gsig], dtype=torch.float32)
        return x, y


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, x):
        return self.net(x)


def train(args: argparse.Namespace):
    labels_csv = Path(args.labels_csv)
    images_dir = Path(args.images_dir)
    out_dir = ensure_dir(Path(args.out))

    ds = ParamDataset(images_dir, labels_csv)
    if len(ds) == 0:
        raise RuntimeError("No labelled rows found in labels CSV")

    # simple split
    n = len(ds)
    n_train = int(n * 0.8)
    indices = np.arange(n)
    np.random.shuffle(indices)
    train_idx = indices[:n_train].tolist()
    val_idx = indices[n_train:].tolist()

    train_ds = torch.utils.data.Subset(ds, train_idx)
    val_ds = torch.utils.data.Subset(ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = SmallCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
        train_loss = running / len(train_loader.dataset)

        model.eval()
        running = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                running += loss.item() * xb.size(0)
        val_loss = running / len(val_loader.dataset)

        print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), out_dir / "cnn_param_model.pth")

    print("Training complete. Best val loss:", best_val)


def nearest_ksize(pred_ksize: float, ksizes: List[int]) -> int:
    ks = np.array(ksizes)
    idx = int(np.argmin(np.abs(ks - pred_ksize)))
    return int(ks[idx])


def predict_and_apply(args: argparse.Namespace):
    images_dir = Path(args.images_dir)
    labels_csv = Path(args.labels_csv)
    out_dir = ensure_dir(Path(args.out))

    model = SmallCNN()
    model.load_state_dict(torch.load(Path(args.model_path), map_location="cpu"))
    model.eval()

    ksizes = args.ksizes

    out_file = out_dir / "cnn_pred_results.csv"
    fieldnames = [
        "image",
        "sigma",
        "pred_ksize",
        "pred_centile",
        "pred_gauss_sigma",
        "psnr",
        "ssim",
    ]
    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        with open(labels_csv, newline="") as lf:
            reader = csv.DictReader(lf)
            for r in reader:
                stem = r["image"]
                sigma = float(r["sigma"]) if r.get("sigma") is not None else float(r.get("sigma", 25.0))
                img_path = find_image_by_stem(images_dir, stem)
                if img_path is None:
                    continue
                img = imread_gray_float(img_path)
                rng = np.random.default_rng(seed=(hash((stem, sigma)) & 0xFFFFFFFF))
                noisy = add_gaussian_noise_sigma255(img, sigma_255=sigma, rng=rng)

                xb = torch.from_numpy(noisy.astype(np.float32))[None, None, ...]
                with torch.no_grad():
                    pred = model(xb).squeeze(0).numpy()

                pred_ks, pred_cent, pred_gsig = float(pred[0]), float(pred[1]), float(pred[2])
                chosen_ks = nearest_ksize(pred_ks, ksizes)
                pred_cent = max(0.0, pred_cent)
                pred_gsig = max(0.0, pred_gsig)

                deno = bitonic_filter_tiled(
                    noisy,
                    ksize=chosen_ks,
                    centile=pred_cent,
                    m=args.m,
                    gauss_sigma=pred_gsig,
                    gauss_alpha=args.galpha,
                )

                p = psnr(img, deno)
                s = ssim(img, deno)

                writer.writerow(
                    {
                        "image": stem,
                        "sigma": sigma,
                        "pred_ksize": chosen_ks,
                        "pred_centile": float(pred_cent),
                        "pred_gauss_sigma": float(pred_gsig),
                        "psnr": float(p),
                        "ssim": float(s),
                    }
                )

    print("Prediction + application results written to:", out_file)


def main(argv: List[str] | None = None):
    p = argparse.ArgumentParser(description="Train or predict filter params with a small CNN")
    sub = p.add_subparsers(dest="mode", required=True)

    t = sub.add_parser("train")
    t.add_argument("--images_dir", type=Path, default=Path("exp1-images"))
    t.add_argument("--labels_csv", type=Path, default=Path("outputs/best_params_labels.csv"))
    t.add_argument("--out", type=Path, default=Path("outputs"))
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--batch_size", type=int, default=8)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--device", type=str, default="cuda")

    p2 = sub.add_parser("predict")
    p2.add_argument("--images_dir", type=Path, default=Path("exp1-images"))
    p2.add_argument("--labels_csv", type=Path, default=Path("outputs/best_params_labels.csv"))
    p2.add_argument("--model_path", type=Path, default=Path("outputs/cnn_param_model.pth"))
    p2.add_argument("--out", type=Path, default=Path("outputs"))
    p2.add_argument("--ksizes", type=int, nargs="+", default=[3, 5, 7])
    p2.add_argument("--m", type=float, default=3.0)
    p2.add_argument("--galpha", type=float, default=1.0)

    args = p.parse_args(argv)
    if args.mode == "train":
        train(args)
    elif args.mode == "predict":
        predict_and_apply(args)


if __name__ == "__main__":
    main()
