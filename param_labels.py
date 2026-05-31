from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np

from io_utils import ensure_dir, imread_gray_float, imwrite_gray_float
from noise import add_gaussian_noise_sigma255
from bitonic_filter import bitonic_filter_tiled
from metrics import psnr, ssim


def parse_floats(values: List[str]) -> List[float]:
    return [float(v) for v in values]


def parse_ints(values: List[str]) -> List[int]:
    return [int(v) for v in values]


def all_image_paths(images_dir: Path) -> List[Path]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    files: List[Path] = []
    for e in exts:
        files.extend(sorted(images_dir.glob(e)))
    return files


def run_grid_search(
    images_dir: Path,
    out_dir: Path,
    sigmas: List[float],
    ksizes: List[int],
    centiles: List[float],
    gsigmas: List[float],
    m: float = 3.0,
    galpha: float = 1.0,
    repeats: int = 1,
    metric: str = "psnr",
):
    out_dir = ensure_dir(out_dir)
    details_dir = ensure_dir(out_dir / "details")

    # Prepare best labels CSV (appendable) and fieldnames
    best_csv = out_dir / "best_params_labels.csv"
    fieldnames = [
        "image",
        "sigma",
        "ksize",
        "centile",
        "gauss_sigma",
        "m",
        "gauss_alpha",
        "psnr_mean",
        "psnr_std",
        "ssim_mean",
        "ssim_std",
    ]
    if not best_csv.exists():
        with open(best_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    image_paths = all_image_paths(images_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    summary_rows = []

    # compute total runs for progress reporting
    total_runs = (
        len(image_paths)
        * len(sigmas)
        * len(ksizes)
        * len(centiles)
        * len(gsigmas)
        * max(1, repeats)
    )
    global_idx = 0
    width = len(str(total_runs))

    for img_path in image_paths:
        name = img_path.stem
        img = imread_gray_float(img_path)

        for sigma in sigmas:
            combos_results = []
            # prepare details file for this (image,sigma)
            details_file = details_dir / f"{name}_sigma{int(sigma)}_results.csv"
            if not details_file.exists():
                with open(details_file, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
            for ksize in ksizes:
                for cent in centiles:
                    for gsig in gsigmas:
                        psnr_list = []
                        ssim_list = []
                        for r in range(repeats):
                            rng = np.random.default_rng(seed=(hash((name, sigma, ksize, cent, gsig, r)) & 0xFFFFFFFF))
                            noisy = add_gaussian_noise_sigma255(img, sigma_255=sigma, rng=rng)

                            deno = bitonic_filter_tiled(
                                noisy,
                                ksize=ksize,
                                centile=cent,
                                m=m,
                                gauss_sigma=gsig,
                                gauss_alpha=galpha,
                            )

                            ps = psnr(img, deno)
                            ss = ssim(img, deno)
                            psnr_list.append(ps)
                            ssim_list.append(ss)

                            # progress counter and console output
                            global_idx += 1
                            idx_str = str(global_idx).zfill(width)
                            print(f"[{idx_str}/{total_runs}] {name} ksize={ksize} cent={cent} gsig={gsig} r={r} -> PSNR={ps:.3f} SSIM={ss:.4f}")

                            # flush to ensure Kaggle shows live output
                            try:
                                import sys

                                sys.stdout.flush()
                            except Exception:
                                pass

                        row = {
                            "image": name,
                            "sigma": float(sigma),
                            "ksize": int(ksize),
                            "centile": float(cent),
                            "gauss_sigma": float(gsig),
                            "m": float(m),
                            "gauss_alpha": float(galpha),
                            "psnr_mean": float(np.mean(psnr_list)),
                            "psnr_std": float(np.std(psnr_list, ddof=0)),
                            "ssim_mean": float(np.mean(ssim_list)),
                            "ssim_std": float(np.std(ssim_list, ddof=0)),
                        }
                        combos_results.append(row)

                        # append this combo result immediately to details CSV
                        with open(details_file, "a", newline="") as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writerow(row)

            # Choose best by metric
            if metric.lower() == "psnr":
                best = max(combos_results, key=lambda x: x["psnr_mean"])
            elif metric.lower() == "ssim":
                best = max(combos_results, key=lambda x: x["ssim_mean"])
            else:
                raise ValueError("metric must be 'psnr' or 'ssim'")

            # append best to global best CSV immediately
            with open(best_csv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(best)

            summary_rows.append(best)

    return best_csv


def main(argv: List[str] | None = None):
    p = argparse.ArgumentParser(description="Grid search ksize/centile/gauss_sigma to produce best-param labels")
    p.add_argument("--images_dir", type=Path, default=Path("exp1-images"))
    p.add_argument("--out", type=Path, default=Path("outputs"))
    p.add_argument("--sigmas", type=float, nargs="+", default=[25.0])
    p.add_argument("--ksizes", type=int, nargs="+", default=[3, 5, 7])
    p.add_argument("--centiles", type=float, nargs="+", default=[1.0, 5.0, 10.0])
    p.add_argument("--gsigmas", type=float, nargs="+", default=[0.0, 1.0, 2.0])
    p.add_argument("--m", type=float, default=3.0)
    p.add_argument("--galpha", type=float, default=1.0)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--metric", type=str, default="psnr", choices=["psnr", "ssim"])

    args = p.parse_args(argv)

    best_csv = run_grid_search(
        images_dir=args.images_dir,
        out_dir=args.out,
        sigmas=args.sigmas,
        ksizes=args.ksizes,
        centiles=args.centiles,
        gsigmas=args.gsigmas,
        m=args.m,
        galpha=args.galpha,
        repeats=args.repeats,
        metric=args.metric,
    )

    print(f"Best-params labels written to: {best_csv}")


if __name__ == "__main__":
    main()
