from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bitonic_filter import bitonic_filter_blockwise
from io_utils import ensure_dir, imread_gray_float, imwrite_gray_float
from metrics import psnr, ssim
from noise import add_gaussian_noise_sigma255


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Exp1 block-size experiment: split image into blocks, denoise each block independently"
    )
    p.add_argument("--images_dir", type=str, default="exp1-images", help="folder containing input images")
    p.add_argument("--out", type=str, default="outputs_blocksize", help="output folder")
    p.add_argument("--sigma", type=float, default=25.0, help="Gaussian sigma on 0..255")
    p.add_argument("--block_sizes", type=int, nargs="+", default=[16, 32, 64, 128], help="non-overlapping block sizes")
    p.add_argument("--ksize", type=int, default=5, help="window size (odd) for Bitonic filter")
    p.add_argument("--centile", type=float, default=5.0, help="rank centile c in [0,100]")
    p.add_argument("--m", type=float, default=3.0, help="sharpness parameter m")
    p.add_argument("--gsigma", type=float, default=1.0, help="Gaussian sigma for residual smoothing")
    p.add_argument("--galpha", type=float, default=1.0, help="Gaussian anisotropy alpha")
    p.add_argument("--seed", type=int, default=123, help="random seed")
    p.add_argument("--tile", type=int, default=192, help="internal tile size for the Bitonic filter")
    p.add_argument("--only", type=str, default="", help="only process one image filename (e.g. lena.png)")
    return p.parse_args()


def iter_images(images_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)


def residual_to_vis(residual: np.ndarray) -> np.ndarray:
    vis = 0.5 + 0.5 * residual
    return np.clip(vis.astype(np.float32), 0.0, 1.0)


def main() -> None:
    args = parse_args()

    images_dir = Path(args.images_dir)
    out_root = ensure_dir(args.out)
    gray_dir = ensure_dir(out_root / "grayscale")

    rng = np.random.default_rng(int(args.seed))
    image_paths = iter_images(images_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found under {images_dir}. Expected common image extensions.")

    if args.only:
        only_lower = args.only.lower()
        image_paths = [p for p in image_paths if p.name.lower() == only_lower]
        if not image_paths:
            raise FileNotFoundError(f"No image named '{args.only}' under {images_dir}")

    rows: list[dict] = []
    total = len(image_paths) * len(args.block_sizes)
    done = 0

    for img_path in image_paths:
        name = img_path.stem
        original = imread_gray_float(img_path)
        imwrite_gray_float(gray_dir / f"{name}_gray.png", original)

        noisy = add_gaussian_noise_sigma255(original, sigma_255=float(args.sigma), rng=rng)
        imwrite_gray_float(out_root / name / f"sigma{int(args.sigma)}" / "noisy.png", noisy)

        for block_size in args.block_sizes:
            denoised = bitonic_filter_blockwise(
                noisy,
                block_size=int(block_size),
                ksize=int(args.ksize),
                centile=float(args.centile),
                m=float(args.m),
                gauss_sigma=float(args.gsigma),
                gauss_alpha=float(args.galpha),
                tile_size=int(args.tile),
            )
            residual = noisy - denoised

            row = {
                "image": name,
                "sigma_255": float(args.sigma),
                "block_size": int(block_size),
                "ksize": int(args.ksize),
                "centile": float(args.centile),
                "m": float(args.m),
                "gauss_sigma": float(args.gsigma),
                "gauss_alpha": float(args.galpha),
                "psnr_noisy": psnr(original, noisy),
                "ssim_noisy": ssim(original, noisy),
                "psnr_denoised": psnr(original, denoised),
                "ssim_denoised": ssim(original, denoised),
            }
            rows.append(row)

            out_dir = ensure_dir(out_root / name / f"sigma{int(args.sigma)}" / f"block{int(block_size)}")
            imwrite_gray_float(out_dir / "denoised.png", denoised)
            imwrite_gray_float(out_dir / "residual.png", residual_to_vis(residual))
            (out_dir / "metrics.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")

            done += 1
            print(
                f"[{done}/{total}] {name} block={int(block_size)} -> PSNR={row['psnr_denoised']:.3f} SSIM={row['ssim_denoised']:.4f}"
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_root / "summary.csv", index=False, encoding="utf-8-sig")

    block_summary = (
        df.groupby("block_size")
        .agg(
            mean_psnr_denoised=("psnr_denoised", "mean"),
            std_psnr_denoised=("psnr_denoised", "std"),
            mean_ssim_denoised=("ssim_denoised", "mean"),
            std_ssim_denoised=("ssim_denoised", "std"),
            count=("psnr_denoised", "size"),
        )
        .reset_index()
        .sort_values("block_size")
    )
    block_summary.to_csv(out_root / "blocksize_summary.csv", index=False, encoding="utf-8-sig")

    print(f"Done. Wrote: {out_root / 'summary.csv'}")
    print(f"Done. Wrote: {out_root / 'blocksize_summary.csv'}")


if __name__ == "__main__":
    main()