from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bitonic_filter import bitonic_filter_tiled
from io_utils import ensure_dir, imread_gray_float, imwrite_gray_float
from metrics import psnr, ssim
from noise import add_gaussian_noise_sigma255


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp1 batch: grayscale -> noise -> bitonic filter -> metrics -> save")
    p.add_argument("--images_dir", type=str, default="exp1-images", help="folder containing input images")
    p.add_argument("--out", type=str, default="outputs", help="output folder")
    p.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="single Gaussian sigma on 0..255 (overrides --sigmas if provided)",
    )
    p.add_argument("--sigmas", type=float, nargs="+", default=[15, 25, 50], help="Gaussian sigma values on 0..255")
    p.add_argument("--ksizes", type=int, nargs="+", default=[3, 5, 7], help="window sizes (odd)")
    p.add_argument("--centile", type=float, default=5.0, help="rank centile c in [0,100] for Bitonic filter")
    p.add_argument("--m", type=float, default=3.0, help="sharpness parameter m (paper uses m=3)")
    p.add_argument("--gsigma", type=float, default=1.0, help="Gaussian sigma for residual smoothing")
    p.add_argument("--galpha", type=float, default=1.0, help="Gaussian anisotropy alpha (1.0 = isotropic)")
    p.add_argument("--seed", type=int, default=123, help="random seed")
    p.add_argument("--tile", type=int, default=192, help="tile size for bitonic filter (smaller uses less memory)")
    p.add_argument("--only", type=str, default="", help="only process one image filename (e.g. lena.png or lenna.jpeg)")
    return p.parse_args()


def iter_images(images_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)


def residual_to_vis(residual: np.ndarray) -> np.ndarray:
    """Map residual (could be negative) to a viewable [0,1] image."""
    vis = 0.5 + 0.5 * residual
    return np.clip(vis.astype(np.float32), 0.0, 1.0)


def main() -> None:
    args = parse_args()

    images_dir = Path(args.images_dir)
    out_root = ensure_dir(args.out)

    rng = np.random.default_rng(int(args.seed))

    gray_dir = ensure_dir(out_root / "grayscale")

    rows: list[dict] = []

    image_paths = iter_images(images_dir)
    if not image_paths:
        raise FileNotFoundError(
            f"No images found under {images_dir}. "
            "Expected files with extensions: .png/.jpg/.jpeg/.bmp/.tif/.tiff. "
            "If you are on Kaggle, make sure your dataset is added and pass --images_dir to that folder."
        )
    if args.only:
        only_lower = args.only.lower()
        image_paths = [p for p in image_paths if p.name.lower() == only_lower]
        if not image_paths:
            raise FileNotFoundError(f"No image named '{args.only}' under {images_dir}")

    sigmas = [float(args.sigma)] if args.sigma is not None else [float(s) for s in args.sigmas]
    total = len(image_paths) * len(sigmas) * len(args.ksizes)
    done = 0

    for img_path in image_paths:
        name = img_path.stem
        original = imread_gray_float(img_path)
        imwrite_gray_float(gray_dir / f"{name}_gray.png", original)

        for sigma in sigmas:
            noisy = add_gaussian_noise_sigma255(original, sigma_255=float(sigma), rng=rng)

            for ksize in args.ksizes:
                denoised = bitonic_filter_tiled(
                    noisy,
                    ksize=int(ksize),
                    centile=float(args.centile),
                    m=float(args.m),
                    gauss_sigma=float(args.gsigma),
                    gauss_alpha=float(args.galpha),
                    tile_size=int(args.tile),
                )
                residual = noisy - denoised

                m = {
                    "image": name,
                    "sigma_255": float(sigma),
                    "ksize": int(ksize),
                    "centile": float(args.centile),
                    "m": float(args.m),
                    "gauss_sigma": float(args.gsigma),
                    "gauss_alpha": float(args.galpha),
                    "psnr_noisy": psnr(original, noisy),
                    "ssim_noisy": ssim(original, noisy),
                    "psnr_denoised": psnr(original, denoised),
                    "ssim_denoised": ssim(original, denoised),
                }
                rows.append(m)

                out_dir = ensure_dir(out_root / name / f"sigma{int(sigma)}" / f"ksize{int(ksize)}")
                imwrite_gray_float(out_dir / "noisy.png", noisy)
                imwrite_gray_float(out_dir / "denoised.png", denoised)
                imwrite_gray_float(out_dir / "residual.png", residual_to_vis(residual))

                (out_dir / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")

                done += 1
                print(f"[{done}/{total}] {name} sigma={int(sigma)} ksize={int(ksize)} -> PSNR={m['psnr_denoised']:.3f} SSIM={m['ssim_denoised']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_root / "summary.csv", index=False, encoding="utf-8-sig")

    print(f"Done. Wrote: {out_root / 'summary.csv'}")


if __name__ == "__main__":
    main()
