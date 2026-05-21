from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bitonic_filter import bitonic_filter_tiled
from io_utils import ensure_dir, imread_gray_float, imwrite_gray_float
from metrics import psnr, ssim
from noise import add_gaussian_noise_sigma255


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Exp1 sensitivity: grayscale -> noise -> bitonic filter (grid over params) -> metrics -> summary CSV"
        )
    )
    p.add_argument("--images_dir", type=str, default="exp1-images", help="folder containing input images")
    p.add_argument("--out", type=str, default="outputs_sensitivity", help="output folder")
    p.add_argument("--sigma", type=float, default=25.0, help="Gaussian sigma on 0..255")

    p.add_argument("--ksizes", type=int, nargs="+", default=[3, 5, 7], help="window sizes (odd)")
    p.add_argument("--centiles", type=float, nargs="+", default=[1.0, 5.0, 10.0], help="rank centiles c in [0,100]")
    p.add_argument("--ms", type=float, nargs="+", default=[1.5, 3.0, 6.0], help="sharpness parameter m")
    p.add_argument("--gsigmas", type=float, nargs="+", default=[0.0, 1.0, 2.0], help="Gaussian sigma for residual smoothing")
    p.add_argument("--galphas", type=float, nargs="+", default=[1.0], help="Gaussian anisotropy alpha")

    p.add_argument("--seed", type=int, default=123, help="random seed")
    p.add_argument("--tile", type=int, default=192, help="tile size for bitonic filter")
    p.add_argument(
        "--only",
        type=str,
        default="",
        help="only process one image filename (e.g. lena.png or lenna.jpeg)",
    )
    p.add_argument(
        "--save_images",
        action="store_true",
        help="also save noisy/denoised images for every parameter combo (can be large)",
    )

    return p.parse_args()


def iter_images(images_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)


def main() -> None:
    args = parse_args()

    images_dir = Path(args.images_dir)
    out_root = ensure_dir(args.out)
    gray_dir = ensure_dir(out_root / "grayscale")

    rng = np.random.default_rng(int(args.seed))

    image_paths = iter_images(images_dir)
    if not image_paths:
        raise FileNotFoundError(
            f"No images found under {images_dir}. Expected: .png/.jpg/.jpeg/.bmp/.tif/.tiff"
        )

    if args.only:
        only_lower = args.only.lower()
        image_paths = [p for p in image_paths if p.name.lower() == only_lower]
        if not image_paths:
            raise FileNotFoundError(f"No image named '{args.only}' under {images_dir}")

    ksizes = [int(k) for k in args.ksizes]
    centiles = [float(c) for c in args.centiles]
    ms = [float(m) for m in args.ms]
    gsigmas = [float(s) for s in args.gsigmas]
    galphas = [float(a) for a in args.galphas]

    total = len(image_paths) * len(ksizes) * len(centiles) * len(ms) * len(gsigmas) * len(galphas)
    done = 0

    rows: list[dict] = []

    for img_path in image_paths:
        name = img_path.stem
        original = imread_gray_float(img_path)
        imwrite_gray_float(gray_dir / f"{name}_gray.png", original)

        noisy = add_gaussian_noise_sigma255(original, sigma_255=float(args.sigma), rng=rng)

        # Save a single noisy image per input for reference.
        if args.save_images:
            img_out_dir = ensure_dir(out_root / name / f"sigma{int(args.sigma)}")
            imwrite_gray_float(img_out_dir / "noisy.png", noisy)

        for ksize in ksizes:
            for centile in centiles:
                for m in ms:
                    for gsigma in gsigmas:
                        for galpha in galphas:
                            denoised = bitonic_filter_tiled(
                                noisy,
                                ksize=ksize,
                                centile=centile,
                                m=m,
                                gauss_sigma=gsigma,
                                gauss_alpha=galpha,
                                tile_size=int(args.tile),
                            )

                            row = {
                                "image": name,
                                "sigma_255": float(args.sigma),
                                "ksize": int(ksize),
                                "centile": float(centile),
                                "m": float(m),
                                "gauss_sigma": float(gsigma),
                                "gauss_alpha": float(galpha),
                                "psnr_noisy": psnr(original, noisy),
                                "ssim_noisy": ssim(original, noisy),
                                "psnr_denoised": psnr(original, denoised),
                                "ssim_denoised": ssim(original, denoised),
                            }
                            rows.append(row)

                            if args.save_images:
                                out_dir = ensure_dir(
                                    out_root
                                    / name
                                    / f"sigma{int(args.sigma)}"
                                    / f"ksize{ksize}_c{centile:g}_m{m:g}_gs{gsigma:g}_ga{galpha:g}"
                                )
                                imwrite_gray_float(out_dir / "denoised.png", denoised)

                            done += 1
                            if done % 20 == 0 or done == total:
                                print(
                                    f"[{done}/{total}] {name} k={ksize} c={centile:g} m={m:g} gs={gsigma:g} ga={galpha:g} -> PSNR={row['psnr_denoised']:.3f} SSIM={row['ssim_denoised']:.4f}"
                                )

    df = pd.DataFrame(rows)
    df.to_csv(out_root / "summary.csv", index=False, encoding="utf-8-sig")

    # Also output the best config per image (by PSNR) for quick inspection.
    best_rows = []
    for image, g in df.groupby("image", sort=False):
        best_rows.append(g.sort_values("psnr_denoised", ascending=False).iloc[0].to_dict())
    pd.DataFrame(best_rows).to_csv(out_root / "best_by_image.csv", index=False, encoding="utf-8-sig")

    print(f"Done. Wrote: {out_root / 'summary.csv'}")
    print(f"Done. Wrote: {out_root / 'best_by_image.csv'}")


if __name__ == "__main__":
    main()
