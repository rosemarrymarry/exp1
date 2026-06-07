from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PARAMS = ["ksize", "centile", "m", "gauss_sigma", "gauss_alpha"]
METRICS = [
    "psnr_noisy",
    "ssim_noisy",
    "psnr_denoised",
    "ssim_denoised",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot parameter impact figures from Exp1 sensitivity summary.csv")
    p.add_argument("--csv", type=str, default="outputs_sensitivity/summary.csv", help="path to summary.csv")
    p.add_argument("--out", type=str, default="outputs_sensitivity", help="output folder for figures")
    p.add_argument(
        "--metric",
        choices=["psnr_denoised", "ssim_denoised"],
        default="psnr_denoised",
        help="metric to visualize",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="output figure DPI",
    )
    return p.parse_args()


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required for plotting. Install it via: pip install matplotlib"
        ) from e


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt = _require_matplotlib()

    df = pd.read_csv(csv_path)

    missing = [c for c in (PARAMS + [args.metric, "image"]) if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Ensure numeric columns are numeric
    for c in PARAMS + METRICS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    metric = args.metric
    metric_label = "PSNR (dB)" if metric.startswith("psnr") else "SSIM"

    # 1) Main effect range bar plot (mean across levels)
    ranges = []
    for p in PARAMS:
        g = df.groupby(p)[metric].mean().reset_index(name="mean")
        rng = float(g["mean"].max() - g["mean"].min())
        best_level = float(g.sort_values("mean", ascending=False).iloc[0][p])
        worst_level = float(g.sort_values("mean", ascending=True).iloc[0][p])
        ranges.append({"param": p, "range": rng, "best": best_level, "worst": worst_level})
    ranges_df = pd.DataFrame(ranges).sort_values("range", ascending=False)

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.bar(ranges_df["param"], ranges_df["range"], color="#4C78A8")
    ax.set_title(f"Parameter sensitivity (range of mean {metric})")
    ax.set_ylabel(f"Range of mean {metric_label}")
    ax.set_xlabel("Parameter")
    ax.grid(True, axis="y", alpha=0.3)
    for i, row in enumerate(ranges_df.itertuples(index=False)):
        ax.text(i, row.range, f"{row.range:.3g}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig_path = fig_dir / f"main_effect_ranges_{metric}.png"
    fig.savefig(fig_path, dpi=int(args.dpi))
    plt.close(fig)

    # 2) Per-parameter main effect curves with error bars
    for p in PARAMS:
        g = (
            df.groupby(p)[metric]
            .agg(mean="mean", std="std", n="size")
            .reset_index()
            .sort_values(p)
        )
        # standard error; if std is NaN (n=1), make it 0
        se = (g["std"].fillna(0.0) / np.sqrt(g["n"].clip(lower=1))).to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        ax.errorbar(g[p], g["mean"], yerr=se, fmt="-o", color="#F58518", ecolor="#F58518", capsize=3)
        ax.set_title(f"Main effect: {p} → mean {metric}")
        ax.set_xlabel(p)
        ax.set_ylabel(metric_label)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / f"effect_{p}_{metric}.png", dpi=int(args.dpi))
        plt.close(fig)

    # 3) Best-worst delta per image (shows how much parameters matter per image)
    best = df.sort_values(["image", metric], ascending=[True, False]).groupby("image").head(1)
    worst = df.sort_values(["image", metric], ascending=[True, True]).groupby("image").head(1)
    delta = best[["image", metric]].merge(worst[["image", metric]], on="image", suffixes=("_best", "_worst"))
    delta["delta"] = delta[f"{metric}_best"] - delta[f"{metric}_worst"]
    delta = delta.sort_values("delta", ascending=False)

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.bar(delta["image"], delta["delta"], color="#54A24B")
    ax.set_title(f"Best–worst gap by image ({metric})")
    ax.set_ylabel(f"Δ {metric_label}")
    ax.set_xlabel("Image")
    ax.grid(True, axis="y", alpha=0.3)
    for i, row in enumerate(delta.itertuples(index=False)):
        ax.text(i, row.delta, f"{row.delta:.3g}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / f"best_worst_delta_by_image_{metric}.png", dpi=int(args.dpi))
    plt.close(fig)

    # Save the numeric tables too (helps reporting)
    ranges_df.to_csv(out_dir / f"main_effect_ranges_{metric}.csv", index=False, encoding="utf-8-sig")
    delta.to_csv(out_dir / f"best_worst_delta_by_image_{metric}.csv", index=False, encoding="utf-8-sig")

    print("Wrote figures to:")
    print(f"- {fig_dir}")
    print("Wrote tables to:")
    print(f"- {out_dir / f'main_effect_ranges_{metric}.csv'}")
    print(f"- {out_dir / f'best_worst_delta_by_image_{metric}.csv'}")


if __name__ == "__main__":
    main()
