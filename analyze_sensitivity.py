from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze Exp1 sensitivity CSV and summarize parameter effects")
    p.add_argument("--csv", type=str, default="outputs_sensitivity/summary.csv", help="path to summary.csv")
    p.add_argument(
        "--out",
        type=str,
        default="outputs_sensitivity",
        help="output folder for derived reports (default: alongside csv)",
    )
    p.add_argument(
        "--score",
        choices=["psnr_denoised", "ssim_denoised"],
        default="psnr_denoised",
        help="metric used to rank best/worst configs",
    )
    return p.parse_args()


PARAMS = ["ksize", "centile", "m", "gauss_sigma", "gauss_alpha"]


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    # 1) Best/worst per image (using chosen score)
    best = df.sort_values(["image", args.score], ascending=[True, False]).groupby("image").head(1)
    worst = df.sort_values(["image", args.score], ascending=[True, True]).groupby("image").head(1)

    delta = best[["image", args.score]].merge(
        worst[["image", args.score]], on="image", suffixes=("_best", "_worst")
    )
    delta["delta_best_minus_worst"] = delta[f"{args.score}_best"] - delta[f"{args.score}_worst"]
    delta.to_csv(out_dir / f"delta_best_worst_by_image_{args.score}.csv", index=False, encoding="utf-8-sig")

    best.to_csv(out_dir / f"best_by_image_{args.score}.csv", index=False, encoding="utf-8-sig")
    worst.to_csv(out_dir / f"worst_by_image_{args.score}.csv", index=False, encoding="utf-8-sig")

    # 2) Main effect tables: average metric for each parameter level
    effects_rows = []
    for param in PARAMS:
        g = (
            df.groupby(param)
            .agg(
                mean_psnr=("psnr_denoised", "mean"),
                mean_ssim=("ssim_denoised", "mean"),
                std_psnr=("psnr_denoised", "std"),
                std_ssim=("ssim_denoised", "std"),
                count=("psnr_denoised", "size"),
            )
            .reset_index()
        )
        g.insert(0, "param", param)
        g.rename(columns={param: "level"}, inplace=True)
        effects_rows.append(g)

    effects = pd.concat(effects_rows, ignore_index=True)
    effects.to_csv(out_dir / "main_effects.csv", index=False, encoding="utf-8-sig")

    # 3) A compact "proof" summary: range of mean score across levels
    proof_rows = []
    for param in PARAMS:
        sub = effects[effects["param"] == param].copy()
        metric = "mean_psnr" if args.score == "psnr_denoised" else "mean_ssim"
        rng = float(sub[metric].max() - sub[metric].min())
        best_level = sub.sort_values(metric, ascending=False).iloc[0]["level"]
        worst_level = sub.sort_values(metric, ascending=True).iloc[0]["level"]
        proof_rows.append(
            {
                "param": param,
                "metric": metric,
                "range_across_levels": rng,
                "best_level": best_level,
                "worst_level": worst_level,
            }
        )

    proof = pd.DataFrame(proof_rows).sort_values("range_across_levels", ascending=False)
    proof.to_csv(out_dir / f"proof_param_impacts_{args.score}.csv", index=False, encoding="utf-8-sig")

    print("Wrote:")
    print(f"- {out_dir / 'main_effects.csv'}")
    print(f"- {out_dir / f'delta_best_worst_by_image_{args.score}.csv'}")
    print(f"- {out_dir / f'proof_param_impacts_{args.score}.csv'}")


if __name__ == "__main__":
    main()
