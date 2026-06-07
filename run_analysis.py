import pandas as pd
from pathlib import Path

csv_path = Path('summary (1).csv')
df = pd.read_csv(csv_path)

out_dir = Path('outputs_analysis_summary1')
out_dir.mkdir(parents=True, exist_ok=True)

params = ['ksize','centile','m','gauss_sigma','gauss_alpha']

# 0) Basic info
info = {
    'rows': [len(df)],
    'images': [df['image'].nunique()],
    'sigma_unique': [sorted(df['sigma_255'].unique())],
}
(pd.DataFrame(info)).to_csv(out_dir/'info.csv', index=False, encoding='utf-8-sig')

# 1) Best/worst per image by PSNR
score='psnr_denoised'
best = df.sort_values(['image', score], ascending=[True, False]).groupby('image').head(1)
worst = df.sort_values(['image', score], ascending=[True, True]).groupby('image').head(1)

delta = best[['image', score]].merge(worst[['image', score]], on='image', suffixes=('_best','_worst'))
delta['delta_best_minus_worst'] = delta[f'{score}_best'] - delta[f'{score}_worst']
delta = delta.sort_values('delta_best_minus_worst', ascending=False)
delta.to_csv(out_dir/'delta_best_worst_by_image_psnr.csv', index=False, encoding='utf-8-sig')

# 2) Main effects (mean across levels)
effects_rows=[]
for p in params:
    g = df.groupby(p).agg(
        mean_psnr=('psnr_denoised','mean'),
        mean_ssim=('ssim_denoised','mean'),
        std_psnr=('psnr_denoised','std'),
        std_ssim=('ssim_denoised','std'),
        count=('psnr_denoised','size'),
    ).reset_index()
    g.insert(0,'param',p)
    g.rename(columns={p:'level'}, inplace=True)
    effects_rows.append(g)

effects = pd.concat(effects_rows, ignore_index=True)
effects.to_csv(out_dir/'main_effects.csv', index=False, encoding='utf-8-sig')

# 3) Proof table: range across levels for each param
proof=[]
for p in params:
    sub = effects[effects['param']==p]
    rng_psnr = float(sub['mean_psnr'].max() - sub['mean_psnr'].min())
    rng_ssim = float(sub['mean_ssim'].max() - sub['mean_ssim'].min())
    best_level_psnr = sub.sort_values('mean_psnr', ascending=False).iloc[0]['level']
    worst_level_psnr = sub.sort_values('mean_psnr', ascending=True).iloc[0]['level']
    proof.append({
        'param': p,
        'range_mean_psnr': rng_psnr,
        'best_level_psnr': best_level_psnr,
        'worst_level_psnr': worst_level_psnr,
        'range_mean_ssim': rng_ssim,
    })

proof_df = pd.DataFrame(proof).sort_values('range_mean_psnr', ascending=False)
proof_df.to_csv(out_dir/'proof_param_impacts_psnr.csv', index=False, encoding='utf-8-sig')

# 4) Overall best config by mean PSNR across images
cfg_cols = params
cfg = df.groupby(cfg_cols).agg(
    mean_psnr=('psnr_denoised','mean'),
    mean_ssim=('ssim_denoised','mean'),
    n=('psnr_denoised','size'),
).reset_index().sort_values('mean_psnr', ascending=False)

cfg.head(30).to_csv(out_dir/'top_configs_by_mean_psnr.csv', index=False, encoding='utf-8-sig')

# Print concise summary
print("CSV:", csv_path)
print("rows=", len(df), "images=", df["image"].nunique(), "sigma_unique=", sorted(df["sigma_255"].unique()))

# overall gain
psnr_gain = (df['psnr_denoised'] - df['psnr_noisy']).mean()
ssim_gain = (df['ssim_denoised'] - df['ssim_noisy']).mean()
print(f"mean gain: PSNR +{psnr_gain:.3f} dB, SSIM +{ssim_gain:.3f}")

print("\nParam sensitivity ranking (by range of mean PSNR across levels):")
print(proof_df.to_string(index=False))

print("\nBest-worst delta per image (PSNR):")
print(delta.to_string(index=False))

print("\nTop 5 configs by mean PSNR:")
print(cfg.head(5).to_string(index=False))

print("\nWrote reports to:", out_dir)
