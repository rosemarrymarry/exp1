# Exp1 — Bitonic Filter 参数实验

数据集：`exp1-images/`（4 张经典图片）

实验流程：
1. 统一格式：读入后全部转为灰度图（float32, 范围 [0,1]）
2. 添加高斯噪声：σ=15、25、50（按 0..255 灰度尺度定义）
3. Bitonic Filter：按论文定义进行 rank-opening/closing，并用平滑残差加权融合（参数包含 `centile`、`m`、高斯平滑 `gsigma/galpha`）
4. 评价指标：PSNR、SSIM（参考原始灰度图）
5. 保存图像：noisy / denoise / residual（residual = noisy - denoised，可视化为 0.5 + residual/2）
6. 参数实验：对多个 `ksize`（默认 3,5,7）× 多个 σ 组合批量跑并汇总 CSV

## 安装

```bash
pip install -r requirements.txt
```

## 运行（批量）

```bash
python run_exp1_batch.py --images_dir exp1-images --out outputs
```

可选：修改窗口大小列表

```bash
python run_exp1_batch.py --ksizes 3 5 7 9
```

## 参数敏感性测试：固定噪声强度，只改变窗口大小

例如固定 σ=25，只扫描不同 `ksize`：

```bash
python run_exp1_batch.py --sigma 25 --ksizes 3 5 7 9 --centile 5 --m 3 --gsigma 1.0 --galpha 1.0
```

（等价写法：`--sigmas 25` 只给一个值也可以。）

## 参数敏感性实验（多参数网格）

如果你要证明“Bitonic 滤波的不同参数会显著影响去噪效果”，建议用网格扫描一次性生成一个汇总表，再用分析脚本输出主效应与最优/最差差值。

### 1) 运行网格扫描

固定噪声强度（例如 σ=25），对 `ksize/centile/m/gauss_sigma/gauss_alpha` 做组合扫描：

```bash
python run_exp1_sensitivity.py --sigma 25 \
  --ksizes 3 5 7 \
  --centiles 1 5 10 \
  --ms 1.5 3 6 \
  --gsigmas 0 1 2 \
  --galphas 1.0 2.0 \
  --out outputs_sensitivity
```

默认只输出指标汇总（更省空间）。如需保存每组参数的去噪图，额外加 `--save_images`（注意会产生大量文件）。

### 2) 分析结果（给出“参数影响”的证据）

```bash
python analyze_sensitivity.py --csv outputs_sensitivity/summary.csv --out outputs_sensitivity --score psnr_denoised
```

输出：

- `outputs_sensitivity/main_effects.csv`：每个参数各取值的均值/方差（主效应）
- `outputs_sensitivity/proof_param_impacts_psnr_denoised.csv`：每个参数“不同取值导致的均值范围”与最优/最差取值
- `outputs_sensitivity/delta_best_worst_by_image_psnr_denoised.csv`：每张图在本次网格中“最优-最差”的差值

## 输出

- `outputs/grayscale/<name>_gray.png`
- `outputs/<name>/sigmaXX/ksizeK/`
  - `noisy.png`
  - `denoised.png`
  - `residual.png`
  - `metrics.json`
- `outputs/summary.csv`（所有图片×参数组合的 PSNR/SSIM 汇总）

## 放到 GitHub（推荐做法）

建议把代码放 GitHub，把实验输出（`outputs/`）留在本地或 Kaggle，不要提交到仓库。

> 注意：`exp1-images/` 里的“经典图片”可能涉及版权/许可问题。若仓库要公开，建议不要把图片原文件提交到 GitHub；改为在本地或 Kaggle 以数据集方式提供你有权使用的图片，然后用 `--images_dir` 指向它。

### 1) 本地初始化并提交

在本项目根目录（有 `run_exp1_batch.py` 那层）打开终端：

```bash
git init
git add .
git commit -m "exp1: bitonic filter batch"
```

### 2) 关联远端并推送

去 GitHub 新建一个空仓库（不要勾选添加 README/.gitignore，因为本地已有），然后执行：

```bash
git branch -M main
git remote add origin https://github.com/<YOUR_NAME>/<REPO_NAME>.git
git push -u origin main
```

如果你用 SSH，则把 remote 换成 `git@github.com:<YOUR_NAME>/<REPO_NAME>.git`。

## 在 Kaggle 更快出结果

最稳妥的方式是：Kaggle Notebook 里 `git clone` 你的 GitHub 仓库（代码），然后把图片作为 Kaggle Dataset 挂载（数据），最后运行脚本。

### A. 准备 Kaggle Dataset（图片）

- 在 Kaggle 创建 Dataset，把你的输入图片上传进去。
- 图片可以是任意名字/格式（png/jpg/...），脚本会自动遍历。

### B. Kaggle Notebook 运行

1) 新建 Kaggle Notebook

2) 右侧设置：
- 开启 GPU 与否对这个实验未必有明显收益（主要是 CPU 运算）；可以先用默认 CPU。
- 如果需要从 GitHub `clone`，通常需要开启 Internet（Kaggle 的开关在 Notebook 设置里）。

3) 在 Notebook 里执行（把占位符换成你的实际仓库与数据集路径）：

```bash
!git clone https://github.com/<YOUR_NAME>/<REPO_NAME>.git
%cd <REPO_NAME>
!pip install -r requirements.txt

# 查看你的 dataset 挂载路径（一般在 /kaggle/input/ 下）
!ls /kaggle/input

# 假设你的 dataset 目录是 /kaggle/input/exp1-images
!python run_exp1_batch.py --images_dir /kaggle/input/exp1-images --out outputs
!ls outputs
```

4) 结果导出

- `outputs/summary.csv` 可直接在 Kaggle 页面下载
- 单张图片的各参数输出也都在 `outputs/<name>/.../` 下

## Exp2 — 参数敏感性实验

目的：系统评估 Bitonic 滤波器中各个超参数对去噪性能的影响（主效应、最优/最差差值、每张图的敏感性）。

运行示例：在固定噪声强度下对 `ksize/centile/m/gauss_sigma/gauss_alpha` 做网格扫描并输出汇总：

```bash
python run_exp1_sensitivity.py --images_dir exp1-images --out outputs_sensitivity --sigma 25 \
  --ksizes 3 5 7 --centiles 1 5 10 --ms 1.5 3 6 --gsigmas 0 1 2 --galphas 1.0 2.0
```

后处理（分析与可视化）：

```bash
python analyze_sensitivity.py --csv outputs_sensitivity/summary.csv --out outputs_sensitivity --score psnr_denoised
python plot_sensitivity.py --csv outputs_sensitivity/summary.csv --out outputs_sensitivity --metric psnr_denoised
```

主要输出：
- `outputs_sensitivity/summary.csv`：参数网格的逐条结果
- `outputs_sensitivity/main_effects.csv`、`proof_param_impacts_*.csv`：主效应与参数影响证明表
- `outputs_sensitivity/figures/`：每个参数的效果图（PNG）

建议：在网格搜索时如不需要保存每组去噪图，可省略 `--save_images` 以节省空间。

## Exp3 — 图像块大小影响实验

目的：研究将图像切为非重叠块（block-wise）独立去噪时，块大小对 PSNR/SSIM 的影响（衡量边界伪影与上下文缺失的影响）。

运行示例：对多个非重叠块大小运行（示例：16/32/64/128）：

```bash
python run_exp1_blocksize.py --images_dir exp1-images --out outputs_blocksize --sigma 25 \
  --block_sizes 16 32 64 128 --ksize 5 --centile 5 --m 3 --gsigma 1.0 --galpha 1.0
```

主要输出：
- `outputs_blocksize/summary.csv`：每张图、每种 block_size 的 PSNR/SSIM 明细（行数≈N_images × N_block_sizes）
- `outputs_blocksize/blocksize_summary.csv`：按 block_size 聚合的均值/标准差汇总

建议的验证流程：先用一组小样本快速跑通，再在大数据集（如 DIV2K）上跑完整实验；记录每张图的多次噪声重复以提高统计稳健性。

---

以上两个实验均与主流程共用核心实现（`bitonic_filter.py`）。若需要把新的实验脚本、结果文件或分析图加入仓库，请在提交前排除大体积输出文件（`outputs/`）或者把结果另存为独立数据集/Artifact。
