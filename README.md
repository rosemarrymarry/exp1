# Exp1 — Bitonic Filter 参数实验

数据集：`exp1-images/`（4 张经典图片）

实验流程：
1. 统一格式：读入后全部转为灰度图（float32, 范围 [0,1]）
2. 添加高斯噪声：σ=15、25、50（按 0..255 灰度尺度定义）
3. Bitonic Filter：输入 noisy image，输出 denoised image（窗口秩统计，默认取中位数）
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
