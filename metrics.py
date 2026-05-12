from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(reference01: np.ndarray, test01: np.ndarray) -> float:
    ref = reference01.astype(np.float32)
    tst = test01.astype(np.float32)
    return float(peak_signal_noise_ratio(ref, tst, data_range=1.0))


def ssim(reference01: np.ndarray, test01: np.ndarray) -> float:
    ref = reference01.astype(np.float32)
    tst = test01.astype(np.float32)
    return float(structural_similarity(ref, tst, data_range=1.0))
