from __future__ import annotations

import numpy as np


def add_gaussian_noise_sigma255(
    image01: np.ndarray,
    sigma_255: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add zero-mean Gaussian noise; sigma is defined on 0..255 intensity scale.

    Args:
        image01: grayscale float32 image in [0,1]
        sigma_255: e.g. 15, 25, 50

    Returns:
        noisy image in [0,1] (float32)
    """
    if rng is None:
        rng = np.random.default_rng()

    sigma01 = float(sigma_255) / 255.0
    noise = rng.normal(loc=0.0, scale=sigma01, size=image01.shape).astype(np.float32)
    out = image01.astype(np.float32) + noise
    return np.clip(out, 0.0, 1.0)
