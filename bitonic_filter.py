from __future__ import annotations

import numpy as np


def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def _bitonic_sort_last_axis(values: np.ndarray) -> np.ndarray:
    """Bitonic sort network over the last axis; vectorized via numpy.

    Note: last-axis length must be power-of-two.
    """
    out = values.copy()
    n = out.shape[-1]
    if n & (n - 1) != 0:
        raise ValueError("Bitonic sort requires power-of-two length")

    idx = np.arange(n, dtype=np.int32)
    lead_shape = (1,) * (out.ndim - 1)

    k = 2
    while k <= n:
        j = k // 2
        while j > 0:
            partner = idx ^ j
            mask = partner > idx

            i1 = idx[mask]
            i2 = partner[mask]

            asc = ((idx & k) == 0)[mask]
            asc = asc.reshape(lead_shape + (asc.size,))

            v1 = out[..., i1]
            v2 = out[..., i2]
            lo = np.minimum(v1, v2)
            hi = np.maximum(v1, v2)

            out[..., i1] = np.where(asc, lo, hi)
            out[..., i2] = np.where(asc, hi, lo)

            j //= 2
        k *= 2

    return out


def bitonic_median_filter(image01: np.ndarray, ksize: int = 3, pad_mode: str = "reflect") -> np.ndarray:
    """Median-like Bitonic Filter for grayscale images.

    Args:
        image01: grayscale float32 image in [0,1], shape (H,W)
        ksize: odd window size

    Returns:
        denoised image in [0,1] (float32)
    """
    if image01.ndim != 2:
        raise ValueError("This Exp1 implementation expects grayscale (H,W)")
    if ksize <= 0 or ksize % 2 == 0:
        raise ValueError("ksize must be a positive odd integer")

    return bitonic_median_filter_tiled(image01=image01, ksize=ksize, pad_mode=pad_mode, tile_size=192)


def bitonic_median_filter_tiled(
    image01: np.ndarray,
    ksize: int = 3,
    pad_mode: str = "reflect",
    tile_size: int = 192,
) -> np.ndarray:
    """Bitonic median filter with tiling to reduce peak memory.

    This computes the same result as the full-image window expansion, but processes
    the output in tiles so `sliding_window_view` never materializes HxW windows at once.
    """
    if image01.ndim != 2:
        raise ValueError("This Exp1 implementation expects grayscale (H,W)")
    if ksize <= 0 or ksize % 2 == 0:
        raise ValueError("ksize must be a positive odd integer")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")

    img = image01.astype(np.float32)
    h, w = img.shape
    n = ksize * ksize
    rank_idx = (n - 1) // 2

    pad = ksize // 2
    padded = np.pad(img, pad_width=pad, mode=pad_mode)

    n_pad = _next_pow2(n)

    out = np.empty((h, w), dtype=np.float32)
    for y0 in range(0, h, tile_size):
        y1 = min(h, y0 + tile_size)
        for x0 in range(0, w, tile_size):
            x1 = min(w, x0 + tile_size)

            # For output region [y0:y1, x0:x1], take padded patch large enough
            # so every kxk neighborhood is available.
            patch = padded[y0 : y1 + 2 * pad, x0 : x1 + 2 * pad]
            windows = np.lib.stride_tricks.sliding_window_view(patch, (ksize, ksize))
            # windows shape: (tile_h, tile_w, ksize, ksize)
            flat = windows.reshape(y1 - y0, x1 - x0, n)

            if n_pad != n:
                flat = np.concatenate(
                    [flat, np.full((y1 - y0, x1 - x0, n_pad - n), np.inf, np.float32)],
                    axis=-1,
                )

            sorted_vals = _bitonic_sort_last_axis(flat)
            out[y0:y1, x0:x1] = sorted_vals[..., rank_idx]

    return np.clip(out, 0.0, 1.0)
