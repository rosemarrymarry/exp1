from __future__ import annotations

import numpy as np


def _gaussian_blur_anisotropic(image01: np.ndarray, sigma: float, alpha: float = 1.0) -> np.ndarray:
    """Anisotropic Gaussian smoothing used by the Bitonic filter weighting.

    The paper denotes this as G_{\sigma,\alpha}. Here we implement a simple
    axis-aligned anisotropic Gaussian: sigma_x = sigma * alpha, sigma_y = sigma.
    If sigma <= 0, the input is returned unchanged.
    """
    img = image01.astype(np.float32)
    if sigma is None or float(sigma) <= 0.0:
        return img

    sigma = float(sigma)
    alpha = float(alpha)
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")

    try:
        import cv2  # type: ignore

        sigma_x = sigma * alpha
        sigma_y = sigma
        out = cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=sigma_x, sigmaY=sigma_y, borderType=cv2.BORDER_REFLECT)
        return out.astype(np.float32)
    except Exception:
        # Fallback: skimage (isotropic only). If cv2 is unavailable, use isotropic smoothing.
        try:
            from skimage.filters import gaussian  # type: ignore

            out = gaussian(img, sigma=sigma, preserve_range=True)
            return out.astype(np.float32)
        except Exception as e:
            raise RuntimeError("Gaussian smoothing requires opencv-python(-headless) or scikit-image") from e


def _rank_index_from_centile(n: int, centile: float) -> int:
    """Convert centile in [0,100] to a rank index in [0,n-1].

    We use a simple "nearest-rank" mapping on the sorted list indices:
        k = round((centile/100) * (n-1))
    which ensures 0 -> min, 50 -> median (for odd n), 100 -> max.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    c = float(centile)
    if not (0.0 <= c <= 100.0):
        raise ValueError("centile must be within [0,100]")
    k = int(np.round((c / 100.0) * (n - 1)))
    return int(np.clip(k, 0, n - 1))


def rank_filter_tiled(
    image01: np.ndarray,
    ksize: int,
    centile: float,
    pad_mode: str = "reflect",
    tile_size: int = 192,
) -> np.ndarray:
    """Rank filter R_{w,c} over a ksize x ksize neighborhood.

    This corresponds to the paper's R_{w,c}(I)(x) = c-th centile of {I(x+y)} for y in w.
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
    k = _rank_index_from_centile(n, centile)

    pad = ksize // 2
    padded = np.pad(img, pad_width=pad, mode=pad_mode)

    out = np.empty((h, w), dtype=np.float32)
    for y0 in range(0, h, tile_size):
        y1 = min(h, y0 + tile_size)
        for x0 in range(0, w, tile_size):
            x1 = min(w, x0 + tile_size)

            patch = padded[y0 : y1 + 2 * pad, x0 : x1 + 2 * pad]
            windows = np.lib.stride_tricks.sliding_window_view(patch, (ksize, ksize))
            flat = windows.reshape(y1 - y0, x1 - x0, n)

            # Use partial selection instead of full sorting.
            kth = np.partition(flat, k, axis=-1)[..., k]
            out[y0:y1, x0:x1] = kth

    return np.clip(out, 0.0, 1.0)


def bitonic_filter_tiled(
    image01: np.ndarray,
    ksize: int = 5,
    centile: float = 5.0,
    m: float = 3.0,
    gauss_sigma: float = 1.0,
    gauss_alpha: float = 1.0,
    pad_mode: str = "reflect",
    tile_size: int = 192,
) -> np.ndarray:
    """Bitonic filter as defined in the paper excerpt.

    Definitions (constant, symmetric mask w):
        R_{w,c}: rank filter returning the c-th centile in a ksize x ksize window
        O_{w,c} = R_{w,100-c}( R_{w,c}(I) )   (opening)
        C_{w,c} = R_{w,c}( R_{w,100-c}(I) )   (closing)

    Weighting:
        eO = | G_{sigma,alpha}( I - O ) |
        eC = | G_{sigma,alpha}( C - I ) |
        b  = (eO^m * (C - eC) + eC^m * (O + eO)) / (eO^m + eC^m)

    Notes:
        - The reverse-rank operator R^{-1} equals R here because we use a constant,
          symmetric square mask.
        - gauss_alpha controls axis-aligned anisotropy; alpha=1 gives isotropic blur.
    """
    if image01.ndim != 2:
        raise ValueError("This Exp1 implementation expects grayscale (H,W)")
    if ksize <= 0 or ksize % 2 == 0:
        raise ValueError("ksize must be a positive odd integer")
    if not (0.0 <= float(centile) <= 100.0):
        raise ValueError("centile must be within [0,100]")
    if float(m) <= 0.0:
        raise ValueError("m must be positive")

    img = image01.astype(np.float32)

    r_c = rank_filter_tiled(img, ksize=ksize, centile=float(centile), pad_mode=pad_mode, tile_size=tile_size)
    r_100c = rank_filter_tiled(img, ksize=ksize, centile=100.0 - float(centile), pad_mode=pad_mode, tile_size=tile_size)

    # Opening and closing (constant symmetric mask => reverse rank == rank)
    opening = rank_filter_tiled(r_c, ksize=ksize, centile=100.0 - float(centile), pad_mode=pad_mode, tile_size=tile_size)
    closing = rank_filter_tiled(r_100c, ksize=ksize, centile=float(centile), pad_mode=pad_mode, tile_size=tile_size)

    eO = np.abs(_gaussian_blur_anisotropic(img - opening, sigma=float(gauss_sigma), alpha=float(gauss_alpha)))
    eC = np.abs(_gaussian_blur_anisotropic(closing - img, sigma=float(gauss_sigma), alpha=float(gauss_alpha)))

    wO = np.power(eO, float(m), dtype=np.float32)
    wC = np.power(eC, float(m), dtype=np.float32)
    denom = wO + wC

    out = (wO * (closing - eC) + wC * (opening + eO)) / (denom + 1e-12)
    return np.clip(out.astype(np.float32), 0.0, 1.0)


def bitonic_filter_blockwise(
    image01: np.ndarray,
    block_size: int,
    ksize: int = 5,
    centile: float = 5.0,
    m: float = 3.0,
    gauss_sigma: float = 1.0,
    gauss_alpha: float = 1.0,
    pad_mode: str = "reflect",
    tile_size: int = 192,
) -> np.ndarray:
    """Apply the Bitonic filter independently on non-overlapping image blocks.

    This is the experiment-oriented variant for studying how block/batch size
    changes denoising quality and boundary artifacts.
    """
    if image01.ndim != 2:
        raise ValueError("This Exp1 implementation expects grayscale (H,W)")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    img = image01.astype(np.float32)
    h, w = img.shape
    out = np.empty((h, w), dtype=np.float32)

    step = int(block_size)
    for y0 in range(0, h, step):
        y1 = min(h, y0 + step)
        for x0 in range(0, w, step):
            x1 = min(w, x0 + step)
            block = img[y0:y1, x0:x1]
            out[y0:y1, x0:x1] = bitonic_filter_tiled(
                block,
                ksize=ksize,
                centile=centile,
                m=m,
                gauss_sigma=gauss_sigma,
                gauss_alpha=gauss_alpha,
                pad_mode=pad_mode,
                tile_size=tile_size,
            )

    return np.clip(out.astype(np.float32), 0.0, 1.0)
