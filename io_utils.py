from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def imread_gray_float(path: str | Path) -> np.ndarray:
    """Read image and convert to grayscale float32 in [0,1]."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    if img.ndim == 2:
        gray = img
    else:
        # Handle possible alpha
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if gray.dtype == np.uint8:
        out = gray.astype(np.float32) / 255.0
    elif gray.dtype == np.uint16:
        out = gray.astype(np.float32) / 65535.0
    else:
        out = gray.astype(np.float32)
        out = np.clip(out, 0.0, 1.0)

    return out


def imwrite_gray_float(path: str | Path, image01: np.ndarray) -> None:
    """Write grayscale float image in [0,1] to PNG (uint8)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    img = np.clip(image01.astype(np.float32), 0.0, 1.0)
    u8 = (img * 255.0 + 0.5).astype(np.uint8)
    cv2.imwrite(str(p), u8)
