from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from skimage.restoration import denoise_tv_chambolle


TEXTURE_SCORE_METRICS = [
    "fg_flat_noise_tv",
    "fg_noise_tv",
    "fg_starved_ratio",
    "fg_edge_spread",
    "fg_mf_median",
]

# Defaults are shared between image_quality_metrics.py and texture_clean_scorer.py.
TEXTURE_SCORE_DEFAULT_GOOD_BAD: dict[str, tuple[float, float]] = {
    "fg_flat_noise_tv": (0.0045, 0.0075),
    "fg_noise_tv": (0.0045, 0.0075),
    "fg_starved_ratio": (0.020, 0.050),
    "fg_edge_spread": (0.100, 0.130),
    "fg_mf_median": (0.018, 0.024),
}

TEXTURE_SCORE_DEFAULT_WEIGHTS: dict[str, float] = {
    "fg_flat_noise_tv": 1.0,
    "fg_noise_tv": 1.0,
    "fg_starved_ratio": 2.0,
    "fg_edge_spread": 0.5,
    "fg_mf_median": 0.5,
}


def score_lower_better(value: float, good: float, bad: float) -> float:
    if bad <= good:
        return 0.0
    t = (value - good) / (bad - good)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return float(1.0 - t)


def texture_clean_score_absolute(
    metrics: dict[str, float],
    good_bad: dict[str, tuple[float, float]] | None = None,
    weights: dict[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    """
    Returns (score_0_10, components) using the absolute per-image scoring.
    """
    good_bad = good_bad or TEXTURE_SCORE_DEFAULT_GOOD_BAD
    weights = weights or TEXTURE_SCORE_DEFAULT_WEIGHTS

    comps: dict[str, float] = {}
    total = 0.0
    wsum = 0.0
    for key in TEXTURE_SCORE_METRICS:
        v = float(metrics[key])
        g, b = good_bad[key]
        s = score_lower_better(v, g, b)
        comps[key] = s
        w = float(weights.get(key, 1.0))
        total += s * w
        wsum += w
    avg = (total / wsum) if wsum > 1e-8 else 0.0
    return float(avg * 10.0), comps


def build_rembg_mask(
    img: Image.Image,
    model: str,
    post_process_mask: bool,
    session: Optional[object] = None,
) -> np.ndarray:
    from rembg import new_session
    from rembg.bg import post_process

    if session is None:
        session = new_session(model)
    masks = session.predict(img)
    if not masks:
        return np.zeros((img.height, img.width), dtype=bool)

    merged = None
    for mask in masks:
        arr = np.array(mask)
        if post_process_mask:
            arr = post_process(arr)
        if merged is None:
            merged = arr.astype(np.uint8)
        else:
            merged = np.maximum(merged, arr.astype(np.uint8))
    return merged >= 128


def compute_noise_robust_metrics(
    gray: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    if mask is None or not np.any(mask):
        return {
            "edge_sharpness": 0.0,
            "edge_spread": 0.0,
            "noise_tv": 0.0,
            "hf_clean": 0.0,
        }

    denoised = denoise_tv_chambolle(gray, weight=0.10)

    gx = cv2.Sobel(denoised, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(denoised, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.sqrt(gx * gx + gy * gy)
    gvals = gmag[mask]
    if gvals.size < 64:
        return {
            "edge_sharpness": 0.0,
            "edge_spread": 0.0,
            "noise_tv": 0.0,
            "hf_clean": 0.0,
        }

    thresh = float(np.percentile(gvals, 90))
    edge_mask = (gmag >= thresh) & mask
    if edge_mask.sum() < 16:
        edge_mask = (gmag >= np.percentile(gvals, 80)) & mask

    kernel = np.ones((3, 3), np.uint8)
    dil = cv2.dilate(edge_mask.astype(np.uint8), kernel, iterations=2) > 0
    ring = dil & (~edge_mask) & mask
    if ring.sum() < 16:
        ring = mask & (~edge_mask)

    edge_mean = float(np.mean(gmag[edge_mask])) if edge_mask.sum() else 0.0
    ring_mean = float(np.mean(gmag[ring])) if ring.sum() else 0.0
    edge_sharpness = edge_mean / (ring_mean + 1e-6)
    edge_spread = ring_mean / (edge_mean + 1e-6)

    mean = cv2.blur(gray, (5, 5))
    mean_sq = cv2.blur(gray * gray, (5, 5))
    lvar = np.maximum(0.0, mean_sq - mean * mean)
    s_lvar = lvar[mask]
    v_thresh = float(np.percentile(s_lvar, 25)) if s_lvar.size else 0.0
    flat_mask = (lvar <= v_thresh) & mask
    residual = np.abs(gray - denoised)
    if flat_mask.sum() < 64:
        noise_tv = float(np.std(residual[mask]))
    else:
        noise_tv = float(np.std(residual[flat_mask]))

    lap = cv2.Laplacian(denoised, cv2.CV_32F)
    hf_clean = float(np.mean(np.abs(lap[mask])))

    return {
        "edge_sharpness": float(edge_sharpness),
        "edge_spread": float(edge_spread),
        "noise_tv": float(noise_tv),
        "hf_clean": float(hf_clean),
    }


def compute_texture_metrics(
    gray: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    if mask is None or not np.any(mask):
        return {
            "starved_ratio": 0.0,
            "flat_noise_tv": 0.0,
            "lvar_p50": 0.0,
            "lvar_p75": 0.0,
            "lvar_p90": 0.0,
            "detail_ratio": 0.0,
            "mf_median": 0.0,
        }

    mean = cv2.blur(gray, (5, 5))
    mean_sq = cv2.blur(gray * gray, (5, 5))
    lvar = np.maximum(0.0, mean_sq - mean * mean)
    s_lvar = lvar[mask]
    if s_lvar.size < 64:
        return {
            "starved_ratio": 0.0,
            "flat_noise_tv": 0.0,
            "lvar_p50": 0.0,
            "lvar_p75": 0.0,
            "lvar_p90": 0.0,
            "detail_ratio": 0.0,
            "mf_median": 0.0,
        }

    v_baseline = float(np.percentile(s_lvar, 20))
    starved_ratio = float(np.mean((lvar < (v_baseline * 0.4)) & mask))

    denoised = denoise_tv_chambolle(gray, weight=0.10)
    residual = np.abs(gray - denoised)
    flat_mask = (lvar < np.percentile(s_lvar, 25)) & mask
    if flat_mask.sum() < 64:
        flat_noise_tv = float(np.std(residual[mask]))
    else:
        flat_noise_tv = float(np.std(residual[flat_mask]))

    lvar_p50 = float(np.percentile(s_lvar, 50))
    lvar_p75 = float(np.percentile(s_lvar, 75))
    lvar_p90 = float(np.percentile(s_lvar, 90))

    hf = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    gauss1 = cv2.GaussianBlur(gray, (0, 0), 1.0)
    gauss5 = cv2.GaussianBlur(gray, (0, 0), 5.0)
    mf = np.abs(gauss1 - gauss5)

    s_hf = hf[mask]
    s_mf = mf[mask]
    avg_hf = float(np.mean(s_hf))
    avg_mf = float(np.mean(s_mf))
    detail_ratio = avg_hf / (avg_mf + 1e-6)
    mf_median = float(np.median(s_mf))

    return {
        "starved_ratio": float(starved_ratio),
        "flat_noise_tv": float(flat_noise_tv),
        "lvar_p50": float(lvar_p50),
        "lvar_p75": float(lvar_p75),
        "lvar_p90": float(lvar_p90),
        "detail_ratio": float(detail_ratio),
        "mf_median": float(mf_median),
    }

