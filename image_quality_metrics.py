import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
from PIL import Image
from skimage.restoration import denoise_tv_chambolle


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class MetricsRow:
    path: Path
    width: int
    height: int
    mean_luma: float
    std_luma: float
    entropy: float
    lap_var: float
    tenengrad: float
    edge_density: float
    noise_mad: float
    snr_db: float
    colorfulness: float
    pct_black: float
    pct_white: float
    fg_pixel_count: Optional[int] = None
    fg_pixel_frac: Optional[float] = None
    fg_mean_luma: Optional[float] = None
    fg_std_luma: Optional[float] = None
    fg_entropy: Optional[float] = None
    fg_lap_var: Optional[float] = None
    fg_tenengrad: Optional[float] = None
    fg_edge_density: Optional[float] = None
    fg_noise_mad: Optional[float] = None
    fg_snr_db: Optional[float] = None
    fg_colorfulness: Optional[float] = None
    fg_pct_black: Optional[float] = None
    fg_pct_white: Optional[float] = None
    fg_subject_count: Optional[int] = None
    fg_edge_sharpness: Optional[float] = None
    fg_edge_spread: Optional[float] = None
    fg_noise_tv: Optional[float] = None
    fg_hf_clean: Optional[float] = None
    fg_starved_ratio: Optional[float] = None
    fg_flat_noise_tv: Optional[float] = None
    fg_lvar_p50: Optional[float] = None
    fg_lvar_p75: Optional[float] = None
    fg_lvar_p90: Optional[float] = None
    fg_detail_ratio: Optional[float] = None
    fg_mf_median: Optional[float] = None
    fg_texture_clean_score: Optional[float] = None
    fg_texture_clean_score_0_10: Optional[float] = None
    fg_starved_ratio: Optional[float] = None
    fg_flat_noise_tv: Optional[float] = None
    fg_lvar_p50: Optional[float] = None
    fg_lvar_p75: Optional[float] = None
    fg_lvar_p90: Optional[float] = None
    fg_detail_ratio: Optional[float] = None
    fg_mf_median: Optional[float] = None


def iter_images(path: Path) -> Iterable[Path]:
    if path.is_file():
        if path.suffix.lower() in IMAGE_EXTS:
            yield path
        return
    for p in sorted(path.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def default_image_paths() -> list[Path]:
    cwd = Path.cwd()
    paths: list[Path] = []
    for p in sorted(cwd.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            paths.append(p)
    for sub in ("test", "tests"):
        subdir = cwd / sub
        if subdir.is_dir():
            paths.extend(iter_images(subdir))
    return paths


def image_entropy(gray_u8: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    if mask is not None:
        values = gray_u8[mask]
        if values.size == 0:
            return 0.0
        hist = np.bincount(values, minlength=256).astype(np.float64)
    else:
        hist = np.bincount(gray_u8.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.0
    p = hist / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def colorfulness_score(
    img_rgb: np.ndarray, mask: Optional[np.ndarray] = None
) -> float:
    img = img_rgb.astype(np.float32)
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    if mask is not None:
        if mask.sum() == 0:
            return 0.0
        r = r[mask]
        g = g[mask]
        b = b[mask]
    rg = r - g
    yb = 0.5 * (r + g) - b
    rg_mean = float(np.mean(rg))
    yb_mean = float(np.mean(yb))
    rg_std = float(np.std(rg))
    yb_std = float(np.std(yb))
    return float(np.sqrt(rg_std * rg_std + yb_std * yb_std) + 0.3 * np.sqrt(rg_mean * rg_mean + yb_mean * yb_mean))


def estimate_noise_mad(gray: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    # Use low-variance (flat) regions to estimate noise via MAD.
    mean = cv2.blur(gray, (5, 5))
    mean_sq = cv2.blur(gray * gray, (5, 5))
    local_var = np.maximum(0.0, mean_sq - mean * mean)
    if mask is not None and mask.any():
        thresh = np.percentile(local_var[mask], 20)
    else:
        thresh = np.percentile(local_var, 20)
    flat_mask = local_var <= thresh
    if mask is not None:
        flat_mask = np.logical_and(flat_mask, mask)

    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    residual = gray - blur
    flat_res = residual[flat_mask]
    if flat_res.size < 64:
        flat_res = residual[mask] if mask is not None and mask.any() else residual.reshape(-1)
    med = np.median(flat_res)
    mad = np.median(np.abs(flat_res - med))
    return float(1.4826 * mad)


def compute_metrics_from_arrays(
    img_rgb: np.ndarray,
    gray: np.ndarray,
    gray_u8: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Optional[dict[str, float]]:
    if mask is not None:
        mask = mask.astype(bool)
        if mask.sum() < 64:
            return None
        gray_vals = gray[mask]
        gray_u8_vals = gray_u8[mask]
    else:
        gray_vals = gray.reshape(-1)
        gray_u8_vals = gray_u8.reshape(-1)

    mean_luma = float(np.mean(gray_vals)) if gray_vals.size else 0.0
    std_luma = float(np.std(gray_vals)) if gray_vals.size else 0.0
    entropy = image_entropy(gray_u8, mask)

    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap_vals = lap[mask] if mask is not None else lap.reshape(-1)
    lap_var = float(np.var(lap_vals)) if lap_vals.size else 0.0

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    g2 = gx * gx + gy * gy
    g2_vals = g2[mask] if mask is not None else g2.reshape(-1)
    tenengrad = float(np.mean(g2_vals)) if g2_vals.size else 0.0

    med = float(np.median(gray_u8_vals)) if gray_u8_vals.size else 0.0
    lower = int(max(0, 0.66 * med))
    upper = int(min(255, 1.33 * med))
    if upper <= lower:
        lower, upper = 20, 60
    edges = cv2.Canny(gray_u8, lower, upper)
    if mask is not None:
        edge_density = float(edges[mask].sum() / (255.0 * mask.sum()))
    else:
        edge_density = float(edges.sum() / (255.0 * edges.size))

    noise_mad = estimate_noise_mad(gray, mask)
    snr_db = float(20.0 * np.log10((std_luma + 1e-8) / (noise_mad + 1e-8)))

    colorfulness = colorfulness_score(img_rgb, mask)
    pct_black = float(np.mean(gray_vals <= 0.01)) if gray_vals.size else 0.0
    pct_white = float(np.mean(gray_vals >= 0.99)) if gray_vals.size else 0.0

    return {
        "mean_luma": mean_luma,
        "std_luma": std_luma,
        "entropy": entropy,
        "lap_var": lap_var,
        "tenengrad": tenengrad,
        "edge_density": edge_density,
        "noise_mad": noise_mad,
        "snr_db": snr_db,
        "colorfulness": colorfulness,
        "pct_black": pct_black,
        "pct_white": pct_white,
    }


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


def count_subjects(
    mask: np.ndarray,
    min_frac: float = 0.01,
    merge_radius: int = 3,
) -> int:
    if mask is None or not np.any(mask):
        return 0

    work = mask.astype(np.uint8)
    if merge_radius > 0:
        k = 2 * merge_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel, iterations=1)

    num_labels, labels = cv2.connectedComponents(work, connectivity=8)
    if num_labels <= 1:
        return 0

    total = int(work.sum())
    min_area = max(64, int(total * min_frac))
    count = 0
    for label in range(1, num_labels):
        area = int(np.sum(labels == label))
        if area >= min_area:
            count += 1
    return count


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


def apply_texture_clean_scores(rows: list[MetricsRow]) -> None:
    metrics = [
        "fg_flat_noise_tv",
        "fg_noise_tv",
        "fg_starved_ratio",
        "fg_edge_spread",
        "fg_mf_median",
    ]

    stats: dict[str, tuple[float, float]] = {}
    for m in metrics:
        values = np.array(
            [getattr(r, m) for r in rows if getattr(r, m) is not None],
            dtype=float,
        )
        if values.size == 0:
            return
        med = float(np.median(values))
        mad = float(np.median(np.abs(values - med)))
        scale = 1.4826 * mad
        if scale <= 1e-8:
            scale = float(np.std(values))
        if scale <= 1e-8:
            scale = 1.0
        stats[m] = (med, scale)

    raw_scores: list[float] = []
    for r in rows:
        vals = [getattr(r, m) for m in metrics]
        if any(v is None for v in vals):
            continue
        zsum = 0.0
        for v, m in zip(vals, metrics):
            med, scale = stats[m]
            zsum += (float(v) - med) / scale
        raw = -(zsum / len(metrics))
        r.fg_texture_clean_score = float(raw)
        raw_scores.append(raw)

    if not raw_scores:
        return
    p5, p95 = np.percentile(raw_scores, [5, 95])
    for r in rows:
        if r.fg_texture_clean_score is None:
            continue
        if p95 > p5:
            norm = (r.fg_texture_clean_score - p5) / (p95 - p5)
            r.fg_texture_clean_score_0_10 = float(np.clip(norm, 0.0, 1.0) * 10.0)
        else:
            r.fg_texture_clean_score_0_10 = 5.0


def compute_metrics(
    path: Path,
    use_rembg: bool = False,
    rembg_model: str = "u2net",
    rembg_post_process: bool = True,
    rembg_session: Optional[object] = None,
    fg_min_pixels: int = 4096,
    fg_component_min_frac: float = 0.01,
    fg_component_merge_radius: int = 3,
) -> MetricsRow:
    img = Image.open(path).convert("RGB")
    img_rgb = np.array(img)
    h, w = img_rgb.shape[:2]

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gray_u8 = (gray * 255.0).astype(np.uint8)

    base = compute_metrics_from_arrays(img_rgb, gray, gray_u8)
    if base is None:
        raise RuntimeError(f"Failed to compute metrics for {path}")

    fg_mask = None
    fg_pixel_count: Optional[int] = None
    fg_pixel_frac: Optional[float] = None
    fg_metrics: Optional[dict[str, float]] = None
    fg_subject_count: Optional[int] = None
    fg_edge_sharpness: Optional[float] = None
    fg_edge_spread: Optional[float] = None
    fg_noise_tv: Optional[float] = None
    fg_hf_clean: Optional[float] = None

    if use_rembg:
        fg_mask = build_rembg_mask(img, rembg_model, rembg_post_process, rembg_session)
        fg_pixel_count = int(fg_mask.sum())
        fg_pixel_frac = float(fg_pixel_count / fg_mask.size) if fg_mask.size else 0.0
        fg_subject_count = count_subjects(
            fg_mask,
            min_frac=fg_component_min_frac,
            merge_radius=fg_component_merge_radius,
        )
        if fg_pixel_count >= fg_min_pixels:
            fg_metrics = compute_metrics_from_arrays(img_rgb, gray, gray_u8, fg_mask)
            robust = compute_noise_robust_metrics(gray, fg_mask)
            fg_edge_sharpness = robust["edge_sharpness"]
            fg_edge_spread = robust["edge_spread"]
            fg_noise_tv = robust["noise_tv"]
            fg_hf_clean = robust["hf_clean"]
            texture = compute_texture_metrics(gray, fg_mask)
            fg_starved_ratio = texture["starved_ratio"]
            fg_flat_noise_tv = texture["flat_noise_tv"]
            fg_lvar_p50 = texture["lvar_p50"]
            fg_lvar_p75 = texture["lvar_p75"]
            fg_lvar_p90 = texture["lvar_p90"]
            fg_detail_ratio = texture["detail_ratio"]
            fg_mf_median = texture["mf_median"]

    return MetricsRow(
        path=path,
        width=w,
        height=h,
        mean_luma=base["mean_luma"],
        std_luma=base["std_luma"],
        entropy=base["entropy"],
        lap_var=base["lap_var"],
        tenengrad=base["tenengrad"],
        edge_density=base["edge_density"],
        noise_mad=base["noise_mad"],
        snr_db=base["snr_db"],
        colorfulness=base["colorfulness"],
        pct_black=base["pct_black"],
        pct_white=base["pct_white"],
        fg_pixel_count=fg_pixel_count,
        fg_pixel_frac=fg_pixel_frac,
        fg_mean_luma=fg_metrics["mean_luma"] if fg_metrics else None,
        fg_std_luma=fg_metrics["std_luma"] if fg_metrics else None,
        fg_entropy=fg_metrics["entropy"] if fg_metrics else None,
        fg_lap_var=fg_metrics["lap_var"] if fg_metrics else None,
        fg_tenengrad=fg_metrics["tenengrad"] if fg_metrics else None,
        fg_edge_density=fg_metrics["edge_density"] if fg_metrics else None,
        fg_noise_mad=fg_metrics["noise_mad"] if fg_metrics else None,
        fg_snr_db=fg_metrics["snr_db"] if fg_metrics else None,
        fg_colorfulness=fg_metrics["colorfulness"] if fg_metrics else None,
        fg_pct_black=fg_metrics["pct_black"] if fg_metrics else None,
        fg_pct_white=fg_metrics["pct_white"] if fg_metrics else None,
        fg_subject_count=fg_subject_count,
        fg_edge_sharpness=fg_edge_sharpness,
        fg_edge_spread=fg_edge_spread,
        fg_noise_tv=fg_noise_tv,
        fg_hf_clean=fg_hf_clean,
        fg_starved_ratio=fg_starved_ratio,
        fg_flat_noise_tv=fg_flat_noise_tv,
        fg_lvar_p50=fg_lvar_p50,
        fg_lvar_p75=fg_lvar_p75,
        fg_lvar_p90=fg_lvar_p90,
        fg_detail_ratio=fg_detail_ratio,
        fg_mf_median=fg_mf_median,
    )


def write_csv(rows: list[MetricsRow], out_path: Path) -> None:
    def fmt(value: Optional[float]) -> str:
        if value is None:
            return ""
        return f"{value:.6f}"

    def fmt_int(value: Optional[int]) -> str:
        return "" if value is None else str(value)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image",
            "width",
            "height",
            "mean_luma",
            "std_luma",
            "entropy",
            "lap_var",
            "tenengrad",
            "edge_density",
            "noise_mad",
            "snr_db",
            "colorfulness",
            "pct_black",
            "pct_white",
            "fg_pixel_count",
            "fg_pixel_frac",
            "fg_mean_luma",
            "fg_std_luma",
            "fg_entropy",
            "fg_lap_var",
            "fg_tenengrad",
            "fg_edge_density",
            "fg_noise_mad",
            "fg_snr_db",
            "fg_colorfulness",
            "fg_pct_black",
            "fg_pct_white",
            "fg_subject_count",
            "fg_edge_sharpness",
            "fg_edge_spread",
            "fg_noise_tv",
            "fg_hf_clean",
            "fg_starved_ratio",
            "fg_flat_noise_tv",
            "fg_lvar_p50",
            "fg_lvar_p75",
            "fg_lvar_p90",
            "fg_detail_ratio",
            "fg_mf_median",
            "fg_texture_clean_score",
            "fg_texture_clean_score_0_10",
        ])
        for r in rows:
            writer.writerow([
                r.path.as_posix(),
                r.width,
                r.height,
                fmt(r.mean_luma),
                fmt(r.std_luma),
                fmt(r.entropy),
                fmt(r.lap_var),
                fmt(r.tenengrad),
                fmt(r.edge_density),
                fmt(r.noise_mad),
                fmt(r.snr_db),
                fmt(r.colorfulness),
                fmt(r.pct_black),
                fmt(r.pct_white),
                fmt_int(r.fg_pixel_count),
                fmt(r.fg_pixel_frac),
                fmt(r.fg_mean_luma),
                fmt(r.fg_std_luma),
                fmt(r.fg_entropy),
                fmt(r.fg_lap_var),
                fmt(r.fg_tenengrad),
                fmt(r.fg_edge_density),
                fmt(r.fg_noise_mad),
                fmt(r.fg_snr_db),
                fmt(r.fg_colorfulness),
                fmt(r.fg_pct_black),
                fmt(r.fg_pct_white),
                fmt_int(r.fg_subject_count),
                fmt(r.fg_edge_sharpness),
                fmt(r.fg_edge_spread),
                fmt(r.fg_noise_tv),
                fmt(r.fg_hf_clean),
                fmt(r.fg_starved_ratio),
                fmt(r.fg_flat_noise_tv),
                fmt(r.fg_lvar_p50),
                fmt(r.fg_lvar_p75),
                fmt(r.fg_lvar_p90),
                fmt(r.fg_detail_ratio),
                fmt(r.fg_mf_median),
                fmt(r.fg_texture_clean_score),
                fmt(r.fg_texture_clean_score_0_10),
            ])


def print_table(rows: list[MetricsRow], limit: int | None) -> None:
    header = (
        f"{'image':36}  {'WxH':>9}  {'lap':>9}  {'ten':>9}  {'noise':>8}  "
        f"{'snr':>7}  {'ent':>6}  {'std':>6}  {'edge':>6}"
    )
    print(header)
    print("-" * len(header))
    show = rows if not limit or limit <= 0 else rows[:limit]
    for r in show:
        name = r.path.name[:36]
        print(
            f"{name:36}  {r.width:4d}x{r.height:<4d}  "
            f"{r.lap_var:9.4f}  {r.tenengrad:9.4f}  {r.noise_mad:8.4f}  "
            f"{r.snr_db:7.2f}  {r.entropy:6.2f}  {r.std_luma:6.3f}  {r.edge_density:6.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute objective, no-reference image quality metrics."
    )
    parser.add_argument(
        "--paths",
        nargs="*",
        default=None,
        help="File or directory paths to scan. Defaults to root files + test/tests folders.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="image_quality_metrics.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--sort",
        choices=[
            "name",
            "lap_var",
            "tenengrad",
            "noise_mad",
            "snr_db",
            "entropy",
            "std_luma",
            "edge_density",
        ],
        default="name",
        help="Sort output table by a metric.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit printed rows (0 = all). CSV always includes all rows.",
    )
    parser.add_argument(
        "--use-rembg",
        action="store_true",
        help="Compute foreground (subject) metrics using rembg segmentation.",
    )
    parser.add_argument(
        "--rembg-model",
        type=str,
        default="u2net",
        help="rembg model name (e.g., isnet-anime, u2net, bria-rmbg).",
    )
    parser.add_argument(
        "--rembg-post-process",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Post-process rembg masks for smoother edges.",
    )
    parser.add_argument(
        "--fg-min-pixels",
        type=int,
        default=4096,
        help="Minimum foreground pixels required to compute fg_* metrics.",
    )
    parser.add_argument(
        "--fg-component-min-frac",
        type=float,
        default=0.01,
        help="Minimum component size as a fraction of the foreground mask for subject counting.",
    )
    parser.add_argument(
        "--fg-component-merge-radius",
        type=int,
        default=3,
        help="Radius for merging nearby components before counting.",
    )
    args = parser.parse_args()

    if args.paths:
        scan_paths: list[Path] = []
        for raw in args.paths:
            scan_paths.append(Path(raw))
        images: list[Path] = []
        for p in scan_paths:
            images.extend(iter_images(p))
    else:
        images = default_image_paths()

    if not images:
        print("No images found.")
        return

    rembg_session = None
    if args.use_rembg:
        try:
            from rembg import new_session

            rembg_session = new_session(args.rembg_model)
        except Exception as exc:
            print(f"rembg init failed: {exc}")
            return

    rows = [
        compute_metrics(
            p,
            use_rembg=args.use_rembg,
            rembg_model=args.rembg_model,
            rembg_post_process=args.rembg_post_process,
            rembg_session=rembg_session,
            fg_min_pixels=args.fg_min_pixels,
            fg_component_min_frac=args.fg_component_min_frac,
            fg_component_merge_radius=args.fg_component_merge_radius,
        )
        for p in images
    ]
    if args.use_rembg:
        apply_texture_clean_scores(rows)

    key_map = {
        "name": lambda r: r.path.name.lower(),
        "lap_var": lambda r: r.lap_var,
        "tenengrad": lambda r: r.tenengrad,
        "noise_mad": lambda r: r.noise_mad,
        "snr_db": lambda r: r.snr_db,
        "entropy": lambda r: r.entropy,
        "std_luma": lambda r: r.std_luma,
        "edge_density": lambda r: r.edge_density,
    }
    rows.sort(key=key_map[args.sort])

    print_table(rows, args.limit)
    write_csv(rows, Path(args.out))
    print(f"\nWrote CSV to {args.out}")


if __name__ == "__main__":
    main()
