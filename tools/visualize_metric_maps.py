import argparse
import io
import math
from pathlib import Path
from typing import Iterable, List, Optional

import cv2
import numpy as np
from PIL import Image
from skimage.restoration import denoise_tv_chambolle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import rembg


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def iter_images(path: Path) -> Iterable[Path]:
    if path.is_file():
        if path.suffix.lower() in IMAGE_EXTS:
            yield path
        return
    for p in sorted(path.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def resolve_image_path(raw: str) -> Optional[Path]:
    if not raw:
        return None
    p = Path(raw)
    if p.exists():
        return p
    raw_norm = raw.replace("\\", "/")
    if len(raw_norm) >= 2 and raw_norm[1] == ":":
        drive = raw_norm[0].lower()
        rest = raw_norm[2:].lstrip("/")
        mapped = Path(f"/mnt/{drive}/{rest}")
        if mapped.exists():
            return mapped
    return None


def load_images_from_csv(csv_path: Path) -> List[Path]:
    import csv

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    images: List[Path] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            raw = r.get("image", "")
            p = resolve_image_path(raw)
            if p:
                images.append(p)
    return images


def subject_mask(pil_image: Image.Image, session: object, post_process: bool) -> np.ndarray:
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    output_bytes = rembg.remove(
        buffer.getvalue(),
        session=session,
        only_mask=True,
        post_process_mask=post_process,
    )
    mask_img = Image.open(io.BytesIO(output_bytes)).convert("L")
    mask = np.array(mask_img) > 128
    return mask


def normalize_map(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask is not None and np.any(mask):
        ref = values[mask]
    else:
        ref = values.reshape(-1)
    if ref.size == 0:
        return np.zeros_like(values)
    lo, hi = np.percentile(ref, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(ref))
        hi = float(np.max(ref)) if np.max(ref) > lo else lo + 1e-6
    out = (values - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    if mask is not None:
        out = out.copy()
        out[~mask] = 0.0
    return out


def compute_maps(img_rgb: np.ndarray, mask: np.ndarray) -> dict:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gray_u8 = (gray * 255.0).astype(np.uint8)

    # High-frequency detail
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap_abs = np.abs(lap)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.sqrt(gx * gx + gy * gy)

    mean = cv2.blur(gray, (5, 5))
    mean_sq = cv2.blur(gray * gray, (5, 5))
    lvar = np.maximum(0.0, mean_sq - mean * mean)

    denoised = denoise_tv_chambolle(gray, weight=0.10)
    tv_residual = np.abs(gray - denoised)

    gauss1 = cv2.GaussianBlur(gray, (0, 0), 1.0)
    gauss5 = cv2.GaussianBlur(gray, (0, 0), 5.0)
    mf = np.abs(gauss1 - gauss5)
    hf = np.abs(lap)
    ratio = hf / (mf + 1e-6)
    blur_map = np.clip(1.0 - (ratio / 1.0), 0.0, 1.0)

    med = float(np.median(gray_u8[mask])) if np.any(mask) else float(np.median(gray_u8))
    lower = int(max(10, 0.66 * med))
    upper = int(min(255, 1.33 * med))
    if upper <= lower:
        lower, upper = 20, 60
    edges = cv2.Canny(gray_u8, lower, upper) / 255.0
    edges = edges * mask

    return {
        "gray": gray,
        "lap_abs": lap_abs,
        "gmag": gmag,
        "lvar": lvar,
        "tv_residual": tv_residual,
        "blur_map": blur_map,
        "edges": edges,
    }


def render_panel(
    img_rgb: np.ndarray,
    mask: np.ndarray,
    maps: dict,
    out_path: Path,
) -> None:
    subject = img_rgb.copy()
    subject[~mask] = 0

    lap_n = normalize_map(maps["lap_abs"], mask)
    gmag_n = normalize_map(maps["gmag"], mask)
    lvar_n = normalize_map(maps["lvar"], mask)
    tv_n = normalize_map(maps["tv_residual"], mask)
    blur_n = normalize_map(maps["blur_map"], mask)
    edges = maps["edges"]

    panels = [
        ("Original", img_rgb),
        ("Subject Cutout", subject),
        ("Laplacian |HF|", lap_n, "magma"),
        ("Gradient Mag", gmag_n, "inferno"),
        ("Local Variance", lvar_n, "viridis"),
        ("TV Residual", tv_n, "plasma"),
        ("Blur Map", blur_n, "magma"),
        ("Edge Map", edges, "gray"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, item in zip(axes.ravel(), panels):
        if len(item) == 2:
            title, data = item
            ax.imshow(data)
        else:
            title, data, cmap = item
            ax.imshow(data, cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize subject-focused metric maps per image."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="image_quality_metrics_2.csv",
        help="CSV path to read image list from.",
    )
    parser.add_argument(
        "--paths",
        nargs="*",
        default=None,
        help="Image files or directories (optional, overrides --csv if provided).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="metric_maps",
        help="Output directory for panel PNGs.",
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
        "--max-images",
        type=int,
        default=0,
        help="Limit number of images processed (0 = all).",
    )
    args = parser.parse_args()

    if args.paths:
        images: List[Path] = []
        for raw in args.paths:
            images.extend(iter_images(Path(raw)))
    else:
        images = load_images_from_csv(Path(args.csv))

    if not images:
        raise SystemExit("No images found.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    session = rembg.new_session(args.rembg_model)

    if args.max_images > 0:
        images = images[: args.max_images]

    for img_path in images:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as exc:
            print(f"Skip {img_path}: {exc}")
            continue
        img_rgb = np.array(img)
        mask = subject_mask(img, session, args.rembg_post_process)
        if not np.any(mask):
            print(f"Skip {img_path}: empty mask")
            continue
        maps = compute_maps(img_rgb, mask)
        out_path = out_dir / f"{img_path.stem}_maps.png"
        render_panel(img_rgb, mask, maps, out_path)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
