import argparse
import io
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
from skimage.restoration import denoise_tv_chambolle
import matplotlib.pyplot as plt
import rembg


@dataclass
class TextureMetrics:
    flat_noise_tv: float
    noise_tv: float
    starved_ratio: float
    edge_spread: float
    mf_median: float


class TextureCleanScorer:
    def __init__(
        self,
        rembg_model: str = "u2net",
        post_process_mask: bool = True,
    ):
        self.rembg_model = rembg_model
        self.post_process_mask = post_process_mask
        self.session = rembg.new_session(rembg_model)

    def _subject_mask(self, pil_image: Image.Image) -> np.ndarray:
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        output_bytes = rembg.remove(
            buffer.getvalue(),
            session=self.session,
            only_mask=True,
            post_process_mask=self.post_process_mask,
        )
        mask_img = Image.open(io.BytesIO(output_bytes)).convert("L")
        return np.array(mask_img) > 128

    def compute(self, pil_image: Image.Image) -> tuple[TextureMetrics, dict]:
        img_rgb = np.array(pil_image.convert("RGB"))
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        mask = self._subject_mask(pil_image)

        mean = cv2.blur(gray, (5, 5))
        mean_sq = cv2.blur(gray * gray, (5, 5))
        lvar = np.maximum(0.0, mean_sq - mean * mean)
        s_lvar = lvar[mask]
        v_baseline = float(np.percentile(s_lvar, 20)) if s_lvar.size else 0.0
        denom = v_baseline * 1.5 + 1e-8
        blotch_map = np.zeros_like(gray, dtype=np.float32)
        if mask.any():
            blotch_map[mask] = np.clip(1.0 - (lvar[mask] / denom), 0.0, 1.0)
        starved_ratio = float(np.mean(blotch_map[mask])) if mask.any() else 0.0

        denoised = denoise_tv_chambolle(gray, weight=0.10)
        residual = np.abs(gray - denoised)
        flat_mask = (lvar < np.percentile(s_lvar, 25)) & mask if s_lvar.size else mask
        if flat_mask.sum() < 64:
            flat_noise_tv = float(np.std(residual[mask])) if mask.any() else 0.0
            noise_tv = flat_noise_tv
        else:
            flat_noise_tv = float(np.std(residual[flat_mask]))
            noise_tv = flat_noise_tv

        gx = cv2.Sobel(denoised, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(denoised, cv2.CV_32F, 0, 1, ksize=3)
        gmag = np.sqrt(gx * gx + gy * gy)
        gvals = gmag[mask]
        if gvals.size:
            thresh = float(np.percentile(gvals, 90))
            edge_mask = (gmag >= thresh) & mask
            if edge_mask.sum() < 16:
                edge_mask = (gmag >= np.percentile(gvals, 80)) & mask
        else:
            edge_mask = np.zeros_like(mask)
        kernel = np.ones((3, 3), np.uint8)
        dil = cv2.dilate(edge_mask.astype(np.uint8), kernel, iterations=2) > 0
        ring = dil & (~edge_mask) & mask
        if ring.sum() < 16:
            ring = mask & (~edge_mask)
        edge_mean = float(np.mean(gmag[edge_mask])) if edge_mask.sum() else 0.0
        ring_mean = float(np.mean(gmag[ring])) if ring.sum() else 0.0
        edge_spread = ring_mean / (edge_mean + 1e-6)

        gauss1 = cv2.GaussianBlur(gray, (0, 0), 1.0)
        gauss5 = cv2.GaussianBlur(gray, (0, 0), 5.0)
        mf = np.abs(gauss1 - gauss5)
        mf_median = float(np.median(mf[mask])) if mask.any() else 0.0

        starved_mask = (lvar < (v_baseline * 0.4)) & mask if s_lvar.size else mask

        # Edge visualization (edge vs ring)
        edge_viz = np.zeros_like(img_rgb)
        edge_viz[edge_mask] = (255, 50, 50)
        edge_viz[ring] = (50, 220, 50)

        maps = {
            "img_rgb": img_rgb,
            "mask": mask,
            "lvar": lvar,
            "residual": residual,
            "gmag": gmag,
            "edge_mask": edge_mask,
            "ring_mask": ring,
            "edge_viz": edge_viz,
            "mf": mf,
            "flat_mask": flat_mask,
            "starved_mask": starved_mask,
            "blotch_map": blotch_map,
        }
        metrics = TextureMetrics(
            flat_noise_tv=flat_noise_tv,
            noise_tv=noise_tv,
            starved_ratio=starved_ratio,
            edge_spread=edge_spread,
            mf_median=mf_median,
        )
        return metrics, maps

    def score(self, metrics: TextureMetrics, maps: dict) -> tuple[float, float, dict]:
        mask = maps["mask"]

        blotch_map = maps["blotch_map"]
        if mask.any():
            blotch_mean = float(np.mean(blotch_map[mask]))
        else:
            blotch_mean = 0.0

        residual_flat = maps["residual"] * maps["flat_mask"]
        if maps["flat_mask"].any():
            flat_noise_norm = normalize_map(residual_flat, maps["flat_mask"])
            flat_noise_mean = float(np.mean(flat_noise_norm[maps["flat_mask"]]))
        else:
            flat_noise_mean = 0.0

        if mask.any():
            mf_norm = normalize_map(maps["mf"], mask)
            mf_mean = float(np.mean(mf_norm[mask]))
        else:
            mf_mean = 0.0

        edge_spread_norm = float(metrics.edge_spread / (metrics.edge_spread + 1.0))

        # Use flat_noise_mean for both flat and noise components to match metric definitions.
        parts = {
            "flat_noise": flat_noise_mean,
            "noise": flat_noise_mean,
            "starved": blotch_mean,
            "edge_spread": edge_spread_norm,
            "mf_energy": mf_mean,
        }

        badness = float(np.mean(list(parts.values()))) if parts else 0.0
        score_raw = float(np.clip(1.0 - badness, 0.0, 1.0))
        return score_raw, score_raw * 10.0, parts


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Texture clean scorer with visualization.")
    parser.add_argument("image_path", type=str, help="Path to the image file.")
    parser.add_argument("--rembg-model", type=str, default="u2net")
    parser.add_argument("--no-rembg-post-process", action="store_true")
    args = parser.parse_args()

    img = Image.open(args.image_path)
    scorer = TextureCleanScorer(
        rembg_model=args.rembg_model,
        post_process_mask=not args.no_rembg_post_process,
    )

    metrics, maps = scorer.compute(img)
    score_raw, score_0_10, parts = scorer.score(metrics, maps)

    print("\n--- Texture Clean Score ---")
    print(f"Score: {score_0_10:.3f} / 10.0  (raw={score_raw:.4f})")
    print(f"fg_flat_noise_tv: {metrics.flat_noise_tv:.6f}")
    print(f"fg_noise_tv:      {metrics.noise_tv:.6f}")
    print(f"fg_starved_ratio: {metrics.starved_ratio:.6f}")
    print(f"fg_edge_spread:   {metrics.edge_spread:.6f}")
    print(f"fg_mf_median:     {metrics.mf_median:.6f}")
    print("\nScore parts (0..1, lower is worse):")
    for k, v in parts.items():
        print(f"{k:12} {v:.4f}")

    img_rgb = maps["img_rgb"]
    mask = maps["mask"]
    subject = img_rgb.copy()
    subject[~mask] = 0

    residual_flat = maps["residual"] * maps["flat_mask"]
    residual_n = normalize_map(residual_flat, maps["flat_mask"])
    mf_n = normalize_map(maps["mf"], mask)
    blotch_n = maps["blotch_map"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Texture Clean Scorer", fontsize=16)

    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(subject)
    axes[0, 1].set_title("Subject Cutout")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(blotch_n, cmap="magma", vmin=0, vmax=1)
    axes[0, 2].set_title("Oversmoothing Map (Blotch)")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(residual_n, cmap="plasma", vmin=0, vmax=1)
    axes[1, 0].set_title("Flat-Region Noise (TV)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(maps["edge_viz"])
    axes[1, 1].set_title("Edge vs Ring (Spread)")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(mf_n, cmap="magma", vmin=0, vmax=1)
    axes[1, 2].set_title("Mid-Frequency Energy")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
