import cv2
import numpy as np
from PIL import Image
import rembg
import argparse
import matplotlib.pyplot as plt
import io


class BlotchBlurScorerV16:
    def __init__(self, rembg_session=None):
        if rembg_session is None:
            self.rembg_session = rembg.new_session(providers=["CPUExecutionProvider"])
        else:
            self.rembg_session = rembg_session

    def _get_subject_mask(self, pil_image: Image.Image) -> np.ndarray:
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        output_bytes = rembg.remove(buffer.getvalue(), session=self.rembg_session)
        output_mask = Image.open(io.BytesIO(output_bytes)).convert("RGBA")
        mask = np.array(output_mask)[:, :, 3] > 128
        return mask

    def _line_integrity(self, gray: np.ndarray, mask: np.ndarray) -> dict:
        gray_u8 = (gray * 255.0).astype(np.uint8)
        blurred_u8 = cv2.GaussianBlur(gray_u8, (0, 0), 1.0)
        masked_vals = blurred_u8[mask]

        if masked_vals.size == 0:
            return {
                "line_score": 0.0,
                "edge_strength": 0.0,
                "edge_density": 0.0,
                "edge_spread": 0.0,
                "edge_sharpness": 0.0,
                "edge_map": np.zeros_like(gray),
            }

        v = float(np.median(masked_vals))
        lower = int(max(10, 0.66 * v))
        upper = int(min(255, 1.33 * v))
        if upper <= lower:
            lower, upper = 20, 60

        edges = cv2.Canny(blurred_u8, lower, upper)
        edge_mask = (edges > 0) & mask
        edge_density = float(np.sum(edge_mask) / (np.sum(mask) + 1e-6))

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gmag = np.sqrt(gx * gx + gy * gy)

        if np.any(edge_mask):
            edge_g = float(np.mean(gmag[edge_mask]))
        else:
            edge_g = 0.0

        dil = cv2.dilate(edge_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2) > 0
        ring_mask = dil & (~edge_mask) & mask
        ring_g = float(np.mean(gmag[ring_mask])) if np.any(ring_mask) else 0.0

        edge_sharpness = edge_g / (ring_g + 1e-6)
        edge_spread = ring_g / (edge_g + 1e-6)

        edge_strength = float(np.clip((edge_g - 0.03) / 0.12, 0.0, 1.0))
        edge_density_n = float(np.clip(edge_density / 0.08, 0.0, 1.0))
        sharpness_n = float(np.clip((edge_sharpness - 1.2) / 1.8, 0.0, 1.0))

        line_score = 10.0 * (
            0.50 * sharpness_n
            + 0.30 * edge_strength
            + 0.20 * edge_density_n
        )

        edge_map = np.zeros_like(gray)
        edge_map[mask] = edges[mask] / 255.0

        return {
            "line_score": float(line_score),
            "edge_strength": edge_strength,
            "edge_density": edge_density,
            "edge_spread": edge_spread,
            "edge_sharpness": edge_sharpness,
            "edge_map": edge_map,
        }

    def analyze(self, image: Image.Image) -> dict:
        img_rgb = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        mask = self._get_subject_mask(image)

        if not np.any(mask):
            return {"texture_score": 0.0, "detail_score": 0.0, "overall_score": 0.0}

        # --- Base Metrics (Restricted to Subject) ---
        mean_l = cv2.blur(gray, (5, 5))
        sq_mean_l = cv2.blur(gray**2, (5, 5))
        lvar = np.maximum(0, sq_mean_l - mean_l**2)

        hf = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
        gauss1 = cv2.GaussianBlur(gray, (0, 0), 1.0)
        gauss5 = cv2.GaussianBlur(gray, (0, 0), 5.0)
        mf = np.abs(gauss1 - gauss5)

        # Subject-only stats
        s_lvar = lvar[mask]
        s_hf = hf[mask]
        s_mf = mf[mask]

        # --- 1. Global Noise Floor Estimation ---
        # mf_median is the "jitter noise" floor.
        # Clean (00166): 0.0206. Noisy (038-04): 0.0233.
        mf_median = np.median(s_mf)

        # --- 2. Detail Score (Clean Detail Ratio) ---
        avg_hf = np.mean(s_hf)
        avg_mf = np.mean(s_mf)
        raw_ratio = avg_hf / (avg_mf + 1e-6)

        # In noisy images (038), high ratio comes from high-energy MF jitter.
        # We penalize high-energy MF detail specifically.
        clean_detail_mult = np.clip(1.0 - (avg_mf - 0.045) * 50.0, 0.5, 1.0)

        # Calibrated Detail Curve
        d_score = 10.0 * np.clip((raw_ratio - 0.9) / 0.12, 0, 1) * clean_detail_mult

        # --- 3. Texture Score (Continuous: detail-gated + band-pass + voids) ---
        v_baseline = np.percentile(s_lvar, 20)
        starv_mask = (lvar < (v_baseline * 0.4)) & mask
        starved_ratio = np.sum(starv_mask) / np.sum(mask)

        detail_norm = np.clip(d_score / 10.0, 0.0, 1.0)
        detail_gate = 0.30 + 0.70 * detail_norm
        void_score = np.exp(-((starved_ratio / 0.08) ** 2))
        mf_balance = np.exp(-(((avg_mf - 0.05) / 0.015) ** 2))
        noise_score = 1.0 / (1.0 + np.exp((mf_median - 0.022) / 0.003))

        t_score = 10.0 * detail_gate * void_score * mf_balance * noise_score

        # --- Final Scoring ---
        line_metrics = self._line_integrity(gray, mask)
        line_score = line_metrics["line_score"]

        overall = (t_score * 0.45) + (d_score * 0.45) + (line_score * 0.10)

        # Maps
        blotch_f = np.zeros_like(gray)
        blotch_f[mask] = np.clip(1.0 - (lvar[mask] / (v_baseline * 1.5 + 1e-8)), 0, 1)

        blur_f = np.zeros_like(gray)
        local_ratio = hf / (mf + 1e-6)
        blur_f[mask] = np.clip(1.0 - (local_ratio[mask] / 1.0), 0, 1)

        return {
            "texture_score": float(t_score),
            "detail_score": float(d_score),
            "line_score": float(line_score),
            "overall_score": float(overall),
            "blotch_map": blotch_f,
            "blur_map": blur_f,
            "edge_map": line_metrics["edge_map"],
            "mask": mask,
            "img_rgb": img_rgb,
            "noise_floor": float(mf_median),
            "avg_mf": float(avg_mf),
            "edge_strength": float(line_metrics["edge_strength"]),
            "edge_density": float(line_metrics["edge_density"]),
            "edge_spread": float(line_metrics["edge_spread"]),
            "edge_sharpness": float(line_metrics["edge_sharpness"]),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Forensic analysis V16.1 (Noise-Specific)."
    )
    parser.add_argument("image_path", type=str, help="Path to image")
    parser.add_argument("--no-plot", action="store_true", help="Skip visualization")
    args = parser.parse_args()

    try:
        pil_img = Image.open(args.image_path).convert("RGB")
    except Exception as e:
        print(f"Error: {e}")
        exit()

    print(f"Analyzing {args.image_path} with V16.1 logic...")
    scorer = BlotchBlurScorerV16()
    res = scorer.analyze(pil_img)

    print(f"\n--- Forensic Analysis Results for {args.image_path} ---")
    print(f"Overall Score:  {res['overall_score']:.2f} / 10.0")
    print(f"Texture Score:  {res['texture_score']:.2f} (Clean Consistency)")
    print(f"Detail Score:   {res['detail_score']:.2f} (Signal Ratio)")
    print(f"Line Score:     {res['line_score']:.2f} (Edge Integrity)")
    print(f"Noise Floor:    {res['noise_floor']:.4f}")
    print(f"Avg MF Energy:  {res['avg_mf']:.4f}")
    print(f"Edge Strength:  {res['edge_strength']:.3f}")
    print(f"Edge Density:   {res['edge_density']:.3f}")
    print(f"Edge Spread:    {res['edge_spread']:.3f}")
    print(f"Edge Sharpness: {res['edge_sharpness']:.3f}")
    print(f"---------------------------------\n")

    if args.no_plot:
        exit()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Forensic Filter V16.1: {args.image_path}", fontsize=16)

    axes[0].imshow(res["img_rgb"])
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    im_blotch = axes[1].imshow(res["blotch_map"], cmap="magma", vmin=0, vmax=1)
    axes[1].set_title("Oversmoothing Map\n(Bright = Voids)")
    axes[1].axis("off")

    im_blur = axes[2].imshow(res["blur_map"], cmap="viridis", vmin=0, vmax=1)
    axes[2].set_title("Energy Inconsistency Map\n(Bright = Poor SNR)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()
