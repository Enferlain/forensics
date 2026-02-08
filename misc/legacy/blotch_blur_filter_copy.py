import cv2
import numpy as np
from PIL import Image
import rembg
import argparse
import matplotlib.pyplot as plt
import io


class BlotchBlurScorerV14:
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

    def analyze(self, image: Image.Image) -> dict:
        img_rgb = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        mask = self._get_subject_mask(image)

        if not np.any(mask):
            return {"texture_score": 0.0, "detail_score": 0.0, "overall_score": 0.0}

        # --- 1. Detail Score (Multi-Band Frequency Ratio) ---
        # High Freq (Laplacian)
        hf = np.abs(cv2.Laplacian(gray, cv2.CV_32F))

        # Mid Freq (Gaussian Difference, 1px vs 5px) - captures "clumps"
        gauss1 = cv2.GaussianBlur(gray, (0, 0), 1.0)
        gauss5 = cv2.GaussianBlur(gray, (0, 0), 5.0)
        mf = np.abs(gauss1 - gauss5)

        s_hf = np.mean(hf[mask])
        s_mf = np.mean(mf[mask])

        # Ratio HF/MF: Clean images are ~1.0+, AI artifacts (clumpy) are ~0.88
        hf_mf_ratio = s_hf / (s_mf + 1e-6)

        # Non-linear mapping to widen the gap between 0.9 and 1.0
        # Steep curve: 1.0 -> 9.0, 0.9 -> 4.0, 0.85 -> 1.0
        d_score = 10.0 * np.clip((hf_mf_ratio - 0.85) / 0.15, 0, 1)

        # --- 2. Texture Score (Clumpiness / Regularity) ---
        # Instead of richness, we penalize "Spatial Inconsistency"
        mean_l = cv2.blur(gray, (5, 5))
        sq_mean_l = cv2.blur(gray**2, (5, 5))
        lvar = np.maximum(0, sq_mean_l - mean_l**2)

        # Measure variance of the local variance (V-Var) in a 15x15 window
        mean_v = cv2.blur(lvar, (15, 15))
        sq_mean_v = cv2.blur(lvar**2, (15, 15))
        vvar = np.maximum(0, sq_mean_v - mean_v**2)

        # Clumpiness Index: std(lvar) / mean(lvar)
        clumpiness = np.sqrt(vvar[mask]) / (mean_v[mask] + 1e-6)
        avg_clumpiness = np.mean(clumpiness)

        # Texture Score: higher is better (lower clumpiness)
        # Clean/Clean grain usually has lower relative deviation in texture floor
        t_score = 10.0 * np.clip(1.0 - (avg_clumpiness - 1.25) * 4.0, 0, 1)

        # Overall Score
        overall = (t_score * 0.6) + (d_score * 0.4)

        # Prepare heatmaps
        blotch_full = np.zeros_like(gray)
        blotch_full[mask] = np.clip((clumpiness - 1.1) / 0.8, 0, 1)

        # Local HF/MF ratio for the blur map
        local_ratio = hf / (mf + 1e-6)
        blur_full = np.clip(1.0 - (local_ratio / 1.05), 0, 1)
        blur_full[~mask] = 0

        return {
            "texture_score": float(t_score),
            "detail_score": float(d_score),
            "overall_score": float(overall),
            "blotch_map": blotch_full,
            "blur_map": blur_full,
            "mask": mask,
            "img_rgb": img_rgb,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forensic analysis V14 (Band-Ratio).")
    parser.add_argument("image_path", type=str, help="Path to image")
    parser.add_argument("--no-plot", action="store_true", help="Skip visualization")
    args = parser.parse_args()

    try:
        pil_img = Image.open(args.image_path).convert("RGB")
    except Exception as e:
        print(f"Error: {e}")
        exit()

    print(f"Analyzing {args.image_path} with V14 logic...")
    scorer = BlotchBlurScorerV14()
    res = scorer.analyze(pil_img)

    print(f"\n--- Forensic Analysis Results for {args.image_path} ---")
    print(f"Overall Score:  {res['overall_score']:.2f} / 10.0")
    print(f"Texture Score:  {res['texture_score']:.2f} (Uniformity)")
    print(f"Detail Score:   {res['detail_score']:.2f} (HF/MF Ratio)")
    print(f"---------------------------------\n")

    if args.no_plot:
        exit()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Forensic Filter V14: {args.image_path}", fontsize=16)

    axes[0].imshow(res["img_rgb"])
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    im_blotch = axes[1].imshow(res["blotch_map"], cmap="magma", vmin=0, vmax=1)
    axes[1].set_title("Clumpiness/Blotch Map\n(Bright = Unnatural texture clusters)")
    axes[1].axis("off")

    im_blur = axes[2].imshow(res["blur_map"], cmap="viridis", vmin=0, vmax=1)
    axes[2].set_title("Frequency Starvation\n(Bright = Lacks Clean Detail)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()
