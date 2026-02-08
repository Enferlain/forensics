import cv2
import numpy as np
from PIL import Image
import rembg
import io


def get_mask(p, sess):
    img = Image.open(p).convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    out = rembg.remove(buffer.getvalue(), session=sess)
    mask = np.array(Image.open(io.BytesIO(out)))[:, :, 3] > 128
    return mask


def investigate_clumpiness(p, name, sess):
    img = np.array(Image.open(p).convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    mask = get_mask(p, sess)

    # 1. Local Variance (Small Radius)
    mean = cv2.blur(gray, (5, 5))
    sq_mean = cv2.blur(gray**2, (5, 5))
    lvar = np.maximum(0, sq_mean - mean**2)

    # 2. Variance of Variance (Clumpiness)
    # How much does the local variance change in a larger window (15x15)?
    mean_v = cv2.blur(lvar, (15, 15))
    sq_mean_v = cv2.blur(lvar**2, (15, 15))
    vvar = np.maximum(0, sq_mean_v - mean_v**2)

    # Subject stats
    s_lvar = lvar[mask]
    s_vvar = vvar[mask]

    # Relative Clumpiness: std(lvar) / mean(lvar)
    # High = inconsistent texture floor (bad AI)
    # Low = consistent grain floor (natural or very clean)
    clumpiness = np.sqrt(s_vvar) / (mean_v[mask] + 1e-6)

    print(f"\n--- Clumpiness Analysis for {name} ---")
    print(f"  L-Var (Mean): {np.mean(s_lvar):.6f}")
    print(f"  V-Var (Mean): {np.mean(s_vvar):.10f}")
    print(f"  Relative Clumpiness (Avg): {np.mean(clumpiness):.4f}")
    print(f"  Clumpiness (90th): {np.percentile(clumpiness, 90):.4f}")


sess = rembg.new_session(providers=["CPUExecutionProvider"])
investigate_clumpiness("00166-4154668875.png", "CLEAN (00166)", sess)
investigate_clumpiness(
    "d:/Projects/forensics/test/242-00-shiro_black-5.456.png", "WORST (242)", sess
)
investigate_clumpiness("116-00-shiro_black-7.595.png", "BLURRY (116)", sess)
