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


def investigate(p, name, sess):
    img = np.array(Image.open(p).convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    mask = get_mask(p, sess)

    # 1. Local Variance
    mean = cv2.blur(gray, (5, 5))
    sq_mean = cv2.blur(gray**2, (5, 5))
    lvar = np.maximum(0, sq_mean - mean**2)
    s_var = lvar[mask]

    # 2. Laplacian (Sharpness)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    s_lap = lap[mask]

    print(f"\n--- Analysis for {name} ({p}) ---")
    print(f"  Mask Size: {mask.sum()}")
    print(
        f"  Variance: Mean={np.mean(s_var):.6f}, 25th={np.percentile(s_var, 25):.6f}, 50th={np.percentile(s_var, 50):.6f}, 75th={np.percentile(s_var, 75):.6f}"
    )
    print(
        f"  Sharpness: Mean={np.mean(s_lap):.6f}, 90th={np.percentile(s_lap, 90):.6f}, 95th={np.percentile(s_lap, 95):.6f}"
    )


sess = rembg.new_session(providers=["CPUExecutionProvider"])
investigate("116-00-shiro_black-7.595.png", "BLURRY/NOISY", sess)
investigate("00166-4154668875.png", "CLEAN", sess)
investigate("038-04-noob10b-6.081.png", "NOISY BG", sess)
