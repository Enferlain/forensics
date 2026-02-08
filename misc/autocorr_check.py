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


def investigate_autocorr(p, name, sess):
    img = np.array(Image.open(p).convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    mask = get_mask(p, sess)

    # 1. Extract Noise (High-Pass)
    # Use a small bilateral filter or tiny blur to remove grain
    signal = cv2.GaussianBlur(gray, (3, 3), 0)
    noise = gray - signal

    # Take a 128x128 patch from a textured area (middle of subject)
    # (Simplified for now: take a patch where variance is median-ish)
    mean = cv2.blur(gray, (5, 5))
    sq_mean = cv2.blur(gray**2, (5, 5))
    lvar = np.maximum(0, sq_mean - mean**2)
    lvar[~mask] = 0

    coords = np.argwhere(lvar > np.percentile(lvar[mask], 50))
    if len(coords) == 0:
        return
    center = coords[len(coords) // 2]

    patch_size = 64
    y, x = center
    y_s, y_e = max(0, y - patch_size), min(gray.shape[0], y + patch_size)
    x_s, x_e = max(0, x - patch_size), min(gray.shape[1], x + patch_size)

    noise_patch = noise[y_s:y_e, x_s:x_e]

    # Autocorrelation
    # (Manual shift and dot product for 1-pixel offset)
    def corr_at_offset(n, dy, dx):
        n1 = n[
            max(0, dy) : n.shape[0] + min(0, dy), max(0, dx) : n.shape[1] + min(0, dx)
        ]
        n2 = n[
            max(0, -dy) : n.shape[0] + min(0, -dy),
            max(0, -dx) : n.shape[1] + min(0, -dx),
        ]
        return np.mean(n1 * n2) / (np.var(n) + 1e-8)

    c1 = corr_at_offset(noise_patch, 1, 0)
    c2 = corr_at_offset(noise_patch, 0, 1)
    c3 = corr_at_offset(noise_patch, 2, 2)

    print(f"\n--- Noise Correlation for {name} ---")
    print(f"  Offset (1,0): {c1:.4f}")
    print(f"  Offset (0,1): {c2:.4f}")
    print(f"  Offset (2,2): {c3:.4f}")


sess = rembg.new_session(providers=["CPUExecutionProvider"])
investigate_autocorr("00166-4154668875.png", "CLEAN (00166)", sess)
investigate_autocorr(
    "d:/Projects/forensics/test/242-00-shiro_black-5.456.png", "WORST (242)", sess
)
investigate_autocorr("116-00-shiro_black-7.595.png", "BLURRY (116)", sess)
