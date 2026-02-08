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


def investigate_bands(p, name, sess):
    img = np.array(Image.open(p).convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    mask = get_mask(p, sess)

    # High Freq (Laplacian / Detail)
    hf = np.abs(cv2.Laplacian(gray, cv2.CV_32F))

    # Mid Freq (Gaussian Difference - e.g. 1px vs 5px)
    # This captures the "viscous clumps" (3-7px scale)
    gauss1 = cv2.GaussianBlur(gray, (0, 0), 1.0)
    gauss5 = cv2.GaussianBlur(gray, (0, 0), 5.0)
    mf = np.abs(gauss1 - gauss5)

    # Low Freq (Base shading)
    lf = cv2.GaussianBlur(gray, (0, 0), 10.0)

    s_hf = np.mean(hf[mask])
    s_mf = np.mean(mf[mask])

    # Ratio HF/MF.
    # Sharp/Clean images should have high energy in HF (grain/clean edges)
    # vs MF (blobs).
    ratio = s_hf / (s_mf + 1e-6)

    print(f"\n--- Band Analysis for {name} ---")
    print(f"  HF Energy (Avg): {s_hf:.6f}")
    print(f"  MF Energy (Avg): {s_mf:.6f}")
    print(f"  HF/MF Ratio: {ratio:.4f}")


sess = rembg.new_session(providers=["CPUExecutionProvider"])
investigate_bands("00166-4154668875.png", "CLEAN (00166)", sess)
investigate_bands(
    "d:/Projects/forensics/test/242-00-shiro_black-5.456.png", "WORST (242)", sess
)
investigate_bands("116-00-shiro_black-7.595.png", "BLURRY (116)", sess)
