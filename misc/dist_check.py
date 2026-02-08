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


def investigate_distribution(p, name, sess):
    img = np.array(Image.open(p).convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    mask = get_mask(p, sess)

    mean = cv2.blur(gray, (5, 5))
    sq_mean = cv2.blur(gray**2, (5, 5))
    lvar = np.maximum(0, sq_mean - mean**2)
    s_var = lvar[mask]

    # Histogram of variance
    # We care about the "Zero floor" (absolute dead zones)
    # and the "Mid-range" (natural grain)
    dead_zones = (s_var < 0.00001).sum() / mask.sum()
    natural_grain = ((s_var > 0.00005) & (s_var < 0.0005)).sum() / mask.sum()
    high_contrast = (s_var > 0.002).sum() / mask.sum()

    print(f"\n--- Distribution for {name} ---")
    print(f"  Dead Zones (<1e-5): {dead_zones:.2%}")
    print(f"  Natural Grain (5e-5 to 5e-4): {natural_grain:.2%}")
    print(f"  High Contrast (>2e-3): {high_contrast:.2%}")
    print(
        f"  Kurtosis of Var: {((s_var - np.mean(s_var)) ** 4).mean() / (np.var(s_var) ** 2):.2f}"
    )


sess = rembg.new_session(providers=["CPUExecutionProvider"])
investigate_distribution("00166-4154668875.png", "CLEAN (00166)", sess)
investigate_distribution(
    "d:/Projects/forensics/test/242-00-shiro_black-5.456.png", "WORST (242)", sess
)
investigate_distribution("116-00-shiro_black-7.595.png", "BLURRY (116)", sess)
