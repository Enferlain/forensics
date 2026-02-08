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


def check_one(p, sess):
    img = np.array(Image.open(p).convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    mask = get_mask(p, sess)

    hf = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    gauss1 = cv2.GaussianBlur(gray, (0, 0), 1.0)
    gauss5 = cv2.GaussianBlur(gray, (0, 0), 5.0)
    mf = np.abs(gauss1 - gauss5)

    s_hf = np.mean(hf[mask])
    s_mf = np.mean(mf[mask])
    ratio = s_hf / (s_mf + 1e-6)

    print(f"\nStats for {p}:")
    print(f"  HF: {s_hf:.6f}")
    print(f"  MF: {s_mf:.6f}")
    print(f"  Ratio: {ratio:.4f}")


sess = rembg.new_session(providers=["CPUExecutionProvider"])
check_one("00166-4154668875.png", sess)
check_one("038-04-noob10b-6.081.png", sess)
