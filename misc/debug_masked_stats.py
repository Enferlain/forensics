import cv2
import numpy as np
from PIL import Image
import rembg
from skimage.restoration import denoise_tv_chambolle
import io


def get_mask(p, sess):
    img = Image.open(p).convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    out = rembg.remove(buffer.getvalue(), session=sess)
    mask = np.array(Image.open(io.BytesIO(out)))[:, :, 3] > 128
    return mask


def stats(p, sess):
    img = np.array(Image.open(p).convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    mask = get_mask(p, sess)

    d = denoise_tv_chambolle(gray, weight=0.1)
    res = np.abs(gray - d)

    res_mean = cv2.blur(res, (5, 5))
    res_sq_mean = cv2.blur(res**2, (5, 5))
    lvar = np.maximum(0, res_sq_mean - res_mean**2)

    lv_mean = cv2.blur(lvar, (11, 11))
    lv_sq_mean = cv2.blur(lvar**2, (11, 11))
    lv_var = np.maximum(0, lv_sq_mean - lv_mean**2)

    print(f"Stats for {p} (MASKED):")
    print(f"  Res Energy: {np.mean(res[mask]):.6f}")
    print(f"  Clumpiness: {np.mean(lv_var[mask]):.10f}")


sess = rembg.new_session(providers=["CPUExecutionProvider"])
stats("116-00-shiro_black-7.595.png", sess)
stats("00166-4154668875.png", sess)
