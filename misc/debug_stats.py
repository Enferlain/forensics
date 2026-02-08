import cv2
import numpy as np
from PIL import Image
from skimage.restoration import denoise_tv_chambolle
import sys


def analyze(p):
    img = np.array(Image.open(p).convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    # Residual
    d = denoise_tv_chambolle(gray, weight=0.1)
    res = np.abs(gray - d)

    # Local Var
    mean = cv2.blur(res, (5, 5))
    sq_mean = cv2.blur(res**2, (5, 5))
    lvar = np.maximum(0, sq_mean - mean**2)

    # Local Var of Var (Clumpiness)
    lv_mean = cv2.blur(lvar, (11, 11))
    lv_sq_mean = cv2.blur(lvar**2, (11, 11))
    lv_var = np.maximum(0, lv_sq_mean - lv_mean**2)

    print(f"Stats for {p}:")
    print(f"  Res Mean: {np.mean(res):.6f}")
    print(f"  LVar Mean: {np.mean(lvar):.6f}")
    print(f"  LVarVar Mean (Clumpiness): {np.mean(lv_var):.10f}")
    print(f"  LVarVar Max: {np.max(lv_var):.10f}")


analyze("116-00-shiro_black-7.595.png")
analyze("00166-4154668875.png")
