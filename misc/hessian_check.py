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


def investigate_hessian(p, name, sess):
    img = np.array(Image.open(p).convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    mask = get_mask(p, sess)

    # Hessian components
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    dxx = cv2.Sobel(dx, cv2.CV_32F, 1, 0, ksize=3)
    dyy = cv2.Sobel(dy, cv2.CV_32F, 0, 1, ksize=3)
    dxy = cv2.Sobel(dx, cv2.CV_32F, 0, 1, ksize=3)

    # Trace and Determinant for Eigenvalues
    # L1, L2 = (T/2) +/- sqrt((T/2)^2 - D)
    tr = dxx + dyy
    det = dxx * dyy - dxy**2

    # Discriminant
    disc = np.maximum(0, (tr / 2) ** 2 - det)
    l1 = (tr / 2) + np.sqrt(disc)
    l2 = (tr / 2) - np.sqrt(disc)

    # Coherence: (L1 - L2) / (L1 + L2)
    # High coherence = Strong lines/structure. Low = Isotropic Noise.
    coherence = np.abs(l1 - l2) / (np.abs(l1) + np.abs(l2) + 1e-6)

    # Subject stats
    s_coh = coherence[mask & (np.abs(l1) > 0.05)]  # Only look at active detail

    print(f"\n--- Hessian Analysis for {name} ---")
    print(f"  Coherence (Avg on active): {np.mean(s_coh):.4f}")
    print(f"  L1 Mean (Active): {np.mean(np.abs(l1)[mask & (np.abs(l1) > 0.05)]):.4f}")


sess = rembg.new_session(providers=["CPUExecutionProvider"])
investigate_hessian("00166-4154668875.png", "CLEAN (00166)", sess)
investigate_hessian("038-04-noob10b-6.081.png", "NOISY (038)", sess)
investigate_hessian(
    "d:/Projects/forensics/test/242-00-shiro_black-5.456.png", "CLUMPY (242)", sess
)
