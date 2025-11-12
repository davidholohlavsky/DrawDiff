from pathlib import Path
from typing import Tuple

import cv2
import fitz  # PyMuPDF
import numpy as np


# ---------- PDF -> RGB image ----------
def pdf_page_to_image(pdf_path: Path, dpi: int = 400) -> np.ndarray:
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    return img


# ---------- alignment (registration) using ORB + homography ----------
def align_images(img_ref: np.ndarray, img_to_align: np.ndarray) -> np.ndarray:
    """
    Align `img_to_align` to `img_ref` via ORB features and homography.
    Returns the warped (aligned) image with the same size as `img_ref`.
    """
    gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    gray_align = cv2.cvtColor(img_to_align, cv2.COLOR_BGR2GRAY)

    # more features -> better robustness on technical drawings
    orb = cv2.ORB_create(2000)
    kp1, des1 = orb.detectAndCompute(gray_ref, None)
    kp2, des2 = orb.detectAndCompute(gray_align, None)

    if des1 is None or des2 is None:
        # not enough features, return original
        return img_to_align

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    if len(matches) < 10:
        # too few matches for a stable homography
        return img_to_align

    matches = sorted(matches, key=lambda m: m.distance)[:500]

    pts_ref = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_aln = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, _mask = cv2.findHomography(
        pts_aln, pts_ref, method=cv2.RANSAC, ransacReprojThreshold=5.0
    )
    if H is None:
        return img_to_align

    aligned = cv2.warpPerspective(
        img_to_align, H, (img_ref.shape[1], img_ref.shape[0]), flags=cv2.INTER_LINEAR
    )
    return aligned


# ---------- line-art extraction (binary mask of lines only) ----------
def extract_line_mask(
    gray: np.ndarray,
    block_size: int = 25,  # must be odd; 21–31 works well
    C: int = 15,
    canny_lo: int = 50,
    canny_hi: int = 150,
) -> np.ndarray:
    # strengthen thin lines via local contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)

    # adaptive threshold -> dark strokes -> 1
    bw = cv2.adaptiveThreshold(
        src=eq,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=block_size,
        C=C,
    )

    # edges help to preserve very thin vectors
    edges = cv2.Canny(eq, canny_lo, canny_hi)

    mask = cv2.bitwise_or(bw, edges)

    # light denoise + optional thinning
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.erode(mask, k, iterations=1)

    return mask


# ---------- compose: lines only (green old on top, red new overwrites) ----------
def compose_lines_only(
    img_old: np.ndarray, img_new_aligned: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    h = min(img_old.shape[0], img_new_aligned.shape[0])
    w = min(img_old.shape[1], img_new_aligned.shape[1])
    old = img_old[:h, :w]
    new = img_new_aligned[:h, :w]

    gray_old = cv2.cvtColor(old, cv2.COLOR_BGR2GRAY)
    gray_new = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)

    old_mask = extract_line_mask(gray_old)
    new_mask = extract_line_mask(gray_new)

    # white canvas
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    # draw old lines green
    canvas[old_mask > 0] = (0, 255, 0)

    # draw new lines red (overwrite green where both present)
    canvas[new_mask > 0] = (0, 0, 255)

    # change metric: XOR of masks
    change_mask = cv2.bitwise_xor(old_mask, new_mask)

    return canvas, change_mask


# ---------- main entry ----------
def run_drawdiff(old_pdf: Path, new_pdf: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) load first pages
    img_old = pdf_page_to_image(pdf_path=old_pdf, dpi=400)
    img_new = pdf_page_to_image(pdf_path=new_pdf, dpi=400)

    # 2) align NEW to OLD
    img_new_aligned = align_images(img_ref=img_old, img_to_align=img_new)

    # 3) compose lines-only overlay (green old, red new overwriting)
    overlay, change_mask = compose_lines_only(
        img_old=img_old, img_new_aligned=img_new_aligned
    )

    # 4) save result
    overlay_path = out_dir / "overlay.png"
    cv2.imwrite(str(overlay_path), overlay)

    return {
        "summary": "Lines-only diff with alignment "
        "(green=original on top, red=new overwrites).",
        "change_pixels": int(np.count_nonzero(change_mask)),
        "overlay_path": str(overlay_path),
    }
