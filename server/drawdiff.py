from pathlib import Path
from typing import Tuple

import cv2
import fitz  # PyMuPDF
import numpy as np

# ================== RYCHLÉ NASTAVENÍ ==================
DPI = 220  # rychlý raster (případně 260–300 pro citlivější linky)
DEBUG_SAVE = True  # ukládá mezivýstupy pro diagnostiku
SAFE_FALLBACK_MIN = 2000  # když masky čar mají méně než ~2000 px, použije se SAFE režim
# ======================================================


# ---------- PDF -> GRAY image (rychlejší než RGB) ----------
def pdf_page_to_gray(pdf_path: Path, dpi: int = DPI) -> np.ndarray:
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)  # 0 = první strana
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False, colorspace=fitz.csGRAY)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    return img


# ---------- Lehké narovnání stránky (deskew) ----------
def normalize_page(gray: np.ndarray) -> np.ndarray:
    g = gray
    # kontrast čar
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(g)

    edges = cv2.Canny(g, 60, 180)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=400)
    if lines is None:
        return gray

    # přibližný úhel k 0°/90°
    angles = []
    for rho, theta in lines[:, 0]:
        deg = theta * 180.0 / np.pi
        d0 = (deg % 180) - 0
        d90 = (deg % 180) - 90
        d = d0 if abs(d0) < abs(d90) else d90
        angles.append(d)
    mean_angle = float(np.median(angles))

    # dorovnání jen když je odchylka znatelná
    if abs(mean_angle) < 0.35:
        return gray

    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D(center=(w / 2, h / 2), angle=mean_angle, scale=1.0)
    rot = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
    return rot


# ---------- Odstranění rámu (crop), aby posun neovlivňovaly okraje ----------
def safe_crop(gray: np.ndarray, pad: int = 40) -> np.ndarray:
    h, w = gray.shape[:2]
    pad = max(0, min(pad, h // 10, w // 10))
    return gray[pad : h - pad, pad : w - pad]


# ---------- Rychlé zarovnání pouze posunem (dx, dy) přes phase correlation ----------
def align_images_shift(
    gray_ref: np.ndarray, gray_to_align: np.ndarray, return_shift: bool = False
):
    h, w = gray_ref.shape[:2]

    # downscale pro robustní a rychlý odhad
    s = 0.25
    ref_sm = cv2.resize(
        gray_ref, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA
    )
    aln_sm = cv2.resize(
        gray_to_align, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA
    )

    # hrany → ignorujeme plochy a šrafy tlačíme dolů
    ref_e = cv2.Canny(ref_sm, 50, 140)
    aln_e = cv2.Canny(aln_sm, 50, 140)

    # Hann okno zmenšuje vliv okrajových artefaktů
    win = cv2.createHanningWindow((ref_e.shape[1], ref_e.shape[0]), cv2.CV_64F)
    a = (ref_e.astype(np.float32) * win).astype(np.float32)
    b = (aln_e.astype(np.float32) * win).astype(np.float32)

    # phase correlace vrací (dy, dx) v pořadí (y, x)
    (shift_yx, _resp) = cv2.phaseCorrelate(a, b)
    dy_sm, dx_sm = shift_yx[1], shift_yx[0]
    dx, dy = dx_sm / s, dy_sm / s

    # aplikace na plné rozlišení
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    aligned = cv2.warpAffine(
        gray_to_align, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=255
    )

    if return_shift:
        return aligned, float(dx), float(dy)
    return aligned


# ---------- Extrakce line-art (čáry) ----------
def extract_line_mask(
    gray: np.ndarray,
    block_size: int = 23,  # menší okno = citlivější
    C: int = 12,
    canny_lo: int = 35,  # uvolněné prahy aby „něco“ chytly
    canny_hi: int = 110,
) -> np.ndarray:
    # lehké odšumění bez ztráty hran
    blur = cv2.bilateralFilter(gray, 5, 45, 45)

    # zvýraznit tenké linky (lokální kontrast)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(blur)

    # adaptivní práh: tmavé čáry -> 1
    bw = cv2.adaptiveThreshold(
        src=eq,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=block_size,
        C=C,
    )

    # hrany (pro velmi tenké vektory)
    edges = cv2.Canny(eq, canny_lo, canny_hi)

    mask = cv2.bitwise_or(bw, edges)

    # potlač tečky/šrafy
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    return mask


# ---------- Kompozice „jen čáry“ ----------
def compose_lines_only(
    gray_old: np.ndarray, gray_new_aligned: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    h = min(gray_old.shape[0], gray_new_aligned.shape[0])
    w = min(gray_old.shape[1], gray_new_aligned.shape[1])
    g_old = gray_old[:h, :w]
    g_new = gray_new_aligned[:h, :w]

    old_mask = extract_line_mask(g_old)
    new_mask = extract_line_mask(g_new)

    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    canvas[old_mask > 0] = (0, 255, 0)  # zelená = původní
    canvas[new_mask > 0] = (0, 0, 255)  # červená = nová (přepisuje)

    change_mask = cv2.bitwise_xor(old_mask, new_mask)
    return canvas, change_mask


# ---------- Hlavní běh ----------
def run_drawdiff(old_pdf: Path, new_pdf: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) rychlá rasterizace do gray
    g_old = pdf_page_to_gray(old_pdf, dpi=DPI)
    g_new = pdf_page_to_gray(new_pdf, dpi=DPI)

    # 2) jemný deskew (narovnání)
    g_old = normalize_page(g_old)
    g_new = normalize_page(g_new)

    # 3) crop okrajů (rám neovlivní korelaci)
    g_old_c = safe_crop(g_old, 40)
    g_new_c = safe_crop(g_new, 40)

    # 4) odhad posunu na cropech; použij na plné rozlišení
    _, dx, dy = align_images_shift(g_old_c, g_new_c, return_shift=True)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    g_new_aln = cv2.warpAffine(
        g_new,
        M,
        (g_new.shape[1], g_new.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderValue=255,
    )

    # 5) DEBUG mezivýstupy
    if DEBUG_SAVE:
        cv2.imwrite(str(out_dir / "debug_01_old_gray.png"), g_old)
        cv2.imwrite(str(out_dir / "debug_02_new_aligned_gray.png"), g_new_aln)

    # 6) Primární masky čar
    old_mask = extract_line_mask(g_old)
    new_mask = extract_line_mask(g_new_aln)

    if DEBUG_SAVE:
        cv2.imwrite(str(out_dir / "debug_03_old_mask.png"), old_mask)
        cv2.imwrite(str(out_dir / "debug_04_new_mask.png"), new_mask)

    # 7) SAFE fallback – pokud „nevidíme“ téměř žádné čáry
    need_safe = (cv2.countNonZero(old_mask) < SAFE_FALLBACK_MIN) and (
        cv2.countNonZero(new_mask) < SAFE_FALLBACK_MIN
    )
    if need_safe:
        SAFE_DPI = 260
        g_old = pdf_page_to_gray(old_pdf, dpi=SAFE_DPI)
        g_new = pdf_page_to_gray(new_pdf, dpi=SAFE_DPI)

        # SAFE: bez deskew/crop (zjednodušení, rychlejší)
        _, dx, dy = align_images_shift(g_old, g_new, return_shift=True)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        g_new_aln = cv2.warpAffine(
            g_new,
            M,
            (g_new.shape[1], g_new.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderValue=255,
        )

        old_mask = extract_line_mask(
            g_old, block_size=21, C=10, canny_lo=25, canny_hi=95
        )
        new_mask = extract_line_mask(
            g_new_aln, block_size=21, C=10, canny_lo=25, canny_hi=95
        )

        if DEBUG_SAVE:
            cv2.imwrite(str(out_dir / "debug_SAFE_01_old_gray.png"), g_old)
            cv2.imwrite(str(out_dir / "debug_SAFE_02_new_aligned_gray.png"), g_new_aln)
            cv2.imwrite(str(out_dir / "debug_SAFE_03_old_mask.png"), old_mask)
            cv2.imwrite(str(out_dir / "debug_SAFE_04_new_mask.png"), new_mask)

    # 8) Kompozice „jen čáry“
    h, w = g_old.shape[:2]
    overlay = np.full((h, w, 3), 255, dtype=np.uint8)
    overlay[old_mask > 0] = (0, 255, 0)
    overlay[new_mask > 0] = (0, 0, 255)

    # 9) Debug – hrany pro rychlou vizuální kontrolu
    if DEBUG_SAVE:
        cv2.imwrite(str(out_dir / "debug_edges_ref.png"), cv2.Canny(g_old, 50, 140))
        cv2.imwrite(
            str(out_dir / "debug_edges_aligned.png"), cv2.Canny(g_new_aln, 50, 140)
        )

    # 10) Ulož výsledek
    overlay_path = out_dir / "overlay.png"
    cv2.imwrite(str(overlay_path), overlay)

    # úklid paměti mezi joby
    try:
        cv2.destroyAllWindows()
        import gc

        gc.collect()
    except Exception:
        pass

    return {
        "summary": f"FAST lines-only diff (DPI={DPI}, safe_fallback={'ON' if need_safe else 'OFF'}).",
        "change_pixels": int(np.count_nonzero(cv2.bitwise_xor(old_mask, new_mask))),
        "overlay_path": str(overlay_path),
    }
