# ---------------------------------------------------------
# drawdiff.py – BASIC VERZE
#
# Jednoduché porovnání dvou výkresů v PDF:
#  - PDF -> šedotónový raster (jen první stránka)
#  - z šedotónu vytáhneme čáry (linework)
#  - starý výkres = zelené čáry na bílém podkladu
#  - nový výkres = červené čáry na bílém podkladu
#  - obojí složíme přes sebe do overlay.png
#
# Poznámka:
#  - žádné zarovnávání (align), žádné deskew, žádné HoughLines
#  - cílem je mít stabilní, pochopitelný a funkční základ
# ---------------------------------------------------------

from pathlib import Path
from typing import Tuple

import cv2
import fitz  # PyMuPDF pro čtení PDF
import numpy as np


# ===================== NASTAVENÍ =====================

# Rozlišení pro rasterizaci PDF (vyšší = ostřejší, ale pomalejší)
DPI = 220

# Pokud je True, ukládají se mezikroky do výstupní složky
DEBUG_SAVE = True

# Prah pro jednoduché prahování čar
# (čím nižší číslo, tím citlivější na tmavší pixely)
LINE_THRESHOLD = 200

# =====================================================


# -----------------------------------------------------
# PDF -> šedotónový numpy obrázek
# -----------------------------------------------------
def pdf_page_to_gray(
    pdf_path: Path,
    dpi: int = DPI,
) -> np.ndarray:
    """
    Načte první stránku PDF jako šedotónový obrázek (numpy pole).

    Kroky:
      1) Otevře PDF pomocí PyMuPDF.
      2) Vyrenderuje první stránku na zadané DPI.
      3) Výsledek převede na 2D numpy pole (výška x šířka, uint8).
    """

    doc = fitz.open(pdf_path)
    page = doc.load_page(0)

    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    # Rastrování přímo do šedotónového obrázku
    pix = page.get_pixmap(
        matrix=mat,
        alpha=False,
        colorspace=fitz.csGRAY,
    )

    img = np.frombuffer(
        pix.samples,
        dtype=np.uint8,
    ).reshape(
        pix.height,
        pix.width,
    )

    return img


# -----------------------------------------------------
# Extrakce jednoduché masky čar
# -----------------------------------------------------
def extract_line_mask(
    gray: np.ndarray,
    threshold: int = LINE_THRESHOLD,
) -> np.ndarray:
    """
    Vytáhne čáry z šedotónového obrázku.

    Princip:
      - lehce rozmažeme, aby zmizel jemný šum
      - použijeme jednoduché prahování:
          tmavé pixely -> 255 (čára)
          světlé pixely -> 0  (pozadí)
    Výstup:
      - binární maska typu uint8, hodnoty 0 nebo 255
    """

    # Odstranění drobného šumu (ne agresivní)
    blur = cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
    )

    # Jednoduché globální prahování
    _ret, mask = cv2.threshold(
        blur,
        threshold,
        255,
        cv2.THRESH_BINARY_INV,
    )

    return mask


# -----------------------------------------------------
# Složení barevného overlay
# -----------------------------------------------------
def compose_overlay(
    old_mask: np.ndarray,
    new_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Složí barevný overlay ze dvou masek čar.

    Vstup:
      - old_mask: binární maska čar z původního výkresu
      - new_mask: binární maska čar z nového výkresu

    Výstup:
      - overlay: RGB obrázek:
          staré čáry = zelené
          nové čáry = červené (přepisují zelené)
      - change_mask:
          XOR obou masek – kde se liší, tam je změna
    """

    if old_mask.shape != new_mask.shape:
        # Bez zarovnávání by se to stát nemělo,
        # ale pro jistotu ořežeme na společný minimální rozměr.
        h = min(old_mask.shape[0], new_mask.shape[0])
        w = min(old_mask.shape[1], new_mask.shape[1])
        old_mask = old_mask[:h, :w]
        new_mask = new_mask[:h, :w]
    else:
        h, w = old_mask.shape

    # Bílý podklad
    canvas = np.full(
        (h, w, 3),
        255,
        dtype=np.uint8,
    )

    # Staré čáry -> zelená
    canvas[old_mask > 0] = (0, 255, 0)

    # Nové čáry -> červená (přepisují zelenou)
    canvas[new_mask > 0] = (0, 0, 255)

    # Kde se masky liší, tam je změna (XOR)
    change_mask = cv2.bitwise_xor(
        old_mask,
        new_mask,
    )

    return canvas, change_mask


# -----------------------------------------------------
# Hlavní funkce, kterou volá FastAPI endpoint
# -----------------------------------------------------
def run_drawdiff(
    old_pdf: Path,
    new_pdf: Path,
    out_dir: Path,
) -> dict:
    """
    Hlavní vstupní bod pro server (FastAPI).

    Vstup:
      - old_pdf: cesta k původnímu PDF
      - new_pdf: cesta k novému PDF
      - out_dir: výstupní adresář pro konkrétní job

    Kroky:
      1) Vytvoří výstupní složku.
      2) Načte z obou PDF první stránku jako šedotón.
      3) Z obou šedotónů vytvoří binární masky čar.
      4) Složí barevný overlay (starý = zeleně, nový = červeně).
      5) Uloží overlay.png do výstupní složky.
      6) Vrátí souhrnné info jako slovník pro API.
    """

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 1) Načtení prvních stran obou PDF jako šedotón
    g_old = pdf_page_to_gray(
        old_pdf,
        dpi=DPI,
    )
    g_new = pdf_page_to_gray(
        new_pdf,
        dpi=DPI,
    )

    if DEBUG_SAVE:
        cv2.imwrite(
            str(out_dir / "debug_01_old_gray.png"),
            g_old,
        )
        cv2.imwrite(
            str(out_dir / "debug_02_new_gray.png"),
            g_new,
        )

    # 2) Výpočet masek čar
    old_mask = extract_line_mask(g_old)
    new_mask = extract_line_mask(g_new)

    if DEBUG_SAVE:
        cv2.imwrite(
            str(out_dir / "debug_03_old_mask.png"),
            old_mask,
        )
        cv2.imwrite(
            str(out_dir / "debug_04_new_mask.png"),
            new_mask,
        )

    # 3) Složení overlay obrázku + masky změn
    overlay, change_mask = compose_overlay(
        old_mask,
        new_mask,
    )

    # 4) Uložení výsledného obrázku
    overlay_path = out_dir / "overlay.png"
    cv2.imwrite(
        str(overlay_path),
        overlay,
    )

    # 5) Úklid paměti (pro jistotu)
    try:
        cv2.destroyAllWindows()
        import gc

        gc.collect()
    except Exception:
        pass

    # 6) Výstup pro API
    return {
        "summary": (f"Basic lines-only diff (DPI={DPI}, threshold={LINE_THRESHOLD})."),
        "change_pixels": int(
            np.count_nonzero(change_mask),
        ),
        "overlay_path": str(overlay_path),
    }
