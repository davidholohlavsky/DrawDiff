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
import math


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
        # Rozdílné rozměry sjednotíme doplněním bílých okrajů, ne ořezem
        h = max(old_mask.shape[0], new_mask.shape[0])
        w = max(old_mask.shape[1], new_mask.shape[1])

        def pad(img):
            canvas = np.full((h, w), 255, dtype=np.uint8)  # pozadí = bílé
            canvas[: img.shape[0], : img.shape[1]] = img
            return canvas

        old_mask = pad(old_mask)
        new_mask = pad(new_mask)

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
# čisté FFT porovnání hran
# -----------------------------------------------------
def align_images_shift(
    gray_ref: np.ndarray,
    gray_to_align: np.ndarray,
) -> tuple[float, float]:
    """
    Najde posun (dx, dy) mezi dvěma obrázky.

    Princip:
      - zmenší oba obrázky (rychlejší výpočet)
      - spočítá hrany (Canny)
      - použije phaseCorrelate (FFT) nad hranami
      - vrátí posun přepočtený zpět na původní měřítko

    Výhoda:
      - funguje i pro obrázky různé velikosti
      - robustní proti šumu
      - otestované v původním prototypu
    """

    h, w = gray_ref.shape

    # měřítko pro zmenšení (rychlost)
    scale = 0.25

    # zmenšené verze na stejné rozměry
    ref_sm = cv2.resize(
        gray_ref,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA,
    )
    aln_sm = cv2.resize(
        gray_to_align,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA,
    )

    # hrany na zmenšených obrázcích
    ref_e = cv2.Canny(ref_sm, 50, 140)
    aln_e = cv2.Canny(aln_sm, 50, 140)

    # Hanning okno proti artefaktům
    win = cv2.createHanningWindow(
        (ref_e.shape[1], ref_e.shape[0]),
        cv2.CV_64F,
    )

    a = (ref_e.astype(np.float32) * win).astype(np.float32)
    b = (aln_e.astype(np.float32) * win).astype(np.float32)

    # phase correlation – vrátí (dy, dx) ve škále zmenšeniny
    (shift_yx, _resp) = cv2.phaseCorrelate(a, b)  # type: ignore

    dy_sm, dx_sm = shift_yx[1], shift_yx[0]

    # přepočet na plné rozlišení
    dx = float(dx_sm / scale)
    dy = float(dy_sm / scale)

    return dx, dy


# -----------------------------------------------------
# Rozšíří menší obrázek tak, aby oba obrázky měly stejný rozměr.
# -----------------------------------------------------
def pad_to_match(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Rozšíří menší obrázek tak, aby oba obrázky měly stejný rozměr.

    PROBLÉM:
      - PDF výkresy často nemají stejné rozměry (jiné okraje, tiskové rámečky).
      - Funkce phaseCorrelate vyžaduje, aby oba obrázky měly shodnou velikost.
      - Ořezávání většího obrázku je riskantní (hrozí ztráta důležitých dat).

    ŘEŠENÍ:
      - Spočítáme maximální výšku a šířku.
      - Vytvoříme nový bílý plátno (255 = bílé pozadí).
      - Menší obrázek do něj vložíme do levého horního rohu beze změny obsahu.
      - Nikde neškálujeme ani nedeformujeme — jen doplňujeme bílé pixely.

    Výhody:
      - Zachováme 100 % původního obsahu obou výkresů.
      - Získáme dva obrázky se stejnými rozměry → FFT výpočet posunu funguje.
      - Robustní řešení pro stavební PDF s různými okraji.

    Návratová hodnota:
      (obrazek_A_padded, obrazek_B_padded)
    """

    # Získáme největší výšku a šířku, které se musí použít
    h = max(a.shape[0], b.shape[0])
    w = max(a.shape[1], b.shape[1])

    def pad(img: np.ndarray) -> np.ndarray:
        # Nové bílé plátno odpovídající největším rozměrům
        padded = np.full(
            (h, w),
            255,  # bílá výplň
            dtype=img.dtype,  # typ zůstává stejný (uint8)
        )

        # Vložení původního obrázku do levého horního rohu
        padded[: img.shape[0], : img.shape[1]] = img
        return padded

    return pad(a), pad(b)


# -----------------------------------------------------
# Vypočítá posun v ose X a Y zvlášť.
# -----------------------------------------------------


def axis_projection_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """
    Vypočítá posun v ose X a Y zvlášť.

    Metoda:
      - Pro osu X: zprůměruje obraz přes řádky → vznikne 1D signál délky W
      - Pro osu Y: zprůměruje obraz přes sloupce → vznikne 1D signál délky H
      - Udělá korelaci mezi signály → najde shift s nejvyšší shodou
    """

    # Průměr přes řádky (X signál)
    sig_x_a = np.mean(a, axis=0)
    sig_x_b = np.mean(b, axis=0)

    # Průměr přes sloupce (Y signál)
    sig_y_a = np.mean(a, axis=1)
    sig_y_b = np.mean(b, axis=1)

    # Korelace osy X
    corr_x = np.correlate(
        sig_x_a - sig_x_a.mean(), sig_x_b - sig_x_b.mean(), mode="full"
    )

    dx = corr_x.argmax() - (len(sig_x_a) - 1)

    # Korelace osy Y
    corr_y = np.correlate(
        sig_y_a - sig_y_a.mean(), sig_y_b - sig_y_b.mean(), mode="full"
    )

    dy = corr_y.argmax() - (len(sig_y_a) - 1)

    return float(dx), float(dy)


def _run_lengths(
    values: list[bool],
) -> list[tuple[bool, int]]:
    """
    Z binární řady True/False udělá run-length kódování.
    Např. [T,T,F,F,F,T] -> [(T,2), (F,3), (T,1)]
    """
    if not values:
        return []

    runs: list[tuple[bool, int]] = []
    current = values[0]
    length = 1

    for v in values[1:]:
        if v == current:
            length += 1
        else:
            runs.append((current, length))
            current = v
            length = 1

    runs.append((current, length))
    return runs


def _is_axis_dashdot_pattern(
    mask: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    samples: int = 400,
    min_runs: int = 10,
) -> bool:
    """
    Ověří, jestli daný úsek čáry připomíná osu (čárka–tečka).
    """
    h, w = mask.shape

    xs = np.linspace(
        x1,
        x2,
        samples,
    )
    ys = np.linspace(
        y1,
        y2,
        samples,
    )

    vals: list[bool] = []
    for x, y in zip(xs, ys):
        xi = int(round(x))
        yi = int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            vals.append(mask[yi, xi] > 0)
        else:
            vals.append(False)

    runs = _run_lengths(vals)
    if not runs:
        return False

    ones = [length for value, length in runs if value]
    zeros = [length for value, length in runs if not value]

    if len(ones) < min_runs:
        return False

    ones_sorted = sorted(ones)
    k = max(
        1,
        len(ones_sorted) // 3,
    )
    short = ones_sorted[:k]
    long = ones_sorted[-k:]

    if not short or not long:
        return False

    mean_short = sum(short) / len(short)
    mean_long = sum(long) / len(long)

    if mean_long < mean_short * 3.0:
        return False

    if len(zeros) < len(ones) / 2:
        return False

    return True


def _cluster_positions(
    positions: list[float],
    tol: float = 15.0,
) -> list[float]:
    """
    Sloučí blízké pozice čar do jedné „osy“.
    """
    if not positions:
        return []

    positions_sorted = sorted(positions)
    clusters: list[list[float]] = []
    current: list[float] = [positions_sorted[0]]

    for p in positions_sorted[1:]:
        if abs(p - current[-1]) <= tol:
            current.append(p)
        else:
            clusters.append(current)
            current = [p]

    clusters.append(current)

    return [float(sum(c) / len(c)) for c in clusters]


def detect_axes(
    mask: np.ndarray,
    min_line_length: int = 1800,
    max_line_gap: int = 25,
) -> tuple[list[float], list[float]]:
    """
    Najde pravděpodobné osy (čárka–tečka) v binární masce čar.
    Přidána filtrace pro přesnější výběr:
      - povoleny jen úhly 0° ±2° nebo 90° ±2°
      - čáry kratší než min_line_length se ignorují
      - osy s rozestupem menším než 400 px se slučují
    """

    edges = mask
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=90,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    axes_x: list[float] = []
    axes_y: list[float] = []

    if lines is None:
        return [], []

    for line in lines:
        x1, y1, x2, y2 = line[0]  # type: ignore[index]

        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < min_line_length:
            continue

        # úhel čáry v stupních
        angle_deg = math.degrees(math.atan2(dy, dx))
        angle_abs = abs(angle_deg)
        if angle_abs > 90:
            angle_abs = 180 - angle_abs

        is_vertical = 88 <= angle_abs <= 92
        is_horizontal = -2 <= angle_deg <= 2 or 178 <= angle_deg <= 180

        if not (is_vertical or is_horizontal):
            continue

        # kontrola patternu čárka–tečka
        if not _is_axis_dashdot_pattern(mask, x1, y1, x2, y2, samples=600, min_runs=10):
            continue

        if is_vertical:
            x_mid = 0.5 * (x1 + x2)
            axes_x.append(x_mid)
        elif is_horizontal:
            y_mid = 0.5 * (y1 + y2)
            axes_y.append(y_mid)

    # sloučení os, které jsou příliš blízko (např. kóty)
    axes_x_clustered = _cluster_positions(axes_x, tol=40.0)
    axes_y_clustered = _cluster_positions(axes_y, tol=40.0)

    # odstranění "kótových" os – ignoruj pokud jsou blíž než 400 px
    def filter_spacing(values: list[float], min_gap: float = 400.0) -> list[float]:
        if len(values) < 2:
            return values
        filtered = [values[0]]
        for v in values[1:]:
            if all(abs(v - f) >= min_gap for f in filtered):
                filtered.append(v)
        return filtered

    axes_x_final = filter_spacing(sorted(axes_x_clustered))
    axes_y_final = filter_spacing(sorted(axes_y_clustered))

    return axes_x_final, axes_y_final


def detect_axis_markers(mask: np.ndarray) -> list[tuple[int, int, int]]:
    """
    Detekuje kruhové značky (kolečko s číslem osy) na výkrese.
    Výstup: [(x, y, r)] pro všechny nalezené kruhy.
    """
    blurred = cv2.medianBlur(mask, 5)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=80,
        param1=100,
        param2=20,
        minRadius=15,
        maxRadius=60,
    )
    results = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for c in circles[0, :]:  # type: ignore[index]
            x, y, r = c
            results.append((int(x), int(y), int(r)))
    return results


def is_dashdot_pattern(mask: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> bool:
    """
    Ověří, jestli čára mezi (x1,y1)-(x2,y2) je čerchovaná (čárka-tečka-čárka).
    """
    samples = 600
    xs = np.linspace(x1, x2, samples)
    ys = np.linspace(y1, y2, samples)
    h, w = mask.shape
    vals = []
    for x, y in zip(xs, ys):
        xi, yi = int(round(x)), int(round(y))
        vals.append(0 <= xi < w and 0 <= yi < h and mask[yi, xi] > 0)

    runs = []
    current, length = vals[0], 1
    for v in vals[1:]:
        if v == current:
            length += 1
        else:
            runs.append((current, length))
            current, length = v, 1
    runs.append((current, length))

    on_runs = [l for v, l in runs if v]
    off_runs = [l for v, l in runs if not v]
    if len(on_runs) < 5 or len(off_runs) < 4:
        return False

    mean_on = np.mean(on_runs)
    short_on = [l for l in on_runs if l < mean_on * 0.6]
    long_on = [l for l in on_runs if l > mean_on * 1.4]
    if not short_on or not long_on:
        return False
    if np.std(off_runs) > np.mean(off_runs) * 0.5:
        return False
    return True


def detect_axes_with_markers(
    mask: np.ndarray, debug_dir: Path | None = None
) -> tuple[list[float], list[float]]:
    """
    Detekce os s využitím čerchovaného patternu + koleček na konci.
    Výsledek se uloží do debug obrázku, pokud je zadán debug_dir.
    """
    markers = detect_axis_markers(mask)
    lines = cv2.HoughLinesP(
        mask,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=1500,
        maxLineGap=30,
    )
    axes_x, axes_y = [], []
    if lines is None:
        return [], []

    debug_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    for line in lines:
        x1, y1, x2, y2 = line[0]  # type: ignore[index]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1500:
            continue
        angle_deg = abs(math.degrees(math.atan2(dy, dx)))
        if angle_deg > 90:
            angle_deg = 180 - angle_deg
        is_vertical = 88 <= angle_deg <= 92
        is_horizontal = angle_deg <= 2 or angle_deg >= 178
        if not (is_vertical or is_horizontal):
            continue
        if not is_dashdot_pattern(mask, x1, y1, x2, y2):
            continue

        def near_marker(x, y, markers, tol=50):
            for mx, my, r in markers:
                if (x - mx) ** 2 + (y - my) ** 2 <= (r + tol) ** 2:
                    return True
            return False

        if not (near_marker(x1, y1, markers) or near_marker(x2, y2, markers)):
            continue

        color = (0, 255, 255) if is_vertical else (255, 255, 0)
        cv2.line(debug_img, (x1, y1), (x2, y2), color, 1)

        if is_vertical:
            axes_x.append(0.5 * (x1 + x2))
        else:
            axes_y.append(0.5 * (y1 + y2))

    for mx, my, r in markers:
        cv2.circle(debug_img, (mx, my), r, (0, 0, 255), 1)

    if debug_dir:
        debug_path = debug_dir / "debug_axes_markers.png"
        cv2.imwrite(str(debug_path), debug_img)

    return axes_x, axes_y


def debug_draw_axes(
    gray: np.ndarray,
    axes_x: list[float],
    axes_y: list[float],
) -> np.ndarray:
    """
    Vykreslí nalezené osy do kopie šedotónového obrázku.
    """
    rgb = cv2.cvtColor(
        gray,
        cv2.COLOR_GRAY2BGR,
    )
    h, w = gray.shape

    for x in axes_x:
        xi = int(round(x))
        if 0 <= xi < w:
            cv2.line(
                rgb,
                (xi, 0),
                (xi, h - 1),
                (0, 255, 255),
                1,
            )

    for y in axes_y:
        yi = int(round(y))
        if 0 <= yi < h:
            cv2.line(
                rgb,
                (0, yi),
                (w - 1, yi),
                (255, 255, 0),
                1,
            )

    return rgb


def compute_shift_from_axes(
    old_axes_x: list[float],
    old_axes_y: list[float],
    new_axes_x: list[float],
    new_axes_y: list[float],
) -> tuple[float, float]:
    """
    Spočítá posun (dx, dy) mezi dvěma sadami os, robustněji než medián.
    Používá histogram nejčastějších posunů (mode).
    """

    import numpy as np

    def mode_shift(ref, target, bin_size=5):
        if not ref or not target:
            return 0.0
        diffs = []
        for r in ref:
            nearest = min(target, key=lambda t: abs(t - r))
            diffs.append(nearest - r)
        if not diffs:
            return 0.0

        # histogram posunů (zaokrouhlený na bin_size)
        bins = np.arange(
            int(min(diffs)) - bin_size,
            int(max(diffs)) + bin_size,
            bin_size,
        )
        hist, edges = np.histogram(diffs, bins=bins)
        mode_index = np.argmax(hist)
        mode_center = 0.5 * (edges[mode_index] + edges[mode_index + 1])

        return float(mode_center)

    dx = mode_shift(old_axes_x, new_axes_x)
    dy = mode_shift(old_axes_y, new_axes_y)

    return dx, dy


def merge_and_autocrop(
    g_old: np.ndarray, g_new: np.ndarray, dx: float, dy: float
) -> np.ndarray:
    """
    Spojí dva výkresy na společné plátno, posune nový výkres o (dx, dy)
    a nakonec automaticky ořízne oblast, kde je alespoň jeden pixel jiný než bílý.


    Parametry:
    g_old: původní výkres (šedotón)
    g_new: nový výkres (šedotón)
    dx, dy: posun nového výkresu vůči původnímu (v pixelech)


    Vrací:
    Výsledný oříznutý obrázek s oběma výkresy na společném plátně.
    """

    # Rozměry původních výkresů
    h_old, w_old = g_old.shape
    h_new, w_new = g_new.shape

    # Rozšíříme plátno tak, aby se oba výkresy vešly vedle sebe i s posunem
    margin = 200  # rezerva okolo (pixely)
    canvas_w = int(max(w_old, w_new + abs(dx)) + margin * 2)
    canvas_h = int(max(h_old, h_new + abs(dy)) + margin * 2)

    # Vytvoříme bílé plátno pro oba výkresy
    canvas_old = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)
    canvas_new = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)

    # Umístíme starý výkres na střed plátna
    offset_x = margin
    offset_y = margin
    canvas_old[offset_y : offset_y + h_old, offset_x : offset_x + w_old] = g_old

    # Posuneme nový výkres o (dx, dy)
    M = np.float32([[1, 0, dx + offset_x], [0, 1, dy + offset_y]])  # type: ignore[index]
    cv2.warpAffine(
        g_new,
        M,  # type: ignore[index]
        (canvas_w, canvas_h),
        canvas_new,
        borderValue=255,
        flags=cv2.INTER_LINEAR,
    )  # type: ignore[index]

    # Sloučíme vrstvy do RGB pro kontrolu
    overlay = cv2.merge(
        [
            canvas_new,  # červený kanál
            canvas_old,  # zelený kanál
            np.full_like(canvas_new, 255),  # modrý (bílý podklad)
        ]
    )

    # Automatické oříznutí – najdeme oblast, kde nejsou samé bílé pixely
    gray_overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
    mask = gray_overlay < 250  # vše, co není úplně bílé

    coords = cv2.findNonZero(mask.astype(np.uint8))
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        overlay_cropped = overlay[y : y + h, x : x + w]
    else:
        overlay_cropped = overlay

    return overlay_cropped


def merge_on_fixed_canvas(
    g_old: np.ndarray,
    g_new: np.ndarray,
) -> np.ndarray:
    """
    Položí dva výkresy vedle sebe na barevné plátno.
    Levý = starý výkres (zeleně)
    Pravý = nový výkres (červeně)
    """

    h_old, w_old = g_old.shape
    h_new, w_new = g_new.shape

    # převod do masek čar
    old_mask = extract_line_mask(g_old)
    new_mask = extract_line_mask(g_new)

    # rozměry plátna
    canvas_w = w_old + w_new
    canvas_h = max(h_old, h_new)

    # bílé pozadí (RGB)
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    # levá polovina – starý výkres (zelený)
    canvas[:h_old, :w_old][old_mask > 0] = (0, 255, 0)

    # pravá polovina – nový výkres (červený)
    offset_x = w_old
    canvas[:h_new, offset_x : offset_x + w_new][new_mask > 0] = (0, 0, 255)

    return canvas


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

    fixed_canvas = merge_on_fixed_canvas(g_old, g_new)
    cv2.imwrite(str(out_dir / "debug_fixed_canvas.png"), fixed_canvas)

    # --- DETEKCE OS (debugovací krok pro kontrolu) ---
    if DEBUG_SAVE:
        # Masky čar
        old_mask = extract_line_mask(g_old)
        new_mask = extract_line_mask(g_new)

        # Detekce os na obou výkresech
        axes_old_x, axes_old_y = detect_axes_with_markers(old_mask, debug_dir=out_dir)
        axes_new_x, axes_new_y = detect_axes_with_markers(new_mask, debug_dir=out_dir)

        # Konverze do barevných obrázků
        debug_old_axes = cv2.cvtColor(g_old, cv2.COLOR_GRAY2BGR)
        debug_new_axes = cv2.cvtColor(g_new, cv2.COLOR_GRAY2BGR)

        # Nastavení barvy (oranžová v BGR)
        axis_color = (0, 165, 255)  # oranžová (BGR)
        thickness = 3

        # Vykreslení všech os oranžově
        for x in axes_old_x:
            cv2.line(
                debug_old_axes,
                (int(x), 0),
                (int(x), old_mask.shape[0]),
                axis_color,
                thickness,
            )
        for y in axes_old_y:
            cv2.line(
                debug_old_axes,
                (0, int(y)),
                (old_mask.shape[1], int(y)),
                axis_color,
                thickness,
            )

        for x in axes_new_x:
            cv2.line(
                debug_new_axes,
                (int(x), 0),
                (int(x), new_mask.shape[0]),
                axis_color,
                thickness,
            )
        for y in axes_new_y:
            cv2.line(
                debug_new_axes,
                (0, int(y)),
                (new_mask.shape[1], int(y)),
                axis_color,
                thickness,
            )

        # Uložení výstupů
        cv2.imwrite(str(out_dir / "debug_axes_old.png"), debug_old_axes)
        cv2.imwrite(str(out_dir / "debug_axes_new.png"), debug_new_axes)

        print(
            f"Old drawing axes: {len(axes_old_x)} vertical, {len(axes_old_y)} horizontal"
        )
        print(
            f"New drawing axes: {len(axes_new_x)} vertical, {len(axes_new_y)} horizontal"
        )

    return {
        "summary": "Placed side-by-side on 3x3 fixed canvas (no alignment).",
        "overlay_path": str(out_dir / "debug_fixed_canvas.png"),
    }

    if DEBUG_SAVE:
        cv2.imwrite(
            str(out_dir / "debug_01_old_gray.png"),
            g_old,
        )
        cv2.imwrite(
            str(out_dir / "debug_02_new_gray.png"),
            g_new,
        )

    # 2.0) sjednotíme velikost (doplň bílá pole, nic neřežeme)
    g_old_p, g_new_p = pad_to_match(g_old, g_new)

    # 2.1) Spočítej posun mezi starým a novým výkresem
    # detekce os na obou výkresech
    axes_old_x, axes_old_y = detect_axes(extract_line_mask(g_old))
    axes_new_x, axes_new_y = detect_axes(extract_line_mask(g_new))

    # výpočet globálního posunu podle os
    dx, dy = compute_shift_from_axes(
        axes_old_x,
        axes_old_y,
        axes_new_x,
        axes_new_y,
    )
    print(f"Axis-based shift detected: dx={dx:.2f}, dy={dy:.2f}")

    # 2.2) Spojení a automatické oříznutí výkresů na společném plátně
    overlay_canvas = merge_and_autocrop(
        g_old,
        g_new,
        dx,
        dy,
    )

    cv2.imwrite(str(out_dir / "debug_overlay_autocrop.png"), overlay_canvas)

    # 2.3) Aplikuj posun na nový výkres – bez ořezu, zachová plný rozsah

    h_old, w_old = g_old.shape
    h_new, w_new = g_new.shape

    # Maximální rozměry obou výkresů
    canvas_w = max(w_old, w_new)
    canvas_h = max(h_old, h_new)

    # Bílé plátno pro celý rozsah výkresu
    canvas = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)

    # Matice posunu
    M = np.float32([[1, 0, dx], [0, 1, dy]])  # type: ignore

    # Posuneme nový výkres na širší plátno (bez ztráty dat)
    g_new_aligned = cv2.warpAffine(
        g_new,
        M,  # type: ignore
        (canvas_w, canvas_h),
        flags=cv2.INTER_LINEAR,
        borderValue=255,
    )  # type: ignore

    # Starý výkres vložíme do stejného rozměru (žádný crop)
    if g_old.shape != (canvas_h, canvas_w):
        g_old_full = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)
        g_old_full[:h_old, :w_old] = g_old
        g_old = g_old_full

    if DEBUG_SAVE:
        cv2.imwrite(str(out_dir / "debug_new_aligned.png"), g_new_aligned)

    # 2.4) Výpočet masek čar – nejdřív sjednotíme velikost, ať se nic neořízne
    g_old_matched, g_new_aligned_matched = pad_to_match(g_old, g_new_aligned)

    old_mask = extract_line_mask(g_old_matched)
    new_mask = extract_line_mask(g_new_aligned_matched)

    if DEBUG_SAVE:
        cv2.imwrite(
            str(out_dir / "debug_03_old_mask.png"),
            old_mask,
        )
        cv2.imwrite(
            str(out_dir / "debug_04_new_mask.png"),
            new_mask,
        )

    # 2.5) Debug: detekce os na staré i nové výkresu
    axes_old_x, axes_old_y = detect_axes_with_markers(old_mask, debug_dir=out_dir)
    axes_new_x, axes_new_y = detect_axes_with_markers(new_mask, debug_dir=out_dir)

    # Výpočet posunu podle shody os
    dx, dy = compute_shift_from_axes(
        axes_old_x,
        axes_old_y,
        axes_new_x,
        axes_new_y,
    )

    print(f"Axis+marker shift detected: dx={dx:.2f}, dy={dy:.2f}")

    # Debug vizualizace – uloží staré i nové osy s kolečky
    debug_old_axes = cv2.cvtColor(old_mask, cv2.COLOR_GRAY2BGR)
    debug_new_axes = cv2.cvtColor(new_mask, cv2.COLOR_GRAY2BGR)

    for x in axes_old_x:
        cv2.line(
            debug_old_axes, (int(x), 0), (int(x), old_mask.shape[0]), (0, 255, 0), 1
        )
    for y in axes_old_y:
        cv2.line(
            debug_old_axes, (0, int(y)), (old_mask.shape[1], int(y)), (0, 255, 0), 1
        )

    for x in axes_new_x:
        cv2.line(
            debug_new_axes, (int(x), 0), (int(x), new_mask.shape[0]), (0, 0, 255), 1
        )
    for y in axes_new_y:
        cv2.line(
            debug_new_axes, (0, int(y)), (new_mask.shape[1], int(y)), (0, 0, 255), 1
        )

    cv2.imwrite(str(out_dir / "debug_05_old_axes.png"), debug_old_axes)
    cv2.imwrite(str(out_dir / "debug_06_new_axes.png"), debug_new_axes)

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
