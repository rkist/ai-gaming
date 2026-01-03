from __future__ import annotations

from typing import Dict, Optional, List

import cv2
import numpy as np


def extract_hud(frame: np.ndarray) -> dict:
    """
    Extract HUD info (time, score, lives, coins) from a frame using simple template OCR.
    Coordinates are approximated from SNES 256x224 base resolution and scaled.
    """
    h, w, _ = frame.shape
    sx = w / 256.0
    sy = h / 224.0

    def crop(x, y, cw, ch):
        return frame[
            int(y * sy) : int((y + ch) * sy),
            int(x * sx) : int((x + cw) * sx),
        ]

    # Regions (approx, based on SMW HUD layout):
    # lives near top-left after the "x" icon
    lives_region = crop(56, 8, 24, 18)
    # time digits (3) after TIME text (top-right)
    time_region = crop(200, 10, 40, 18)
    # coin count near top-right under coin icon
    coins_region = crop(200, 18, 32, 18)
    # score (up to 6 digits) just below coins
    score_region = crop(188, 36, 72, 20)

    lives_digits = _extract_digits(lives_region, 1)
    time_digits = _extract_digits(time_region, 3)
    coins_digits = _extract_digits(coins_region, 2)
    score_digits = _extract_digits(score_region, 6)

    lives = int(lives_digits[0]) if lives_digits and lives_digits[0].isdigit() else None
    time = int("".join(time_digits)) if time_digits and all(d.isdigit() for d in time_digits) else None
    coins = int("".join(coins_digits)) if coins_digits and all(d.isdigit() for d in coins_digits) else None
    score = int("".join(score_digits)) if score_digits and all(d.isdigit() for d in score_digits) else None

    return {"time": time, "score": score, "lives": lives, "coins": coins}


def _digit_templates() -> Dict[str, np.ndarray]:
    """
    Rough SMW HUD digit templates (binary) drawn as 12x16 masks.
    They need not be perfect; resized matching is tolerant.
    """
    raw = {
        "0": [
            " 1111     ",
            "11  11    ",
            "11  11    ",
            "11  11    ",
            "11  11    ",
            "11  11    ",
            "11  11    ",
            "11  11    ",
            "11  11    ",
            "11  11    ",
            "11  11    ",
            "11  11    ",
            "11  11    ",
            " 1111     ",
        ],
        "1": [
            "  11      ",
            " 111      ",
            "11 1      ",
            "   1      ",
            "   1      ",
            "   1      ",
            "   1      ",
            "   1      ",
            "   1      ",
            "   1      ",
            "   1      ",
            "   1      ",
            "   1      ",
            " 1111     ",
        ],
        "2": [
            " 1111     ",
            "11  11    ",
            "    11    ",
            "    11    ",
            "    11    ",
            "   11     ",
            "  11      ",
            " 11       ",
            "11        ",
            "11        ",
            "11        ",
            "11        ",
            "11        ",
            "111111    ",
        ],
        "3": [
            " 1111     ",
            "11  11    ",
            "    11    ",
            "    11    ",
            "    11    ",
            "  111     ",
            "    11    ",
            "    11    ",
            "    11    ",
            "    11    ",
            "    11    ",
            "    11    ",
            "11  11    ",
            " 1111     ",
        ],
        "4": [
            "   111    ",
            "  1 11    ",
            " 1  11    ",
            "1   11    ",
            "1   11    ",
            "1   11    ",
            "1   11    ",
            "111111    ",
            "    11    ",
            "    11    ",
            "    11    ",
            "    11    ",
            "    11    ",
            "    11    ",
        ],
        "5": [
            "111111    ",
            "11        ",
            "11        ",
            "11        ",
            "11        ",
            "11111     ",
            "    11    ",
            "    11    ",
            "    11    ",
            "    11    ",
            "    11    ",
            "    11    ",
            "11  11    ",
            " 1111     ",
        ],
        "6": [
            " 1111     ",
            "11  11    ",
            "11        ",
            "11        ",
            "11        ",
            "11111     ",
            "11  11    ",
            "11  11    ",
            "11  11    ",
            "11  11    ",
            "11  11    ",
            "11  11    ",
            "11  11    ",
            " 1111     ",
        ],
        "7": [
            "111111    ",
            "    11    ",
            "    11    ",
            "   11     ",
            "   11     ",
            "   11     ",
            "  11      ",
            "  11      ",
            "  11      ",
            " 11       ",
            " 11       ",
            " 11       ",
            " 11       ",
            " 11       ",
        ],
        "8": [
            " 1111     ",
            "11  11    ",
            "11  11    ",
            "11  11    ",
            "11  11    ",
            " 1111     ",
            "11  11    ",
            "11  11    ",
            "11  11    ",
            "11  11    ",
            "11  11    ",
            "11  11    ",
            "11  11    ",
            " 1111     ",
        ],
        "9": [
            " 1111     ",
            "11  11    ",
            "11  11    ",
            "11  11    ",
            "11  11    ",
            "11  11    ",
            "11  11    ",
            " 11111    ",
            "    11    ",
            "    11    ",
            "    11    ",
            "    11    ",
            "11  11    ",
            " 1111     ",
        ],
    }
    tmpl = {}
    for digit, lines in raw.items():
        arr = np.array([[1 if ch == "1" else 0 for ch in line] for line in lines], dtype=np.uint8)
        tmpl[digit] = cv2.resize(arr, (12, 16), interpolation=cv2.INTER_NEAREST)
    return tmpl


def _extract_digits(region: np.ndarray, count: int, threshold: int = 160) -> Optional[List[str]]:
    """
    Extract 'count' digits from a HUD subregion using simple binarize + template match.
    Returns list of digit characters or None if detection fails.
    """
    if region is None or region.size == 0:
        return None
    gray = cv2.cvtColor(region, cv2.COLOR_RGBA2GRAY)
    _, bw = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # find contours/digits
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes = sorted(boxes, key=lambda b: b[0])  # left-to-right

    # If we didn't get expected count, bail
    if len(boxes) != count:
        # fallback: split evenly
        h, w = bw.shape
        step = max(1, w // max(1, count))
        boxes = [(i * step, 0, step, h) for i in range(count)]

    tmpl = _digit_templates()
    digits: List[str] = []
    for (x, y, w, h) in boxes[:count]:
        digit_img = bw[y : y + h, x : x + w]
        if digit_img.size == 0:
            digits.append("?")
            continue
        digit_img = cv2.resize(digit_img, (12, 16), interpolation=cv2.INTER_AREA)
        best_char = None
        best_score = 1e9
        for ch, ref in tmpl.items():
            diff = np.mean(np.abs(digit_img.astype(np.int16) - ref * 255))
            if diff < best_score:
                best_score = diff
                best_char = ch
        digits.append(best_char or "?")
    return digits

