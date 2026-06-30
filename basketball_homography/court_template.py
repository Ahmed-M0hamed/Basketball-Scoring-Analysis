"""
Basketball Court Template & Keypoint Definitions
=================================================
Defines all standard court keypoints on a canonical 2D template.
NBA standard court: 94ft × 50ft  →  940 × 500 px  (10 px/ft)

Keypoint IDs are stable — every downstream module refers to them by ID,
so you can safely add new IDs at the end without breaking anything.
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import IntEnum


# ─────────────────────────────────────────────────────────────────────────────
# Court dimensions  (pixels, 10 px / ft)
# ─────────────────────────────────────────────────────────────────────────────
SCALE          = 10          # px per foot
COURT_W        = 94 * SCALE  # 940 px  (length, x-axis)
COURT_H        = 50 * SCALE  # 500 px  (width,  y-axis)
MARGIN         = 60          # drawing margin (px)
IMG_W          = COURT_W + 2 * MARGIN
IMG_H          = COURT_H + 2 * MARGIN


# ─────────────────────────────────────────────────────────────────────────────
# Stable keypoint IDs
# ─────────────────────────────────────────────────────────────────────────────
class KP(IntEnum):
    # ── Corners ──────────────────────────────────────────────────────────────
    CORNER_BL        =  0   # bottom-left  corner
    CORNER_BR        =  1   # bottom-right corner
    CORNER_TR        =  2   # top-right    corner
    CORNER_TL        =  3   # top-left     corner

    # ── Half-court ───────────────────────────────────────────────────────────
    HALF_BOTTOM      =  4   # half-court line, bottom sideline
    HALF_TOP         =  5   # half-court line, top    sideline
    CENTER_CIRCLE_B  =  6   # center circle, bottom  tangent
    CENTER_CIRCLE_T  =  7   # center circle, top     tangent
    CENTER_CIRCLE_L  =  8   # center circle, left    tangent
    CENTER_CIRCLE_R  =  9   # center circle, right   tangent
    CENTER_DOT       = 10   # jump-ball dot

    # ── Left paint / key ─────────────────────────────────────────────────────
    L_PAINT_BL       = 11   # left paint, bottom-left  (baseline)
    L_PAINT_BR       = 12   # left paint, bottom-right (baseline)
    L_PAINT_TR       = 13   # left paint, top-right
    L_PAINT_TL       = 14   # left paint, top-left
    L_FT_LINE_B      = 15   # left free-throw line, bottom end
    L_FT_LINE_T      = 16   # left free-throw line, top    end
    L_FT_CIRCLE_B    = 17   # left FT circle, bottom tangent
    L_FT_CIRCLE_T    = 18   # left FT circle, top    tangent
    L_BASKET         = 19   # left basket centre
    L_BACKBOARD_B    = 20   # left backboard, bottom end
    L_BACKBOARD_T    = 21   # left backboard, top    end
    L_RESTRICTED_B   = 22   # left restricted-area arc, bottom
    L_RESTRICTED_T   = 23   # left restricted-area arc, top

    # ── Left 3-point line ────────────────────────────────────────────────────
    L_3PT_CORNER_B   = 24   # left 3-pt corner, bottom (baseline)
    L_3PT_CORNER_T   = 25   # left 3-pt corner, top    (baseline)
    L_3PT_ARC_B      = 26   # left 3-pt arc, bottom side tangent
    L_3PT_ARC_T      = 27   # left 3-pt arc, top    side tangent
    L_3PT_TOP        = 28   # left 3-pt arc, apex (furthest point)

    # ── Right paint / key ────────────────────────────────────────────────────
    R_PAINT_BL       = 29
    R_PAINT_BR       = 30
    R_PAINT_TR       = 31
    R_PAINT_TL       = 32
    R_FT_LINE_B      = 33
    R_FT_LINE_T      = 34
    R_FT_CIRCLE_B    = 35
    R_FT_CIRCLE_T    = 36
    R_BASKET         = 37
    R_BACKBOARD_B    = 38
    R_BACKBOARD_T    = 39
    R_RESTRICTED_B   = 40
    R_RESTRICTED_T   = 41

    # ── Right 3-point line ───────────────────────────────────────────────────
    R_3PT_CORNER_B   = 42
    R_3PT_CORNER_T   = 43
    R_3PT_ARC_B      = 44
    R_3PT_ARC_T      = 45
    R_3PT_TOP        = 46


# ─────────────────────────────────────────────────────────────────────────────
# Template coordinate computation  (NBA dimensions)
# ─────────────────────────────────────────────────────────────────────────────
def _build_template_points() -> Dict[int, Tuple[float, float]]:
    """
    Returns {keypoint_id: (x, y)} in the canonical template image space.
    Origin (0, 0) is the bottom-left corner of the court.
    Y grows upward (court convention); flip for OpenCV image coords later.
    """
    S  = SCALE
    CW = COURT_W
    CH = COURT_H

    # ── Basket positions (NBA: 5.25 ft from baseline) ────────────────────────
    L_BASKET_X  = 5.25  * S
    R_BASKET_X  = CW - 5.25 * S
    BASKET_Y    = CH / 2

    # ── Paint / key (NBA: 16 ft wide, 19 ft deep from baseline) ─────────────
    PAINT_W     = 16  * S / 2   # half-width from center
    PAINT_DEPTH = 19  * S

    # ── Free-throw circle radius = 6 ft ──────────────────────────────────────
    FT_R = 6 * S

    # ── 3-point line (NBA corner: 14 ft deep, arc radius 23.75 ft) ──────────
    PT3_CORNER_DEPTH = 14  * S
    PT3_R            = 23.75 * S
    PT3_CORNER_Y_OFF = 3   * S   # 3 ft from sideline

    # ── Restricted area arc radius = 4 ft ────────────────────────────────────
    RESTRICT_R = 4 * S

    pts: Dict[int, Tuple[float, float]] = {}

    # Corners
    pts[KP.CORNER_BL] = (0,   0)
    pts[KP.CORNER_BR] = (CW,  0)
    pts[KP.CORNER_TR] = (CW, CH)
    pts[KP.CORNER_TL] = (0,  CH)

    # Half-court
    pts[KP.HALF_BOTTOM]     = (CW/2,  0)
    pts[KP.HALF_TOP]        = (CW/2, CH)
    pts[KP.CENTER_DOT]      = (CW/2, CH/2)
    pts[KP.CENTER_CIRCLE_B] = (CW/2, CH/2 - 6*S)
    pts[KP.CENTER_CIRCLE_T] = (CW/2, CH/2 + 6*S)
    pts[KP.CENTER_CIRCLE_L] = (CW/2 - 6*S, CH/2)
    pts[KP.CENTER_CIRCLE_R] = (CW/2 + 6*S, CH/2)

    # ── LEFT side ────────────────────────────────────────────────────────────
    # Paint
    pts[KP.L_PAINT_BL]  = (0,                      BASKET_Y - PAINT_W)
    pts[KP.L_PAINT_BR]  = (0,                      BASKET_Y + PAINT_W)
    pts[KP.L_PAINT_TL]  = (PAINT_DEPTH,            BASKET_Y - PAINT_W)
    pts[KP.L_PAINT_TR]  = (PAINT_DEPTH,            BASKET_Y + PAINT_W)
    pts[KP.L_FT_LINE_B] = (PAINT_DEPTH,            BASKET_Y - PAINT_W)
    pts[KP.L_FT_LINE_T] = (PAINT_DEPTH,            BASKET_Y + PAINT_W)

    # FT circle tangent points (top/bottom of circle at FT line)
    pts[KP.L_FT_CIRCLE_B] = (PAINT_DEPTH,          BASKET_Y - FT_R)
    pts[KP.L_FT_CIRCLE_T] = (PAINT_DEPTH,          BASKET_Y + FT_R)

    # Basket & backboard
    pts[KP.L_BASKET]      = (L_BASKET_X,            BASKET_Y)
    pts[KP.L_BACKBOARD_B] = (4 * S,                 BASKET_Y - 3*S)
    pts[KP.L_BACKBOARD_T] = (4 * S,                 BASKET_Y + 3*S)

    # Restricted area arc (bottom/top tangents)
    pts[KP.L_RESTRICTED_B] = (L_BASKET_X,           BASKET_Y - RESTRICT_R)
    pts[KP.L_RESTRICTED_T] = (L_BASKET_X,           BASKET_Y + RESTRICT_R)

    # 3-point corners
    pts[KP.L_3PT_CORNER_B] = (0,               PT3_CORNER_Y_OFF)
    pts[KP.L_3PT_CORNER_T] = (0,           CH - PT3_CORNER_Y_OFF)

    # 3-pt arc side tangents  (where arc meets the straight corner segments)
    pts[KP.L_3PT_ARC_B] = (PT3_CORNER_DEPTH,       PT3_CORNER_Y_OFF)
    pts[KP.L_3PT_ARC_T] = (PT3_CORNER_DEPTH,   CH - PT3_CORNER_Y_OFF)

    # 3-pt arc apex  (point furthest from basket along the arc midline)
    pts[KP.L_3PT_TOP] = (L_BASKET_X + PT3_R,        BASKET_Y)

    # ── RIGHT side  (mirror of left) ─────────────────────────────────────────
    def mirror_x(x): return CW - x

    pairs = [
        (KP.L_PAINT_BL,     KP.R_PAINT_BR),
        (KP.L_PAINT_BR,     KP.R_PAINT_BL),
        (KP.L_PAINT_TL,     KP.R_PAINT_TR),
        (KP.L_PAINT_TR,     KP.R_PAINT_TL),
        (KP.L_FT_LINE_B,    KP.R_FT_LINE_T),
        (KP.L_FT_LINE_T,    KP.R_FT_LINE_B),
        (KP.L_FT_CIRCLE_B,  KP.R_FT_CIRCLE_T),
        (KP.L_FT_CIRCLE_T,  KP.R_FT_CIRCLE_B),
        (KP.L_BASKET,       KP.R_BASKET),
        (KP.L_BACKBOARD_B,  KP.R_BACKBOARD_T),
        (KP.L_BACKBOARD_T,  KP.R_BACKBOARD_B),
        (KP.L_RESTRICTED_B, KP.R_RESTRICTED_T),
        (KP.L_RESTRICTED_T, KP.R_RESTRICTED_B),
        (KP.L_3PT_CORNER_B, KP.R_3PT_CORNER_T),
        (KP.L_3PT_CORNER_T, KP.R_3PT_CORNER_B),
        (KP.L_3PT_ARC_B,    KP.R_3PT_ARC_T),
        (KP.L_3PT_ARC_T,    KP.R_3PT_ARC_B),
        (KP.L_3PT_TOP,      KP.R_3PT_TOP),
    ]
    for src, dst in pairs:
        sx, sy = pts[src]
        pts[dst] = (mirror_x(sx), sy)

    return pts


# ── Public constant ───────────────────────────────────────────────────────────
TEMPLATE_KEYPOINTS: Dict[int, Tuple[float, float]] = _build_template_points()


# ─────────────────────────────────────────────────────────────────────────────
# Human-readable labels  (for visualisation / debugging)
# ─────────────────────────────────────────────────────────────────────────────
KP_LABELS: Dict[int, str] = {k: k.name for k in KP}

# Semantic groups  →  used for selective rendering or group-based matching
KP_GROUPS: Dict[str, List[int]] = {
    "corners":       [KP.CORNER_BL, KP.CORNER_BR, KP.CORNER_TR, KP.CORNER_TL],
    "half_court":    [KP.HALF_BOTTOM, KP.HALF_TOP, KP.CENTER_DOT,
                      KP.CENTER_CIRCLE_B, KP.CENTER_CIRCLE_T,
                      KP.CENTER_CIRCLE_L, KP.CENTER_CIRCLE_R],
    "left_paint":    [KP.L_PAINT_BL, KP.L_PAINT_BR, KP.L_PAINT_TR, KP.L_PAINT_TL,
                      KP.L_FT_LINE_B, KP.L_FT_LINE_T,
                      KP.L_FT_CIRCLE_B, KP.L_FT_CIRCLE_T],
    "left_basket":   [KP.L_BASKET, KP.L_BACKBOARD_B, KP.L_BACKBOARD_T,
                      KP.L_RESTRICTED_B, KP.L_RESTRICTED_T],
    "left_3pt":      [KP.L_3PT_CORNER_B, KP.L_3PT_CORNER_T,
                      KP.L_3PT_ARC_B, KP.L_3PT_ARC_T, KP.L_3PT_TOP],
    "right_paint":   [KP.R_PAINT_BL, KP.R_PAINT_BR, KP.R_PAINT_TR, KP.R_PAINT_TL,
                      KP.R_FT_LINE_B, KP.R_FT_LINE_T,
                      KP.R_FT_CIRCLE_B, KP.R_FT_CIRCLE_T],
    "right_basket":  [KP.R_BASKET, KP.R_BACKBOARD_B, KP.R_BACKBOARD_T,
                      KP.R_RESTRICTED_B, KP.R_RESTRICTED_T],
    "right_3pt":     [KP.R_3PT_CORNER_B, KP.R_3PT_CORNER_T,
                      KP.R_3PT_ARC_B, KP.R_3PT_ARC_T, KP.R_3PT_TOP],
}
