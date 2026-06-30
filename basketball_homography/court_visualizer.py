"""
Court Visualizer
================
Draws the canonical basketball-court template and overlays homography
results on top of original video frames.

All drawing is pure OpenCV — no matplotlib required at runtime.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

from court_template import (
    TEMPLATE_KEYPOINTS, KP, KP_LABELS, KP_GROUPS,
    COURT_W, COURT_H, MARGIN, IMG_W, IMG_H, SCALE,
)
from court_homography import HomographyResult, project_points


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette  (BGR)
# ─────────────────────────────────────────────────────────────────────────────
C = {
    "court_bg"      : (  34,  85,  34),   # dark green
    "court_line"    : ( 255, 255, 255),   # white
    "margin_bg"     : (  20,  50,  20),
    "kp_default"    : (   0, 255, 255),   # cyan
    "kp_inlier"     : (   0, 255,   0),   # green
    "kp_outlier"    : (   0,   0, 255),   # red
    "kp_undetected" : ( 100, 100, 100),   # grey
    "label_text"    : ( 255, 255,   0),   # yellow
    "projected_overlay": (0, 200, 255),   # orange-ish
    "confidence_bar": (  50, 200,  50),
    "black"         : (   0,   0,   0),
}


def _to_img(court_pt: Tuple[float, float]) -> Tuple[int, int]:
    """Convert court-space (origin=bottom-left, y-up) to image px (y-down)."""
    x, y = court_pt
    ix = int(round(x)) + MARGIN
    iy = IMG_H - MARGIN - int(round(y))
    return (ix, iy)


# ─────────────────────────────────────────────────────────────────────────────
# Template image builder
# ─────────────────────────────────────────────────────────────────────────────
class CourtTemplateRenderer:
    """Renders a clean 2-D top-down basketball court template image."""

    def __init__(self, line_thickness: int = 2):
        self.lt = line_thickness
        self._base: Optional[np.ndarray] = None   # cached clean court

    # ── Public API ────────────────────────────────────────────────────────────
    def get_base_court(self) -> np.ndarray:
        """Return a clean court image (cached after first call)."""
        if self._base is None:
            self._base = self._draw_court()
        return self._base.copy()

    def draw_keypoints(
        self,
        img                : np.ndarray,
        highlight_ids      : Optional[List[int]] = None,
        highlight_color    : Tuple[int,int,int]  = C["kp_inlier"],
        default_color      : Tuple[int,int,int]  = C["kp_default"],
        show_labels        : bool                = True,
        radius             : int                 = 6,
    ) -> np.ndarray:
        """Overlay all template keypoints on *img* (drawn in-place clone)."""
        img = img.copy()
        for kp_id, coord in TEMPLATE_KEYPOINTS.items():
            pt = _to_img(coord)
            col = highlight_color if (highlight_ids and kp_id in highlight_ids) \
                  else default_color
            cv2.circle(img, pt, radius, col, -1)
            cv2.circle(img, pt, radius, C["black"], 1)
            if show_labels:
                label = KP_LABELS.get(kp_id, str(kp_id))
                cv2.putText(img, label, (pt[0] + 6, pt[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, C["label_text"], 1,
                            cv2.LINE_AA)
        return img

    # ── Internal court drawing ────────────────────────────────────────────────
    def _draw_court(self) -> np.ndarray:
        img = np.full((IMG_H, IMG_W, 3), C["margin_bg"], dtype=np.uint8)

        # Court rectangle (filled green)
        tl = _to_img((0,       COURT_H))
        br = _to_img((COURT_W, 0))
        cv2.rectangle(img, tl, br, C["court_bg"], -1)

        lt = self.lt
        S  = SCALE

        def line(p1, p2):
            cv2.line(img, _to_img(p1), _to_img(p2), C["court_line"], lt)

        def circle(cx, cy, r, full=False):
            c = _to_img((cx, cy))
            if full:
                cv2.circle(img, c, int(round(r)), C["court_line"], lt)
            else:
                # Draw only arcs via polyline approximation
                cv2.circle(img, c, int(round(r)), C["court_line"], lt)

        def arc(cx, cy, r, start_deg, end_deg):
            """Draw a partial circle arc."""
            pts = []
            for deg in np.linspace(start_deg, end_deg, 60):
                rad = np.deg2rad(deg)
                px  = cx + r * np.cos(rad)
                py  = cy + r * np.sin(rad)
                pts.append(_to_img((px, py)))
            pts = np.array(pts, dtype=np.int32)
            cv2.polylines(img, [pts], False, C["court_line"], lt)

        CW, CH = COURT_W, COURT_H
        MID_Y  = CH / 2

        # ── Court boundary ────────────────────────────────────────────────────
        line((0, 0), (CW, 0))
        line((CW, 0), (CW, CH))
        line((CW, CH), (0, CH))
        line((0, CH), (0, 0))

        # ── Half-court line ───────────────────────────────────────────────────
        line((CW/2, 0), (CW/2, CH))

        # ── Centre circle (r = 6 ft) ──────────────────────────────────────────
        circle(CW/2, MID_Y, 6*S, full=True)

        # ── Left paint ────────────────────────────────────────────────────────
        line((0, MID_Y - 8*S), (19*S, MID_Y - 8*S))
        line((0, MID_Y + 8*S), (19*S, MID_Y + 8*S))
        line((19*S, MID_Y - 8*S), (19*S, MID_Y + 8*S))

        # FT circle
        arc(19*S, MID_Y, 6*S,  -90,  90)   # solid half toward basket
        pts_dashed = []                      # dashed half away from basket
        for i, deg in enumerate(np.linspace(90, 270, 20)):
            rad = np.deg2rad(deg)
            px  = 19*S + 6*S * np.cos(rad)
            py  = MID_Y + 6*S * np.sin(rad)
            if i % 2 == 0:
                pts_dashed.append(_to_img((px, py)))
            else:
                if len(pts_dashed) > 1:
                    cv2.polylines(img, [np.array(pts_dashed, np.int32)],
                                  False, C["court_line"], lt)
                pts_dashed = []

        # ── Right paint ───────────────────────────────────────────────────────
        line((CW, MID_Y - 8*S), (CW - 19*S, MID_Y - 8*S))
        line((CW, MID_Y + 8*S), (CW - 19*S, MID_Y + 8*S))
        line((CW - 19*S, MID_Y - 8*S), (CW - 19*S, MID_Y + 8*S))
        arc(CW - 19*S, MID_Y, 6*S, 90, 270)

        # ── Baskets (5.25 ft from baseline) ──────────────────────────────────
        for bx in [5.25*S, CW - 5.25*S]:
            cv2.circle(img, _to_img((bx, MID_Y)), int(0.75*S),
                       C["court_line"], lt)
            # Backboard
            bbd_x = 4*S if bx < CW/2 else CW - 4*S
            line((bbd_x, MID_Y - 3*S), (bbd_x, MID_Y + 3*S))
            # Restricted area arc (r = 4 ft)
            if bx < CW/2:
                arc(bx, MID_Y, 4*S, -90, 90)
            else:
                arc(bx, MID_Y, 4*S, 90, 270)

        # ── 3-point lines ─────────────────────────────────────────────────────
        for bx in [5.25*S, CW - 5.25*S]:
            left = bx < CW/2
            # Corner straight segments (3 ft from sidelines)
            if left:
                line((0, 3*S),        (14*S, 3*S))
                line((0, CH - 3*S),   (14*S, CH - 3*S))
                arc(bx, MID_Y, 23.75*S, -68, 68)
            else:
                line((CW, 3*S),       (CW - 14*S, 3*S))
                line((CW, CH - 3*S),  (CW - 14*S, CH - 3*S))
                arc(bx, MID_Y, 23.75*S, 112, 248)

        # ── Lane dots / hash marks (cosmetic) ────────────────────────────────
        for side in [-1, 1]:
            for lane_x in [7*S, 11*S, 14*S, 17*S]:
                for dy in [3*S, 8*S]:
                    y_off = MID_Y + side * dy
                    cv2.circle(img, _to_img((lane_x, y_off)), 2,
                               C["court_line"], -1)
            for lane_x in [CW - x for x in [7*S, 11*S, 14*S, 17*S]]:
                for dy in [3*S, 8*S]:
                    y_off = MID_Y + side * dy
                    cv2.circle(img, _to_img((lane_x, y_off)), 2,
                               C["court_line"], -1)

        return img


# ─────────────────────────────────────────────────────────────────────────────
# Frame overlay renderer
# ─────────────────────────────────────────────────────────────────────────────
class FrameOverlayRenderer:
    """
    Overlays homography visualisations directly on a video frame.
    """

    def __init__(self, template_renderer: CourtTemplateRenderer):
        self.tpl = template_renderer

    def draw_detected_keypoints(
        self,
        frame       : np.ndarray,
        detections  : Dict[int, Tuple[float, float]],
        result      : HomographyResult,
        radius      : int  = 8,
        show_labels : bool = True,
    ) -> np.ndarray:
        """
        Draw detected keypoints on a frame.
        Inliers → green, outliers → red, undetected (not in detections) → skipped.
        """
        img  = frame.copy()
        inlier_ids = set(result.used_keypoints)

        for kp_id, (x, y) in detections.items():
            pt  = (int(round(x)), int(round(y)))
            col = C["kp_inlier"] if kp_id in inlier_ids else C["kp_outlier"]
            cv2.circle(img, pt, radius, col, -1)
            cv2.circle(img, pt, radius, C["black"], 1)
            if show_labels:
                label = KP_LABELS.get(kp_id, str(kp_id))
                cv2.putText(img, label, (pt[0] + 8, pt[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, C["label_text"], 1,
                            cv2.LINE_AA)
        return img

    def draw_court_overlay(
        self,
        frame  : np.ndarray,
        result : HomographyResult,
        alpha  : float = 0.35,
    ) -> np.ndarray:
        """
        Warp the template court back into frame space and blend it as a
        semi-transparent overlay (requires a valid H_inv).
        """
        if not result.success or result.H_inv is None:
            return frame.copy()

        h, w = frame.shape[:2]
        court_img = self.tpl.get_base_court()

        # Warp template → frame
        warped = cv2.warpPerspective(court_img, result.H_inv, (w, h))

        # Create mask (non-black pixels in warped image)
        mask   = (warped.sum(axis=2) > 0).astype(np.uint8)[:, :, None]

        blended = frame.copy().astype(np.float32)
        blended = blended * (1 - alpha * mask) + warped.astype(np.float32) * alpha * mask
        return np.clip(blended, 0, 255).astype(np.uint8)

    def draw_hud(
        self,
        frame  : np.ndarray,
        result : HomographyResult,
    ) -> np.ndarray:
        """Draw a heads-up display with homography quality metrics."""
        img = frame.copy()
        H, W = img.shape[:2]
        y0   = 24

        def put(text, y, col=(255, 255, 255)):
            cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, col,    1, cv2.LINE_AA)

        status_col = C["kp_inlier"] if result.success else C["kp_outlier"]
        put(f"Frame {result.frame_id}", y0)
        put(f"Status  : {'OK' if result.success else 'FAILED'}", y0 + 24, status_col)
        put(f"Matches : {result.n_matches}", y0 + 48)
        put(f"Inliers : {result.n_inliers}", y0 + 72)
        if result.success:
            put(f"Reproj  : {result.reproj_error:.2f} px", y0 + 96)
            put(f"Conf    : {result.confidence:.3f}",       y0 + 120)

            # Confidence bar
            bar_x, bar_y, bar_w, bar_h = 12, y0 + 134, 160, 10
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                          (60, 60, 60), -1)
            filled = int(bar_w * result.confidence)
            bar_col = (0, int(200 * result.confidence), int(200 * (1 - result.confidence)))
            cv2.rectangle(img, (bar_x, bar_y),
                          (bar_x + filled, bar_y + bar_h), bar_col, -1)

        return img


# ─────────────────────────────────────────────────────────────────────────────
# Side-by-side comparison helper
# ─────────────────────────────────────────────────────────────────────────────
def make_side_by_side(
    frame           : np.ndarray,
    template_img    : np.ndarray,
    result          : HomographyResult,
    detections      : Dict[int, Tuple[float, float]],
    target_height   : int = 500,
) -> np.ndarray:
    """
    Create a side-by-side visualisation:
      left  – original frame with detected keypoints
      right – template with inlier keypoints highlighted
    """
    tpl_renderer = CourtTemplateRenderer()
    frm_renderer = FrameOverlayRenderer(tpl_renderer)

    # Left panel
    left = frm_renderer.draw_detected_keypoints(frame, detections, result)
    left = frm_renderer.draw_hud(left, result)

    # Right panel
    inlier_ids = result.used_keypoints
    right = tpl_renderer.draw_keypoints(
        template_img,
        highlight_ids=inlier_ids,
        show_labels=False,
    )

    # Resize both to same height
    def resize_h(img, h):
        ratio = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * ratio), h))

    left  = resize_h(left,  target_height)
    right = resize_h(right, target_height)

    # Separator
    sep = np.full((target_height, 4, 3), 200, dtype=np.uint8)
    return np.hstack([left, sep, right])
