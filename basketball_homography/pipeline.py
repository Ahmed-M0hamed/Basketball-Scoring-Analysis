"""
Basketball Homography Pipeline
==============================
End-to-end pipeline that wires together:
  1. Keypoint detector    (plug in your own model here)
  2. Homography estimator
  3. Temporal smoother    (EMA on H matrix)
  4. Visualiser

Usage
-----
    pipeline = BasketballHomographyPipeline()
    for frame in video_frames:
        out = pipeline.process_frame(frame)
        cv2.imshow("result", out.visualization)
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from court_template import TEMPLATE_KEYPOINTS, KP, KP_LABELS
from court_homography import (
    CourtHomographyEstimator, DetectedKeypoints,
    HomographyResult, frame_to_template, template_to_frame,
)
from court_visualizer import (
    CourtTemplateRenderer, FrameOverlayRenderer,
    make_side_by_side,
)


# ─────────────────────────────────────────────────────────────────────────────
# Temporal homography smoother  (EMA on the 3×3 matrix elements)
# ─────────────────────────────────────────────────────────────────────────────
class HomographySmoother:
    """
    Exponential moving average over consecutive homography matrices.
    Falls back to the raw estimate when too few good frames have been seen.

    Parameters
    ----------
    alpha       : EMA weight for the newest frame  (higher = more reactive)
    min_conf    : only update EMA when result.confidence ≥ this value
    """

    def __init__(self, alpha: float = 0.4, min_conf: float = 0.3):
        self.alpha    = alpha
        self.min_conf = min_conf
        self._ema_H : Optional[np.ndarray] = None

    def update(self, result: HomographyResult) -> HomographyResult:
        """Return a new HomographyResult whose H has been EMA-smoothed."""
        if not result.success or result.H is None:
            return result

        if result.confidence < self.min_conf:
            return result

        if self._ema_H is None:
            self._ema_H = result.H.copy()
        else:
            self._ema_H = self.alpha * result.H + (1 - self.alpha) * self._ema_H

        # Return a copy with the smoothed matrix
        smoothed          = HomographyResult(**result.__dict__)
        smoothed.H        = self._ema_H.copy()
        smoothed.H_inv    = np.linalg.inv(self._ema_H)
        return smoothed

    def reset(self):
        self._ema_H = None


# ─────────────────────────────────────────────────────────────────────────────
# Keypoint detector interface  (swap with your real model)
# ─────────────────────────────────────────────────────────────────────────────
DetectorFn = Callable[
    [np.ndarray],
    Dict[int, Tuple[float, float]]
]
"""
Type alias for a keypoint detector function.
Signature: (frame: np.ndarray) -> {kp_id: (x, y)}
Only VISIBLE keypoints should be returned.
"""


class StubKeypointDetector:
    """
    Placeholder detector that returns a random visible subset of the template
    keypoints projected back into a simulated camera view.

    Replace this class with your actual trained model (e.g. HRNet, ViTPose,
    or a custom CNN fine-tuned on basketball court images).
    """

    def __init__(self, frame_shape: Tuple[int, int] = (720, 1280)):
        self.H, self.W = frame_shape
        # Simulated camera homography  (template → frame)
        src = np.array([
            [0,        0       ],
            [TEMPLATE_KEYPOINTS[KP.CORNER_BR][0], 0],
            [TEMPLATE_KEYPOINTS[KP.CORNER_BR][0],
             TEMPLATE_KEYPOINTS[KP.CORNER_TR][1]],
            [0, TEMPLATE_KEYPOINTS[KP.CORNER_TL][1]],
        ], dtype=np.float32)

        dst = np.array([
            [50,       self.H - 80],
            [self.W - 50, self.H - 80],
            [self.W - 200,  200     ],
            [200,           200     ],
        ], dtype=np.float32)

        self._H_sim, _ = cv2.findHomography(src, dst)
        rng = np.random.default_rng(42)
        all_ids = list(TEMPLATE_KEYPOINTS.keys())
        # Simulate a fixed "visible" subset  (in real code, the model decides)
        n_visible = max(6, len(all_ids) // 2)
        self._visible_ids = rng.choice(all_ids, n_visible, replace=False).tolist()

    def detect(self, frame: np.ndarray) -> Dict[int, Tuple[float, float]]:
        """Return simulated noisy detections for the visible subset."""
        result: Dict[int, Tuple[float, float]] = {}
        rng = np.random.default_rng()   # fresh per-frame noise

        for kp_id in self._visible_ids:
            tx, ty = TEMPLATE_KEYPOINTS[kp_id]
            pt_h   = self._H_sim @ np.array([tx, ty, 1.0])
            pt_h  /= pt_h[2]
            # Add Gaussian noise to simulate detector uncertainty
            noise_x = rng.normal(0, 2.5)
            noise_y = rng.normal(0, 2.5)
            fx = float(pt_h[0]) + noise_x
            fy = float(pt_h[1]) + noise_y

            # Only keep point if it's inside the frame
            if 0 <= fx < self.W and 0 <= fy < self.H:
                result[kp_id] = (fx, fy)

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame pipeline result
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PipelineResult:
    frame_id      : int                               = 0
    detections    : Dict[int, Tuple[float, float]]    = field(default_factory=dict)
    raw_result    : Optional[HomographyResult]        = None
    smooth_result : Optional[HomographyResult]        = None
    visualization : Optional[np.ndarray]              = None   # BGR image

    @property
    def H(self) -> Optional[np.ndarray]:
        """Best available homography (smoothed > raw)."""
        r = self.smooth_result or self.raw_result
        return r.H if (r and r.success) else None

    @property
    def success(self) -> bool:
        r = self.smooth_result or self.raw_result
        return bool(r and r.success)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────
class BasketballHomographyPipeline:
    """
    Full per-frame pipeline.

    Parameters
    ----------
    detector_fn     : callable (frame → {kp_id: (x,y)}).
                      Pass None to use the built-in stub detector.
    estimator_cfg   : kwargs forwarded to CourtHomographyEstimator
    smoother_alpha  : EMA alpha for temporal smoothing
    show_overlay    : blend court template onto frame output
    show_hud        : draw HUD metrics on frame output
    side_by_side    : produce a split-screen instead of single-frame output
    """

    def __init__(
        self,
        detector_fn     : Optional[DetectorFn] = None,
        estimator_cfg   : Optional[dict]        = None,
        smoother_alpha  : float                 = 0.4,
        show_overlay    : bool                  = True,
        show_hud        : bool                  = True,
        side_by_side    : bool                  = True,
    ):
        self.estimator  = CourtHomographyEstimator(**(estimator_cfg or {}))
        self.smoother   = HomographySmoother(alpha=smoother_alpha)
        self.tpl_render = CourtTemplateRenderer()
        self.frm_render = FrameOverlayRenderer(self.tpl_render)
        self._detector  = detector_fn            # None → set on first frame
        self.show_overlay   = show_overlay
        self.show_hud       = show_hud
        self.side_by_side   = side_by_side
        self._frame_id  = 0

    # ── Public API ────────────────────────────────────────────────────────────
    def process_frame(self, frame: np.ndarray) -> PipelineResult:
        """Process a single BGR frame. Returns a PipelineResult."""
        self._frame_id += 1

        # 1. Detect keypoints
        if self._detector is None:
            self._detector = StubKeypointDetector(
                frame_shape=(frame.shape[0], frame.shape[1])
            ).detect

        detections = self._detector(frame)

        detected = DetectedKeypoints(
            frame_id   = self._frame_id,
            detections = detections,
        )

        # 2. Estimate homography
        raw = self.estimator.estimate(detected)

        # 3. Temporal smoothing
        smoothed = self.smoother.update(raw)

        # 4. Visualise
        vis = self._build_visualization(frame, detections, smoothed)

        return PipelineResult(
            frame_id      = self._frame_id,
            detections    = detections,
            raw_result    = raw,
            smooth_result = smoothed,
            visualization = vis,
        )

    def reset_smoother(self):
        """Call on scene cuts / camera changes."""
        self.smoother.reset()

    def get_template_image(self) -> np.ndarray:
        return self.tpl_render.get_base_court()

    # ── Private ───────────────────────────────────────────────────────────────
    def _build_visualization(
        self,
        frame      : np.ndarray,
        detections : Dict[int, Tuple[float, float]],
        result     : HomographyResult,
    ) -> np.ndarray:
        active = result if result.success else (
            self.smoother._ema_H is not None and result
        )

        if self.side_by_side:
            tpl_img = self.tpl_render.get_base_court()
            return make_side_by_side(frame, tpl_img, result, detections)

        vis = frame.copy()
        if self.show_overlay and result.success:
            vis = self.frm_render.draw_court_overlay(vis, result, alpha=0.3)
        vis = self.frm_render.draw_detected_keypoints(vis, detections, result)
        if self.show_hud:
            vis = self.frm_render.draw_hud(vis, result)
        return vis
