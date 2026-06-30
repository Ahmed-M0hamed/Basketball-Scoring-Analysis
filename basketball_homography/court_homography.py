"""
Court Homography Engine
=======================
Computes H: frame_pixel → template_pixel from a *partial* set of detected
keypoints (not every frame shows every keypoint).

Key design choices
------------------
* Accepts any dict {keypoint_id: (x, y)} for both template and detected pts.
* Requires ≥ 4 matched keypoints; works robustly with 6-10+.
* Uses cv2.findHomography with RANSAC to discard outlier detections.
* Returns a CourtHomographyResult that bundles H, confidence, and provenance.
* Exposes helpers to project arbitrary (x, y) pairs in both directions.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from court_template import TEMPLATE_KEYPOINTS, KP, KP_LABELS


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DetectedKeypoints:
    """
    Container for keypoints detected in a single video frame.

    Attributes
    ----------
    frame_id   : int  – frame index (for logging / caching)
    detections : dict – {KP_id: (x_pixel, y_pixel)} for visible keypoints only
    confidences: dict – optional per-keypoint detection confidence in [0, 1]
    """
    frame_id   : int                             = 0
    detections : Dict[int, Tuple[float, float]]  = field(default_factory=dict)
    confidences: Dict[int, float]                = field(default_factory=dict)


@dataclass
class HomographyResult:
    """
    Result of a single homography estimation.

    Attributes
    ----------
    success         : estimation converged with enough inliers
    H               : 3×3 homography matrix  (frame → template)
    H_inv           : 3×3 inverse             (template → frame)
    used_keypoints  : IDs that contributed to the final H
    inlier_mask     : bool array, one entry per matched keypoint
    reproj_error    : mean reprojection error of inliers (px in template space)
    confidence      : heuristic score in [0, 1]
    n_inliers       : number of RANSAC inliers
    n_matches       : total matched keypoints before RANSAC
    frame_id        : echoed from DetectedKeypoints
    """
    success        : bool                       = False
    H              : Optional[np.ndarray]       = None
    H_inv          : Optional[np.ndarray]       = None
    used_keypoints : List[int]                  = field(default_factory=list)
    inlier_mask    : Optional[np.ndarray]       = None
    reproj_error   : float                      = float("inf")
    confidence     : float                      = 0.0
    n_inliers      : int                        = 0
    n_matches      : int                        = 0
    frame_id       : int                        = 0


# ─────────────────────────────────────────────────────────────────────────────
# Homography estimator
# ─────────────────────────────────────────────────────────────────────────────
class CourtHomographyEstimator:
    """
    Estimates H from partial per-frame keypoint detections.

    Parameters
    ----------
    min_points       : minimum matched keypoints to attempt estimation (≥ 4)
    ransac_threshold : reprojection error threshold for RANSAC (pixels)
    min_inliers      : minimum RANSAC inliers to accept result
    max_reproj_error : reject results with mean reproj error above this value
    """

    def __init__(
        self,
        min_points       : int   = 4,
        ransac_threshold : float = 8.0,
        min_inliers      : int   = 4,
        max_reproj_error : float = 15.0,
    ):
        assert min_points >= 4, "Homography needs at least 4 point correspondences."
        self.min_points       = min_points
        self.ransac_threshold = ransac_threshold
        self.min_inliers      = min_inliers
        self.max_reproj_error = max_reproj_error

    # ── Main entry point ─────────────────────────────────────────────────────
    def estimate(self, detected: DetectedKeypoints) -> HomographyResult:
        """
        Estimate H from the detected keypoints in one frame.

        detected.detections  →  {kp_id: (x, y)} in FRAME pixel space
        TEMPLATE_KEYPOINTS   →  {kp_id: (x, y)} in TEMPLATE pixel space

        Only keypoints present in BOTH dicts are used.
        """
        result = HomographyResult(frame_id=detected.frame_id)

        # ── 1.  Build matched point arrays ───────────────────────────────────
        frame_pts, template_pts, matched_ids = self._build_correspondences(
            detected.detections
        )
        result.n_matches = len(matched_ids)

        if result.n_matches < self.min_points:
            return result   # not enough points

        # ── 2.  Weighting  (optional – skew RANSAC via detection confidence) ─
        #       cv2.findHomography doesn't take weights natively; we replicate
        #       high-confidence points to give them more RANSAC votes.
        src = frame_pts.astype(np.float32)
        dst = template_pts.astype(np.float32)

        # ── 3.  RANSAC homography ─────────────────────────────────────────────
        H, mask = cv2.findHomography(
            src, dst,
            method    = cv2.RANSAC,
            ransacReprojThreshold = self.ransac_threshold,
            maxIters  = 2000,
            confidence= 0.995,
        )

        if H is None:
            return result

        mask = mask.ravel().astype(bool)
        result.n_inliers = int(mask.sum())

        if result.n_inliers < self.min_inliers:
            return result

        # ── 4.  Refit on inliers only ─────────────────────────────────────────
        H, _ = cv2.findHomography(
            src[mask], dst[mask],
            method = 0,         # least-squares on inliers
        )
        if H is None:
            return result

        # ── 5.  Quality metrics ───────────────────────────────────────────────
        reproj_err = self._mean_reprojection_error(
            src[mask], dst[mask], H
        )
        if reproj_err > self.max_reproj_error:
            return result

        # ── 6.  Pack result ───────────────────────────────────────────────────
        result.success        = True
        result.H              = H
        result.H_inv          = np.linalg.inv(H)
        result.used_keypoints = [matched_ids[i] for i in range(len(matched_ids)) if mask[i]]
        result.inlier_mask    = mask
        result.reproj_error   = reproj_err
        result.confidence     = self._compute_confidence(result)
        return result

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _build_correspondences(
        self,
        detections: Dict[int, Tuple[float, float]],
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """Return parallel arrays of frame pts, template pts, and their IDs."""
        ids, frame_pts, template_pts = [], [], []
        for kp_id, frame_xy in detections.items():
            if kp_id in TEMPLATE_KEYPOINTS:
                ids.append(kp_id)
                frame_pts.append(frame_xy)
                template_pts.append(TEMPLATE_KEYPOINTS[kp_id])
        return (
            np.array(frame_pts,    dtype=np.float64),
            np.array(template_pts, dtype=np.float64),
            ids,
        )

    @staticmethod
    def _mean_reprojection_error(
        src: np.ndarray, dst: np.ndarray, H: np.ndarray
    ) -> float:
        """Mean Euclidean reprojection error in template space."""
        n   = len(src)
        src_h = np.hstack([src, np.ones((n, 1))])      # (N, 3)
        proj  = (H @ src_h.T).T                        # (N, 3)
        proj  = proj[:, :2] / proj[:, 2:3]             # dehomogenise
        return float(np.linalg.norm(proj - dst, axis=1).mean())

    @staticmethod
    def _compute_confidence(r: HomographyResult) -> float:
        """
        Heuristic confidence ∈ [0, 1].
        Rewards more inliers and lower reprojection error.
        """
        inlier_score  = min(r.n_inliers / 12.0, 1.0)       # saturates at 12 pts
        reproj_score  = max(0.0, 1.0 - r.reproj_error / 15.0)
        return round(0.6 * inlier_score + 0.4 * reproj_score, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Projection utilities
# ─────────────────────────────────────────────────────────────────────────────
def project_points(
    pts: np.ndarray,
    H  : np.ndarray,
) -> np.ndarray:
    """
    Project Nx2 pixel coordinates through homography H.

    Parameters
    ----------
    pts : (N, 2) float array  – input  coordinates
    H   : (3, 3) float array  – homography matrix

    Returns
    -------
    (N, 2) float array – projected coordinates
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])   # (N, 3)
    out   = (H @ pts_h.T).T                            # (N, 3)
    return (out[:, :2] / out[:, 2:3]).astype(np.float64)


def frame_to_template(pts: np.ndarray, result: HomographyResult) -> Optional[np.ndarray]:
    """Project frame pixels → template pixels."""
    if not result.success or result.H is None:
        return None
    return project_points(pts, result.H)


def template_to_frame(pts: np.ndarray, result: HomographyResult) -> Optional[np.ndarray]:
    """Project template pixels → frame pixels."""
    if not result.success or result.H_inv is None:
        return None
    return project_points(pts, result.H_inv)
