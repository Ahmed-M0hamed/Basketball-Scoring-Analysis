"""
Demo: Basketball Court Homography
==================================
Generates:
  1. court_template.png  – clean annotated court template
  2. demo_frame_N.png    – simulated frames with homography overlay
  3. demo_pipeline.mp4   – (if ffmpeg is available) assembled video

Run from the project root:
    python demo.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np

from court_template  import TEMPLATE_KEYPOINTS, KP, KP_GROUPS, SCALE
from court_homography import (
    CourtHomographyEstimator, DetectedKeypoints,
    HomographyResult, frame_to_template, template_to_frame,
)
from court_visualizer import CourtTemplateRenderer, FrameOverlayRenderer
from pipeline import BasketballHomographyPipeline, StubKeypointDetector


OUT = "outputs"
os.makedirs(OUT, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Save annotated template
# ─────────────────────────────────────────────────────────────────────────────
def save_template():
    renderer = CourtTemplateRenderer()
    base     = renderer.get_base_court()
    annotated = renderer.draw_keypoints(base, show_labels=True)

    # draw group legend
    import itertools
    colors = [(255,100,100),(100,255,100),(100,100,255),
              (255,255,0),(255,0,255),(0,255,255),(200,150,50),(50,200,150)]
    for (group, ids), col in zip(KP_GROUPS.items(), itertools.cycle(colors)):
        for kp_id in ids:
            from court_template import TEMPLATE_KEYPOINTS, MARGIN, COURT_H
            x, y = TEMPLATE_KEYPOINTS[kp_id]
            ix = int(round(x)) + MARGIN
            iy = int(round(COURT_H)) + MARGIN - int(round(y))
            cv2.circle(annotated, (ix, iy), 8, col, 2)

    path = f"{OUT}/court_template.png"
    cv2.imwrite(path, annotated)
    print(f"[✓] Saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Simulate frames and run pipeline
# ─────────────────────────────────────────────────────────────────────────────
def simulate_broadcast_frame(
    frame_id   : int,
    shape      : tuple = (720, 1280, 3),
    n_visible  : int   = 18,
    noise_std  : float = 3.5,
    seed       : int   = None,
) -> np.ndarray:
    """
    Create a synthetic BGR frame that looks roughly like a broadcast overhead shot.
    In production you'd read a real video frame with cv2.VideoCapture.
    """
    H, W, _ = shape
    img = np.zeros(shape, dtype=np.uint8)

    # Green court surface
    pts = np.array([[80, H-60], [W-80, H-60], [W-220, 160], [220, 160]], np.int32)
    cv2.fillPoly(img, [pts], (34, 100, 34))

    # Slight texture gradient to make it realistic-looking
    noise = np.random.RandomState(frame_id).randint(-8, 8, shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


def save_demo_frames(n_frames: int = 5):
    """Run pipeline on synthetic frames and save side-by-side visuals."""
    pipeline = BasketballHomographyPipeline(
        show_overlay = True,
        show_hud     = True,
        side_by_side = True,
    )

    paths = []
    for i in range(n_frames):
        frame = simulate_broadcast_frame(frame_id=i)
        result = pipeline.process_frame(frame)

        status = "OK" if result.success else "FAIL"
        n_det  = len(result.detections)
        n_in   = result.smooth_result.n_inliers if result.smooth_result else 0
        conf   = result.smooth_result.confidence if result.smooth_result else 0.0
        print(f"  Frame {i+1:02d} | status={status} | detected={n_det} "
              f"| inliers={n_in} | conf={conf:.3f}")

        path = f"{OUT}/demo_frame_{i+1:02d}.png"
        cv2.imwrite(path, result.visualization)
        paths.append(path)

    print(f"[✓] Saved {n_frames} demo frames to {OUT}/")
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Coordinate projection example
# ─────────────────────────────────────────────────────────────────────────────
def projection_example():
    """Show how to project player positions to the template plane."""
    print("\n── Projection example ──────────────────────────────────")

    # Simulate one frame
    frame = simulate_broadcast_frame(frame_id=99)
    shape = (frame.shape[0], frame.shape[1])
    stub  = StubKeypointDetector(shape)
    dets  = stub.detect(frame)

    detected = DetectedKeypoints(frame_id=99, detections=dets)
    estimator = CourtHomographyEstimator()
    result    = estimator.estimate(detected)

    if not result.success:
        print("  [!] Homography failed (need more visible keypoints)")
        return

    # Fake player pixel positions (from a tracker, e.g.)
    player_frame_pts = np.array([
        [400, 500],
        [800, 400],
        [640, 350],
    ], dtype=np.float64)

    template_pts = frame_to_template(player_frame_pts, result)
    if template_pts is not None:
        for i, (fp, tp) in enumerate(zip(player_frame_pts, template_pts)):
            print(f"  Player {i}: frame ({fp[0]:.0f}, {fp[1]:.0f}) → "
                  f"template ({tp[0]:.1f}, {tp[1]:.1f})  "
                  f"[{tp[0]/SCALE:.1f} ft, {tp[1]/SCALE:.1f} ft]")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("═" * 55)
    print("  Basketball Court Homography Demo")
    print("═" * 55)

    print("\n[1] Generating court template …")
    tpl_path = save_template()

    print("\n[2] Running pipeline on synthetic frames …")
    frame_paths = save_demo_frames(n_frames=6)

    projection_example()

    print("\n═" * 28)
    print("  All outputs saved to ./outputs/")
    print("═" * 28)
