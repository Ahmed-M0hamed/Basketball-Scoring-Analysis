import cv2
import numpy as np


def draw_signal(
    frame,
    signal_history,
    x=20,
    y=20,
    width=600,
    height=250,
    line_color=(0, 0, 255),
    fill_color=(0, 0, 255),
    background=(35, 35, 35),
    background_alpha=.05,   # Transparency of panel
    fill_alpha=0.20,         # Transparency of filled area
):

    # -----------------------------
    # Create transparent background
    # -----------------------------
    roi = frame[y:y+height, x:x+width]

    background_img = np.full(
        (height, width, 3),
        background,
        dtype=np.uint8,
    )

    canvas = cv2.addWeighted(
        background_img,
        background_alpha,
        roi,
        1 - background_alpha,
        0,
    )

    if len(signal_history) > 1:

        values = np.asarray(signal_history, dtype=np.float32)

        mn = values.min()
        mx = values.max()

        if mx - mn < 1e-6:
            mx = mn + 1

        values = (values - mn) / (mx - mn)

        xs = np.linspace(
            0,
            width - 1,
            len(values),
        )

        ys = height - 10 - values * (height - 20)

        pts = np.column_stack((xs, ys)).astype(np.int32)

        # -----------------------------
        # Semi-transparent filled area
        # -----------------------------
        polygon = np.vstack([
            [0, height - 1],
            pts,
            [width - 1, height - 1]
        ])

        overlay = canvas.copy()

        cv2.fillPoly(
            overlay,
            [polygon],
            fill_color,
        )

        canvas = cv2.addWeighted(
            overlay,
            fill_alpha,
            canvas,
            1 - fill_alpha,
            0,
        )

        # Signal line
        cv2.polylines(
            canvas,
            [pts],
            False,
            line_color,
            2,
            cv2.LINE_AA,
        )

        # Current point
        cv2.circle(
            canvas,
            tuple(pts[-1]),
            5,
            line_color,
            -1,
        )

    # Border

    # Title
    cv2.putText(
        canvas,
        "Game Momentum",
        (15, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,0,255),
        2,
        cv2.LINE_AA,
    )

    # Put graph back on frame
    frame[y:y+height, x:x+width] = canvas

    return frame