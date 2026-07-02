import cv2
import numpy as np


class StatsPanel:
    def __init__( self,x=30, y=30, width=420, row_height=40, alpha=0.80,):
        self.x = x
        self.y = y
        self.width = width
        self.row_height = row_height
        self.alpha = alpha

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.65
        self.thickness = 2

    def _center_text(self, img, text, cx, cy, color):
        (w, h), _ = cv2.getTextSize(
            text,
            self.font,
            self.font_scale,
            self.thickness,
        )

        cv2.putText(
            img,
            text,
            (int(cx - w / 2), int(cy + h / 2)),
            self.font,
            self.font_scale,
            color,
            self.thickness,
            cv2.LINE_AA,
        )

    def draw(self, frame, stats, title="TEAM STATS"):
        teams = list(stats.keys())
        metrics = list(next(iter(stats.values())).keys())

        rows = len(metrics) + 2  # title + header
        height = rows * self.row_height

        x = self.x
        y = self.y
        w = self.width

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (x, y),
            (x + w, y + height),
            (35, 35, 35),
            -1,
        )

        frame[:] = cv2.addWeighted(
            overlay,
            self.alpha,
            frame,
            1 - self.alpha,
            0,
        )

        # ---------------------------
        # Column widths
        # ---------------------------

        metric_w = int(w * 0.45)
        team_w = (w - metric_w) // len(teams)

        # ---------------------------
        # Grid
        # ---------------------------

        for r in range(rows + 1):
            yy = y + r * self.row_height
            cv2.line(frame, (x, yy), (x + w, yy), (255, 255, 255), 1)

        cv2.line(
            frame,
            (x + metric_w, y + self.row_height),
            (x + metric_w, y + height),
            (255, 255, 255),
            1,
        )

        for i in range(1, len(teams)):
            xx = x + metric_w + i * team_w
            cv2.line(
                frame,
                (xx, y + self.row_height),
                (xx, y + height),
                (255, 255, 255),
                1,
            )

        # Border
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + height),
            (255, 255, 255),
            2,
        )

        # ---------------------------
        # Title
        # ---------------------------

        self._center_text(
            frame,
            title,
            x + w / 2,
            y + self.row_height / 2,
            (0, 255, 255),
        )

        # ---------------------------
        # Team Names
        # ---------------------------

        header_y = y + self.row_height + self.row_height / 2

        for i, team in enumerate(teams):
            cx = x + metric_w + i * team_w + team_w / 2

            self._center_text(
                frame,
                team.replace("_", " ").title(),
                cx,
                header_y,
                (255, 255, 255),
            )

        # ---------------------------
        # Metrics
        # ---------------------------

        for r, metric in enumerate(metrics):

            cy = (
                y
                + (r + 2) * self.row_height
                + self.row_height / 2
            )

            # Metric name
            cv2.putText(
                frame,
                metric.replace("_", " ").title(),
                (x + 12, int(cy + 6)),
                self.font,
                self.font_scale,
                (255, 255, 255),
                self.thickness,
                cv2.LINE_AA,
            )

            # Values
            for c, team in enumerate(teams):

                value = str(stats[team][metric])

                cx = x + metric_w + c * team_w + team_w / 2

                self._center_text(
                    frame,
                    value,
                    cx,
                    cy,
                    (0, 255, 0),
                )

        return frame