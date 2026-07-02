import cv2 
import numpy as np
import sys 
from utils import get_bottom_center_of_player , get_center_of_box
import math 
def draw_traingle(frame,bbox,color):

    y= int(bbox[1])
    x,_ = get_center_of_box(bbox)
    triangle_points = np.array([
        [x,y],
        [x-20,y-25],
        [x+20,y-25],
    ])
    cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
    cv2.drawContours(frame, [triangle_points],0,(0,0,0), 3)

    return frame

def draw_ellipse(frame,bbox,color,track_id=None):

    y2 = int(bbox[3])
    x_center, _ = get_center_of_box(bbox)
    width = bbox[2]-bbox[0]

    cv2.ellipse(
        frame,
        center=(x_center,y2),
        axes=(int(width), int(0.35*width)),
        angle=0.0,
        startAngle=-45,
        endAngle=235,
        color = color,
        thickness=3,
        lineType=cv2.LINE_4
    )

    

    return frame

def draw_arc_from_three_points(
    image,
    p1,
    p2,
    p3,
    color=(0, 255, 0),
    thickness=2,
    n_points=100,
    max_radius_ratio=50,  # NEW: radius vs chord-length sanity cap
):
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    p3 = np.asarray(p3, dtype=np.float64)

    A = np.array([
        [2*(p2[0]-p1[0]), 2*(p2[1]-p1[1])],
        [2*(p3[0]-p1[0]), 2*(p3[1]-p1[1])]
    ])
    b = np.array([
        p2[0]**2 + p2[1]**2 - p1[0]**2 - p1[1]**2,
        p3[0]**2 + p3[1]**2 - p1[0]**2 - p1[1]**2
    ])

    # --- Scale-aware collinearity check ---
    scale = np.linalg.norm(p2 - p1) + np.linalg.norm(p3 - p1) + 1e-6
    det = np.linalg.det(A)
    if abs(det) < 1e-6 * scale**2:
        cv2.line(image, tuple(p1.astype(int)), tuple(p2.astype(int)), color, thickness, cv2.LINE_AA)
        return image

    center = np.linalg.solve(A, b)
    radius = np.linalg.norm(center - p1)

    # --- Radius sanity cap: fall back to straight line if arc is basically flat ---
    chord = np.linalg.norm(p2 - p1)
    if radius > max_radius_ratio * chord:
        cv2.line(image, tuple(p1.astype(int)), tuple(p2.astype(int)), color, thickness, cv2.LINE_AA)
        return image

    def angle(point):
        return math.atan2(point[1] - center[1], point[0] - center[0])

    a1, a2, a3 = angle(p1) % (2*np.pi), angle(p2) % (2*np.pi), angle(p3) % (2*np.pi)

    def angle_between(start, end, test):
        return ((test - start) % (2*np.pi)) <= ((end - start) % (2*np.pi))

    if angle_between(a1, a2, a3):
        start, end = a1, a2
    else:
        start, end = a2, a1

    if end < start:
        end += 2*np.pi

    theta = np.linspace(start, end, n_points)
    pts = np.stack([
        center[0] + radius*np.cos(theta),
        center[1] + radius*np.sin(theta)
    ], axis=1)
    pts = np.round(pts).astype(np.int32)

    cv2.polylines(image, [pts], False, color, thickness, cv2.LINE_AA)
    return image

def make_bulge_point(p1, p2, bulge=0.2, side=1):
    """
    Create a third point offset perpendicular to the p1-p2 line,
    to be used as p3 in draw_arc_from_three_points.

    bulge : float
        Fraction of the p1-p2 distance to offset (0.2 = 20% bulge)
    side : 1 or -1
        Which side of the line to bulge toward
    """
    p1 = np.array(p1, dtype=np.float64)
    p2 = np.array(p2, dtype=np.float64)

    mid = (p1 + p2) / 2
    direction = p2 - p1
    length = np.linalg.norm(direction)

    # perpendicular vector
    perp = np.array([-direction[1], direction[0]]) / length

    p3 = mid + side * bulge * length * perp
    return tuple(p3)