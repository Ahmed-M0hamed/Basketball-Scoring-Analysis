import cv2
import math
import numpy as np


def draw_arc_from_three_points(
    image,
    p1,
    p2,
    p3,
    color=(0, 255, 0),
    thickness=2,
    n_points=100,
):
    """
    Draw the unique circular arc passing through p1 -> p3 -> p2.

    Parameters
    ----------
    p1 : tuple
        Arc start point.
    p2 : tuple
        Arc end point.
    p3 : tuple
        Any point lying on the desired arc.
    """

    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    p3 = np.asarray(p3, dtype=np.float64)

    # ----------------------------
    # Compute circle center
    # ----------------------------
    A = np.array([
        [2*(p2[0]-p1[0]), 2*(p2[1]-p1[1])],
        [2*(p3[0]-p1[0]), 2*(p3[1]-p1[1])]
    ])

    b = np.array([
        p2[0]**2 + p2[1]**2 - p1[0]**2 - p1[1]**2,
        p3[0]**2 + p3[1]**2 - p1[0]**2 - p1[1]**2
    ])

    # Three collinear points
    if abs(np.linalg.det(A)) < 1e-8:
        cv2.line(
            image,
            tuple(p1.astype(int)),
            tuple(p2.astype(int)),
            color,
            thickness,
        )
        return None

    center = np.linalg.solve(A, b)
    radius = np.linalg.norm(center - p1)

    # ----------------------------
    # Compute angles
    # ----------------------------
    def angle(point):
        return math.atan2(
            point[1] - center[1],
            point[0] - center[0]
        )

    a1 = angle(p1)
    a2 = angle(p2)
    a3 = angle(p3)

    # normalize to [0, 2pi)
    a1 %= 2*np.pi
    a2 %= 2*np.pi
    a3 %= 2*np.pi

    def angle_between(start, end, test):
        return ((test - start) % (2*np.pi)) <= ((end - start) % (2*np.pi))

    # choose direction that passes through p3
    if angle_between(a1, a2, a3):
        start = a1
        end = a2
    else:
        start = a2
        end = a1

    # unwrap
    if end < start:
        end += 2*np.pi

    theta = np.linspace(start, end, n_points)

    pts = np.stack([
        center[0] + radius*np.cos(theta),
        center[1] + radius*np.sin(theta)
    ], axis=1)

    pts = np.round(pts).astype(np.int32)

    cv2.polylines(
        image,
        [pts],
        False,
        color,
        thickness,
        cv2.LINE_AA,
    )

    return  image  , pts

def check_which_team_offencing(ball_possitions_frames , detections , teams_classifications) :
    offensive_map = [0] * len(detections) 
    last_ball_pos_index = 0 
    for ball_bosition_frame in  ball_possitions_frames : 
        teams = teams_classifications[ball_bosition_frame] 
        detection = detections[ball_bosition_frame]
        player_posses_index = np.where(detection.class_id == 4)
        track_id_player_possess = detection.tracker_id[player_posses_index] 
        if len(track_id_player_possess) == 1  and track_id_player_possess in teams['boston'] : 
            offensive_map[last_ball_pos_index:ball_bosition_frame+1 ] = [0] * (ball_bosition_frame  - last_ball_pos_index)
            last_ball_pos_index = ball_bosition_frame+1
        elif len(track_id_player_possess) == 1 and  track_id_player_possess in teams['new_york'] : 
            offensive_map[last_ball_pos_index:ball_bosition_frame+1 ] = [1] * (ball_bosition_frame  - last_ball_pos_index)
            last_ball_pos_index = ball_bosition_frame+1
    
    offensive_map[ ball_possitions_frames[-1]  : len(detections) ]  = [0] * (len(detections) - ball_possitions_frames[-1])

    return offensive_map 

def smooth_curve(signal_history) : 
    WINDOW = 5
    if len(signal_history ) == WINDOW : 
        smoothed_distance = np.mean(signal_history[-WINDOW:])
        signal_history[-1] = smoothed_distance
    return signal_history 