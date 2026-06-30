import cv2 
import os 
import numpy as np 
import math

def read_frames(path) : 
    frames = [] 
    cap = cv2.VideoCapture(path) 
    while True : 
        ret , frame = cap.read()  
        if not ret : 
            break 
        frames.append(frame) 
    cap.release() 
    return frames 

def write_video(  output_path , frames ) : 
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc,30 , (frames[0].shape[1], frames[0].shape[0]))
        for frame in frames :
            out.write(frame)
        out.release()

def distance_between_two_points(point_1 , point_2) : 
    x1 , y1 = point_1 
    x2, y2 = point_2 
    distance = np.sqrt((x2-x1) **2 + (y2-y1)**2)
    return distance 
def get_bottom_center_of_player(points) : 
    x1 , y1 , x2, y2 = points 
    x_center = int(x1 + (x2-x1  )/2 ) 
    y_center = int(y1-(y1-y2)) 
    return x_center, y_center

def draw_arc(image, pt_start, center_pt, pt_end , r , color, thickness=2):

    p1 = pt_start
    p2 = pt_end
    center = center_pt
    radius = r
    cx , cy = center_pt
    angle1 = math.degrees(math.atan2(p1[1] - cy, p1[0] - cx))
    angle2 = math.degrees(math.atan2(p2[1] - cy, p2[0] - cx))

    # Convert from [-180,180] to [0,360)
    angle1 %= 360
    angle2 %= 360
    cv2.ellipse(
        image,
        center,
        (radius, radius),   # circle
        0,                  # rotation
        angle1,
        angle2,
        color=color,
        thickness=thickness
        )
    return image 