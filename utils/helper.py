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

def get_center_of_box(points ) : 
    x1 , y1 , x2, y2 = points 
    x_center = int(x1 + (x2-x1  )/2 ) 
    y_center = int(y1-(y1-y2)/2 ) 
    return x_center, y_center 

def center_of_two_points(point_1 , point_2) : 
    x1, y1 = point_1 
    x2 , y2 = point_2 
    return (int( (x1+x2) /2) , int( (y1+y2) /2))