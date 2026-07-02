import cv2 
from .scoring_effects_function import text_poping , flash_screen 

def effects(frame_n , frame , popup_events , duration ) : 
    for event in popup_events : 
        if frame_n >= event['frame'] and frame_n <= (event['frame']+duration) : 
            color = (255 , 0 ,0 ) if event['team'] == 'new_york' else (0 ,255,0) 
            frame =text_poping(frame , frame_n , event['event'] , event['frame'] ,color, duration=duration) 
            frame = flash_screen(frame , frame_n , color , event['frame'] , duration = duration)

    return frame 