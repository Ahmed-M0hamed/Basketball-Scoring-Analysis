import numpy as np 
from typing import List, Tuple, Dict, Any

# def frames_to_ranges(frames):
#     if not frames:
#         return []

#     frames = sorted(frames)  # Optional if already sorted

#     ranges = []
#     start = end = frames[0]

#     for frame in frames[1:]:
#         if frame == end + 1:
#             end = frame
#         else:
#             ranges.append({"start": start, "end": end})
#             start = end = frame

#     ranges.append({"start": start, "end": end})

#     return ranges
def event_start_frames(frames):
    if not frames:
        return []

    frames = sorted(frames)  # Optional if already sorted

    starts = [frames[0]]

    for prev, curr in zip(frames, frames[1:]):
        if curr != prev + 1:
            starts.append(curr)

    return starts
def find_events(detections) : 
    '''
        main events : 
            player-in-possestion of ball : class 4 
            player shot block : class 7 
            player jump shot : class 5
            ball-in-basket :  class 1 
            player-layup-dunk : 6 
    '''
    events = {'player-in-possestion' : [] , 'player-shot-block' : [] , 'player-jump-shot' : [] , 'ball-in-basket' : [] , 'player-layup-dunk' : []}
    for i ,detection in enumerate(detections) : 
        for box , track_id ,class_id ,  name in zip(detection.xyxy , detection.tracker_id , detection.class_id , detection.data['class_name']) : 
            if class_id == 4 : 
                events['player-in-possestion'].append(i)
            elif class_id == 7 : 
                events['player-shot-block'].append(i)
            elif class_id == 5 : 
                events['player-jump-shot'].append(i )
            elif class_id == 1 : 
                events['ball-in-basket'].append(i)
            elif class_id == 6 : 
                events['player-layup-dunk'].append(i)
    events = {k: event_start_frames(v) for k, v in events.items()}
    return events 

