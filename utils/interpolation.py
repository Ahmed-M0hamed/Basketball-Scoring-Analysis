import pandas as pd 
import numpy as np 
from .helper import distance_between_two_points ,get_bottom_center_of_player
def tracker_hallosination(detections) : 
    max_detection_num = max([len(detection.xyxy) for detection in detections]) 
    player_detection_classes_ids =  [1,3,4,5,6,7]
    previous_tracker_ids = None 
    previous_player_boxes = None 
    for  detection in detections : 
        filtered_ids = []
        filtered_ids_indexes = []
        filtered_player_boxes = []
        for i , (tracker_id , class_id , box) in enumerate(zip(detection.tracker_id , detection.class_id , detection.xyxy) ): 
            if class_id in player_detection_classes_ids : 
                filtered_ids.append(tracker_id)
                filtered_player_boxes.append(box)
                filtered_ids_indexes.append(i)
        if not any(track_id > max_detection_num for track_id in filtered_ids) :
            previous_tracker_ids = filtered_ids 
            previous_player_boxes = filtered_player_boxes 

        else : 
            mis_tracked = list(set(filtered_ids) - set(previous_tracker_ids)) 
            targets_for_comparasion = list(set(previous_tracker_ids) - set(filtered_ids)) 
            if not targets_for_comparasion : 
                for val in mis_tracked : 
                    filtered_player_boxes.pop(filtered_ids.index(val))
                    ind = np.where(detection.tracker_id == val)
                    detection.xyxy = np.delete( detection.xyxy,  ind , axis = 0) 
                    filtered_ids.remove(val) 
                    detection.tracker_id = detection.tracker_id[detection.tracker_id!=val]
                previous_tracker_ids = filtered_ids 
                previous_player_boxes = filtered_player_boxes
                continue

            mis_tracked_indexes = [filtered_ids.index(element) for element in mis_tracked] 
            targets_for_comparasion_indexes = [previous_tracker_ids.index(element) for element in targets_for_comparasion]
            # print(filtered_ids)
            # print(previous_tracker_ids)
            # print(mis_tracked)
            # print(targets_for_comparasion)
            # print('-------------------')
            distances = [] 
            for mis_tracked_index in mis_tracked_indexes : 
                mis_tracked_index_distances = []
                for targets_for_comparasion_index in targets_for_comparasion_indexes : 
                    mis_tracked_index_distances.append(distance_between_two_points( get_bottom_center_of_player(filtered_player_boxes[mis_tracked_index]) , get_bottom_center_of_player(previous_player_boxes[targets_for_comparasion_index])) )
                distances.append(mis_tracked_index_distances) 
            # print(distances)
            indexes_min_distances = [np.argmin(d) for d in distances] 
            correction_ids = [previous_tracker_ids[index] for index in indexes_min_distances]
            corrected_tracker_ids = filtered_ids
            corrected_player_boxes = filtered_player_boxes 
            for mis_tracked_index , correction_id in zip(mis_tracked_indexes , correction_ids) : 
                corrected_tracker_ids[mis_tracked_index] = correction_id
                corrected_player_boxes[mis_tracked_index] = previous_player_boxes[previous_tracker_ids.index(correction_id)]
            previous_tracker_ids = corrected_tracker_ids 
            previous_player_boxes = corrected_player_boxes
            for filtered_ids_index , corrected_tracker_id in zip(filtered_ids_indexes,previous_tracker_ids) : 
                detection.tracker_id[filtered_ids_index] =  corrected_tracker_id 
            
        print(detection.xyxy) 
        