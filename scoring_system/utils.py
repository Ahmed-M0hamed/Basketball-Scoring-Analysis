import numpy as np 
from utils import get_bottom_center_of_player , distance_between_two_points 
import cv2 
from bisect import bisect_left,bisect_right

def get_less_than(sorted_list, target):
    # Find the index of the first element that is >= target
    index = bisect_left(sorted_list, target)
    
    # Slice the list from the beginning to that index
    return sorted_list[:index]
def get_more_than(sorted_list, target):
    # Find the index of the first element that is >= target
    index = bisect_right(sorted_list, target)
    
    # Slice the list from the beginning to that index
    return sorted_list[index:]

def check_3pts_or_2pts(position_player_made_shot , top_arc_pts , bottom_arc_pts    , THREE_LINE_TOP_LEFT , THREE_LINE_TOP_RIGHT , THREE_LINE_BOTTOM_LEFT , THREE_LINE_BOTTOM_RIGHT ) : 
    distance_to_arc = [distance_between_two_points(position_player_made_shot , point) for point in [THREE_LINE_TOP_LEFT , THREE_LINE_BOTTOM_LEFT]] 
    if np.argmin(distance_to_arc) == 0 : 
        # player closest to top arc 
        distances_to_arc = [distance_between_two_points(position_player_made_shot , point) for point in top_arc_pts] 
        closest_point_on_arc = top_arc_pts[np.argmin(distances_to_arc)] 
        x_player , y_player = position_player_made_shot
        x_arc , y_arc = closest_point_on_arc 
        if y_player > THREE_LINE_TOP_LEFT[1] and y_player < THREE_LINE_TOP_RIGHT[1] and x_player > x_arc :  
            return 2
        return 3 
    else : 
        # player closest to bottom arc 
        distances_to_arc = [distance_between_two_points(position_player_made_shot , point) for point in bottom_arc_pts] 
        closest_point_on_arc = bottom_arc_pts[np.argmin(distances_to_arc)] 
        x_player , y_player = position_player_made_shot
        x_arc , y_arc = closest_point_on_arc 
        if y_player > THREE_LINE_BOTTOM_LEFT[1] and y_player < THREE_LINE_BOTTOM_RIGHT[1] and x_player < x_arc :  
            return 2
        return 3
    
def check_assist(track_id_player_made_shot , frame_player_shot_ball  ,player_in_possession , detections , H , teams_classification ) :
    proj = None 
    track_id = None 
    for i in range(1 , len(get_less_than(player_in_possession , frame_player_shot_ball ) ) +1) : 
        closest_player_position_shot = get_less_than(player_in_possession , frame_player_shot_ball )[-i] if len(get_less_than(player_in_possession , frame_player_shot_ball )) > 0 else -1
        detection = detections[closest_player_position_shot]
        previous_player_in_position_index =np.where( detection.class_id == 4) 
        track_id_previous_player_in_postition = detection.tracker_id[previous_player_in_position_index]
        
        if len(track_id_previous_player_in_postition) == 1 :  
            if track_id_previous_player_in_postition != track_id_player_made_shot  :
                if (track_id_previous_player_in_postition in teams_classification['boston'] and track_id_player_made_shot in teams_classification['boston']) or (track_id_previous_player_in_postition in teams_classification['new_york'] and track_id_player_made_shot in teams_classification['new_york']) :
                    
                    position_player_made_shot = detection.xyxy[previous_player_in_position_index]
                    center_bottom_player_made_shot = np.array([get_bottom_center_of_player(position_player_made_shot[0])])
                    proj_center_bottom_player_made_shot = cv2.perspectiveTransform(
                        center_bottom_player_made_shot.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)
                    # print(f'player {track_id_previous_player_in_postition} made an assist to player {track_id_player_made_shot} at frame {frame_player_shot_ball} ')
                    proj = proj_center_bottom_player_made_shot[0]
                    track_id = track_id_previous_player_in_postition
                    break 
    return proj , track_id  