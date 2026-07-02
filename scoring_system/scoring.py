import numpy as np 
from utils import get_bottom_center_of_player , distance_between_two_points 
import cv2 


from bisect import bisect_left
from .utils import get_less_than , check_3pts_or_2pts , check_assist , get_more_than

  




def find_stats(stats ,foot_prints, popup_events , events , frame_n , teams_classifications , detections , H , top_arc_pts , bottom_arc_pts , THREE_LINE_TOP_LEFT , THREE_LINE_TOP_RIGHT , THREE_LINE_BOTTOM_LEFT , THREE_LINE_BOTTOM_RIGHT ) : 
    baskets =  events['ball-in-basket']
    jump_shots = events['player-jump-shot']
    layup_dunks = events['player-layup-dunk'] 
    player_in_possession = events['player-in-possestion']
    player_shot_block = events['player-shot-block']

    if frame_n in baskets :  
        closest_jump_shot = get_less_than(jump_shots , frame_n )[-1] if len(get_less_than(jump_shots , frame_n )) > 0 else -1
        closest_layup_dunk = get_less_than(layup_dunks , frame_n )[-1] if len(get_less_than(layup_dunks , frame_n )) > 0 else -1
        get_next_ball_possession = get_more_than(player_in_possession , frame_n )[0] if len(get_more_than(player_in_possession , frame_n )) > 0 else -1
        if  closest_jump_shot >  closest_layup_dunk : 
            detection = detections[closest_jump_shot] 
            player_made_shot =np.where( detection.class_id == 5)
            position_player_made_shot = detection.xyxy[player_made_shot]
            # print(f'position_player_made_shot: {position_player_made_shot}')
            track_id_player_made_shot = detection.tracker_id[player_made_shot]
            center_bottom_player_made_shot = np.array([get_bottom_center_of_player(position_player_made_shot[0])])
            proj_center_bottom_player_made_shot = cv2.perspectiveTransform(
                center_bottom_player_made_shot.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)
            shot = check_3pts_or_2pts(proj_center_bottom_player_made_shot[0] , top_arc_pts , bottom_arc_pts    , THREE_LINE_TOP_LEFT , THREE_LINE_TOP_RIGHT , THREE_LINE_BOTTOM_LEFT , THREE_LINE_BOTTOM_RIGHT )
            assist_proj, assist_track_id = check_assist(track_id_player_made_shot ,closest_jump_shot , player_in_possession , detections ,H  , teams_classifications[closest_jump_shot] )
            if track_id_player_made_shot in teams_classifications[closest_jump_shot]['boston'] :
                stats['boston']['score'] += shot
                foot_prints.append({'frame' : frame_n  , 'team' : 'boston' , 'scorer_id' : track_id_player_made_shot , 'shot_type' : shot  , 'scorer_position' : proj_center_bottom_player_made_shot[0] , 'assist_id' : assist_track_id , 'assist_position' : assist_proj})
                popup_events.append({'frame' : frame_n  , 'team' : 'boston' , 'event' : f'{shot} points Boston'})
            else : 
                stats['new_york']['score'] += shot
                foot_prints.append({'frame' : frame_n  , 'team' : 'new_york' , 'scorer_id' : track_id_player_made_shot , 'shot_type' : shot  , 'scorer_position' : proj_center_bottom_player_made_shot[0] , 'assist_id' : assist_track_id , 'assist_position' : assist_proj})
                popup_events.append({'frame' : frame_n  , 'team' : 'new_york' , 'event' : f'{shot} points Newyork'})
                # print(f'frame {frame_n} new_york made a jump shot player {track_id_player_made_shot} {shot_type}' )

        elif closest_layup_dunk >  closest_jump_shot : 
            detection = detections[closest_layup_dunk] 
            player_made_shot = np.where( detection.class_id == 6) 
            position_player_made_shot = detection.xyxy[player_made_shot]
            track_id_player_made_shot = detection.tracker_id[player_made_shot]
            center_bottom_player_made_shot = np.array([get_bottom_center_of_player(position_player_made_shot[0])])
            proj_center_bottom_player_made_shot = cv2.perspectiveTransform(
                center_bottom_player_made_shot.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)
            shot = check_3pts_or_2pts(proj_center_bottom_player_made_shot[0] , top_arc_pts , bottom_arc_pts    , THREE_LINE_TOP_LEFT , THREE_LINE_TOP_RIGHT , THREE_LINE_BOTTOM_LEFT , THREE_LINE_BOTTOM_RIGHT )
            assist_proj, assist_track_id = check_assist(track_id_player_made_shot ,closest_layup_dunk , player_in_possession , detections ,H , teams_classifications[closest_layup_dunk] )
            if track_id_player_made_shot in  teams_classifications[closest_layup_dunk]['boston'] :
                stats['boston']['score'] += shot
                foot_prints.append({'frame' : frame_n  , 'team' : 'boston' , 'scorer_id' : track_id_player_made_shot , 'shot_type' : shot  , 'scorer_position' : proj_center_bottom_player_made_shot[0] , 'assist_id' : assist_track_id , 'assist_position' : assist_proj})
                popup_events.append({'frame' : frame_n  , 'team' : 'boston' , 'event' : f'{shot} points Boston'})
            if track_id_player_made_shot in  teams_classifications[closest_layup_dunk]['new_york'] :
                stats['new_york']['score'] += shot
                foot_prints.append({'frame' : frame_n  , 'team' : 'new_york' , 'scorer_id' : track_id_player_made_shot , 'shot_type' : shot  , 'scorer_position' : proj_center_bottom_player_made_shot[0] , 'assist_id' : assist_track_id , 'assist_position' : assist_proj})
                popup_events.append({'frame' : frame_n  , 'team' : 'new_york' , 'event' : f'{shot} points Newyork'})
    # CHECK REPOND 
    if frame_n in jump_shots or frame_n in layup_dunks :
        get_next_ball_possession = get_more_than(player_in_possession , frame_n )[0] if len(get_more_than(player_in_possession , frame_n )) > 0 else -1
        get_next_basket_made = get_more_than(baskets , frame_n )[0] if len(get_more_than(baskets , frame_n )) > 0 else -1
        
        if get_next_ball_possession != -1 and get_next_basket_made != -1 :
            if get_next_ball_possession < get_next_basket_made : 
                detection = detections[get_next_ball_possession] 
                shot_detection = detections[frame_n]
                player_in_possession = np.where( detection.class_id == 4) 
                palyer_made_shot = np.where( shot_detection.class_id == 6 or shot_detection.class_id == 5) 
                track_id_player_in_possession = detection.tracker_id[player_in_possession]
                track_id_player_made_shot = shot_detection.tracker_id[palyer_made_shot]
                if track_id_player_made_shot in teams_classifications[frame_n]['boston'] and  track_id_player_in_possession in teams_classifications[get_next_ball_possession]['boston'] : 
                    stats['boston']['repond'] += 1
                    popup_events.append({'frame' : frame_n  , 'team' : 'boston' , 'event' : 'boston offensive repond'})
                elif track_id_player_made_shot in teams_classifications[frame_n]['new_york'] and  track_id_player_in_possession in teams_classifications[get_next_ball_possession]['new_york'] : 
                    stats['new_york']['repond'] += 1
                    popup_events.append({'frame' : frame_n  , 'team' : 'new_york' , 'event' : 'new york offensive repond'})
                elif track_id_player_made_shot in teams_classifications[frame_n]['boston'] and  track_id_player_in_possession in teams_classifications[get_next_ball_possession]['new_york'] : 
                    stats['new_york']['repond'] += 1
                    popup_events.append({'frame' : frame_n  , 'team' : 'new_york' , 'event' : 'boston deffensive repond'})
                elif track_id_player_made_shot in teams_classifications[frame_n]['new_york'] and  track_id_player_in_possession in teams_classifications[get_next_ball_possession]['boston'] : 
                    stats['boston']['repond'] += 1
                    popup_events.append({'frame' : frame_n  , 'team' : 'boston' , 'event' : 'new york deffensive repond'})
    # CHECK TURNOVERS and PASSES 
    if frame_n in player_in_possession : 
        get_next_ball_possession = get_more_than(player_in_possession , frame_n )[0] if len(get_more_than(player_in_possession , frame_n )) > 0 else -1
        get_next_jump_shot = get_more_than(jump_shots , frame_n )[0] if len(get_more_than(jump_shots , frame_n )) > 0 else -1
        get_next_layup_dunk = get_more_than(layup_dunks , frame_n )[0] if len(get_more_than(layup_dunks , frame_n )) > 0 else -1
        if get_next_ball_possession != -1 and get_next_jump_shot != -1 and get_next_layup_dunk != -1 : 
            if get_next_ball_possession < get_next_jump_shot and get_next_ball_possession < get_next_layup_dunk : 
                detection = detections[frame_n] 
                next_detection = detections[get_next_ball_possession]
                player_in_possession = np.where( detection.class_id == 4) 
                next_player_in_possession = np.where( next_detection.class_id == 4) 

                if len(player_in_possession) == 1 and len(next_player_in_possession) == 1 :
                    track_id_player_in_possession = detection.tracker_id[player_in_possession]
            
                    track_id_next_player_in_possession = next_detection.tracker_id[next_player_in_possession]

                    if len(track_id_player_in_possession) == 1 and len(track_id_next_player_in_possession) == 1 :
                        if track_id_player_in_possession != track_id_next_player_in_possession : 
                            if (track_id_player_in_possession in teams_classifications[get_next_ball_possession]['boston'] and track_id_next_player_in_possession in teams_classifications[get_next_ball_possession]['new_york']) or (track_id_player_in_possession in teams_classifications[get_next_ball_possession]['new_york'] and track_id_next_player_in_possession in teams_classifications[get_next_ball_possession]['boston']) :
                                if track_id_player_in_possession in teams_classifications[get_next_ball_possession]['boston'] : 
                                    stats['boston']['turnovers'] += 1
                                    popup_events.append({'frame' : frame_n  , 'team' : 'boston' , 'event' : 'turnover boston'})
                                else : 
                                    stats['new_york']['turnovers'] += 1
                                    popup_events.append({'frame' : frame_n  , 'team' : 'new_york' , 'event' : 'turnover newyork'})
                            else : 
                                if track_id_player_in_possession in teams_classifications[get_next_ball_possession]['boston'] : 
                                    stats['boston']['passes'] += 1

                                else : 
                                    stats['new_york']['passes'] += 1


    return stats , foot_prints , popup_events