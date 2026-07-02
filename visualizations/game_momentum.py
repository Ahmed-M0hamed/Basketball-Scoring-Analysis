import cv2 
import numpy as np 
from utils import get_bottom_center_of_player , distance_between_two_points , center_of_two_points



def get_momuntums(frame , frame_n, team_classification , detection , cumulative_momuntum , MAP , H ,THREE_LINE_TOP_LEFT , THREE_LINE_TOP_RIGHT , THREE_LINE_BOTTOM_LEFT , THREE_LINE_BOTTOM_RIGHT) : 
    offensing = 'boston' if  MAP[frame_n] == 0 else 'new_york'
    total_team_distance = 0 
    if offensing == 'boston' and len(detection.xyxy[np.where(detection.tracker_id ==team_classification['boston'][0])]) != 0  : 
        player_from_boston = np.array([get_bottom_center_of_player( detection.xyxy[np.where(detection.tracker_id ==team_classification['boston'][0] )][0] )]) 
        proj_player_from_boston = cv2.perspectiveTransform(
                player_from_boston.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)
        which_court_side_distances = [distance_between_two_points( proj_player_from_boston[0], point) for point in [THREE_LINE_TOP_LEFT , THREE_LINE_BOTTOM_LEFT]] 
        which_court_side = np.argmin(which_court_side_distances) # [0-> top_side , 1->bottom_side]
        boston_players_indexes = np.where(np.isin(detection.tracker_id, team_classification['boston']))[0].tolist()
        boston_players = np.array(detection.xyxy[boston_players_indexes]) 
        proj_players__boston = cv2.perspectiveTransform(
            boston_players.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)
        if which_court_side == 0 : 
            distances = [distance_between_two_points(player  , center_of_two_points(THREE_LINE_TOP_LEFT , THREE_LINE_TOP_RIGHT) ) for player in proj_players__boston]
            total_team_distance = sum(distances) 
        elif which_court_side == 1 : 
            distances = [distance_between_two_points(player  , center_of_two_points(THREE_LINE_BOTTOM_LEFT , THREE_LINE_BOTTOM_RIGHT) ) for player in proj_players__boston]
            total_team_distance = sum(distances) 
    elif offensing == 'new_york' and len( detection.xyxy[np.where(detection.tracker_id ==team_classification['new_york'][0])] ) != 0: 
        player_from_new_york = np.array([get_bottom_center_of_player( detection.xyxy[np.where(detection.tracker_id ==team_classification['new_york'][0])][0] )] ) 
        proj_player_from_new_york = cv2.perspectiveTransform(
                player_from_new_york.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)
        which_court_side_distances = [distance_between_two_points( proj_player_from_new_york[0], point) for point in [THREE_LINE_TOP_LEFT , THREE_LINE_BOTTOM_LEFT]] 
        which_court_side = np.argmin(which_court_side_distances) # [0-> top_side , 1->bottom_side]
        new_york_players_indexes = np.where(np.isin(detection.tracker_id, team_classification['new_york']))[0].tolist()
        new_york_players = np.array(detection.xyxy[new_york_players_indexes]) 
        proj_players_new_york = cv2.perspectiveTransform(
            new_york_players.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)
        if which_court_side == 0 : 
            distances = [distance_between_two_points(player  , center_of_two_points(THREE_LINE_TOP_LEFT , THREE_LINE_TOP_RIGHT) ) for player in proj_players_new_york]
            total_team_distance = sum(distances) 
        elif which_court_side == 1 : 
            distances = [distance_between_two_points(player  , center_of_two_points(THREE_LINE_BOTTOM_LEFT , THREE_LINE_BOTTOM_RIGHT) ) for player in proj_players_new_york]
            total_team_distance = sum(distances) 
    else :
        total_team_distance = 20000
    # print(total_team_distance)
    # print(frame_n)
    cumulative_momuntum.append(1 / total_team_distance)
    return cumulative_momuntum
    

