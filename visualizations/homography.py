import cv2
from visualizations.court_tempelete import build_court_template
import numpy as np 
from .mini_maps import dynamic_map , heat_map
from utils import get_bottom_center_of_player , get_center_of_box
def compute_homography(
    src_pts,
    dst_pts ,
    min_points = 4,
    ransac_thresh = 5.0
) :

    if src_pts is None or len(src_pts) < min_points:
        return None, None

    H, mask = cv2.findHomography(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
    )
    return H, mask


def draw_mini_court(
    frames , 
    detections , 
    court_detectoins , 
    teams_classifications
) :
    # MAP DETECTIONS TO COURT TEMPELETE POINTS
    points_mapping = {
        27: "ctl",
        32: "ctr",
        5: "cbr",
        0: "cbl",
        28: "TTLT",
        31: "TTRT",
        24: "TTLB",
        25: "TTRB",
        19: "TTR",
        1: "TBLB",
        4: "TBRB",
        7: "TBLT",
        8: "TBRT",
        13: "TBR",
        29: "PTLT",
        30: "PTRT",
        21: "PTLB",
        23: "PTRB",
        22: "PTC",
        9: "PBLT",
        11: "PBRT",
        2: "PBLB",
        3: "PBRB",
        10: "PBC",
        15: "CL",
        14: "CR",
        16: "CC",
    }
    court_template_dynamic, points   = build_court_template()
    court_template_heat_map_1 , points = build_court_template()
    court_template_heat_map_2 , points = build_court_template()
    team_1_cumultive = []
    team_2_cumultive = []
    for i , (frame  , detection , court_detectoin , teams_classification ) in enumerate (zip(frames , detections , court_detectoins , teams_classifications )):
        
        # {'x': 413.0, 'y': 609.0, 'conf': 0.001953125, 'class_id': 3},
        frame_court_detections = [] 
        frame_tempelete_points = []


        for p in court_detectoin : 
            if p['class_id'] in points_mapping.keys() and p['conf'] > .5: 
                point_key = points_mapping[p['class_id']] 
                frame_tempelete_points.append(points[point_key]) 
                frame_court_detections.append((p['x'] , p['y'])) 
        frame_court_detections = np.array(frame_court_detections , dtype=np.float32) 
        frame_tempelete_points = np.array(frame_tempelete_points , dtype=np.float32)
        
        H, mask = compute_homography(frame_court_detections, frame_tempelete_points)

        team_1_positions = []
        team_2_positions = [] 
        ball_positions = []
        ref_positiions = [] 
        for box , class_id , tracker_id in zip(detection.xyxy , detection.class_id, detection.tracker_id) : 
            if class_id in [1,3,4,5,6,7] : 
                if tracker_id in teams_classification['boston'] : 
                    x_center , y_center = get_bottom_center_of_player(box) 
                    team_1_positions.append((x_center , y_center))
                elif tracker_id in teams_classification['new_york'] : 
                    x_center , y_center = get_bottom_center_of_player(box) 
                    team_2_positions.append((x_center , y_center))

            elif class_id == 8 : 
                x_center , y_center = get_bottom_center_of_player(box) 
                ref_positiions.append((x_center , y_center))
            elif class_id == 0 : 
                x_center , y_center = get_center_of_box(box) 
                ball_positions.append((x_center , y_center))

        # ── (c) overlays ─────────────────────────────────────
        if H is not None:
            frame = dynamic_map(i , frame , court_template_dynamic ,frame_court_detections , H ,
                                [ (np.array(team_1_positions) , (255,0,0))
                                   ,(np.array(team_2_positions) , (0,0,255) )  
                                   ,(np.array(ball_positions) , (30, 255, 255) )
                                #    ,(np.array(ref_positiions) , (30, 255, 255) )
                                   ] ,name='dynamic')
            frame , team_1_cumultive = heat_map(frame , court_template_heat_map_1 ,H , np.array(team_1_positions ) 
                                                , team_1_cumultive ,   position = 'top-right' , name ="BOSTON HEAT MAP" )
            frame , team_2_cumultive = heat_map(frame , court_template_heat_map_2 ,H , np.array(team_2_positions ) 
                                                , team_2_cumultive ,   position = 'bottom-right' , name='NEW YORK HEAT MAP' )
    return frames 