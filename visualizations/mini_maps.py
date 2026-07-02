import cv2
import numpy as np 
from scoring_system import get_less_than

def dynamic_map(frame_n , 
    frame,
    court_template,
    kp_frame,
    H,
    player_ref_ball , 
    position="top_right",
    size_frac =  0.12,
    name = ''
) : 
    
    fh, fw = frame.shape[:2]
    th, tw = court_template.shape[:2]

    target_w =   int(fw  * size_frac) 
    target_h = int(th * int(fw  * size_frac)  / tw)
    mini = cv2.resize(court_template.copy(), (target_w, target_h))

    sx, sy = target_w / tw, target_h / th 

    if H is not None and kp_frame is not None:
        proj = cv2.perspectiveTransform(
            kp_frame.reshape(-1, 1, 2), H).reshape(-1, 2)
        for px, py in proj:
                mx, my = int(px * sx), int(py * sy)
                # if 0 <= mx < target_w and 0 <= my < target_h:
                #     cv2.circle(mini, (mx, my), 3, (0, 0, 255), -1)
    if player_ref_ball and H is not None:
            for pts, color in player_ref_ball:
                if pts.shape[0] != 0  :
                    proj_extra = cv2.perspectiveTransform(
                        pts.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)
                    for px, py in proj_extra:
                        mx, my = int(px * sx), int(py * sy)
                        cv2.circle(mini, (mx, my), 5, color  , -1)
    
    pad = 80
    if position == "bottom-right":
        x0, y0 = fw - target_w - pad, fh - target_h - pad
    elif position == "bottom-left":
        x0, y0 = pad, fh - target_h - pad
    elif position == "top-right":
        x0, y0 = fw - target_w - pad, pad
    else:   # top-left
        x0, y0 = pad, pad

    roi = frame[y0:y0+target_h, x0:x0+target_w]
    frame[y0:y0+target_h, x0:x0+target_w] = cv2.addWeighted(
        roi, 0.4, mini, 0.6, 0)
    cv2.rectangle(frame,
                  (x0 - 2, y0 - 2),
                  (x0 + target_w + 2, y0 + target_h + 2),
                  (255, 255, 255), 1)
    cv2.putText(frame ,str(name) , (x0 , y0-15),cv2.FONT_HERSHEY_SIMPLEX,.9,(0, 0, 255),2,cv2.LINE_AA  )
    return frame


def heat_map( 
    frame,
    court_template,
    H,
    team , 
    team_cumilative , 
    position="top_right",
    size_frac =  0.12,
    name = ''
) : 
    
    fh, fw = frame.shape[:2]
    th, tw = court_template.shape[:2]

    target_w =   int(fw  * size_frac) 
    target_h = int(th * int(fw  * size_frac)  / tw)
    mini = cv2.resize(court_template.copy(), (target_w, target_h))

    sx, sy = target_w / tw, target_h / th 


    map = np.zeros_like(mini, dtype=np.uint8)
    if  H is not None:
        if team.shape[0] != 0 : 
            proj_extra = cv2.perspectiveTransform(
                team.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)
            team_cumilative.append(proj_extra)
            for frame_points in team_cumilative  : 
                for point in frame_points : 
                    px , py = point 
                    mx, my = int(px * sx), int(py * sy)
                    if 0 <= mx < target_w and 0 <= my < target_h:
                        map[my , mx] += 100
            map = cv2.GaussianBlur(map, (0,0),
                    sigmaX=25,
                    sigmaY=25
                    )
            normalized = np.zeros_like(map, dtype=np.uint8)
            map = cv2.normalize(
                map,
                normalized,
                0, 255, cv2.NORM_MINMAX
            )
            map = cv2.applyColorMap(
            map,
            cv2.COLORMAP_JET ) 
            
            mini = cv2.addWeighted(mini , .5 , map , .5 , 0 )
    
    pad = 80
    if position == "bottom-right":
        x0, y0 = fw - target_w - pad, fh - target_h - pad
    elif position == "bottom-left":
        x0, y0 = pad, fh - target_h - pad
    elif position == "top-right":
        x0, y0 = fw - target_w - pad, pad
    else:   # top-left
        x0, y0 = pad, pad

    roi = frame[y0:y0+target_h, x0:x0+target_w]
    frame[y0:y0+target_h, x0:x0+target_w] = cv2.addWeighted(
        roi, 0.4, mini, 0.6, 0)
    cv2.rectangle(frame,
                  (x0 - 2, y0 - 2),
                  (x0 + target_w + 2, y0 + target_h + 2),
                  (255, 255, 255), 1)
    cv2.putText(frame ,str(name) , (x0 , y0-15),cv2.FONT_HERSHEY_SIMPLEX,.9,(0, 0, 255),2,cv2.LINE_AA  )
    return frame ,team_cumilative

def foot_prints_map(frame_n , 
    frame,
    court_template,
    foot_prints , 
    position="top_right",
    size_frac =  0.12,
    name = '') : 
    fh, fw = frame.shape[:2]
    th, tw = court_template.shape[:2]

    target_w =   int(fw  * size_frac) 
    target_h = int(th * int(fw  * size_frac)  / tw)
    mini = cv2.resize(court_template.copy(), (target_w, target_h))

    sx, sy = target_w / tw, target_h / th 

    foot_prints_frames = [foot['frame'] for foot in foot_prints] 

    previous_foot_prints = get_less_than(foot_prints_frames , frame_n )
    for foot in foot_prints : 
       if foot['frame'] in previous_foot_prints : 
            color = (0 , 0  , 255) if foot['team'] == 'boston' else (255 , 0 , 0)
            scorer_x , scorer_y = foot['scorer_position'] 
            if foot['assist_id'] is not None : 
                assist_x , assist_y = foot['assist_position'] 
                mx, my = int(assist_x * sx), int(assist_y * sy)
                cv2.putText(mini, '+' ,(mx, my), cv2.FONT_HERSHEY_SIMPLEX, .75, color, 2 )
            if scorer_x != -1:
                mx, my = int(scorer_x * sx), int(scorer_y * sy)
                cv2.circle(mini, (mx, my), 7, color, -1)







    pad = 80
    if position == "bottom-right":
        x0, y0 = fw - target_w - pad, fh - target_h - pad
    elif position == "bottom-left":
        x0, y0 = pad, fh - target_h - pad
    elif position == "top-right":
        x0, y0 = fw - target_w - pad, pad
    else:   # top-left
        x0, y0 = pad, pad

    roi = frame[y0:y0+target_h, x0:x0+target_w]
    frame[y0:y0+target_h, x0:x0+target_w] = cv2.addWeighted(
        roi, 0.4, mini, 0.6, 0)
    cv2.rectangle(frame,
                  (x0 - 2, y0 - 2),
                  (x0 + target_w + 2, y0 + target_h + 2),
                  (255, 255, 255), 1)
    cv2.putText(frame ,str(name) , (x0 , y0-15),cv2.FONT_HERSHEY_SIMPLEX,.9,(0, 0, 255),2,cv2.LINE_AA  )
    return frame