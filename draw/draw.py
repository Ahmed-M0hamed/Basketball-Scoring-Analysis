import cv2 
from utils import get_bottom_center_of_player , get_center_of_box
from .utils import draw_ellipse , draw_traingle ,draw_arc_from_three_points , make_bulge_point
class Draw :
 
  
    def draw_players_ball(self , detections  , teams_classifications  , frames ) : 
        valid_team_member = None 
        corrected_team_classifications  = []
        for i , (detection , frame , teams) in enumerate(zip(detections ,frames , teams_classifications)) : 
            if len(teams['boston']) != len(teams['new_york']) and valid_team_member is not None : 
                teams = valid_team_member
            elif len(teams['boston']) != len(teams['new_york']) and valid_team_member is  None : 
                j = i 
                found = True 
                while found : 
                    if len(teams_classifications[j]['boston']) != len(teams_classifications[j]['new_york']) and j < len(teams_classifications): 
                        j += 1 
                    elif len(teams_classifications[j]['boston']) == len(teams_classifications[j]['new_york']) and j < len(teams_classifications)  : 
                        found = False 
                        teams = teams_classifications[j]

            cv2.putText(frame , f'frame_{i}' , (100 , 100) , cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 255), 2  )
            for box , track_id ,class_id ,  name in zip(detection.xyxy , detection.tracker_id , detection.class_id , detection.data['class_name']) : 
                if track_id in teams['boston'] : 
                    frame = draw_ellipse(frame , box , (0 , 0 ,255) )
                    frame = draw_traingle(frame , box , (0 , 0 ,  255))

                    # cv2.rectangle(frame , (int(x1) , int(y1) ) , (int(x2) , int(y2)) , color= (255,0,0) ,thickness=2) 
                    # cv2.putText(frame , f'id_{track_id} ' , (int(x1) , int(y1 - 5)) ,cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 255), 2  )      
                elif track_id in teams['new_york'] : 
                    frame = draw_ellipse(frame , box , (255 , 0 ,0) )
                    frame = draw_traingle(frame , box , (255 , 0 ,  0))
                elif  class_id == 8  : 

                    frame = draw_ellipse(frame , box , (0 , 0 ,0) )
                    frame = draw_traingle(frame , box , (0 , 0 ,0))
                elif class_id == 0 : 
                    x1, y1, x2,y2 = box
                    x_c , y_c = get_center_of_box(box)
                    radius = int(abs((x2 - x1)) /2 ) 
                    cv2.circle(frame , (int(x_c) , int(y_c)) , radius ,(0 ,255,0) , 2 , -1 )
                    frame = draw_traingle(frame , box, (0 , 255,0 )) 
            valid_team_member = teams 
            corrected_team_classifications.append(teams)
        return frames , corrected_team_classifications
    def draw_key_points(self , detections ,frames ) : 
        line_pairs = [(31 ,32)   , (30,29) , (28, 29) , (27 ,28) , (28,24) 
                 , (21,29 ) ,(23,30) ,(31,25) , (27,18) , (18 , 15)
                 , (32,14) , (21,23) , (15,16) , (16,14) , (0,1) ,(1,2) ,(2,3)
                 ,(3,4) ,(4,5) ,(0,12) , (12,15) , (1,7) , (4,8) , (5,14) , (9,11)]
        ellipsis_pairs = [(25 , 24 , 19) , (7,8,13)] 
        for detection , frame in zip(detections , frames ) : 
            exsist_class_ids = {}
            for point in detection : 
                if point['conf'] > .5 : 
                    cv2.circle(frame , (int(point['x'] ) , int(point['y'])), 7, (30, 255, 255)  , -1)
                    cv2.putText(frame , f'p_{point['class_id']}' ,(int(point['x'] ) , int(point['y']-5)) ,cv2.FONT_HERSHEY_SIMPLEX, .75, (30, 255, 255), 2  )
                    exsist_class_ids[point['class_id']] = (int(point['x']) , int(point['y']))

            # for pair in line_pairs : 
            #     p1 ,p2 = pair 
            #     if p1 in exsist_class_ids and p2 in exsist_class_ids:
            #         cv2.line(frame ,exsist_class_ids[p1]  , exsist_class_ids[p2] , 
            #               (30, 255, 255) , 2     )
            
        return frames  
