import cv2 

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
            for box , track_id , name in zip(detection.xyxy , detection.tracker_id , detection.data['class_name']) : 
                if track_id in teams['boston'] : 
                    x1, y1, x2,y2 = box
                    cv2.rectangle(frame , (int(x1) , int(y1) ) , (int(x2) , int(y2)) , color= (255,0,0) ,thickness=2) 
                    cv2.putText(frame , f'id_{track_id}_{name} ' , (int(x1) , int(y1 - 5)) ,cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 255), 2  )      
                elif track_id in teams['new_york'] : 
                    x1, y1, x2,y2 = box
                    cv2.rectangle(frame , (int(x1) , int(y1) ) , (int(x2) , int(y2)) , color= (0,0,255) ,thickness=2) 
                    cv2.putText(frame , f'id_{track_id}_{name} ' , (int(x1) , int(y1 - 5)) ,cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 255), 2  ) 
                else : 
                    x1, y1, x2,y2 = box
                    cv2.rectangle(frame , (int(x1) , int(y1) ) , (int(x2) , int(y2)) , color= (0,255,0) , thickness=2) 
                    cv2.putText(frame , f'id_{track_id}_{name} ' , (int(x1) , int(y1 - 5)) ,cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 255), 2  ) 
            valid_team_member = teams 
            corrected_team_classifications.append(teams)
        return frames , corrected_team_classifications
    def draw_key_points(self , detections ,frames ) : 
        for detection , frame in zip(detections , frames ) : 
            for point in detection : 
                if point['conf'] > .5 : 
                    cv2.circle(frame , (int(point['x'] ) , int(point['y'])), 5, (0 ,0 ,255)  , -1)
                    cv2.putText(frame , f'p_{point['class_id']}' ,(int(point['x'] ) , int(point['y']-5)) ,cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 255), 2  )
        return frames  
