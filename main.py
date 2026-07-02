import os 
# from utils import read_frames 
from utils  import read_frames , write_video , tracker_hallosination
from infering import Inference 
from draw import Draw
from visualizations import draw_mini_court
from scoring_system import find_events  , find_stats 

def main():

    frames = read_frames(os.path.join(os.getcwd() , 'inputs/extended_input.mp4')) 
    inference_class= Inference(frames ) 
    detections = inference_class.player_ball_detectioin('basketball-player-detection-3-ycjdo/4' , presaved=True  , path = os.path.join(os.getcwd() , 'presaved/player_ball.pkl'))
    court_detections = inference_class.court_keypoints('basketball-court-detection-2/14' , presaved=True , path=os.path.join(os.getcwd() , 'presaved/court.pkl'))
    teams = inference_class.one_shot_classification(detections , presaved=True , path = os.path.join(os.getcwd() , 'presaved/teams.pkl'))
    events = find_events(detections)
    # print(events )

    drawer =Draw()
    annotated_frames, corrected_team_classifications =drawer.draw_players_ball(detections , teams , frames) 
    annotated_frames = drawer.draw_key_points(court_detections , annotated_frames)
    annotated_frames  , stats , foot_prints , cum= draw_mini_court(annotated_frames , detections , court_detections ,corrected_team_classifications , events )
    # print(cum)
    write_video(os.path.join(os.getcwd() , 'outputs/out_5.mp4') , annotated_frames)

if __name__ == "__main__":
    main()
