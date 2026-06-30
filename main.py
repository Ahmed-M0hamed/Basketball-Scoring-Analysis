import os 
# from utils import read_frames 
from utils  import read_frames , write_video , tracker_hallosination
from infering import Inference 
from draw import Draw
from visualizations import draw_mini_court
import warnings

def main():

    frames = read_frames(os.path.join(os.getcwd() , 'inputs/extended_input.mp4')) 
    inference_class= Inference(frames ) 
    detections = inference_class.player_ball_detectioin('basketball-player-detection-3-ycjdo/4' , presaved=True  , path = os.path.join(os.getcwd() , 'presaved/player_ball.pkl'))
    court_detections = inference_class.court_keypoints('basketball-court-detection-2/14' , presaved=True , path=os.path.join(os.getcwd() , 'presaved/court.pkl'))
    teams = inference_class.one_shot_classification(detections , presaved=True , path = os.path.join(os.getcwd() , 'presaved/teams.pkl'))
    # inference_class.OCR(detections , frames )
    # tracker_hallosination(detections)

    drawer =Draw()
    annotated_frames, corrected_team_classifications =drawer.draw_players_ball(detections , teams , frames) 
    annotated_frames = drawer.draw_key_points(court_detections , annotated_frames)
    annotated_frames = draw_mini_court(annotated_frames , detections , court_detections ,corrected_team_classifications)
    write_video(os.path.join(os.getcwd() , 'outputs/out_3.mp4') , annotated_frames[:238])

if __name__ == "__main__":
    main()
