# import inference 
import os
import sys 
import pickle
# from inference import get_model
import supervision as sv 
from PIL import Image
import cv2
from transformers import CLIPProcessor, CLIPModel , pipeline ,AutoProcessor, AutoModelForMultimodalLM 
import easyocr
import warnings
warnings.filterwarnings('ignore')
import numpy as np 
import torch 
class Inference : 
    def __init__(self , frames ) : 
        self.frames = frames
        
    
    def player_ball_detectioin(self , model_name : str , presaved:bool = False , path:str = None) :
        if presaved and path is not None : 
            with open(path , 'rb') as f : 
                detections = pickle.load(f) 
            return detections 
        
        #  model  =get_model(model_id = model_name )
        # tracker = sv.ByteTrack()
        # detections = []
        # for frame in self.frames : 
        #     result = model.infer(frame  ,confidence=.4, iou_threshold=.9)[0]  
        #     detection = sv.Detections.from_inference(result)
        #     tracked_detections = tracker.update_with_detections(detection)
        #     detections.append(tracked_detections)

        if path is not None : 
            with open(path , 'wb')as f  : 
                pickle.dump(detections , f ) 
        return detections 
    def court_keypoints(self , model_name : str , presaved:bool = False , path:str = None) :
        if presaved and path is not None : 
            with open(path , 'rb') as f : 
                detections = pickle.load(f) 
            return detections 
        
        # model  =get_model(model_id = model_name )
        # detections = []
        # for frame in self.frames : 
        #     result = model.infer(frame  ,confidence=.4)[0] 
        #     points_list = []
        #     points = result.predictions[0].keypoints
        #     for point in points : 
        #         point_dict = {}
        #         point_dict['x'] = point.x
        #         point_dict['y'] = point.y
        #         point_dict['conf'] = point.confidence
        #         point_dict['class_id'] = point.class_id
        #         points_list.append(point_dict)
        #     detections.append(points_list)

        if path is not None : 
            with open(path , 'wb')as f  : 
                pickle.dump(detections , f ) 
        return detections 
    
    def one_shot_classification(self , detections , presaved = False , path = None   ) : 
        if presaved and path is not None : 
            with open(path , 'rb') as f : 
                team_classifications = pickle.load(f)
            return team_classifications
        mapping = {'white' : 'boston' , 'blue' :'new_york'} 
        pipe = pipeline("zero-shot-image-classification", model="patrickjohncyh/fashion-clip")
        team_classifications = []
        for frame , detection  in zip(self.frames , detections ) : 
            teams = {'boston' : [] , 'new_york' : []}
            for box , class_id , track_id in zip(detection.xyxy , detection.class_id , detection.tracker_id )  : 
                if class_id in [1,3,4,5,6,7] : 
                    x1 , y1 , x2 , y2 = box 
                    player_crop = frame[int(y1):int(y2),int(x1):int(x2)] 
                    rgb_image = cv2.cvtColor(player_crop, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_image)
                    image = pil_image 
                    outputs = pipe(image ,candidate_labels=['white shirt' , 'blue shirt'])
                    label  = outputs[0]['label']
                    color = label.split()[0]
                    team = mapping[color]
                    teams[team].append(track_id)
            team_classifications.append(teams)

        if path is not None : 
            with open(path , 'wb') as f : 
                pickle.dump(team_classifications , f) 
        return team_classifications       

    def OCR(self , detections , frames ) : 
        for detection , frame in zip(detections[:1] , frames[:1] ) : 
            for box , class_id in zip(detection.xyxy ,detection.class_id) : 

                if class_id == 2 : 
                    x1 , y1 , x2 , y2 = box 
                    crop = frame[int(y1):int(y2),int(x1):int(x2)] 
                    rgb_frame = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(rgb_frame)

                    # reader = easyocr.Reader(['en'])
                    # result = reader.readtext(crop) 
                    # print(result)
                    # from transformers import pipeline
                    ocr = pipeline("image-to-text", model="microsoft/trocr-base-printed")
                    result = ocr(image)
