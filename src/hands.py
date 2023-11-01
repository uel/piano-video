import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from draw_landmarks import draw_landmarks_on_image
import cv2
import numpy as np

def DetectHandsByColor(self):
        # Get approximate keyboard location using template matching
        # Calculate average color of keyboard
        # use color as baseline brightness for hand detection
        for i, frame in self.get_video(2):
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (0, 80, 60), (20, 150, 255))
            cv2.imshow('frame', mask)
            cv2.waitKey(0)
    
def landmarker(show_landmarks=False):
    base_options = python.BaseOptions(model_asset_path='models/hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, 
                                            running_mode=mp.tasks.vision.RunningMode.VIDEO, 
                                            num_hands=2, 
                                            min_hand_detection_confidence=0.1,
                                            min_hand_presence_confidence=0.1,
                                            min_tracking_confidence=0.1)

    landmarker = vision.HandLandmarker.create_from_options(options)
    
    def detect(self, frame, timestamp):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        
        results = self.detect_for_video(image, timestamp)

        if show_landmarks:
            if results.hand_landmarks:
                draw_landmarks_on_image(frame, results)
            cv2.imshow('hand_landmarks', frame)
            cv2.waitKey(0)

        return results

    landmarker.detect = detect

    return landmarker