import piano_video
import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from draw_landmarks import draw_landmarks_on_image

#video_path = "data/0_raw/all_videos/flowkey – Learn piano/4PuLjxWdujM.mp4"
video_path = "demo/scarlatti.mp4"

video = piano_video.PianoVideo(video_path)
background = video.background
midi_boxes, masks = video.key_segments
midi = video.transcribed_midi

mask_dict = {midi_boxes[i][1]: masks[i] for i in range(len(midi_boxes))}

base_options = python.BaseOptions(model_asset_path='models/hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, 
                                        running_mode=mp.tasks.vision.RunningMode.VIDEO, 
                                        num_hands=2, 
                                        min_hand_detection_confidence=0.1,
                                        min_hand_presence_confidence=0.1,
                                        min_tracking_confidence=0.1)

landmarker = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(video_path)
# out = cv2.VideoWriter('demo/scarlatti_hands.avi',cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 360))

i = 0
while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        # use mediapipe hands to draw landmarks
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        results = landmarker.detect_for_video(image, (i*1000)//30)

        if midi[i]:
            background_mask = cv2.subtract(background, frame).sum(axis=2) < 100
            colored = frame // 2 + (0, 128, 0)
            for note in midi[i]:
                if note[2][0] in mask_dict:
                    mask = background_mask & mask_dict[note[2][0]]
                    frame[mask] = colored[mask]
        
        frame = draw_landmarks_on_image(frame, results)
        # out.write(frame)

        cv2.imshow('frame', frame)
        cv2.waitKey(1)
        i += 1
    else:
        break

cap.release()
# out.release()
cv2.destroyAllWindows()