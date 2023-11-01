import piano_video
import cv2

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from draw_landmarks import draw_landmarks_on_image

video = piano_video.PianoVideo("demo/scarlatti.mp4")
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

cap = cv2.VideoCapture("demo/scarlatti.mp4")
out = cv2.VideoWriter('demo/scarlatti_hands.avi',cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 360))

i = 0
while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        # use mediapipe hands to draw landmarks
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        results = landmarker.detect_for_video(image, (i*1000)//30)
        frame = draw_landmarks_on_image(frame, results)

        if midi[i]:
            for note in midi[i]:
                frame[mask_dict[note[2][0]]] = (0, 255, 0)
        
        out.write(frame)

        cv2.imshow('frame', frame)
        cv2.waitKey(1)
        i += 1
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()