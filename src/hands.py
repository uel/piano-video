import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from draw import draw_landmarks_on_image
import cv2
    
def landmarker(show_landmarks=False):
    base_options = python.BaseOptions(model_asset_path='models/hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, 
                                            running_mode=mp.tasks.vision.RunningMode.VIDEO, 
                                            num_hands=2, 
                                            min_hand_detection_confidence=0.5,
                                            min_hand_presence_confidence=0.5,
                                            min_tracking_confidence=0.5)

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



if __name__ == "__main__":
    cap = cv2.VideoCapture("data/0_raw/all_videos/Erik C 'Piano Man'/8xJdM4S-fko.mp4")
    hands = landmarker()
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frame_count = 0
    two_hands_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the image and get the hand landmarks.
        results = hands.detect(hands, frame, (1000*frame_count)//fps)

        # Check if there are two hands.
        if results.hand_landmarks and len(results.hand_landmarks) == 2:
            two_hands_count += 1

        if len(results.hand_landmarks) > 2:
            print(f'Found {len(results.hand_landmarks)} hands')

        # Use the landmarker function on the frame
        landmarks = landmarker(frame)

        frame_count += 1

    cap.release()

    coverage = (two_hands_count / frame_count) * 100
    print(f'Coverage of frames with two hands: {coverage}%')