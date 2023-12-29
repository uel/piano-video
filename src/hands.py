import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from draw import draw_landmarks_on_image
import cv2
import numpy as np
    
def landmarker(show_landmarks=False):
    base_options = python.BaseOptions(model_asset_path='models/hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, 
                                            running_mode=mp.tasks.vision.RunningMode.VIDEO, 
                                            num_hands=2, 
                                            min_hand_detection_confidence=0.5,
                                            min_hand_presence_confidence=0.5,
                                            min_tracking_confidence=0.5)

    landmarker = vision.HandLandmarker.create_from_options(options)
    
    def landmarks_to_tuples(hand):
        return tuple( (round(landmark.x, 4), round(landmark.y, 4), round(landmark.z, 4)) for landmark in hand )


    def detect(self, frame, timestamp):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)        
        results = self.detect_for_video(image, timestamp)

        left_hand, right_hand = None, None
        if results.hand_landmarks:
            hand0, cat0 = landmarks_to_tuples(results.hand_landmarks[0]), results.handedness[0][0].category_name
            if len(results.hand_landmarks) == 2:
                hand1, cat1 = landmarks_to_tuples(results.hand_landmarks[1]), results.handedness[1][0].category_name
            else:
                hand1, cat1 = None, ''

            if len(results.hand_landmarks) == 2 and cat0 == cat1:
                # choose left and right depending on average x coordinate
                mean_0 = np.mean([landmark.x for landmark in results.hand_landmarks[0]])
                mean_1 = np.mean([landmark.x for landmark in results.hand_landmarks[1]])
                if mean_0 > mean_1: # will be swapped later
                    left_hand = hand0
                    right_hand = hand1
                else:
                    left_hand = hand1
                    right_hand = hand0
            else:
                if cat0 == 'Left': left_hand = hand0
                elif cat0 == 'Right': right_hand = hand0

                if cat1 == 'Left': left_hand = hand1
                elif cat1 == 'Right': right_hand = hand1

        if show_landmarks:
            if results.hand_landmarks:
                frame = draw_landmarks_on_image(frame, results)
            cv2.imshow('hand_landmarks', frame)
            cv2.waitKey(1)

        return right_hand, left_hand # swap

    landmarker.detect = detect

    return landmarker



def arg_zip(list1, list2):
    '''sorted input lists'''
    result = []
    i, j = 0, 0
    while i < len(list1) and j < len(list2):
        if list1[i][0] == list2[j][0]:
            result.append((list1[i][0], list1[i][1], list2[j][1]))
            i += 1
            j += 1
        elif list1[i][0] < list2[j][0]:
            result.append((list1[i][0], list1[i][1], None))
            i += 1
        else:
            result.append((list2[j][0], None, list2[j][1]))
            j += 1

    while i < len(list1):
        result.append((list1[i][0], list1[i][1], None))
        i += 1
    
    while j < len(list2):
        result.append((list2[j][0], None, list2[j][1]))
        j += 1
    
    return result

def fill_gaps_hand(hand, max_gap_size):
    if len(hand) == 0: return []

    result = []
    for i in range(1, len(hand)):
        result.append(hand[i-1])
        if 1 < hand[i][0] - hand[i-1][0] <= max_gap_size:
            alphas = np.linspace(0, 1, hand[i][0] - hand[i-1][0] + 1)[1:-1]
            landmarks1 = np.array(hand[i-1][1])
            landmarks2 = np.array(hand[i][1]) 

            for j in range(alphas.shape[0]):
                result.append((hand[i-1][0]+1+j, ((1-alphas[j]) * landmarks1 + alphas[j] * landmarks2).tolist()))

    result.append(hand[-1])
    return result

def fill_gaps(hand_landmarks, max_gap_size=15):
    left = [(i, l) for i, l, _ in hand_landmarks if l]
    right = [(i, r) for i, _, r in hand_landmarks if r]
    left = fill_gaps_hand(left, max_gap_size)
    right = fill_gaps_hand(right, max_gap_size)
    return arg_zip(left, right)


if __name__ == "__main__":
    cap = cv2.VideoCapture("data/videos/Erik C 'Piano Man'/8xJdM4S-fko.mp4")
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
        if results and len(results) == 2:
            two_hands_count += 1

        if len(results) > 2:
            print(f'Found {len(results.hand_landmarks)} hands')

        # Draw the landmarks on the frame
        frame = draw_landmarks_on_image(frame, results)
        cv2.imshow('hand_landmarks', frame)
        cv2.waitKey(0)

        frame_count += 1

    cap.release()

    coverage = (two_hands_count / frame_count) * 100
    print(f'Coverage of frames with two hands: {coverage}%')

    # import file_io
    # landmarks = file_io.read_landmarks(r"data\hand_landmarks\nc29R1xYmjQ.bin")
    # fill_gaps(landmarks)