import piano_video
import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from draw import draw_landmarks_on_image, draw_boxes
from keyboard_segmentation import get_key_masks, white_key_mask


def get_demo(video_path, show=True, save=False):
    assert show or save, "Either show or save must be True"
    video = piano_video.PianoVideo(video_path)
    background = video.background
    keyboard_box, lightness_thresh, keys = video.keys
    midi = video.fingers
    landmarks = video.hand_landmarks()


    masks = get_key_masks(white_key_mask(background, lightness_thresh), keys)
    mask_dict = {keys[i][1]: masks[i] for i in range(len(keys))}


    cap = cv2.VideoCapture(video_path)
    if save:
        out = cv2.VideoWriter('demo/'+video.file_name+".avi", cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 360))

    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            hands = next(landmarks) 

            # resize frame to fit background
            frame = cv2.resize(frame, dsize=(background.shape[1], background.shape[0]), interpolation=cv2.INTER_AREA)
            frame = draw_landmarks_on_image(frame, hands)

            if midi[i]:
                background_mask = cv2.subtract(background, frame).sum(axis=2) < 100
                colored = frame // 2 + (0, 128, 0)
                for note in midi[i]:
                    if note[2][0] in mask_dict and note.data[2] is not None:
                        mask = background_mask & mask_dict[note[2][0]]
                        frame[mask] = colored[mask]


                for note in midi[i]:
                    if note.data[2] is not None and note[0]+5 >= i and note.data[2][1]:
                        x, y = (int(note.data[2][1][0]*frame.shape[1]), int(note.data[2][1][1]*frame.shape[0]))
                        frame = cv2.circle(frame, (x, y), 3,              (0, 0, 255), -1)
                        if note[0] == i: print(note.data[2][0])

            frame = draw_boxes(frame, keys, show=False)

            # out.write(frame)

            cv2.imshow('frame', cv2.resize(frame, dsize=(640*2, 360*2), fx=2, fy=2))
            cv2.waitKey(0)
            i += 1
        else:
            break

    cap.release()
    # out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "data/0_raw/all_videos/Jane/ykxAS-P_zHI.webm"
    video_path = "data/0_raw/all_videos/flowkey – Learn piano/zWULIrqQPEk.mp4"
    video_path = "data/0_raw/all_videos/Jane/ykxAS-P_zHI.webm"
    video_path = "demo/scarlatti.mp4"
    video_path = "data/0_raw/all_videos/Jane/YgO-UJDfCZE.webm"
    get_demo(video_path)