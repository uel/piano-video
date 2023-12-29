import piano_video
import cv2
import numpy as np
import json

from intervaltree import IntervalTree
from draw import draw_landmarks_on_image, draw_boxes

from keyboard_segmentation import get_key_masks, white_key_mask

def play_demo(video_path, show=True, save=False, skip_non_piano=False, truth_midi_path=None):
    assert show or save, "Either show or save must be True"
    video = piano_video.PianoVideo(video_path)
    background = video.background
    sections = video.sections
    keyboard_box, keys = video.keys
    keyboard_box = (int(keyboard_box[0]*background.shape[1]), 
                    int(keyboard_box[1]*background.shape[0]), 
                    int(keyboard_box[2]*background.shape[1]), 
                    int(keyboard_box[3]*background.shape[0]))
    midi = video.fingers
    if truth_midi_path is not None:
        midi = IntervalTree().from_tuples(json.load(open(truth_midi_path, "r")))
    landmarks = video.hand_landmarker()


    # masks = get_key_masks(white_key_mask(background, lightness_thresh), keys)
    masks = []
    for key in keys:
        mask = np.zeros(background.shape[:2], dtype=bool)
        mask[int(key[2][1]*background.shape[0]):int(key[2][3]*background.shape[0]), int(key[2][0]*background.shape[1]):int(key[2][2]*background.shape[1])] = 1
        masks.append(mask)

    mask_dict = {keys[i][1]: masks[i] for i in range(len(keys))}


    cap = cv2.VideoCapture(video_path)
    if save:
        out = cv2.VideoWriter('demo/'+video.file_name+".avi", cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 360))

    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            hands = next(landmarks) 
            print(i)

            if skip_non_piano and not sections[i]:
                i += 1
                continue

            frame = cv2.resize(frame, dsize=(background.shape[1], background.shape[0]), interpolation=cv2.INTER_AREA)

            if midi[i]:
                background_mask = cv2.subtract(background, frame).sum(axis=2) < 100
                colored = frame // 2 + (0, 128, 0)
                colored_removed = frame // 2 + (128, 128, 0)
                for note in midi[i]:
                    if note[2][0] in mask_dict:
                        mask = background_mask & mask_dict[note[2][0]]
                        if note.data[2] is not None:
                            frame[mask] = colored[mask]
                        else:
                            frame[mask] = colored_removed[mask]

                for note in midi[i]:
                    if note.data[2] is not None:
                        if isinstance(note.data[2], str):
                            if note[0] == i: print(note.data[2])
                        elif note[0]+2 >= i and note.data[2][1]:
                            x, y = (int(note.data[2][1][0]*frame.shape[1]), int(note.data[2][1][1]*frame.shape[0]))
                            frame = cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
                            if note[0] == i: print(note.data[2][0])


            if sections[i]:
                frame = draw_boxes(frame, keys, show=False)

            frame = draw_landmarks_on_image(frame, hands)

            if save:
                out.write(frame)

            if show:
                cv2.imshow('frame', cv2.resize(frame, dsize=(640, 360), fx=1, fy=1))
                # if hands and bool(hands[0] is None) != bool(hands[1] is None):
                #     cv2.waitKey(0)
                # else: cv2.waitKey(1)
                cv2.waitKey(1)
            i += 1
        else:
            break

    cap.release()
    if save:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "data/videos/Jane/ykxAS-P_zHI.webm"
    video_path = "data/videos/Jane/ykxAS-P_zHI.webm"
    video_path = "data/videos/Paul Barton/s2_9g-dAnT0.mp4"
    video_path = "demo/sections_test.mp4"
    video_path = r"C:\Users\danif\s\BP\data\videos\Paul Barton\NLPxfEMfnVM.mp4"
    video_path = "data/videos/Erik C 'Piano Man'/8xJdM4S-fko.mp4"
    video_path = "recording/rec3.mp4"
    video_path = "data/videos/flowkey – Learn piano/zWULIrqQPEk.mp4"
    video_path = "data/videos/Jane/XYFZFlDK2ko.webm"
    video_path = "data/videos/Paul Barton/nc29R1xYmjQ.mp4"
    video_path = "demo/scarlatti.mp4"
    play_demo(video_path)
    # play_demo(video_path, truth_midi_path="recording/rec3_fingers_truth.json")