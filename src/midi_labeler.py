# used for labeling ground truth fingerings for midi files
import piano_video
import cv2
import numpy as np
import json

from draw import draw_landmarks_on_image, draw_boxes
from keyboard_segmentation import get_key_masks, white_key_mask
from intervaltree import IntervalTree

def LabelMidi(ground_truth_midi, video_path, output_file):
    video = piano_video.PianoVideo(video_path)
    background = video.background
    keyboard_box, lightness_thresh, keys = video.keys
    midi = IntervalTree().from_tuples(json.load(open(ground_truth_midi, "r")))
    landmarks = video.hand_landmarks()


    result = IntervalTree().from_tuples(json.load(open(output_file, "r")))
    target_i = max([note[0] for note in result], default=-1) + 1

    masks = get_key_masks(white_key_mask(background, lightness_thresh), keys)
    mask_dict = {keys[i][1]: masks[i] for i in range(len(keys))}
    cap = cv2.VideoCapture(video_path)

    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret != True: break

        hands = next(landmarks) 
        if i < target_i: i+=1; continue

        frame = cv2.resize(frame, dsize=(background.shape[1], background.shape[0]), interpolation=cv2.INTER_AREA)

        if any([ note[0] == i for note in midi[i]]):
            background_mask = cv2.subtract(background, frame).sum(axis=2) < 100
            colored = frame // 2 + (0, 128, 0)
            colored_removed = frame // 2 + (128, 128, 0)
            
            original_frame = draw_landmarks_on_image(frame, hands)

            for note in midi[i]:
                if note[0] == i:
                    frame = original_frame.copy()

                    if note[2][0] in mask_dict:
                        mask = background_mask & mask_dict[note[2][0]]
                        if note.data[2] is not None:
                            frame[mask] = colored[mask]
                        else:
                            frame[mask] = colored_removed[mask]

                    if note.data[2] and note.data[2][1]:
                        x, y = (int(note.data[2][1][0]*frame.shape[1]), int(note.data[2][1][1]*frame.shape[0]))
                        frame = cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)


                    cv2.imshow('frame', cv2.resize(frame, dsize=(640*2, 360*2), fx=2, fy=2))
                    cv2.waitKey(1)
                    if note.data[2] is not None:
                        correction = input(str(note.data[2][0])+ " ")
                        if correction == "":
                            result[note[0]:note[1]] = [note.data[0], note.data[1], note.data[2][0]]
                        else:
                            result[note[0]:note[1]] = [note.data[0], note.data[1], correction]
                    else:
                        correction = input("None ")
                        if correction == "":
                            result[note[0]:note[1]] = [note.data[0], note.data[1], None]
                        else:
                            result[note[0]:note[1]] = [note.data[0], note.data[1], correction]

        else:
            cv2.imshow('frame', cv2.resize(frame, dsize=(640*2, 360*2), fx=2, fy=2))
            cv2.waitKey(5)
                
        if i % 100 == 0:
            print("Saving...", i)
            json.dump(list(sorted(result)), open(output_file, "w"))

        i += 1

    cap.release()
    cv2.destroyAllWindows()
    json.dump(list(sorted(result)), open(output_file, "w"))



if __name__ == "__main__":
    #"rec1.mp4"
    LabelMidi("recording/rec3_fingers_estimate.json", "recording/rec3.mp4", "recording/rec3_fingers_truth.json")
