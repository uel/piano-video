from draw import draw_landmarks_on_image
import math
import numpy as np
from intervaltree import IntervalTree
import logging

def dist_to_center(rect, query):
    x1, y1, x2, y2 = rect
    x, y = query

    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2

    return math.sqrt((x - x_center)**2 + (y - y_center)**2)

def dist_to_bottom_center(rect, query):
    x1, y1, x2, y2 = rect
    x, y = query

    x_center = (x1 + x2) / 2
    y_center = y2

    return math.sqrt((x - x_center)**2 + (y - y_center)**2)

def dist_to_bounds(rect, query):
    x1, y1, x2, y2 = rect
    x, y = query

    if x1 <= x <= x2 and y1 <= y <= y2:
        return -1/dist_to_center(rect, query)

    x_closest = max(x1, min(x, x2))
    y_closest = max(y1, min(y, y2)) 

    return math.sqrt((x - x_closest)**2 + (y - y_closest)**2)

def dist_to_bounds0(rect, query):
    x1, y1, x2, y2 = rect
    x, y = query

    if x1 <= x <= x2 and y1 <= y <= y2:
        return 0

    x_closest = max(x1, min(x, x2))
    y_closest = max(y1, min(y, y2)) 

    return math.sqrt((x - x_closest)**2 + (y - y_closest)**2)


def closest_finger(key, hands, dist_alg="bounds", finger_alg="tip"):
    finger_tips = []
    for hand, fingers in zip(["L", "R"], hands):
        if fingers is None: continue
        if finger_alg == "tip":
            for f_id, finger in enumerate(np.array(fingers)[[4, 8, 12, 16, 20]], 1): # indicies of finger tips
                finger_tips.append((hand+str(f_id), (finger[0], finger[1])))
        elif finger_alg == "midpoint": # midpoint between finger tips and previous knuckle, 8-7, 12-11, 16-15, 20-19
            fingers_array = np.array(fingers)
            for f_id, landmark_id in enumerate([4, 8, 12, 16, 20], 1):
                finger = (fingers_array[landmark_id] + fingers_array[landmark_id-1])/2
                finger_tips.append((hand+str(f_id), (finger[0], finger[1])))


    name, id, rect = key
    closest_finger = None
    closest_dist = math.inf
    same_dist = False
    for finger in finger_tips:
        if dist_alg == "center":
            dist = dist_to_center(rect, finger[1])  
        elif dist_alg == "bottom_center":
            dist = dist_to_bottom_center(rect, finger[1])
        elif dist_alg == "bounds":
            dist = dist_to_bounds(rect, finger[1])
        elif dist_alg == "bounds0":
            dist = dist_to_bounds0(rect, finger[1])
        else:
            raise ValueError("Invalid distance algorithm")
        
        if dist < closest_dist:
            closest_dist = dist
            closest_finger = finger
        elif dist == closest_dist:
            same_dist = True


    if closest_dist == math.inf: # finger doesn't exist at this point
        closest_dist = 0

    return closest_finger, closest_dist, same_dist

def finger_notes(notes, landmarks, keys, dist_alg="bounds", octave_shift = 0) -> tuple[int, IntervalTree]:
    sorted_notes = list(sorted(notes))
    note = sorted_notes.pop(0) if sorted_notes else None
    finger_notes = IntervalTree()
    dist_sum = 0
    dist_count = 0
    distances = []
    same_dist_count = 0

    for i, hands in enumerate(landmarks):
        while note and note.begin == i:
            box = next((key for key in keys if key[1] == (note.data[0] + octave_shift*12) ), None)
            if box is not None and hands != []:
                finger, dist, same_dist = closest_finger(box, hands, dist_alg=dist_alg)
                if same_dist: same_dist_count += 1
                distances.append(dist)
                finger_notes[note.begin:note.end] = (note.data[0]+octave_shift*12, note.data[1], finger)
                dist_sum += dist
                dist_count += 1
            else: # the played note is not on the keyboard or no hand is detected
                finger_notes[note.begin:note.end] = (note.data[0]+octave_shift*12, note.data[1], None)
            note = sorted_notes.pop(0) if sorted_notes else None

    return finger_notes, dist_sum/dist_count if dist_count > 0 else 0

def remove_outliers(notes, keys, distance_thresh=1): # sets finger of a note to None if it isn't close by any finger
    finger_notes = IntervalTree()

    for note in notes:
        box = next((key for key in keys if key[1] == note.data[0]), None)
        if box is not None:
            if note.data[2] is not None:
                dist = dist_to_bounds(box[2], note.data[2][1])
                if dist <= (box[2][2] - box[2][0])*distance_thresh:
                    finger_notes[note.begin:note.end] = note.data
                    continue

        finger_notes[note.begin:note.end] = (note.data[0], note.data[1], None) # couldn't assign finger
        # logging.info(f"Removed outlier {note.data[0]}")

    return finger_notes