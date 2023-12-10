from draw import draw_landmarks_on_image
import math
import numpy as np
from intervaltree import IntervalTree
import logging

# distance to closest point in rectangle
def closest_point_to_rect(rect, query):
    x1, y1, x2, y2 = rect
    x, y = query

    if x1 <= x <= x2 and y1 <= y <= y2: return 0

    x_closest = max(x1, min(x, x2))
    y_closest = max(y1, min(y, y2))
    return math.sqrt((x_closest - x)**2 + (y_closest - y)**2)

def closest_finger(key, hands):
    finger_tips = []
    for hand, fingers in zip(["L", "R"], hands): # TOOD: hands doesn contain info about handendness
        for f_id, finger in enumerate(np.array(fingers)[[4, 8, 12, 16, 20]], 1): # indicies of finger tips
            finger_tips.append((hand+str(f_id), (finger[0], finger[1])))

    name, id, rect = key
    closest_finger = None
    closest_dist = math.inf
    for finger in finger_tips:
        dist = closest_point_to_rect(rect, finger[1])
        if dist < closest_dist:
            closest_dist = dist
            closest_finger = finger
    
    return closest_finger, closest_dist

def finger_notes(notes, landmarks, keys, midi_id_offset = 0) -> tuple[int, IntervalTree]:
    sorted_notes = list(sorted(notes))
    note = sorted_notes.pop(0) if sorted_notes else None
    finger_notes = IntervalTree()
    dist_sum = 0
    dist_count = 0

    for i, hands in enumerate(landmarks):
        while note and note.begin == i:
            box = next((key for key in keys if key[1] == (note.data[0] + midi_id_offset*12) ), None)
            if box is not None and hands != []:
                finger, dist = closest_finger(box, hands) 
                finger_notes[note.begin:note.end] = (note.data[0]+midi_id_offset*12, note.data[1], finger)
                dist_sum += dist
                dist_count += 1
            else: # the played note is not on the keyboard or no hand is detected
                finger_notes[note.begin:note.end] = (note.data[0]+midi_id_offset*12, note.data[1], None)
            note = sorted_notes.pop(0) if sorted_notes else None

    return finger_notes, dist_sum/dist_count if dist_count > 0 else 0

def remove_outliers(notes, keys): # sets finger of a note to None if it isn't close by any finger
    finger_notes = IntervalTree()

    for note in notes:
        box = next((key for key in keys if key[1] == note.data[0]), None)
        if box is not None:
            if note.data[2] is not None:
                dist = closest_point_to_rect(box[2], note.data[2][1])
                if dist <= (box[2][2] - box[2][0])*2: # double the width of white key
                    finger_notes[note.begin:note.end] = note.data
                    continue

        # TODO: a finger should never play two notes at once?
        finger_notes[note.begin:note.end] = (note.data[0], note.data[1], None) # couldn't assign finger
        logging.info(f"Removed outlier {note.data[0]}")

    return finger_notes