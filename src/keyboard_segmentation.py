import cv2
import numpy as np
import logging

VISUALIZE = False
WHITE_KEY_B_THRESH = 0.7 # lightness multiplier for mean lightness of B/W key area
WHITE_KEY_W_THRESH = 0.6 # lightness multiplier for mean lightness of W key area
HALF_STRIP_HEIGHT = 5 # half the height of the strip used to find edges
BLACK_KEY_BOTTOM_THRESH = 0.85
BLACK_KEY_EDGE_THRESH = 0.5
BLACK_KEY_BIG_GAP = 1.1 # gaps between black keys bigger that BLACK_KEY_BIG_GAP * mean_width are considered big gaps
MIDDLE_C_LOC = 2.3 # middle C is a bit to the left of the middle

def segment_keys(image, keyboard_loc):
    left, top, right, bottom = keyboard_loc
    left = int(left*image.shape[1])
    top = int(top*image.shape[0])
    right = int(right*image.shape[1])
    bottom = int(bottom*image.shape[0])

    y_white = bottom - (bottom-top)//4
    y_black = top + (1*(bottom-top))//3

    black_lightness = mean_lightness(image, y_black) * WHITE_KEY_B_THRESH
    white_lightness = mean_lightness(image, y_white) * WHITE_KEY_W_THRESH

    white_keys_b = white_key_mask(image, black_lightness)
    white_keys_w = white_key_mask(image, white_lightness)

    black_bottom = get_black_key_bottom(white_keys_b, left, right, y_black)

    if VISUALIZE:
        img = cv2.rectangle(image.copy(), (left, top), (right, bottom), (0, 255, 0), 1)
        img = cv2.line(img, (left, y_black), (right, y_black), (0, 255, 0), 1)
        img = cv2.line(img, (left, black_bottom), (right, black_bottom), (0, 255, 0), 1)
        cv2.imshow('img', cv2.resize(img, (0, 0), fx=2, fy=2))
        cv2.waitKey(0)

    labeled_keys = get_key_lines(white_keys_b, left, right, y_black)

    boxes = make_bounding_boxes(labeled_keys, top, bottom, black_bottom, image.shape)

    c_idx = get_middle_c(labeled_keys, left, right)

    midi_boxes = []
    for i, b in enumerate(boxes):
        midi_boxes.append([labeled_keys[i][2], 60 + i - c_idx, b])
        
    return midi_boxes

def mean_lightness(image, y):
    # average brightness of white keys
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    image = image[y-HALF_STRIP_HEIGHT:y+HALF_STRIP_HEIGHT, :] # TODO: black padding will cause issues
    return np.mean(image[:, :, 1])


def white_key_mask(image, mean_lightness):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    mask = cv2.inRange(hls, (0, mean_lightness, 0), (255, 255, 255))//255
    return mask

def get_black_key_bottom(white_key_mask, left, right, black_y):
    white_key_mask = white_key_mask[:, left:right]
    whites = np.sum(white_key_mask, axis=1)
    for i in range(black_y, white_key_mask.shape[0]):
        if whites[i] > (right-left)*BLACK_KEY_BOTTOM_THRESH:
            return i

def get_black_key_lines(top_keys, black_y, left, right):
    # top part of keys
    top_keys = top_keys[black_y-HALF_STRIP_HEIGHT:black_y+HALF_STRIP_HEIGHT, :]
    top_keys = np.sum(top_keys, axis=0)
    top_keys = np.convolve(top_keys, np.ones(5), mode='same')
    top_keys = top_keys <= 2*HALF_STRIP_HEIGHT*HALF_STRIP_HEIGHT*BLACK_KEY_EDGE_THRESH

    if VISUALIZE:
        img = top_keys.copy()*255
        img = np.array([img]*2*HALF_STRIP_HEIGHT, dtype=np.uint8)
        cv2.imshow('img', img)
        cv2.waitKey(0)

    lines = []
    last_line = 0
    for i in range(left, right-HALF_STRIP_HEIGHT):
        if top_keys[i] is not top_keys[i+1] and i > last_line+2:
            is_left = not top_keys[i]
            lines.append((i, is_left))
            last_line = i

    if not lines:
        logging.warning("No lines found")
        return []

    if VISUALIZE:
        for line in lines:
            cv2.line(image, (line[0], black_y-3), (line[0], black_y+3), (0, 255, 0), 1)
        cv2.imshow('img', cv2.resize(image, (0, 0), fx=2, fy=2))
        cv2.waitKey(0)
    
    return lines

def get_black_key_groups(black_keys, avg_width):
    black_key_groups = []
    group = []
    for i in range(len(black_keys)):
        group.append(black_keys[i])
        if i != len(black_keys)-1 and black_keys[i+1][0] - black_keys[i][1] > avg_width*BLACK_KEY_BIG_GAP:
            black_key_groups.append(group)
            group = []

    if group:
        black_key_groups.append(group)

    # middle 2 group
    middle_2_group = -1
    for i in range(len(black_key_groups)//2-1, len(black_key_groups)):
        if len(black_key_groups[i]) == 2:
            middle_2_group = i
            break

    # resize groups so that they are varying between 2 and 3
    groups = []
    middle_2_key = black_keys.index(black_key_groups[middle_2_group][0])

    i = middle_2_key
    while i < len(black_keys):
        group = black_keys[i:min(i+2, len(black_keys))]
        groups.append(group)
        i += 2
        if i >= len(black_keys):
            break

        group = black_keys[i:min(i+3, len(black_keys))]
        groups.append(group)
        i += 3

    middle_2_group = 0
    i = middle_2_key
    while i > 0:
        group = black_keys[max(i-3, 0):i]
        groups.insert(0, group)
        middle_2_group += 1
        i -= 3
        if i <= 0: break

        group = black_keys[max(i-2, 0):i]
        groups.insert(0, group)
        middle_2_group += 1
        i -= 2

    return groups, middle_2_group

def get_black_keys(black_key_groups, middle_2_group):
    labeled_black_keys = []
    note_names_2 = ["C#", "D#"]
    note_names_3 = ["F#", "G#", "A#"]
    for i in range(middle_2_group, len(black_key_groups)):
        if (i & 1) == (middle_2_group & 1):
            for j, key in enumerate(black_key_groups[i]):
                labeled_black_keys.append((key[0], key[1], note_names_2[j]))
        else:
            for j, key in enumerate(black_key_groups[i]):
                labeled_black_keys.append((key[0], key[1], note_names_3[j]))

    note_names_2.reverse()
    note_names_3.reverse()
    for i in range(middle_2_group-1, -1, -1):
        if (i & 1) == (middle_2_group & 1):
            for j, key in enumerate(reversed(black_key_groups[i])):
                labeled_black_keys.append((key[0], key[1], note_names_2[j]))
        else:
            for j, key in enumerate(reversed(black_key_groups[i])):
                labeled_black_keys.append((key[0], key[1], note_names_3[j]))

    labeled_black_keys.sort()
    return labeled_black_keys

def get_white_keys(labeled_black_keys, left, right, avg_width):
    white_lines = [left]
    labeled_white_keys = []

    for i, key in enumerate(labeled_black_keys):
        width = key[1] - key[0]
        if key[2] == "C#":
            white_lines.append(key[0]+round((2*width)/3.))
            labeled_white_keys.append((white_lines[-2], white_lines[-1], "C"))
        elif key[2] == "D#":
            white_lines.append(key[0]+round((width+1)/3.))
            labeled_white_keys.append((white_lines[-2], white_lines[-1], "D"))
        elif key[2] == "F#":
            white_lines.append(key[0]+round((4*width)/5.))
            labeled_white_keys.append((white_lines[-2], white_lines[-1], "F"))
        elif key[2] == "G#":
            white_lines.append(key[0]+round(width/2.))
            labeled_white_keys.append((white_lines[-2], white_lines[-1], "G"))
        elif key[2] == "A#":
            white_lines.append(key[0]+round((width+1)/3.))
            labeled_white_keys.append((white_lines[-2], white_lines[-1], "A"))

        if ( key[2] == "A#" or key[2] == "D#" ):
            if i == len(labeled_black_keys)-1:
                if right - white_lines[-1] > avg_width*2:
                    width = (right - white_lines[-1])//2
                    white_lines.append(white_lines[-1]+width)
            else:
                width = (labeled_black_keys[i+1][0] - key[1] + 1)//2
                white_lines.append(key[1]+width)

            if key[2] == "A#":
                labeled_white_keys.append((white_lines[-2], white_lines[-1], "B"))
            else:
                labeled_white_keys.append((white_lines[-2], white_lines[-1], "E"))

    key_order = ["C", "D", "E", "F", "G", "A", "B"]
    next_white_key = key_order[(key_order.index(labeled_white_keys[-1][2])+1)%7]
    labeled_white_keys.append((white_lines[-1], right, next_white_key))

    return labeled_white_keys

def get_key_lines(white_key_mask, left, right, black_y):
    lines = get_black_key_lines(white_key_mask, black_y, left, right)

    black_keys = []
    for i in range(len(lines)-1):
        if lines[i][1]:
            black_keys.append((lines[i][0], lines[i+1][0]))
    avg_width = np.mean([black_keys[i+1][0] - black_keys[i][1] for i in range(len(black_keys)-1)])

    b_key_groups, middle_2_group = get_black_key_groups(black_keys, avg_width)

    labeled_b_keys = get_black_keys(b_key_groups, middle_2_group)
    labeled_w_keys = get_white_keys(labeled_b_keys, left, right, avg_width)
    labeled_keys = sorted(labeled_b_keys + labeled_w_keys)

    return labeled_keys

def get_middle_c(labeled_keys, left, right):
    best_c_index = None
    best_c_dist = None
    middle = (left+right)//MIDDLE_C_LOC # middle C is a bit to the left of the middle
    for i in range(len(labeled_keys)):
        if labeled_keys[i][2] == "C":
            dist = abs(labeled_keys[i][0] - middle)
            if best_c_dist is None or dist < best_c_dist:
                best_c_dist = dist
                best_c_index = i
    return best_c_index

def make_bounding_boxes(labeled_keys, top, bottom, black_bottom, img_shape):
    boxes = []
    for i in range(len(labeled_keys)):
        if "#" in labeled_keys[i][2]:
            boxes.append((  round(labeled_keys[i][0]/img_shape[1], 5),
                            round(top/img_shape[0], 5), 
                            round(labeled_keys[i][1]/img_shape[1], 5), 
                            round(black_bottom/img_shape[0], 5) ))
        else:
            boxes.append((  round(labeled_keys[i][0]/img_shape[1], 5),
                            round(top/img_shape[0], 5), 
                            round(labeled_keys[i][1]/img_shape[1], 5), 
                            round(bottom/img_shape[0], 5) ))

    return boxes

def get_key_masks(white_key_mask, midi_boxes):
    masks = []
    black_mask = 1 - white_key_mask
    for note, midi, box in midi_boxes:
        box = (int(box[0]*white_key_mask.shape[1]), int(box[1]*white_key_mask.shape[0]), int(box[2]*white_key_mask.shape[1]), int(box[3]*white_key_mask.shape[0]))
        if "#" in note:
            mask = np.zeros(white_key_mask.shape[:2], dtype=bool)
            mask[box[1]:box[3], box[0]:box[2]] = black_mask[box[1]:box[3], box[0]:box[2]]
            masks.append(mask)
        else:
            mask = np.zeros(white_key_mask.shape[:2], dtype=bool)
            mask[box[1]:box[3], box[0]:box[2]] = white_key_mask[box[1]:box[3], box[0]:box[2]]
            masks.append(mask)
    return masks


if __name__ == "__main__":
    import key_matcher
    import os
    #background_path = "data/1_intermediate/background/-cVFo4ujq9k.png"
    #background_path = "data/1_intermediate/background/scarlatti.png"
    background_dir = "data/1_intermediate/background/"
    for filename in os.listdir(background_dir):
        # filename= "tdGW5R7xDxg.png"
        # VISUALIZE=True
        background_path = os.path.join(background_dir, filename)
        print(background_path)
        image = cv2.imread(background_path)
        #mask = white_key_mask(image)
        # cv2.imshow('mask', cv2.resize(mask*255, (0, 0), fx=2, fy=2))
        # cv2.waitKey(0)
        # try:
        matcher = key_matcher.YoloMatcher()
        midi_boxes, masks = segment_keys(image, matcher)
        # image = draw_boxes(image, midi_boxes)
        # except Exception as e:
        #     print(background_path, e)
        #     throw e

        cv2.imwrite("data/visual/segments/"+filename, image)