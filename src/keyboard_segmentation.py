import cv2
import numpy as np
import logging

visualize = False
HALF_STRIP_HEIGHT = 5
TOP_THRESH = 0.3
SIDES_THRESH = 0.5
BLACK_KEY_BOTTOM_THRESH = 0.85
BLACK_KEY_EDGE_THRESH = 0.5
BLACK_KEY_BIG_GAP = 1.1 # gaps between black keys bigger that BLACK_KEY_BIG_GAP * mean_width are considered big gaps

def segment_keys(image, key_matcher):
    res, (top, bottom) = key_matcher.GetBestTemplateMatch(image)

    # bottom 3/4rds of keys
    y_white = bottom - (bottom-top)//4
    y_black = top + (1*(bottom-top))//3

    black_lightness = mean_lightness(image, y_black) * 0.7
    white_lightness = mean_lightness(image, y_white) * 0.6

    white_keys_b = white_key_mask(image, black_lightness)
    white_keys_w = white_key_mask(image, white_lightness)

    left, right = keyboard_sides(white_keys_w, y_white)


    top, bottom = top_bottom(white_keys_b, left, right, y_white)
    black_bottom = black_key_bottom(white_keys_b, left, right, y_black)

    if visualize:
        img = cv2.rectangle(image.copy(), (left, top), (right, bottom), (0, 255, 0), 1)
        img = cv2.line(img, (left, y_black), (right, y_black), (0, 255, 0), 1)
        img = cv2.line(img, (left, black_bottom), (right, black_bottom), (0, 255, 0), 1)
        cv2.imshow('img', cv2.resize(img, (0, 0), fx=2, fy=2))
        cv2.waitKey(0)


    labeled_edges = key_edges(image, white_keys_b, left, right, y_black)

    c_idx = get_middle_c(labeled_edges, left, right)

    boxes = make_bounding_boxes(labeled_edges, top, bottom, black_bottom)

    midi_boxes = []
    for i, b in enumerate(boxes):
        midi_boxes.append((b, 60 + i - c_idx, labeled_edges[i][2]))

    masks = key_masks(white_keys_b, midi_boxes)

    #show_masks(image, masks)
    # show_boxes(image, midi_boxes)

    return midi_boxes, masks

def mean_lightness(image, y):
    # average brightness of white keys
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    image = image[y-HALF_STRIP_HEIGHT:y+HALF_STRIP_HEIGHT, :] # TODO: black padding will cause issues
    return np.mean(image[:, :, 1])


def white_key_mask(image, mean_lightness):
    # use hsv
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    mask = cv2.inRange(hls, (0, mean_lightness, 0), (255, 255, 255))//255
    # cv2.imshow("window", mask)
    # cv2.waitKey(0)
    # mask = cv2.inRange(image, (150, 150, 150), (255, 255, 255))//255
    return mask

def keyboard_sides(white_keys, white_y):
    # pixels around the keys
    white_keys = white_keys[white_y-HALF_STRIP_HEIGHT:white_y+HALF_STRIP_HEIGHT, :]

    if visualize:
        img = white_keys.copy()*255
        cv2.imshow('img', cv2.resize(img, (0, 0), fx=2, fy=2))
        cv2.waitKey(0)

    # move from center until large black area is found
    # if sum of 5x10 area is < 30 set middle x of area as the edge
    # use sum and convolution to find edges
    white_keys = np.sum(white_keys, axis=0)
    white_keys = np.convolve(white_keys, np.ones(HALF_STRIP_HEIGHT), mode='same')
    white_keys = white_keys <= 2*HALF_STRIP_HEIGHT*HALF_STRIP_HEIGHT*SIDES_THRESH

    if visualize:
        img = white_keys.copy()*255
        img = np.array([img]*2*HALF_STRIP_HEIGHT, dtype=np.uint8)
        cv2.imshow('img', img)
        cv2.waitKey(0)

    left = 0
    right = white_keys.shape[0]-1
    for i in range(white_keys.shape[0]):
        if i < white_keys.shape[0]//2 and white_keys[i] and i > left:
            left = i
        elif i >= white_keys.shape[0]//2 and white_keys[i]:
            right = i
            break

    return left, right

def top_bottom(white_key_mask, left, right, white_y):
    white_key_mask = white_key_mask[:, left:right]
    whites = np.sum(white_key_mask, axis=1)

    top = white_y
    for i in range(white_y, -1, -1):
        if whites[i] < (right-left)*TOP_THRESH:
            top = i
            break

    bottom = white_y
    for i in range(white_y, white_key_mask.shape[0]):
        if whites[i] < (right-left)*SIDES_THRESH:
            bottom = i
            break

    return top, bottom

def black_key_bottom(white_key_mask, left, right, black_y):
    white_key_mask = white_key_mask[:, left:right]
    whites = np.sum(white_key_mask, axis=1)
    for i in range(black_y, white_key_mask.shape[0]):
        if whites[i] > (right-left)*BLACK_KEY_BOTTOM_THRESH:
            return i


key_order = ["C", "D", "E", "F", "G", "A", "B"]

def key_edges(image, white_key_mask, left, right, black_y):
    # top part of keys
    top_keys = white_key_mask[black_y-HALF_STRIP_HEIGHT:black_y+HALF_STRIP_HEIGHT, :]
    top_keys = np.sum(top_keys, axis=0)
    top_keys = np.convolve(top_keys, np.ones(5), mode='same')
    top_keys = top_keys <= 2*HALF_STRIP_HEIGHT*HALF_STRIP_HEIGHT*BLACK_KEY_EDGE_THRESH

    if visualize:
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

    if visualize:
        for line in lines:
            cv2.line(image, (line[0], black_y-3), (line[0], black_y+3), (0, 255, 0), 1)
        cv2.imshow('img', cv2.resize(image, (0, 0), fx=2, fy=2))
        cv2.waitKey(0)

    black_keys = []
    for i in range(len(lines)-1):
        if lines[i][1]:
            black_keys.append((lines[i][0], lines[i+1][0]))

    avg_width = np.mean([black_keys[i+1][0] - black_keys[i][1] for i in range(len(black_keys)-1)])

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
    for i in range(len(black_key_groups)//2, len(black_key_groups)):
        if len(black_key_groups[i]) == 2:
            middle_2_group = i

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


    next_white_key = key_order[(key_order.index(labeled_white_keys[-1][2])+1)%7]
    labeled_white_keys.append((white_lines[-1], right, next_white_key))

    # for key in labeled_black_keys:
    #     cv2.line(image, (key[0], black_y-3), (key[0], black_y+3), (0, 255, 0), 1)
    #     cv2.line(image, (key[1], black_y-3), (key[1], black_y+3), (0, 255, 0), 1)
    # cv2.imshow('keys', cv2.resize(image, (0, 0), fx=2, fy=2))
    # cv2.waitKey(0)

    labeled_keys = labeled_black_keys + labeled_white_keys
    labeled_keys.sort()

    return labeled_keys

def get_middle_c(labeled_keys, left, right):
    best_c_index = None
    best_c_dist = None
    middle = (left+right)//2
    for i in range(len(labeled_keys)):
        if labeled_keys[i][2] == "C":
            dist = abs(labeled_keys[i][0] - middle)
            if best_c_dist is None or dist < best_c_dist:
                best_c_dist = dist
                best_c_index = i
    return best_c_index

def make_bounding_boxes(labeled_keys, top, bottom, black_bottom):
    boxes = []
    for i in range(len(labeled_keys)):
        if "#" in labeled_keys[i][2]:
            boxes.append((labeled_keys[i][0], top, labeled_keys[i][1], black_bottom))
        else:
            boxes.append((labeled_keys[i][0], top, labeled_keys[i][1], bottom))

    return boxes

def key_masks(white_key_mask, midi_boxes):
    masks = []
    black_mask = 1 - white_key_mask
    for box, midi, note in midi_boxes:
        if "#" in note:
            mask = np.zeros(white_key_mask.shape[:2], dtype=bool)
            mask[box[1]:box[3], box[0]:box[2]] = black_mask[box[1]:box[3], box[0]:box[2]]
            masks.append(mask)
        else:
            mask = np.zeros(white_key_mask.shape[:2], dtype=bool)
            mask[box[1]:box[3], box[0]:box[2]] = white_key_mask[box[1]:box[3], box[0]:box[2]]
            masks.append(mask)
    return masks

def show_masks(image, masks):
    import random

    for mask in masks:
        random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        image[mask] = random_color

    cv2.imshow('img', cv2.resize(image, (0, 0), fx=2, fy=2))
    cv2.waitKey(0)

def show_boxes(image, midi_boxes):
    for box, midi, note in midi_boxes:
        if not "#" in note:
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)

    for box, midi, note in midi_boxes:
        if "#" in note:
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)

    cv2.imshow('img', cv2.resize(image, (0, 0), fx=2, fy=2))
    cv2.waitKey(0)


if __name__ == "__main__":
    import key_matcher
    import os
    #background_path = "data/1_intermediate/background/-cVFo4ujq9k.png"
    #background_path = "data/1_intermediate/background/scarlatti.png"
    background_dir = "data/1_intermediate/background/"
    for filename in os.listdir(background_dir):
        # filename= "2nlvRLfHuzg.png"
        background_path = os.path.join(background_dir, filename)
        print(background_path)
        image = cv2.imread(background_path)
        #mask = white_key_mask(image)
        # cv2.imshow('mask', cv2.resize(mask*255, (0, 0), fx=2, fy=2))
        # cv2.waitKey(0)
        matcher = key_matcher.KeyMatcher()
        try:
            midi_boxes, masks = segment_keys(image, matcher)
        except Exception as e:
            cv2.imshow('img', cv2.resize(image, (0, 0), fx=2, fy=2))
            cv2.waitKey(0)
            continue
        show_boxes(image, midi_boxes)
