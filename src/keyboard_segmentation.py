import cv2
import numpy as np

def segment_keys(image, key_matcher):  # TODO: test white key detection on different keyboards
    res, (top, bottom) = key_matcher.GetBestTemplateMatch(image)

    # bottom 3/4rds of keys
    y_white = bottom - (bottom-top)//4

    left, right = keyboard_sides(image, y_white)

    y_black = top + (1*(bottom-top))//3

    black_bottom = black_key_bottom(image, left, right, y_black)

    labeled_edges = key_edges(image, left, right, y_black)

    c_idx = get_middle_c(labeled_edges, left, right)

    boxes = make_bounding_boxes(labeled_edges, top, bottom, black_bottom)

    midi_boxes = []
    for i, b in enumerate(boxes):
        midi_boxes.append((b, 60 + i - c_idx, labeled_edges[i][2]))

    masks = key_masks(image, midi_boxes)

    #show_masks(image, masks)
    # show_boxes(image, midi_boxes)

    return midi_boxes, masks

def white_key_mask(image):
    # use hsv
    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # mask = cv2.inRange(hsv, (0, 0, 150), (180, 64, 255))
    # cv2.imshow("window", mask)
    # cv2.waitKey(0)
    return cv2.inRange(image, (150, 150, 150), (255, 255, 255))//255

def keyboard_sides(image, white_y):
    # pixels around the keys
    white_keys = image[white_y-5:white_y+5, :]

    # white mask
    white_keys = white_key_mask(white_keys)

    # move from center until large black area is found
    # if sum of 5x10 area is < 30 set middle x of area as the edge
    # use sum and convolution to find edges
    white_keys = np.sum(white_keys, axis=0) 
    white_keys = np.convolve(white_keys, np.ones(5), mode='same')
    white_keys = white_keys <= 30

    left = 0
    right = None
    for i in range(white_keys.shape[0]):
        if i < white_keys.shape[0]//2 and white_keys[i] and i > left:
            left = i
        elif i >= white_keys.shape[0]//2 and white_keys[i]:
            right = i
            break
    
    return left, right

def black_key_bottom(image,left, right, black_y):
    # y axis average of image
    whites = white_key_mask(image[:, left:right])
    whites = np.sum(whites, axis=1)
    for i in range(black_y, image.shape[0]):
        if whites[i] > (right-left)*0.7:
            return i


key_order = ["C", "D", "E", "F", "G", "A", "B"]

def key_edges(image, left, right, black_y):
    # top part of keys
    top_keys = image[black_y-3:black_y+3, :]
    top_keys = white_key_mask(top_keys)
    top_keys = np.sum(top_keys, axis=0)
    top_keys = np.convolve(top_keys, np.ones(3), mode='same')
    top_keys = top_keys <= 8
    
    lines = []
    last_line = 0
    for i in range(left, right-5): 
        if top_keys[i] is not top_keys[i+1] and i > last_line+2:
            is_left = not top_keys[i]
            lines.append((i, is_left))
            last_line = i

    # for line in lines:
    #     cv2.line(image, (line[0], black_y-3), (line[0], black_y+3), (0, 255, 0), 1)
    # cv2.imshow('keys', cv2.resize(image, (0, 0), fx=2, fy=2))
    # cv2.waitKey(0)
    
    black_keys = []
    for i in range(len(lines)-1):
        if lines[i][1]:
            black_keys.append((lines[i][0], lines[i+1][0]))
    
    avg_width = np.mean([black_keys[i+1][0] - black_keys[i][1] for i in range(len(black_keys)-1)])

    black_key_groups = []
    group = []
    for i in range(len(black_keys)):
        group.append(black_keys[i])
        if i != len(black_keys)-1 and black_keys[i+1][0] - black_keys[i][1] > avg_width*1.2:
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

def key_masks(image, midi_boxes):
    masks = []
    white_mask = white_key_mask(image)
    black_mask = 1 - white_mask
    for box, midi, note in midi_boxes:
        if "#" in note:
            mask = np.zeros(image.shape[:2], dtype=bool)
            mask[box[1]:box[3], box[0]:box[2]] = black_mask[box[1]:box[3], box[0]:box[2]]
            masks.append(mask)
        else:
            mask = np.zeros(image.shape[:2], dtype=bool)
            mask[box[1]:box[3], box[0]:box[2]] = white_mask[box[1]:box[3], box[0]:box[2]]
            masks.append(mask)
    return masks

def show_masks(image, masks):
    import random

    for mask in masks:
        random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        image[mask] = random_color

    cv2.imshow('masks', cv2.resize(image, (0, 0), fx=2, fy=2))
    cv2.waitKey(0)

def show_boxes(image, midi_boxes):
    for box, midi, note in midi_boxes:
        if "#" in note:
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
        else:
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)

    cv2.imshow('boxes', cv2.resize(image, (0, 0), fx=2, fy=2))
    cv2.waitKey(0)