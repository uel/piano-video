
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2 

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
  if isinstance(detection_result, list):
    hand_landmarks_list = []
    for hand in detection_result:
      hand_landmarks_list.append([landmark_pb2.NormalizedLandmark(x=landmark[0], y=landmark[1], z=landmark[2]) for landmark in hand])
  else:
    hand_landmarks_list = detection_result.hand_landmarks

  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])

    hand_landmark_style = solutions.drawing_styles.get_default_hand_landmarks_style()
    hand_connection_style = solutions.drawing_styles.get_default_hand_connections_style()
    spec = solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1)
    for k in hand_connection_style:
      hand_connection_style[k] = spec

    for k in hand_landmark_style:
      hand_landmark_style[k] = spec

    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      hand_landmark_style,
      hand_connection_style
      )

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    # cv2.putText(annotated_image, f"{handedness[0].category_name}",
    #             (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
    #             FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image

def show_masks(image, masks):
    import random

    for mask in masks:
        random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        image[mask] = random_color

    cv2.imshow('img', cv2.resize(image, (0, 0), fx=2, fy=2))
    cv2.waitKey(0)

def draw_boxes(image, midi_boxes, show=False):
    avg_white_width = np.mean([box[2]-box[0] for note, _, box in midi_boxes if not "#" in note])

    for note, midi, box in midi_boxes: # white keys
        if not "#" in note:
            cv2.rectangle(image,(   int(box[0]*image.shape[1]),
                                    int(box[1]*image.shape[0])), 
                                (   int(box[2]*image.shape[1]), 
                                    int(box[3]*image.shape[0])), (0, 0, 255), 1)
            
            fontsize = (avg_white_width/14.)*0.25
            offset = int((avg_white_width/14.)*2.5)
            cv2.putText(image, str(midi), (int(box[0]*image.shape[1])+offset, int(box[3]*image.shape[0])-offset), cv2.FONT_HERSHEY_SIMPLEX, fontsize, (0, 0, 255), 1)

    for note, midi, box in midi_boxes: # black keys
        if "#" in note:
            cv2.rectangle(image,(   int(box[0]*image.shape[1]),
                                    int(box[1]*image.shape[0])), 
                                (   int(box[2]*image.shape[1]), 
                                    int(box[3]*image.shape[0])), (0, 0, 128), 1)

    if show:
        cv2.imshow('img', cv2.resize(image, (0, 0), fx=2, fy=2))
        cv2.waitKey(0)

    return image