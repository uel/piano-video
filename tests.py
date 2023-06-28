import os
import cv2
import main
import numpy as np
from data import GetPointsFromXML
from keys import preprocess_image
import keras
import matplotlib.pyplot as plt

def t2i(t):
    return tuple(int(x) for x in t)


def KeyboardDetectionTest(has_keyboard):
    total_frames = 0
    correct_frames = 0
    for file in os.listdir("data/separated_frames/with_keyboard"):
        if file.endswith(".jpg"):
            img = cv2.imread("data/separated_frames/with_keyboard/" + file)
            total_frames += 1
            if has_keyboard(img):
                correct_frames += 1
            print(f"Accuracy on keyboard frames: {correct_frames/total_frames} on {total_frames} frames", end='\r')

    print()

    total_frames = 0
    correct_frames = 0
    for file in os.listdir("data/separated_frames/without_keyboard"):
        if file.endswith(".jpg"):
            img = cv2.imread("data/separated_frames/without_keyboard/" + file)
            total_frames += 1
            if not has_keyboard(img):
                correct_frames += 1
            print(f"Accuracy on non-keyboard frames: {correct_frames/total_frames}  on {total_frames} frames", end='\r')
    print()

def KeyboardBoundingBoxTest(get_bounding_box):
    # read data/separated_frames/keyboard_annotations.xml

    img_points = GetPointsFromXML("data/separated_frames/keyboard_annotations.xml")

    img_count = 0
    diffs = []
    for points, image in zip(*img_points):
        img = cv2.imread("data/separated_frames/with_keyboard/" + image)
        width_scale = img.shape[1] / 640
        height_scale = img.shape[0] / 480

        top_left, top_right, bottom_right, bottom_left = get_bounding_box(img)
        diff = 0
        diff += abs(top_left[0] - points[0][0]) * width_scale + abs(top_left[1] - points[0][1]) * height_scale
        diff += abs(top_right[0] - points[1][0]) * width_scale + abs(top_right[1] - points[1][1])  * height_scale
        diff += abs(bottom_right[0] - points[2][0]) * width_scale + abs(bottom_right[1] - points[2][1]) * height_scale
        diff += abs(bottom_left[0] - points[3][0]) * width_scale + abs(bottom_left[1] - points[3][1]) * height_scale
        img_count += 1
        diffs.append(diff/8)

        if diff > 5:
            for (x, y) in [top_left, top_right, bottom_right, bottom_left]:
                cv2.circle(img, t2i((x, y)), 5, (0, 0, 255), -1)
            plt.imshow(img[...,::-1])
            plt.show()

        print(f"Average pixel difference in 360p: {np.mean(diffs)} +- {np.std(diffs)} on {img_count} images", end='\r')
    print()

def TemplateMatchingWrapper(img):
    scale = 0.25
    img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
    top_left, bottom_right, max_val = main.DetectKeyboardTemplateMatching(img)
    #print(max_val)
    return max_val > 0.87

def TemplateMatchingBoundingBoxWrapper(img):
    scale = 0.25
    
    img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
    top_left, bottom_right = main.GetPianoBoundingBox(img)
    scale = 1/scale

    bottom_left = (top_left[0]*scale, bottom_right[1]*scale)
    top_right = (bottom_right[0]*scale, top_left[1]*scale)
    bottom_right = (bottom_right[0]*scale, bottom_right[1]*scale)
    top_left = (top_left[0]*scale, top_left[1]*scale)

    return top_left, top_right, bottom_right, bottom_left

bounding_box_model = None
def NNBoundingBoxWrapper(image):
    global bounding_box_model
    if bounding_box_model is None:
        bounding_box_model = keras.models.load_model("models/bounding_box3.h5")
    
    img, _ = preprocess_image(image)

    points = bounding_box_model.predict(np.array([img]), verbose=0)[0]
    points = points.reshape((4, 2))

    points[:, 0] *= image.shape[1]
    points[:, 1] *= image.shape[0]
    points = points.astype(int)

    return points[0], points[1], points[2], points[3]

def NNDetectionWrapper(image):
    global detection_model
    if detection_model is None:
        detection_model = keras.models.load_model("models/detection.h5")
    
    img, _ = preprocess_image(image)

    return detection_model.predict(np.array([img]), verbose=0)[0][0] > 0.5

#KeyboardDetectionTest(TemplateMatchingWrapper)
#KeyboardBoundingBoxTest(TemplateMatchingBoundingBoxWrapper)
#KeyboardDetectionTest(NNDetectionWrapper)

# print("Template matching bounding box model: ")
# KeyboardBoundingBoxTest(TemplateMatchingBoundingBoxWrapper)

print("Neural network bounding box model: ")
KeyboardBoundingBoxTest(NNBoundingBoxWrapper)