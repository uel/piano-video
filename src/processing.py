import numpy as np
import xml.etree.ElementTree as ET
import cv2
import os
import shutil
import glob
import albumentations as A
from keyboard_detector import KeyboardDetector
from piano_video import PianoVideo
import time

def Sort4Points(points):
    # sort points clockwise
    # https://stackoverflow.com/a/6989383
    centroid = np.mean(points, axis=0)
    points = sorted(points, key=lambda p: np.arctan2(p[1] - centroid[1], p[0] - centroid[0]))
    return points

def GetPointsFromXML(file):
    root = ET.parse(file).getroot()

    res = []
    images = []
    for image in root.findall("image"):
        points = image[0].attrib["points"]
        points = points.split(';')
        points = [point.split(',') for point in points]
        points = [(float(point[0]), float(point[1])) for point in points]
        points = [(int(point[0]), int(point[1])) for point in points]
        points = Sort4Points(points)
        res.append(points)
        images.append(image.attrib["name"])
    
    res = np.array(res).astype(np.int32)
    return res, images

def Resize(input_folder, output_folder, target_shape):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        image = cv2.imread(os.path.join(input_folder, filename))
        image = KeyboardDetector.PreprocessInput(image, target_shape=target_shape)
        cv2.imwrite(os.path.join(output_folder, filename), image)

def Augment(input_folder, output_folder, num_augmentations=1):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    transform = A.Compose([
        A.RandomBrightnessContrast(p=1, brightness_limit=0.2, contrast_limit=0.2),
        A.ShiftScaleRotate(p=1, border_mode=cv2.BORDER_CONSTANT, rotate_limit=3, scale_limit=0.1, shift_limit=0.1),
    ])

    for filename in os.listdir(input_folder):
        image = cv2.imread(os.path.join(input_folder, filename))
        cv2.imwrite(os.path.join(output_folder, filename), image)
        name, ext = os.path.splitext(filename)
        for i in range(num_augmentations):
            transformed = transform(image=image)
            transformed = transformed["image"]
            cv2.imwrite(os.path.join(output_folder, f'{name}_aug{i}{ext}'), transformed)

# saves file paths of all videos in input_folder to output_file if they contain enough keyboard frames
def KeyboardVideos(input_folder, output_file):
    with open(output_file, 'w') as f:
        for filename in os.listdir(input_folder):
            video = PianoVideo(os.path.join(input_folder, filename))
            if video.get_piano() is not None:
                f.write(os.path.join(input_folder, filename)+'\n')

def KeyboardFrames(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    subfiles = glob.iglob(os.path.join(input_folder, '**/**'))
    files = glob.iglob(os.path.join(input_folder, '**'))
    all_files = sorted(list(set(list(subfiles) + list(files))))

    for filename in all_files:
        video = PianoVideo(filename)
        t = time.time()
        video.background
        video.hand_landmarks
        video.transcribed_midi
        print(f'{filename} {time.time()-t}')

 


def KeyboardDetectorData():
    data_dir = r'data/1_intermediate/keyboard_detector/'
    Resize(data_dir+'separated/with_keyboard', data_dir+'resized/with_keyboard', (180, 320))
    Resize(data_dir+'separated/without_keyboard', data_dir+'resized/without_keyboard', (180, 320))
    Augment(data_dir+'resized/with_keyboard', data_dir+'augmented/with_keyboard', 1)
    Augment(data_dir+'resized/without_keyboard', data_dir+'augmented/without_keyboard', 1)
    shutil.copytree(data_dir+'augmented/with_keyboard', 'data/2_final/keyboard_detector/with_keyboard', dirs_exist_ok=True)
    shutil.copytree(data_dir+'augmented/without_keyboard', 'data/2_final/keyboard_detector/without_keyboard', dirs_exist_ok=True)

#video = PianoVideo(r"data\0_raw\all_videos\Jane\2cz5qP36g_Y.webm")
# video = PianoVideo(r"data\0_raw\all_videos\Erik C 'Piano Man'\2PtpDnDwBk8.mp4")

# keys = video.DetectHandsByColor()
# cv2.imshow("w", keys)
# cv2.waitKey(0)

KeyboardFrames('data/0_raw/all_videos/flowkey – Learn piano', 'data/1_intermediate/background')