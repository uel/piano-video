import numpy as np
import xml.etree.ElementTree as ET
import cv2
import os
import shutil
import albumentations as A

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

def Resize(input_folder, output_folder, target_shape=(360, 640)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        image = cv2.imread(os.path.join(input_folder, filename))
        target_height = target_shape[0]
        aspect_ratio = image.shape[1] / image.shape[0]
        target_width = int(target_height * aspect_ratio)

        if target_width > target_shape[1]:
            target_width = target_shape[1]
            target_height = int(target_width / aspect_ratio)

        resized_image = cv2.resize(image, (target_width, target_height))

        # Place the resized image on a black background
        background = np.zeros((360, 640, 3), dtype=np.uint8)
        x_offset = (640 - target_width) // 2
        y_offset = (360 - target_height) // 2
        background[y_offset:y_offset+target_height, x_offset:x_offset+target_width] = resized_image

        cv2.imwrite(os.path.join(output_folder, filename), background)

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

# sorted by filename
def CopyNFiles(input_dir, output_dir, n):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = sorted(os.listdir(input_dir))
    for i in range(n):
        shutil.copy(os.path.join(input_dir, files[i]), output_dir)

def KeyboardDetectorData():
    data_dir = r'data/1_intermediate/keyboard_detector/'
    Resize(data_dir+'separated/with_keyboard', data_dir+'resized/with_keyboard')
    Resize(data_dir+'separated/without_keyboard', data_dir+'resized/without_keyboard')
    Augment(data_dir+'resized/with_keyboard', data_dir+'augmented/with_keyboard', 1)
    Augment(data_dir+'resized/without_keyboard', data_dir+'augmented/without_keyboard', 1)

    file_count = min(len(os.listdir(data_dir+'augmented/with_keyboard')), len(os.listdir(data_dir+'augmented/without_keyboard')))
    CopyNFiles(data_dir+'augmented/with_keyboard', 'data/2_final/keyboard_detector/with_keyboard', file_count)
    CopyNFiles(data_dir+'augmented/without_keyboard', 'data/2_final/keyboard_detector/without_keyboard', file_count)
