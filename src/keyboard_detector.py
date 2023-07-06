import os
import cv2
import numpy as np
from data import GetPointsFromXML
import matplotlib.pyplot as plt
import albumentations as A
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

def preprocess_image(image, inverse_mappping=False):
    # Resize image while maintaining the aspect ratio
    target_height = 360
    aspect_ratio = image.shape[1] / image.shape[0]
    target_width = int(target_height * aspect_ratio)

    if target_width > 640:
        target_width = 640
        target_height = int(target_width / aspect_ratio)

    resized_image = cv2.resize(image, (target_width, target_height))
    resize_ratio = target_width / image.shape[1]

    # Place the resized image on a black background
    background = np.zeros((360, 640, 3), dtype=np.uint8)
    x_offset = (640 - target_width) // 2
    y_offset = (360 - target_height) // 2
    background[y_offset:y_offset+target_height, x_offset:x_offset+target_width] = resized_image

    # Normalize pixel values between 0 and 1
    normalized_image = background / 255.0

    # Create function for coordinate mapping
    if inverse_mappping:
        map_coordinates = lambda x, y: ( (x*640 - x_offset)/resize_ratio, (y*360 - y_offset)/resize_ratio )
    else:
        map_coordinates = lambda x, y: ( (x*resize_ratio + x_offset) / 640, (y*resize_ratio + y_offset) / 360 )

    return normalized_image, map_coordinates

def LoadImages(dir, files=[], points=None):

    if len(files) == 0:
        files = os.listdir(dir)

    points_res = []
    images = []
    for file in files:
        img = cv2.imread(dir +"/"+ file)
        normalized_image, map_coordinates = preprocess_image(img)
        images.append(normalized_image)
        if points is not None:
            points_res.append(np.array([map_coordinates(x, y) for (x, y) in points[files.index(file)]]))

    if points is not None:
        return images, points_res
    return images


def augment_data(image, bounding_box, n=5):
    image = (image * 255).astype(np.uint8)
    bounding_box = bounding_box.reshape((4, 2))

    # Convert bounding box coordinates to a list of tuples, scale them to the image size
    bounding_box = [(x * 639, y * 359) for (x, y) in bounding_box]

    transform = A.Compose([
        A.RandomBrightnessContrast(p=1, brightness_limit=0.2, contrast_limit=0.2),
        A.ShiftScaleRotate(p=1, border_mode=cv2.BORDER_CONSTANT, rotate_limit=3, scale_limit=0.1, shift_limit=0.1)
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=True))


    augmented_images = []
    bounding_boxes = []
    
    # Generate n augmentations
    while len(augmented_images) < n:
        # Apply the transformation to the image
        augmented = transform(image=image, keypoints=bounding_box)
        img = augmented['image']
        b_box = augmented['keypoints']
    
        # Append the augmented image and bounding box coordinates to the result
        if len(b_box) == 4:
            augmented_images.append(img/255.0)
            b_box = np.array(b_box)
            b_box[:, 0] /= 639
            b_box[:, 1] /= 359
            b_box = b_box.reshape((8,))
            bounding_boxes.append(b_box)
    
    return augmented_images, bounding_boxes

# for (points, image) in zip(all_points, images):
#     img = cv2.imread("data/separated_frames/with_keyboard/" + image)
#     if img.shape[1]/img.shape[0] == 640/360: continue
#     normalized_image, map_coordinates = preprocess_image(img)
#     points = [map_coordinates(x, y) for (x, y) in points]
#     print(points)
#     for (x, y) in points:
#         cv2.circle(normalized_image, (int(x * 640), int(y * 360)), 3, (0, 255, 0), -1)
#     cv2.imshow(image, normalized_image)
#     cv2.waitKey(0)


# --- Detection ---
def TrainDetectionModel():
    detection = Sequential()
    detection.add(Conv2D(32, (3, 3), activation='relu', input_shape=(360, 640, 3)))
    detection.add(MaxPooling2D((2, 2)))
    detection.add(Conv2D(64, (3, 3), activation='relu'))
    detection.add(MaxPooling2D((2, 2)))
    detection.add(Flatten())
    detection.add(Dense(64, activation='relu'))
    detection.add(Dense(1, activation='sigmoid'))
    detection.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    with_keyboard = LoadImages("data/separated_frames/with_keyboard")
    without_keyboard = LoadImages("data/separated_frames/without_keyboard")
    min_class_count = min(len(with_keyboard), len(without_keyboard))
    combined = np.array(with_keyboard[:min_class_count] + without_keyboard[:min_class_count])
    labels = np.array([1.]*(min_class_count) + [0.]*(min_class_count))
    x_train, x_test, y_train, y_test = train_test_split(combined, labels, test_size=0.3, random_state=0)

    detection.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test), batch_size=4)
    detection.save("models/detection.h5")
    return detection


# --- Bounding box ---
def TrainBoundingBoxModel(model_name, grayscale=False):
    bounding_box = Sequential()

    if grayscale:
        bounding_box.add(Conv2D(32, (3, 3), activation='relu', input_shape=(360, 640, 1)))
    else:
        bounding_box.add(Conv2D(32, (3, 3), activation='relu', input_shape=(360, 640, 3)))

    bounding_box.add(MaxPooling2D((2, 2)))
    
    bounding_box.add(Conv2D(64, (3, 3), activation='relu'))
    bounding_box.add(MaxPooling2D((2, 2)))

    bounding_box.add(Flatten())
    bounding_box.add(Dense(64, activation='relu'))
    bounding_box.add(Dense(8, activation='sigmoid'))
    bounding_box.compile(optimizer='adam', loss='mean_squared_error')

    try:
        bounding_box = load_model("models/" + model_name + ".h5")
    except:
        pass

    points, images = GetPointsFromXML("data/separated_frames/keyboard_annotations.xml")
    images, points = LoadImages("data/separated_frames/with_keyboard", images, points)
    
    if grayscale:
        new_images = []
        for image in images:
            new_image = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]
            new_images.append(new_image)
        images = np.array(new_images)
    else:
        images = np.array(images)

    points = np.array(points)
    points = points.reshape((points.shape[0], 8))

    x_train, x_test, y_train, y_test = train_test_split(images, points, test_size=0.2, random_state=0)

    for i in range(len(x_train)):
        # prediction = bounding_box.predict(np.array([test_img]))[0]
        # prediction = prediction.reshape((4, 2))
        aug_data, pts = augment_data(x_train[i], y_train[i], 4)
        # for (x, y) in pts[0]:
        #     cv2.circle(aug_data[0], (int(x*640), int(y*360)), 3, (0, 255, 0), -1)
        # plt.imshow(aug_data[0])
        # plt.show()
        x_train = np.append(x_train, aug_data, axis=0)
        y_train = np.append(y_train, pts, axis=0)

    for i in range(len(x_test)):
        aug_data, pts = augment_data(x_test[i], y_test[i], 4)
        x_test = np.append(x_test, aug_data, axis=0)
        y_test = np.append(y_test, pts, axis=0)

    bounding_box.fit(x_train, y_train, epochs=15, validation_data=(x_test, y_test), batch_size=8)
    bounding_box.save("models/" + model_name + ".h5")

    return bounding_box

#TrainBoundingBoxModel("bounding_box_grayscale", grayscale=True)