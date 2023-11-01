from keras.models import load_model
import cv2
import numpy as np

class KeyboardDetector:
    def __init__(self, model_path=None):
        if model_path is None:
            self.model = None
        else:
            self.model = load_model(model_path)
        config = self.model.get_config()
        self.image_shape = config["layers"][0]["config"]["batch_input_shape"][1:3]

    def PreprocessInput(image, target_shape):
        target_height = target_shape[0]
        aspect_ratio = image.shape[1] / image.shape[0]
        target_width = int(target_height * aspect_ratio)

        if target_width > target_shape[1]:
            target_width = target_shape[1]
            target_height = int(target_width / aspect_ratio)

        resized_image = cv2.resize(image, (target_width, target_height))

        # Place the resized image on a black background
        background = np.zeros((target_shape[0], target_shape[1], 3), dtype=np.uint8)
        x_offset = (target_shape[1] - target_width) // 2
        y_offset = (target_shape[0] - target_height) // 2
        background[y_offset:y_offset+target_height, x_offset:x_offset+target_width] = resized_image

        return background
    
    def ContainsKeyboard(self, image):
        image = KeyboardDetector.PreprocessInput(image, self.image_shape)
        image = np.expand_dims(image, axis=0)
        pred = self.model.predict(image, verbose=0)[0][0]
        return pred > 0.5
    