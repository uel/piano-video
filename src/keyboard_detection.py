import os
import cv2

class KeyboardDetector:
    def __init__(self, max_image_size) -> None:
        from ultralytics import YOLO
        self.model = YOLO("models/keyboard_detection.pt")
        self.max_image_size = max_image_size

    def DetectKeyboard(self, img):
        result = self.model.predict(img, verbose=False, imgsz=self.max_image_size, conf=0.5)[0].boxes
        if len(result.conf) > 0:
            return result.xyxyn[0].tolist()
        else:
            return None

if __name__ == "__main__":
    matcher = KeyboardDetector()
    dir_path = 'data/keyboard_detector/separated/with_keyboard/'
    files = os.listdir(dir_path)
    count = 0
    correct_count = 0
    for file in files:
        img = cv2.imread(f'{dir_path}/{file}')
        c = matcher.ContainsKeyboard(img)
        count += 1
        if c: correct_count += 1
        else:
            cv2.imshow('img', img)
            cv2.waitKey(0)
            pass
        print(f'{correct_count}/{count}')
            