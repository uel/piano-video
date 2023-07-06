import cv2

class PianoVideo():
    def __init__(self, path) -> None:
        self.path = path


    def GetVideo(self):
        cap = cv2.VideoCapture(self.path)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
        cap.release()

    def FindIntervals(self, contains:function):
        pass


    def GetPiano(self):
        # get all intervals with a piano and average them out
        pass