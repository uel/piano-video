import time
t0 = time.time()
import random
from typing import Callable
import numpy as np
import cv2
from intervaltree import IntervalTree
import os
import json
import logging

class PianoVideo():
    def __init__(self, path, cache_path="data/1_intermediate") -> None:
        self.path = path
        self.cache_path = cache_path
        self.file_name = os.path.basename(path).split('.')[0]
        cap = cv2.VideoCapture(self.path)

        if not cap.isOpened():
            raise FileNotFoundError()

        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self._sections = None # interval tree of sections with piano
        self._background = None
        self._hand_landmarks = None
        self._detector = None
        self._audio_path = None
        self._transcribed_midi = None

        if True:
            if os.path.exists(f"{self.cache_path}/sections/{self.file_name}.json"):
                with open(f"{self.cache_path}/sections/{self.file_name}.json", 'r') as f:
                    self._sections = IntervalTree.from_tuples(json.load(f))

            if os.path.exists(f"{self.cache_path}/audio/{self.file_name}.mp3"):
                self._audio_path = f"{self.cache_path}/audio/{self.file_name}.mp3"

            if os.path.exists(f"{self.cache_path}/background/{self.file_name}.png"):
                self._background = cv2.imread(f"{self.cache_path}/background/{self.file_name}.png")

            if os.path.exists(f"{self.cache_path}/hand_landmarks/{self.file_name}.json"):
                with open(f"{self.cache_path}/hand_landmarks/{self.file_name}.json", 'r') as f:
                    self._hand_landmarks = json.load(f)

            if os.path.exists(f"{self.cache_path}/transcribed_midi/{self.file_name}.json"):
                with open(f"{self.cache_path}/transcribed_midi/{self.file_name}.json", 'r') as f:
                    self._transcribed_midi = IntervalTree.from_tuples(json.load(f))

            if os.path.exists(f"{self.cache_path}/background/{self.file_name}.png"):
                self._background = cv2.imread(f"{self.cache_path}/background/{self.file_name}.png")

    @property
    def detector(self):
        if self._detector is None:
            from key_matcher import FeatureKeyMatcher, KeyMatcher
            self._detector = KeyMatcher()
        return self._detector
    
    # audio path, if doesn't exist use ffmpeg to extract audio
    @property
    def audio_path(self):
        if not os.path.exists(f"{self.cache_path}/audio/{self.file_name}.mp3"):
            os.system(f'ffmpeg -hide_banner -loglevel error -i "{self.path}" "{self.cache_path}/audio/{self.file_name}.mp3"')
        return f"{self.cache_path}/audio/{self.file_name}.mp3"
    
    # sections, if doesn't exist use self.find_intervals(self.detector.ContainsKeyboard) and save to json
    @property
    def sections(self):
        if self._sections is None:
            self._sections = self.find_intervals(self.detector.ContainsKeyboard)
            with open(f"{self.cache_path}/sections/{self.file_name}.json", 'w') as f:
                json.dump(list(self._sections), f)
        return self._sections
    
    @property
    def hand_landmarks(self):
        if not os.path.exists(f"{self.cache_path}/hand_landmarks/{self.file_name}.json"):
            import hands
            self._hand_landmarks = []
            with hands.landmarker() as landmarker:
                for i, frame in self.get_video(): # frame must be in mp format
                    results = landmarker.detect(landmarker, frame, (1000*i)//self.fps)
                    if results.hand_landmarks:
                        d = []
                        for hand in results.hand_landmarks: # TODO: multiply by width and height?
                            d.append([[round(landmark.x, 4), round(landmark.y, 4), round(landmark.z, 4)] for landmark in hand])
                        self._hand_landmarks.append([i, d])
            with open(f"{self.cache_path}/hand_landmarks/{self.file_name}.json", 'w') as f:
                json.dump(self._hand_landmarks, f)
        
        # generator function that returns empty list if no hand landmarks 
        def landmarks_generator():
            current = 0
            for i in range(0, self.frame_count):
                if current < len(self._hand_landmarks) and self._hand_landmarks[current][0] == i:
                    yield self._hand_landmarks[current][1]
                    current += 1
                else:
                    yield []

        return landmarks_generator
    
    
    @property
    def transcribed_midi(self):
        if not os.path.exists(f"{self.cache_path}/transcribed_midi/{self.file_name}.json"):
            import transcription
            result = transcription.transcribe_piano(self.audio_path)
            self._transcribed_midi = []
            for note in result:
                onset = int(note["onset_time"]*self.fps)
                offset = int(note["offset_time"]*self.fps)
                if onset == offset: continue # prevent null intervals in interval tree
                self._transcribed_midi.append([onset, offset, [note["midi_note"], note["velocity"]]])

            with open(f"{self.cache_path}/transcribed_midi/{self.file_name}.json", 'w') as f:
                json.dump(self._transcribed_midi, f)

            self._transcribed_midi = IntervalTree.from_tuples(self._transcribed_midi)

        return self._transcribed_midi

    def get_video(self, skip:float=0.):
        '''Generates frames from video evenly spaced by skip seconds'''
        cap = cv2.VideoCapture(self.path)

        i = 0 # frame number
        t = 0 # time in seconds
        last_t = -skip # last time when a frame was yielded
        while True:
            t = i / self.fps
            ret, frame = cap.read()
            if not ret: break
            if t-last_t >= skip:
                last_t = t
                yield i, frame
            i+=1
        cap.release()

    def find_intervals(self, contains:Callable[[np.ndarray], bool], accuracy=.5):
        '''Finds intervals in video where contains is true given the frame'''
        intervals = IntervalTree()

        left = None
        for i, frame in self.get_video(accuracy):
            if contains(frame):
                if left is None:
                    left = i
            else:
                if left is not None:
                    intervals[left:i] = True
                    left = None

        if left is not None:
            intervals[left:i] = True
        
        return intervals

    @property
    def background(self):
        '''Gets the frame of the piano using mediapipe hand landmarks'''
        if self._background is not None:
            return self._background

        size = sum([i.end-i.begin for i in self.sections])

        if size < 10*self.fps: # at least 10 seconds of keyboard frames
            logging.warning(f"Background not extracted, {self.file_name} has less than 10 seconds of keyboard frames")
            return None

        frames = []
        nan_counts = np.zeros((self.width,)) # make sure there are no NaNs in result

        landmarks = self.hand_landmarks()
        handc = 0
        for i, frame in self.get_video():
            landmark_result = next(landmarks)
            if not self.sections[i]: continue
            #if len(landmark_result) < 2: continue

            frame = frame.astype(float)
            for hand in landmark_result:
                handc+=1
                left = np.inf
                right = 0
                for landmark in hand:
                    # Get horzizontal bounds of landmarks and replace horizontal area in image with Nan values
                    left = min(left, landmark[0])
                    right = max(right, landmark[0])
                left = int(left*frame.shape[1])
                left = max(0, left-10)
                right = int(right*frame.shape[1])
                right = min(frame.shape[1], right+10)
                frame[:,left:right,:] = np.NAN

                frame = cv2.resize(frame, (640, int(640 / frame.shape[1] * frame.shape[0])))

                if len(frames) > 200:
                    frames[random.randint(0, 200)] = frame
                else:
                    frames.append(frame)
                
        
        # print(handc)
        # for i in range(len(frames)):
        #     cv2.imwrite(f".temp/{i}.png", frames[i].astype(np.uint8))

        self._background = np.nanmedian(frames, axis=(0)).astype(np.uint8)
        if np.isnan(self._background).any():
            logging.warning(f"NaN values in {self.file_name} background")

        cv2.imwrite(f"{self.cache_path}/background/{self.file_name}.png", self._background)

        return self._background
    
    @property
    def key_segments(self): # TODO: caching
        import keyboard_segmentation
        return keyboard_segmentation.segment_keys(self.background, self.detector)
                    

#video = PianoVideo("demo/scarlatti.mp4")
# video = PianoVideo(r"C:\Users\danif\s\BP\data\0_raw\all_videos\Erik C 'Piano Man'\8xJdM4S-fko.mp4")
# video = PianoVideo(r"C:\Users\danif\s\BP\data\0_raw\all_videos\flowkey – Learn piano\CRHexNAxnlU.mp4")
# sections = video.sections
# midi_boxes, masks = video.key_segments
# print(time.time()-t0)
# video.key_segments
# print(time.time()-t0)