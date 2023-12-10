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
from draw import draw_landmarks_on_image

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
        self._keys = None

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

            if os.path.exists(f"{self.cache_path}/keys/{self.file_name}.json"):
                with open(f"{self.cache_path}/keys/{self.file_name}.json", 'r') as f:
                    self._keyboard_box, self._white_key_lightness_thresh, self._keys = json.load(f)

    @property
    def detector(self):
        if self._detector is None:
            from key_matcher import FeatureKeyMatcher, KeyMatcher, YoloMatcher
            self._detector = YoloMatcher()
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
                if onset == offset: continue # prevent null intervals in interval tree, TODO: minimal interval length?
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

    def find_intervals(self, contains:Callable[[np.ndarray], bool], accuracy=10., min_length=30.):
        '''Finds intervals in video where contains is true given the frame'''
        intervals = IntervalTree()
        left = None
        for i, frame in self.get_video(accuracy):
            if contains(frame):
                if left is None:
                    left = i
            else:
                if left is not None:
                    intervals[left:i] = True # right is first index with stride accuracy where contains is false
                    left = None

        if left is not None and left != i:
            intervals[left:i] = True

        if len(intervals) == 0: return intervals

        # Refining
        frame_acc = int(self.fps*accuracy)
        sorted_intervals = sorted(intervals)
        new_intervals = IntervalTree()

        left, right, _ = sorted_intervals.pop(0)
        for i, frame in self.get_video():

            if left - frame_acc < i < left:
                if contains(frame):
                    left = i

            if right-frame_acc < i <= right:
                if contains(frame):
                    right = i
                else:
                    if right - left > min_length*self.fps:
                        new_intervals[left:right] = True

                    if sorted_intervals:
                        left, right, _ = sorted_intervals.pop(0)
                    else:
                        break
                    
        if right - left > min_length*self.fps:
            new_intervals[left:right] = True

        return new_intervals

    @property
    def background(self):
        '''Gets the frame of the piano using mediapipe hand landmarks'''
        if self._background is not None:
            return self._background

        size = sum([i.end-i.begin for i in self.sections])

        if size < 0.5*self.frame_count: # at least 50% of video must be piano
            logging.warning(f"Background not extracted, {self.file_name} has less than 50% of keyboard frames")
            return None

        frames = []
        non_nan_counts = np.zeros((self.width,)) # make sure there are no NaNs in result
        frame_count = 0
        landmarks = self.hand_landmarks()
        handc = 0
        for i, frame in self.get_video():
            landmark_result = next(landmarks)
            if not self.sections[i]: continue
            #if len(landmark_result) < 2: continue

            frame = frame.astype(float)
            not_nan = np.ones((self.width,), dtype=np.uint8) # 1 if not NaN, 0 if NaN
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
                not_nan[left:right] = 0

            frame = cv2.resize(frame, (640, int(640 / frame.shape[1] * frame.shape[0]))) # TODO: standard for scaling

            if np.any(not_nan & ( non_nan_counts <= (min(non_nan_counts)+5) )):
                frame_count += 1
                if len(frames) >= 200:
                    if min(non_nan_counts) > 200: break
                    frames[frame_count%200] = frame # counts are not being removed,
                else:
                    frames.append(frame)
                non_nan_counts += not_nan

        self._background = np.nanmedian(frames, axis=(0)).astype(np.uint8)
        if np.isnan(self._background).any():
            logging.warning(f"NaN values in {self.file_name} background")

        # cv2.imshow('img', self._background)
        # cv2.waitKey(0)

        cv2.imwrite(f"{self.cache_path}/background/{self.file_name}.png", self._background)

        return self._background

    @property
    def keys(self):
        if self._keys is not None:
            return [self._keyboard_box, self._white_key_lightness_thresh, self._keys]
        
        import keyboard_segmentation
        keyboard_box, white_key_lightness_thresh, keys = keyboard_segmentation.segment_keys(self.background, self.detector)
        self._keyboard_box = keyboard_box
        self._white_key_lightness_thresh = white_key_lightness_thresh
        self._keys = keys

        with open(f"{self.cache_path}/keys/{self.file_name}.json", 'w') as f:
            json.dump([self._keyboard_box, self._white_key_lightness_thresh, self._keys], f)
    
    @property
    def fingers(self):
        from fingers import finger_notes, remove_outliers

        _, _, keys = self.keys

        # iterate over onsents, get hand landmarks at that time
        min_dist = np.inf
        best_shift = 0
        best_notes = None
        for i in  [0, -1, 1]:
            notes, dist = finger_notes(self.transcribed_midi, self.hand_landmarks(), keys, midi_id_offset=i)
            if dist < min_dist:
                min_dist = dist
                best_shift = i
                best_notes = notes

        if best_shift != 0: logging.info(f"Shifted midi by {best_shift} octaves")

        self._fingers = remove_outliers(best_notes, keys)

        with open(f"{self.cache_path}/fingers/{self.file_name}.json", 'w') as f:
            json.dump(list(sorted(self._fingers)), f)

        return self._fingers


logging.basicConfig(level=logging.DEBUG)
# video = PianoVideo("demo/scarlatti.mp4")
# video = PianoVideo(r"C:\Users\danif\s\BP\data\0_raw\all_videos\Erik C 'Piano Man'\gBMmUVzvl2U.mp4")
# video = PianoVideo(r"C:\Users\danif\s\BP\data\0_raw\all_videos\Liberty Park Music\3psRRVgGYdc.mp4")
# video = PianoVideo(r"C:\Users\danif\s\BP\data\0_raw\all_videos\Paul Barton\NLPxfEMfnVM.mp4")
# video = PianoVideo(r"C:\Users\danif\s\BP\data\0_raw\all_videos\Jane\2cz5qP36g_Y.webm")

# video = PianoVideo(r"C:\Users\danif\s\BP\data\0_raw\all_videos\Jane\YgO-UJDfCZE.webm")
# video.fingers
# pass
# sections = video.sections
# midi_boxes, masks = video.key_segments
# print(time.time()-t0)
# video.key_segments
# print(time.time()-t0)