from typing import Callable
import numpy as np
import os
import json
import logging
from functools import cached_property

import file_io
import setup

import cv2
from intervaltree import IntervalTree

class PianoVideo():
    setup_complete = False

    def __init__(self, path, data_path="data", max_shape=640) -> None:
        self.path = path
        self.data_path = data_path
        self.file_name = os.path.basename(path).split('.')[0] # used as identifier

        if not PianoVideo.setup_complete:
            setup.setup_check(data_path)
            PianoVideo.setup_complete = True

        cap = cv2.VideoCapture(self.path)

        if not cap.isOpened():
            raise FileNotFoundError()

        self.fps = round(cap.get(cv2.CAP_PROP_FPS))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.max_shape = max_shape

    @cached_property
    def detector(self):
        from keyboard_detection import KeyboardDetector
        return KeyboardDetector(self.max_shape)

    @cached_property
    def audio_path(self):
        '''Extracts audio from video and returns path to audio file'''
        if os.path.exists(f"{self.data_path}/audio/{self.file_name}.mp3"):
            return f"{self.data_path}/audio/{self.file_name}.mp3"
        
        os.system(f'ffmpeg -hide_banner -loglevel error -i "{self.path}" "{self.data_path}/audio/{self.file_name}.mp3"')
        
        return f"{self.data_path}/audio/{self.file_name}.mp3"

    @cached_property
    def sections(self):
        if os.path.exists(f"{self.data_path}/sections/{self.file_name}.json"):
                with open(f"{self.data_path}/sections/{self.file_name}.json", 'r') as f:
                    return IntervalTree.from_tuples(json.load(f))

        _sections = self.find_intervals(self.detector.DetectKeyboard)
        with open(f"{self.data_path}/sections/{self.file_name}.json", 'w') as f:
            json.dump(list(_sections), f)

        return _sections

    @cached_property
    def hand_landmarks(self):
        import hands
        if os.path.exists(f"{self.data_path}/hand_landmarks/{self.file_name}.bin"):
            landmarks = file_io.read_landmarks(f"{self.data_path}/hand_landmarks/{self.file_name}.bin")
            return hands.fill_gaps(landmarks)
            # return landmarks
        
        _hand_landmarks = []
        with hands.landmarker() as landmarker:
            for i, frame in self.get_video(): # frame must be in mp format
                left_hand, right_hand = landmarker.detect(landmarker, frame, (1000*i)//self.fps)
                _hand_landmarks.append((i, left_hand, right_hand))
        file_io.write_landmarks(f"{self.data_path}/hand_landmarks/{self.file_name}.bin", _hand_landmarks)

        return _hand_landmarks
        # return _hand_landmarks
        return hands.fill_gaps(_hand_landmarks)
    
    @property
    def hand_landmarker(self):
        # generator function that returns empty list if no hand landmarks
        def landmarks_generator():
            current = 0
            for i in range(0, self.frame_count+1):
                if current < len(self.hand_landmarks) and self.hand_landmarks[current][0] == i:
                    yield self.hand_landmarks[current][1:]
                    current += 1
                else:
                    yield (None, None)

        return landmarks_generator()


    @cached_property
    def transcribed_midi(self):
        if os.path.exists(f"{self.data_path}/transcribed_midi/{self.file_name}.json"):
            with open(f"{self.data_path}/transcribed_midi/{self.file_name}.json", 'r') as f:
                return IntervalTree.from_tuples(json.load(f))

        import transcription
        result = transcription.transcribe_piano(self.audio_path)
        self._transcribed_midi = []
        for note in result:
            onset = int(note["onset_time"]*self.fps)
            offset = int(note["offset_time"]*self.fps)
            if onset == offset: continue # prevent null intervals in interval tree, TODO: minimal interval length?
            self._transcribed_midi.append([onset, offset, [note["midi_note"], note["velocity"]]])

        with open(f"{self.data_path}/transcribed_midi/{self.file_name}.json", 'w') as f:
            json.dump(self._transcribed_midi, f)

        return IntervalTree.from_tuples(self._transcribed_midi)

    def get_video(self, skip:float=0.):
        '''Generates frames from video evenly spaced by skip seconds'''
        cap = cv2.VideoCapture(self.path)

        i = 0 # frame number
        t = 0 # time in seconds
        last_t = -skip # last time when a frame was yielded
        while True:
            t = i / self.fps
            ret, new_frame = cap.read()
            if not ret:
                if t - last_t != 0:
                    yield i, frame # yield last frame
                break
            
            frame = new_frame

            if t-last_t >= skip:
                last_t = t
                yield i, frame
            i+=1
        cap.release()

    def find_intervals(self, contains:Callable[[np.ndarray], bool], accuracy=5.):
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

            if right-frame_acc < i:
                if contains(frame):
                    right = i
                else:
                    new_intervals[left:right] = True

                    if sorted_intervals: left, right, _ = sorted_intervals.pop(0)
                    else: break
                    
        new_intervals[left:right] = True

        return new_intervals

    def resize_frame(self, frame):
        '''Resizes frame to fit in a square with size max_shape while keeping aspect ratio'''
        if max(frame.shape) > self.max_shape:
            scale = self.max_shape / max(frame.shape)
            return cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        else:
            return frame

    @cached_property
    def background(self):
        '''Gets the frame of the piano using hand landmarks'''
        if os.path.exists(f"{self.data_path}/background/{self.file_name}.png"):
            return cv2.imread(f"{self.data_path}/background/{self.file_name}.png")

        frames = []
        non_nan_counts = np.zeros((self.width,)) # make sure there are no NaNs in result
        frame_count = 0
        landmarks = self.hand_landmarker
        handc = 0
        for i, frame in self.get_video():
            landmark_result = next(landmarks)
            if not self.sections[i]: continue
            #if len(landmark_result) < 2: continue

            frame = frame.astype(float)
            not_nan = np.ones((self.width,), dtype=np.uint8) # 1 if not NaN, 0 if NaN
            for hand in landmark_result:
                if hand is None: continue
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

            frame = self.resize_frame(frame)

            if np.any(not_nan & ( non_nan_counts <= (min(non_nan_counts)+5) )):
                frame_count += 1
                if len(frames) >= 200:
                    if min(non_nan_counts) > 200: break
                    frames[frame_count%200] = frame # counts are not being removed,
                else:
                    frames.append(frame)
                non_nan_counts += not_nan

        if len(frames) == 0:
            return None
            # raise Exception("No frames with hands found")

        _background = np.nanmedian(frames, axis=(0)).astype(np.uint8)

        if np.isnan(_background).any():
            logging.warning(f"NaN values in {self.file_name} background")

        cv2.imwrite(f"{self.data_path}/background/{self.file_name}.png", _background)

        return _background
    
    def keyboard_frame_count(self):
        '''Returns the number of frames where the keyboard is detected'''
        return sum([i.end-i.begin for i in self.sections])
    
    def approx_keys(self):
        import keyboard_segmentation
        keyboard_loc = self.detector.DetectKeyboard(self.background)
        keys = keyboard_segmentation.segment_keys(self.background, keyboard_loc)
        return keyboard_loc, keys
    
    @cached_property
    def keys(self):
        if os.path.exists(f"{self.data_path}/keys/{self.file_name}.json"):
            with open(f"{self.data_path}/keys/{self.file_name}.json", 'r') as f:
                return json.load(f) 
            
        from fingers import finger_notes

        keyboard_loc, keys = self.approx_keys()

        # iterate over onsents, get hand landmarks at that time
        min_dist = np.inf
        for i in  [0, -1, 1]:
            _, dist = finger_notes(self.transcribed_midi, self.hand_landmarker, keys, midi_id_offset=i)
            if dist < min_dist:
                min_dist = dist
                best_shift = i 

        if best_shift != 0: logging.info(f"Shifted keys by {-best_shift} octaves")

        for key in keys:
            key[1] += -best_shift*12 # -best_shift because transcribed midi had to be shifted by best_shift

        with open(f"{self.data_path}/keys/{self.file_name}.json", 'w') as f:
            json.dump([keyboard_loc, keys], f)

        return keyboard_loc, keys
    
    @cached_property
    def fingers(self):
        if os.path.exists(f"{self.data_path}/fingers/{self.file_name}.json"):
            with open(f"{self.data_path}/fingers/{self.file_name}.json", 'r') as f:
                return IntervalTree.from_tuples(json.load(f))

        from fingers import remove_outliers, finger_notes

        _, keys = self.keys

        best_notes, _ = finger_notes(self.transcribed_midi, self.hand_landmarker, keys)
        self._fingers = remove_outliers(best_notes, keys)

        with open(f"{self.data_path}/fingers/{self.file_name}.json", 'w') as f:
            json.dump(list(sorted(self._fingers)), f)

        return self._fingers


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    video = PianoVideo("demo/scarlatti.mp4")
    # video = PianoVideo(r"C:\Users\danif\s\BP\data\videos\Erik C 'Piano Man'\gBMmUVzvl2U.mp4")
    # video = PianoVideo(r"C:\Users\danif\s\BP\data\videos\Liberty Park Music\3psRRVgGYdc.mp4")
    # video = PianoVideo(r"C:\Users\danif\s\BP\data\videos\Paul Barton\NLPxfEMfnVM.mp4")
    # video = PianoVideo(r"C:\Users\danif\s\BP\data\videos\Jane\2cz5qP36g_Y.webm")

    # video = PianoVideo(r"C:\Users\danif\s\BP\data\videos\Jane\XYFZFlDK2ko.webm")
    # video.hand_landmarks
    # video = PianoVideo(r"C:\Users\danif\s\BP\recording\rec3.mp4")
    # video = PianoVideo(r"C:\Users\danif\s\BP\demo\sections_test.mp4")
    video.transcribed_midi
    # pass
    # sections = video.sections
    # midi_boxes, masks = video.key_segments

    # print(time.time()-t0)
    # video.key_segments
    # print(time.time()-t0)