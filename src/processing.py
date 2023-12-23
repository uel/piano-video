import os
import glob
from piano_video import PianoVideo
import time

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
        print(filename)
        video = PianoVideo(filename)
        t = time.time()
        if video.background is not None:
            print("sections ", time.time()-t)
            video.hand_landmarks
            print("landmarks ", time.time()-t)
            # video.transcribed_midi
            # print("midi ", time.time()-t)

KeyboardFrames('data/0_raw/all_videos/Jane', 'data/1_intermediate/background')
KeyboardFrames('data/0_raw/all_videos/flowkey – Learn piano', 'data/1_intermediate/background')
KeyboardFrames('data/0_raw/all_videos/Paul Barton', 'data/1_intermediate/background')
KeyboardFrames('data/0_raw/all_videos/Liberty Park Music', 'data/1_intermediate/background')
KeyboardFrames("data/0_raw/all_videos/Erik C 'Piano Man'", 'data/1_intermediate/background')