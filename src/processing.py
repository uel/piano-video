import os
import glob
import random

import ffmpeg
import yt_dlp

from piano_video import PianoVideo

def DownloadChannel(channel, video_folder, video_limit=100):
    '''Downloads 100 most recent videos of a channel'''

    ydl_opts = {
        'format': 'bestvideo[height<=480][fps<=30]+bestaudio/best[height<=480][fps<=30]/best',
        'outtmpl': f'{video_folder}/{channel}/%(id)s.%(ext)s',
        'max_downloads': video_limit,
        'download_archive': f'{video_folder}/downloaded.txt',
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([channel])

def SampleRandomFrames(input_folder, output_folder, frames_per_video=2):
    '''Samples random frames from all videos in input_folder and saves them to output_folder'''
    os.makedirs(output_folder, exist_ok=True)

    for folder in os.listdir(input_folder):
        for filename in os.listdir(os.path.join(input_folder, folder)):
            if filename.endswith('.mp4'):
                file_path = os.path.join(input_folder, folder, filename)
                duration = float(ffmpeg.probe(file_path)['format']['duration'])
                seconds = list(range(0, int(duration+1)))
                sample_seconds = random.sample(seconds, frames_per_video)
                for second in sample_seconds:
                    output_file = os.path.join(output_folder, f"{filename[:-4]}_{second}.jpg")
                    ffmpeg.input(file_path, ss=second)\
                          .output(output_file, vframes=1, loglevel='error')\
                          .run()    

def GetFiles(input_folder):
    subfiles = glob.iglob(os.path.join(input_folder, '**/**'))
    files = glob.iglob(os.path.join(input_folder, '**'))
    all_files = sorted(list(set(list(subfiles) + list(files))))
    all_files = [file for file in all_files if os.path.isfile(file)]
    return all_files

if __name__ == "__main__":
    # CHANNELS = [
    #     "https://www.youtube.com/@flowkey_DE",
    #     "https://www.youtube.com/@PaulBartonPiano",
    #     "https://www.youtube.com/@Libertyparkmusic_LPM",
    #     "https://www.youtube.com/@ErikCPianoman",
    #     "https://www.youtube.com/@janepianotutorials"
    # ]
    # for channel in CHANNELS:
    #     DownloadChannel(channel, 'data/videos')
    # SampleRandomFrames('data/videos', 'data/frames/all_frames', 2)
    
    files = GetFiles("data/videos")
    for filename in files:
        print(filename)
        video = PianoVideo(filename)
        video.sections # saves file into data/background
