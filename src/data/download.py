import os
import ffmpeg
import random

CHANNELS = [
    "https://www.youtube.com/@flowkey_DE",
    "https://www.youtube.com/@PaulBartonPiano",
    "https://www.youtube.com/@Libertyparkmusic_LPM",
    "https://www.youtube.com/@ErikCPianoman",
    "https://www.youtube.com/@janepianotutorials"
]

# using yt-dlp binary, download all videos from CHANNELS, save to data/(channel_name)/(video_id).mp4
# downlaod in 480
def DownloadChannels():
    for channel in CHANNELS:
        os.system(f'yt-dlp -S "res:480,fps" -o "data/all_videos/%(channel)s/%(id)s.%(ext)s" "{channel}" --max-downloads 100 --download-archive "data/downloaded.txt"')


# 1. Download all videos from CHANNELS
# 2. Sample an equal number of frames from each channel
def SampleRandomFrames(frames_per_video=2):
    for folder in os.listdir('data/all_videos'):
        for filename in os.listdir('data/all_videos/' + folder):
            if filename.endswith('.mp4'):
                duration = float(ffmpeg.probe('data/all_videos/' + folder + '/' + filename)['format']['duration'])
                seconds = list(range(0, int(duration+1)))
                sample_seconds = random.sample(seconds, frames_per_video)
                for second in sample_seconds:
                    #os.system(f'ffmpeg -ss {second} -i "data/all_videos/{folder}/{filename}" -frames:v 1 "data/all_frames/{filename[:-4]}_{second}.jpg"')
                    ffmpeg.input(f'data/all_videos/{folder}/{filename}', ss=second)\
                          .output(f'data/all_frames/{filename[:-4]}_{second}.jpg', vframes=1, loglevel='error')\
                          .run()