import os
import ffmpeg
import random
import numpy as np
import xml.etree.ElementTree as ET

CHANNELS = [
    #"https://www.youtube.com/@flowkey_DE",
    #"https://www.youtube.com/@PaulBartonPiano",
    #"https://www.youtube.com/@Libertyparkmusic_LPM",
    #"https://www.youtube.com/@ErikCPianoman",
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

def Sort4Points(points):
    # sort points clockwise
    # https://stackoverflow.com/a/6989383
    centroid = np.mean(points, axis=0)
    points = sorted(points, key=lambda p: np.arctan2(p[1] - centroid[1], p[0] - centroid[0]))
    return points

def GetPointsFromXML(file):
    root = ET.parse(file).getroot()

    res = []
    images = []
    for image in root.findall("image"):
        points = image[0].attrib["points"]
        points = points.split(';')
        points = [point.split(',') for point in points]
        points = [(float(point[0]), float(point[1])) for point in points]
        points = [(int(point[0]), int(point[1])) for point in points]
        points = Sort4Points(points)
        res.append(points)
        images.append(image.attrib["name"])
    
    res = np.array(res).astype(np.int32)
    return res, images