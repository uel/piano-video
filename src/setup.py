import os
from urllib.request import urlretrieve

mediapipe_hands = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
piano_transcription = "https://zenodo.org/record/4034264/files/CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"

def setup_check(data_path):
    os.makedirs(f"{data_path}/videos", exist_ok=True)
    os.makedirs(f"{data_path}/audio", exist_ok=True)
    os.makedirs(f"{data_path}/background", exist_ok=True)
    os.makedirs(f"{data_path}/transcribed_midi", exist_ok=True)
    os.makedirs(f"{data_path}/keys", exist_ok=True)
    os.makedirs(f"{data_path}/sections", exist_ok=True)
    os.makedirs(f"{data_path}/fingers", exist_ok=True)
    os.makedirs(f"{data_path}/hand_landmarks", exist_ok=True)

    os.makedirs("models", exist_ok=True)

    if not os.path.exists("models/piano_transcription.pth"):
        print('Downloading piano transcription model...')
        urlretrieve(piano_transcription, "models/piano_transcription.pth")

    if not os.path.exists("models/hand_landmarker.task"):
        print('Downloading hand landmarker model...')
        urlretrieve(mediapipe_hands, "models/hand_landmarker.task")
    