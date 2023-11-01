from piano_transcription_inference import PianoTranscription, sample_rate, load_audio
import os
from urllib.request import urlretrieve

def transcribe_piano(input_audio, output_midi=None):
    audio, _ = load_audio(input_audio, sr=sample_rate, mono=True)

    if not os.path.exists("models/piano_transcription/note_F1=0.9677_pedal_F1=0.9186.pth"):
        os.makedirs("models/piano_transcription", exist_ok=True)
        print('Downloading piano transcription model...')
        urlretrieve("https://zenodo.org/record/4034264/files/CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1", "models/piano_transcription/note_F1=0.9677_pedal_F1=0.9186.pth")

    # Transcriptor
    transcriptor = PianoTranscription(device='cpu', checkpoint_path='models/piano_transcription/note_F1=0.9677_pedal_F1=0.9186.pth')    # 'cuda' | 'cpu'

    # Transcribe and write out to MIDI file
    transcribed_dict = transcriptor.transcribe(audio, output_midi)

    return transcribed_dict["est_note_events"]