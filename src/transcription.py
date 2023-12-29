from piano_transcription_inference import PianoTranscription, sample_rate, load_audio

def transcribe_piano(input_audio, output_midi=None):
    audio, _ = load_audio(input_audio, sr=sample_rate, mono=True)

    transcriptor = PianoTranscription(device='cpu', checkpoint_path='models/piano_transcription.pth')    # 'cuda' | 'cpu'

    transcribed_dict = transcriptor.transcribe(audio, output_midi)

    return transcribed_dict["est_note_events"]