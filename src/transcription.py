from piano_transcription_inference import PianoTranscription, sample_rate, load_audio
from intervaltree import IntervalTree, Interval

def transcribe_piano(input_audio, output_midi=None):
    audio, _ = load_audio(input_audio, sr=sample_rate, mono=True)

    transcriptor = PianoTranscription(device='cpu', checkpoint_path='models/piano_transcription.pth')    # 'cuda' | 'cpu'

    transcribed_dict = transcriptor.transcribe(audio, output_midi)

    return transcribed_dict["est_note_events"]

def dict_to_intervals(note_events, fps):
    intervals = []
    for note in note_events:
        onset = round(note["onset_time"]*fps)
        offset = round(note["offset_time"]*fps)
        if onset == offset: continue
        intervals.append([onset, offset, [note["midi_note"], note["velocity"]]])
    return IntervalTree.from_tuples(intervals)

def midi_to_intervals(file_path, fps=30):
    import mido

    intervals = IntervalTree()
    last_note_on = {}
    
    time2frame = lambda time: round( time *  fps )
    time = 0
        
    # Parse MIDI file
    with mido.MidiFile(file_path) as midi_file:
        for message in midi_file:
            time += message.time
            if message.type == 'note_on' or message.type == 'note_off':
                if message.velocity == 0 or message.type == 'note_off':
                    if message.note in last_note_on:
                        note_start_time, velocity = last_note_on.pop(message.note)
                        if time2frame(note_start_time) == time2frame(time): continue
                        intervals.addi(time2frame(note_start_time), time2frame(time), [message.note, velocity])
                else:
                    last_note_on[message.note] = (time, message.velocity)

    return intervals