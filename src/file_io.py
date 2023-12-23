import struct
import math
import numpy as np

def write_landmarks(filename, data):
    with open(filename, 'wb') as file:
        for hands in data:
            frame_id, left, right = hands
            if left is None: left = [math.nan]*63
            else: left = np.array(left[:21]).reshape(-1).tolist()

            if right is None: right = [math.nan]*63
            else: right = np.array(right[:21]).reshape(-1).tolist()

            packed_data = struct.pack('I63f63f', frame_id, *left, *right)
            file.write(packed_data)

def read_landmarks(filename):
    result = []
    size = struct.calcsize('I63f63f')
    
    with open(filename, 'rb') as file:
        while True:
            data = file.read(size)
            if not data: break
            
            unpacked_data = struct.unpack('I63f63f', data)
            
            frame_id, left, right = unpacked_data[0], unpacked_data[1:64], unpacked_data[64:]
            
            if math.isnan(left[0]): left = None
            else: left = np.array(left).reshape(-1, 3).tolist()

            if math.isnan(right[0]): right = None
            else: right = np.array(right).reshape(-1, 3).tolist()
            
            result.append([frame_id, left, right])
    
    return result
  
def midi_to_interval_tree(midi_file_path, resolution=1./30):
    import mido
    from intervaltree import Interval, IntervalTree

    # Load the MIDI file
    mid = mido.MidiFile(midi_file_path)

    # Create an empty interval tree
    tree = IntervalTree()

    # Iterate over all tracks in the MIDI file
    for track in mid.tracks:
        # Initialize time counter and tempo
        time_counter = 0
        tempo = mido.bpm2tempo(120)  # Default tempo is 120 BPM

        # Initialize a dictionary to keep track of note_on events
        note_on_events = {}

        # Iterate over all messages in the track
        for msg in track:
            # Convert time to seconds
            time_in_seconds = mido.tick2second(msg.time, mid.ticks_per_beat, tempo)
            # Update time counter
            time_counter += time_in_seconds
            
            # If the message is a set_tempo message, update the tempo
            if msg.type == 'set_tempo':
                tempo = msg.tempo
            # If the message represents a note
            elif msg.type == 'note_on' or msg.type == 'note_off':
                # If the message is a note_on, add it to the note_on_events dictionary
                if msg.type == 'note_on':
                    start_i = int((time_counter)/resolution)
                    note_on_events[msg.note] = (start_i, msg.velocity)
                # If the message is a note_off, create an interval and add it to the tree
                elif msg.type == 'note_off':
                    note_on_event = note_on_events.pop(msg.note, None)
                    if note_on_event is not None:
                        start_i, velocity = note_on_event
                        end_i = int((time_counter + time_in_seconds)/resolution)
                        tree[start_i:end_i] = (msg.note, velocity)


    return tree

if __name__ == "__main__":
    # Example usage
    midi_file_path = 'recording/rec3.mid'
    result_tree = midi_to_interval_tree(midi_file_path)
    import json
    with open('recording/rec3.json', 'w') as f:
        json.dump(list(sorted(result_tree)), f)
