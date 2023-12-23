# https://arxiv.org/pdf/1710.11153.pdf
import json
from intervaltree import Interval, IntervalTree

def len_midi_intersections(midi1, midi2):
    res = 0
    for i in range(min(len(midi1), len(midi2))):
        for note1 in midi1[i]:
            for note2 in midi2[i]:
                if note1.data[0] == note2.data[0]:
                    res += 1
                    break
    return res

def len_midi_difference(midi1, midi2):
    res = 0
    for i in range(min(len(midi1), len(midi2))):
        for note1 in midi1[i]:
            for note2 in midi2[i]:
                if note1.data[0] == note2.data[0]:
                    break
            else:
                res += 1
    return res

def F(true, pred):
    true_positives = len_midi_intersections(true, pred)
    false_positives = len_midi_difference(pred, true)
    false_negatives = len_midi_difference(true, pred)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    return 2 * (precision * recall) / (precision + recall)


def RemoveFingerless(notes):
    res = IntervalTree()
    for note in notes:
        if note.data[2] is not None:
            res.add(note)
    return res

def ShiftIntervals(notes, shift):
    res = IntervalTree()
    for note in notes:
        res.add(Interval(note.begin + shift, note.end + shift, note.data))
    return res


if __name__ == "__main__":
    # file1 = "data/1_intermediate/fingers/scarlatti.json"
    # file2 = "data/1_intermediate/transcribed_midi/scarlatti.json"

    file1 = "recording/rec3_fingers_truth.json"
    file2 = "recording/rec3_estimate.json"
    file3 = "recording/rec3_fingers_estimate.json"

    with open(file1, "r") as f:
        data1 = IntervalTree().from_tuples(json.load(f))
        data1 = ShiftIntervals(data1, 2)
    with open(file2, "r") as f:
        data2 = IntervalTree().from_tuples(json.load(f))
    with open(file3, "r") as f:
        data3 = IntervalTree().from_tuples(json.load(f))
        data3 = RemoveFingerless(data3)

    print(F(data1, data2))
    print(F(data1, data3))