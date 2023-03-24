from music21 import *
from music21.instrument import Instrument
import os


def generate_music_sheet(folder, title, summary, time_signature = '4/4'):
    filename_list=os.listdir(folder)
    for filename in filename_list:
        if filename.endswith('music21.txt'):
            filename_path = os.path.join(folder, filename)
            make_musicscore(filename_path, title, summary, time_signature)


def make_musicscore(filename, title, summary, time_signature):
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()

    p1 = stream.Part([clef.TrebleClef(), meter.TimeSignature(time_signature)])
    s = stream.Score(p1)
    s.insert(0, metadata.Metadata())
    s.metadata.title = title
    s.metadata.composer = summary
    sl_list = []
    found = False
    for line in lines:
        items = line.strip().split('\t')
        idx = items[0]
        midi_str = items[3]
        if ',' in midi_str:
            midi_list = midi_str.split(',')
            tmp_list = []
            for midi in midi_list:
                tmp_note = note.Note(midi=int(midi))
                tmp_list.append(tmp_note)
            mynote = chord.Chord(tmp_list)
        else:
            mynote = note.Note(midi=int(midi_str))
        note_duration = items[4]
        color = items[5]
        articulation = items[6]
        if len(items)>=8:
            comment = items[7]
            mynote.addLyric(comment)

        if note_duration == "quarter":
            mynote.duration = duration.Duration(1.0)
        elif note_duration == "eighth":
            mynote.duration = duration.Duration(0.5)
        elif note_duration == "triplet_eighth":
            mynote.duration = duration.Duration(0.333333333)
        elif note_duration == "dotted_half":
            mynote.duration = duration.Duration(3.0)
        elif note_duration == "half":
            mynote.duration = duration.Duration(2.0)
        mynote.style.color = color
        if articulation in ['Staccato']:
            mynote.articulations = [articulations.Staccato()]
        p1.append(mynote)
        if articulation == 'Legato':
            sl_list.append(mynote)
            found = True
        else:
            if found == True:
                sl_list.append(mynote)
                p1.append(spanner.Slur(sl_list))
                found = False
                sl_list = []
    s.show('musicxml.pdf', fp=filename[:-11]+"score.pdf")

if __name__ == "__main__":

    JINGLE1_PREDICTION_FOLDER = '/home/parallels/PycharmProjects/VirtualPianoTeacher2023/data/StudentMultiLabelPrediction'
    title = "Jingle 1"
    summary = 'Emily Diep\nred - wrong note, blue - extra note, black - correct note, orange - wrong tempo'
    generate_music_sheet(JINGLE1_PREDICTION_FOLDER, title, summary)

