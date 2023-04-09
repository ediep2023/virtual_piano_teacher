import sys, os
from pydub import AudioSegment
from scipy.io import wavfile
import noisereduce as nr

import librosa
import librosa.onset
import numpy as np

CLASS_DICT = {
    '62': 0,
    '66': 1,
    '67': 2,
    '69': 3,
    '71': 4,
    '72': 5,
    '74': 6,
    '76': 7,
    '78': 8,
    '79': 9,
    '66, 69, 74': 10,
    '69, 74': 11,
    '62, 66, 69': 12,
    '67, 71': 13,
    '69, 72': 14,
    '71, 74': 15,
    '72, 76': 16,
    '66, 69': 17,
    '62, 67, 71': 18,
    '71, 74, 79': 19,
    '0': 1000
}

def get_onset_time(full_filename):
    '''
    :param full_filename: full directory of original audio
    :method: reduce noise and trim original audio file to find note onset
    :return: onset times, new audio duration, preprocessed audio filename
    '''
    rate, data = wavfile.read(full_filename)
    # perform noise reduction
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    new_full_filename = full_filename[:-4]+"_noise_filtered.wav"
    wavfile.write(new_full_filename, rate, reduced_noise)

    y, sr = librosa.load(new_full_filename)
    signal, index = librosa.effects.trim(y)
    aud_duration = librosa.get_duration(y[:index[1]])
    o_env = librosa.onset.onset_strength(y[:index[1]], sr=sr, hop_length=512)

    times = librosa.times_like(o_env)
    onset_raw = librosa.onset.onset_detect(onset_envelope=o_env, backtrack=False)
    onset_bt = librosa.onset.onset_backtrack(onset_raw, o_env)

    return times[onset_bt], aud_duration, new_full_filename


def write_onset_file(output_file, onset_times, aud_duration):
    '''
    :param output_file: full directory for output file
    :param onset_times: a list of onset times
    :param aud_duration: audio duration
    :method: create a text file (output_file) with each note's onset time. assumes no note overlaps where offset time
    :        is 10 milliseconds before next onset.
    '''
    f = open(output_file, 'w')
    for i in range(len(onset_times)-1):
        onset_time = round(float(onset_times[i]), 2)
        next_onset_time = round(float(onset_times[i+1]), 2)
        offset_time = next_onset_time - 0.01
        if offset_time > onset_time:
            f.write('%0.2f\t%0.2f\n'%(onset_time, offset_time))
    last_onset_time = round(float(onset_times[-1]), 2)

    if aud_duration>last_onset_time:
        f.write('%0.2f\t%0.2f\n' % (last_onset_time, aud_duration))
    f.close()

def get_audio(input_audio, output_file, onset_time, offset_time):
    '''
    :param input_audio: noise reduced and trimmed audio
    :param output_file: full output path of chopped audio files
    :param onset_time: detected onset time
    :param offset_time: assumed offset time
    :method: chops preprocessed audio by onset time into singular note files
    :return: chopped up audio files in full directory
    '''
    print(input_audio, onset_time, offset_time)
    onset_time = onset_time * 1000 #Works in milliseconds
    offset_time = offset_time * 1000

    new_offset_time = onset_time + 1000
    if new_offset_time <= offset_time:
        offset_time = new_offset_time

    new_audio = input_audio[onset_time:offset_time]
    new_audio.export(output_file, format="wav")
    return new_audio

def read_ref_file(txt_file):
    '''
    :param txt_file: an input txt file containing each note's midi within a song
    :method: reads input file to identify the MIDIs of a given song in sequencial order
    :return: midi list
    '''
    f = open(txt_file,'r')
    lines = f.readlines()
    f.close()
    midi_list = []
    for line in lines:
        line = line.strip()
        midi_list.append(line)
    return midi_list

def read_ref_txt(wfile):
    '''
    :param wfile: wavfile
    :method: search for specific ref midi text file based on the filename of the wavfile
    :return: midi list of wfile
    '''
    JINGLE_ONE_REF_FILE='/home/parallels/PycharmProjects/PianoModelBuilder/data/Minuet/ScienceFair2023/NoiseFiltered/MIDI/JingleOne.txt'
    JINGLE_TWO_REF_FILE='/home/parallels/PycharmProjects/PianoModelBuilder/data/Minuet/ScienceFair2023/NoiseFiltered/MIDI/JingleTwo.txt'
    JINGLE_THREE_REF_FILE = '/home/parallels/PycharmProjects/PianoModelBuilder/data/Minuet/ScienceFair2023/NoiseFiltered/MIDI/JingleThree.txt'
    MINUET_REF_FILE = '/home/parallels/PycharmProjects/PianoModelBuilder/data/Minuet/ScienceFair2023/NoiseFiltered/MIDI/Minuet.txt'
    midilist = []
    if "JingleOne" in wfile:
        midilist = read_ref_file(JINGLE_ONE_REF_FILE)
    elif "JingleTwo" in wfile:
        midilist = read_ref_file(JINGLE_TWO_REF_FILE)
    elif "JingleThree" in wfile:
        midilist = read_ref_file(JINGLE_THREE_REF_FILE)
    elif "Minuet" in wfile:
        midilist = read_ref_file(MINUET_REF_FILE)
    return midilist

def generate_audios(input_file, output_folderpath):
    '''
    :param input_file: preprocessed audio
    :param output_folderpath: full directory of chopped up wav files
    :method: reads onset ref file and loads original audio. splits ref file to create dictionary where key = note number
    :        and value is dicationary containing key value pairs of onset, offset, label, and midi.
    '''
    input_audio = AudioSegment.from_file(input_file)
    audio_len = input_audio.duration_seconds
    ref_file = input_file[:-4] + ".txt"
    fh = open(ref_file, 'r')
    lines = fh.readlines()
    fh.close()
    ref_dict = {}
    idx = 1
    print (input_file)

    midilist = read_ref_txt(input_file)

    for line in lines:
        line=line.strip()
        items = line.split('\t')
        onset = items[0]
        offset = items[1]
        midi = midilist[idx-1]
        label = CLASS_DICT[midi]
        ref_dict[idx] = {'onset': round(float(onset), 2), 'offset':  round(float(offset), 2), 'label': label, 'midi':midi}
        idx += 1

    basename = os.path.basename(input_file)
    for key in ref_dict.keys():
        onset=ref_dict[key]['onset']
        offset=ref_dict[key]['offset']
        midi=ref_dict[key]['midi'].strip()
        label=ref_dict[key]['label']
        out_filename = basename[:-4] + "_" + str(key) + "-999-" + str(label) + "-" + str(label) + ".wav"
        output_file = os.path.join(output_folderpath, out_filename)
        print(onset, offset, midi, label)
        print(input_file)
        get_audio(input_audio, output_file, onset, offset)

if __name__ == '__main__':
    '''
    lists full file directory for wavfiles. puts chopped up files into train folder (input_folder) and valid folder (input_folder2)
    '''

    input_folder = "/home/parallels/PycharmProjects/PianoModelBuilder/data/Minuet/ScienceFair2023/NoiseFiltered/Minuet/ReferenceAudio"
    output_folder = "/home/parallels/PycharmProjects/PianoModelBuilder/data/Minuet/ScienceFair2023/NoiseFiltered/Minuet/train"

    input_folder2 = "/home/parallels/PycharmProjects/PianoModelBuilder/data/Minuet/ScienceFair2023/NoiseFiltered/Minuet/TeacherAudio"
    output_folder2 = "/home/parallels/PycharmProjects/PianoModelBuilder/data/Minuet/ScienceFair2023/NoiseFiltered/Minuet/valid"

    for folder in [input_folder,input_folder2]:
        wavfiles = os.listdir(folder)
        for wfile in wavfiles:
            if wfile.endswith('.wav') and (not ("noise_filtered" in wfile)):
                full_filename = os.path.join(folder, wfile)
                onset_times, aud_duration, new_full_filename = get_onset_time(full_filename)
                output_filepath = new_full_filename[:-4] + ".txt"
                write_onset_file(output_filepath, onset_times, aud_duration)
                if folder == input_folder:
                    generate_audios(new_full_filename, output_folder)
                elif folder == input_folder2:
                    generate_audios(new_full_filename, output_folder2)




