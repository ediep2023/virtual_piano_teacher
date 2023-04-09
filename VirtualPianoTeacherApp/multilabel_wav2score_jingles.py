import librosa, os
import matplotlib.pyplot as plt
import librosa.display
import librosa.onset
import numpy as np
from pydub import AudioSegment
from scipy.io import wavfile
import noisereduce as nr

import torch
import torchaudio
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torchvision import transforms
from torchvision.utils import save_image
import torch.optim as optim


from torchsummary import summary
from MultiLabelDenseNet import DenseNet as CNNNetwork

from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd

import math, statistics

ORIGINAL_WAV_FOLDER = '/home/parallels/PycharmProjects/VirtualPianoTeacher2023/data'
MINI_OUTPUT_FOLDER = "/home/parallels/PycharmProjects/VirtualPianoTeacher2023/data/MiniWav"
PREDICTION_FOLDER = '/home/parallels/PycharmProjects/VirtualPianoTeacher2023/data/StudentMultiLabelPrediction/'
FINAL_MODEL = "/home/parallels/PycharmProjects/VirtualPianoTeacher2023/data/NoiseFilteredJingle1_one_sec_multilabel_densenet_epoch5.pth"
DURATION_REF_FILE = "/home/parallels/PycharmProjects/VirtualPianoTeacher2023/data/MIDI/JingleOne_duration.txt"

BATCH_SIZE = 1
SAMPLE_RATE = 16000
NUM_SAMPLES = 16000
HOP_LENGTH = 512

INDEX2MIDI = {
    0: '62',
    1: '66',
    2: '67',
    3: '69',
    4: '71',
    5: '72',
    6: '74',
    7: '76',
    8: '78',
    9: '79'
}

CLASS_DICT = {
    62: '1,0,0,0,0,0,0,0,0,0',
    66: '0,1,0,0,0,0,0,0,0,0',
    67: '0,0,1,0,0,0,0,0,0,0',
    69: '0,0,0,1,0,0,0,0,0,0',
    71: '0,0,0,0,1,0,0,0,0,0',
    72: '0,0,0,0,0,1,0,0,0,0',
    74: '0,0,0,0,0,0,1,0,0,0',
    76: '0,0,0,0,0,0,0,1,0,0',
    78: '0,0,0,0,0,0,0,0,1,0',
    79: '0,0,0,0,0,0,0,0,0,1',
    0: '0,0,0,0,0,0,0,0,0,0'
}

NOTE_DICT = {
    0: "xxx ",
    62: "D-4 ",
    66: "F#4 ",
    67: "G-4 ",
    69: "A-4 ",
    71: "B-4 ",
    72: "C-5 ",
    74: "D-5 ",
    76: "E-5 ",
    78: "F#5 ",
    79: "G-5 ",

}

REF_DICT = {
    1: 67,
    2: 71,
    3: 74,
    4: 79,
    5: 69,
    6: 78,
    7: 79,
    8: 67,
    9: 67,
    10: 67,
    11: 71,
    12: 74,
    13: 79,
    14: 69,
    15: 78,
    16: 79,
    17: 67,
    18: 67,
    19: 76,
    20: 76,
    21: 76,
    22: 79,
    23: 74,
    24: 74,
    25: 74,
    26: 79,
    27: 72,
    28: 74,
    29: 72,
    30: 71,
    31: 72,
    32: 69,
    33: 67,
    34: 71,
    35: 74,
    36: 79,
    37: 69,
    38: 78,
    39: 79,
    40: 67,
    41: 67,
    42: 67,
    43: 71,
    44: 74,
    45: 79,
    46: 69,
    47: 78,
    48: 79,
    49: 67,
    50: 67,
    51: 76,
    52: 74,
    53: 72,
    54: 71,
    55: 69,
    56: 74,
    57: 72,
    58: 71,
    59: 69,
    60: 67,
    61: 69,
    62: 71,
    63: 72,
    64: 62,
    65: 66,
    66: 67
}

JINGLE_1_DICT = {
    1: '62',
    2: '66',
    3: '67',
    4: '69',
    5: '67',
    6: '66',
    7: '66,69,74'
}

JINGLE_2_DICT = {
    1: '62',
    2: '66',
    3: '67',
    4: '69',
    5: '74',
    6: '78',
    7: '69,74',
    8: '69',
    9: '67',
    10: '66',
    11: '67',
    12: '69',
    13: '67',
    14: '62,66,69'
}

JINGLE_3_DICT = {
    1: '67,71',
    2: '69,72',
    3: '71,74',
    4: '71,74',
    5: '72,76',
    6: '71,74',
    7: '69,72',
    8: '67,71',
    9: '66,69',
    10: '67',
    11: '66,69',
    12: '67,71',
    13: '66,69',
    14: '62,67,71',
    15: '66,69',
    16: '67',
    17: '67',
    18: '71',
    19: '74',
    20: '71,74,79'
}

class SoundSet(Dataset):
    def __init__(self, audio_file, transformation, target_sample_rate, num_samples, device):
        '''
        :param audio_file: input audio
        :param transformation: function to transform audio into mel spectrogram
        :param target_sample_rate: number of samples * time
        :param num_samples: sample count
        :param device: CPU
        :method: initalize SoundDataset with input value
        '''
        self.audio_file = audio_file
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        '''
        :return: 1 test sample every time model is tested
        '''
        return 1

    def __getitem__(self, index):
        '''
        :param index: a note's index within a song
        :method: adjust indesed chopped audio to one second and transform into mel spectorgram
        :return: digital signal of the indexed chopped audio, and its 10 corresponding labels.
        '''
        signal, sr = torchaudio.load(self.audio_file)
        signal = signal.to(self.device)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        return signal

    def _cut_if_necessary(self, signal):
        '''
        :param signal: chopped up note audio
        :method: trim note audio if greater than one second
        :return: adjusted signal
        '''
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        '''
        :param signal: chopped up note audio
        :method: padding 0s to audio if less than one second
        :return: adjusted signal
        '''
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

def create_data_loader(dataset, batch_size, shuffle):
    '''
    :param dataset: dataset input
    :param batch_size: number of samples in a batch
    :param shuffle: shuffle samples
    :method: create data loader
    :return: data loader
    '''
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def test_model(test_dl, model, device):
    '''
    :param test_dl: test data loader
    :param model: CNN model
    :param device: CPU
    :method: Use CNN model
    :return: 10 prediction labels
    '''
    all_labels = []
    for i, (data) in enumerate(test_dl):
        ## Forward Pass
        data = data.to(device=device)
        outputs = model(data)
        # get all the labels
        for out in outputs:
            if out >= 0.5:
                all_labels.append('1')
            else:
                all_labels.append('0')
    return all_labels

def get_onset_time(full_filename):
    '''
    :param full_filename:
    :method: get the onset time of the student's preprocessed audio
    :return: onset times, tempo, sample rate, audio duration, preprocessed audio
    '''
    rate, data = wavfile.read(full_filename)
    # perform noise reduction
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    new_full_filename = full_filename[:-4] + "_noise_filtered.wav"
    wavfile.write(new_full_filename, rate, reduced_noise)

    y, sr = librosa.load(new_full_filename)
    signal, index = librosa.effects.trim(y)

    aud_duration = librosa.get_duration(y=y[:index[1]])
    o_env = librosa.onset.onset_strength(y=y[:index[1]], sr=sr, hop_length=512)
    tempo = librosa.beat.tempo(onset_envelope=o_env, sr=sr)

    times = librosa.times_like(o_env)
    onset_raw = librosa.onset.onset_detect(onset_envelope=o_env, backtrack=False)
    onset_bt = librosa.onset.onset_backtrack(onset_raw, o_env)

    return o_env, times[onset_bt], tempo, sr, aud_duration, new_full_filename

def write_onset_file(output_file, onset_times, aud_duration):
    '''
    :param output_file: new_full_filename
    :param onset_times: each note's onset times
    :param aud_duration: duration of input audio
    :method: create dictionary of each onset times
    :return: dictionary of onset times
    '''
    timing_dict = {}
    f = open(output_file, 'w')
    for i in range(len(onset_times)-1):
        onset_time = round(float(onset_times[i]), 2)
        next_onset_time = round(float(onset_times[i + 1]), 2)
        offset_time = next_onset_time - 0.01
        timing_dict[i+1]={'onset': onset_time, 'offset': offset_time}
        if offset_time > onset_time:
            f.write('%0.2f\t%0.2f\n' % (onset_time, offset_time))
    last_onset_time = round(float(onset_times[-1]), 2)

    timing_dict[len(onset_times)] = {'onset': last_onset_time, 'offset': aud_duration}
    if aud_duration > last_onset_time:
        f.write('%0.2f\t%0.2f\n' % (last_onset_time, aud_duration))
    f.close()
    return timing_dict


def get_audio(input_audio, output_file, onset_time, offset_time):
    '''
    :param input_audio: noise reduced and trimmed audio
    :param output_file: full output path of chopped audio files
    :param onset_time: detected onset time
    :param offset_time: assumed offset time
    :method: chops preprocessed audio by onset time into singular note files
    :return: chopped up audio files in full directory
    '''
    onset_time = onset_time * 1000 #Works in milliseconds
    offset_time = offset_time * 1000

    new_offset_time = onset_time + 1000
    if new_offset_time <= offset_time:
        offset_time = new_offset_time

    new_audio = input_audio[onset_time:offset_time]
    new_audio.export(output_file, format="wav")
    return new_audio

def generate_audios(input_file, timing_file, output_folderpath, wav_id):
    '''
    :param input_file: student audio
    :param timing_file: file with each note onset
    :param output_folderpath: output folder
    :param wav_id: note index
    :method: generate chopped up wav audio of student audio
    '''
    input_audio = AudioSegment.from_file(input_file)
    audio_len = input_audio.duration_seconds

    fh = open(timing_file, 'r')
    lines = fh.readlines()
    fh.close()

    ref_dict = {}
    idx = 1
    print (input_file)
    for line in lines:
        line=line.strip()
        items = line.split('\t')
        onset = items[0]
        offset = items[1]
        ref_dict[idx] = {'onset': round(float(onset), 2), 'offset':  round(float(offset), 2)}
        idx += 1

    basename = os.path.basename(input_file)
    for key in ref_dict.keys():
        onset=ref_dict[key]['onset']
        offset=ref_dict[key]['offset']
        out_filename = basename[:-4] + "_" + str(key) + "-999-000" + "-" + str(wav_id) + ".wav"
        output_file = os.path.join(output_folderpath, out_filename)
        get_audio(input_audio, output_file, onset, offset)

def get_ref_sequence(filename):
    '''
    :param filename: input audio filename
    :method: look for note dict ref list based on song
    :return: sequence list
    '''
    seq_list=[]
    if "Minuet" in filename:
        for i in range(len(REF_DICT)):
            seq_list.append(NOTE_DICT[REF_DICT[i+1]])
    elif "JingleOne" in filename:
        for i in range (len(JINGLE_1_DICT)):
            if ',' in JINGLE_1_DICT[i + 1]:
                jingle1_values = JINGLE_1_DICT[i + 1].split(',')
                note_value = []
                for mynote in jingle1_values:
                    note_value.append(NOTE_DICT[int(mynote)].strip())
                list_value1 = ','.join(note_value) + ' '
                seq_list.append(list_value1)
            else:
                seq_list.append(NOTE_DICT[int(JINGLE_1_DICT[i + 1])])
    elif "JingleTwo" in filename:
        for i in range (len(JINGLE_2_DICT)):
            if ',' in JINGLE_2_DICT[i + 1]:
                jingle2_values = JINGLE_2_DICT[i + 1].split(',')
                note_value = []
                for mynote in jingle2_values:
                    note_value.append(NOTE_DICT[int(mynote)].strip())
                list_value2 = ','.join(note_value) + ' '
                seq_list.append(list_value2)
            else:
                seq_list.append(NOTE_DICT[int(JINGLE_2_DICT[i + 1])])

    elif "JingleThree" in filename:
        for i in range (len(JINGLE_3_DICT)):
            if ',' in JINGLE_3_DICT[i + 1]:
                jingle3_values = JINGLE_3_DICT[i + 1].split(',')
                note_value = []
                for mynote in jingle3_values:
                    note_value.append(NOTE_DICT[int(mynote)].strip())
                list_value3 = ','.join(note_value) + ' '
                seq_list.append(list_value3)
            else:
                seq_list.append(NOTE_DICT[int(JINGLE_3_DICT[i + 1])])

    return seq_list

def get_predicted_sequence(output_filename):
    '''
    :param output_filename: the full path of prediction file
    :                       - first column=.wav filename, second column=note index, third column=midi number
    :method: read prediction file and search NOTE_DICT to convert midi to note name
    :return: list of the predicted note names according to the note index
    '''
    f = open(output_filename, 'r')
    lines = f.readlines()
    f.close()
    seq_list = []
    for i in range(len(lines)):
        seq_list.append("")

    for line in lines:
        items = line.strip().split('\t')
        position = int(items[1])
        midi = items[2]
        print(midi)
        if ',' in midi:
            note_name_list = []
            note_value_list = midi.split(',')
            for note_value in note_value_list:
                note_name = NOTE_DICT[int(note_value)].strip()
                note_name_list.append(note_name)
            midi = ','.join(note_name_list) + ' '
            idx = position - 1
            seq_list[idx] = midi
        else:
            midi = int(midi)
            idx = position - 1
            seq_list[idx] = NOTE_DICT[midi]
    return seq_list

def score(ref, test):
    '''
    :param ref: reference notes
    :param test: predicted notes
    :method: define scoring system: returns +1 for match,
    '''
    if ref==test:
        return 1
    else:
        return -1

def align(ref_list, test_list, gap_penalty):
    '''
    :param ref_list: reference notes list
    :param test_list: predicted notes list
    :param gap_penalty: penalty when gap
    :method: Needleman-Wunsch alignment algorithm - build scoring matrix and tracks optimal alignment in pointer matrix
    :return: scoring matrix and pointer matrix
    '''
    #init matrix
    score_matrix = []
    for j in range(len(test_list) + 1):
        tmp = []
        for i in range(len(ref_list) + 1):
            tmp.append(0)
        score_matrix.append(tmp)

    pointer_matrix = []
    for j in range(len(test_list) + 1):
        tmp = []
        for i in range(len(ref_list) + 1):
            tmp.append("")
        pointer_matrix.append(tmp)
    #set value for first row and first column
    for i in range(len(ref_list)+1):
        score_matrix[0][i] = i * gap_penalty

    for j in range(len(test_list)+1):
        score_matrix[j][0]=j * gap_penalty

    for r in range(len(test_list) + 1)[1:]:
        for c in range(len(ref_list) + 1)[1:]:
            f = [score_matrix[r-1][c-1] + score(test_list[r-1], ref_list[c-1]),
                 score_matrix[r - 1][c] + gap_penalty,  # gap
                 score_matrix[r][c-1] + gap_penalty]  # gap
            score_matrix[r][c] = max(f)
            pointer_matrix[r][c] = f.index(max(f))
    return score_matrix, pointer_matrix

def get_alignment(test_list, ref_list, score_matrix, pointer_matrix):
    '''
    :param test_list: predicted notes list
    :param ref_list: expected notes list
    :param score_matrix: stores scoring system
    :param pointer_matrix: stores optimal alignment for all the scores in the scoring matrix
    :method: backtracking using pointer matrix to find optimal alignment
    :return: alignment of predicted and expected notes in string seperated by space
    '''
    align1 = ""
    align2 = ""
    r = len(test_list)
    c = len(ref_list)
    while True:
        if r == 0 or c == 0:
            if r==0 and c !=0:
                while(c>0):
                    align1 = ref_list[c - 1] + align1
                    align2 = "--- " + align2
                    c = c - 1
            elif r!=0 and c==0:
                while(r > 0):
                    align2 = test_list[r-1] + align2
                    align1 = "--- " + align1
                    r=r-1
            break
        if pointer_matrix[r][c] == 0:
            align1 = ref_list[c-1] + align1
            align2 = test_list[r-1] + align2
            r = r - 1
            c = c - 1
        elif pointer_matrix[r][c] == 1:
            align1 = "--- " + align1
            align2 = test_list[r - 1] + align2
            r = r - 1
        elif pointer_matrix[r][c] == 2:
            align1 = ref_list[c - 1] + align1
            align2 = "--- " + align2
            c = c - 1

    align1_list=align1.split(' ')
    align2_list=align2.split(' ')
    print(align1)
    print(align2)
    for i in range(len(align2_list)):
        if len(align1_list[i]) > len(align2_list[i]):
            space_len2= len(align1_list[i]) - len(align2_list[i])
            for j in range(space_len2):
                align2_list[i] = align2_list[i] + '*'
        elif len(align1_list[i]) < len(align2_list[i]):
            space_len1 = len(align2_list[i]) - len(align1_list[i])
            for j in range(space_len1):
                align1_list[i] = align1_list[i] + '*'

    align1 = ' '.join(align1_list)
    align2 = ' '.join(align2_list)
    return align1, align2

def print_alignment(align1, align2, output_file):
    '''
    :param align1: reference alignment
    :param align2: predicted alignment
    :param output_file: alignment output file
    :method: write alignment file with a maximum of 40 character per line
    '''
    f = open(output_file, "w")
    total_len = len(align1)
    iterations = math.floor(total_len / 40)
    if total_len % 40 > 0:
        iterations += 1
    for i in range(iterations):
        f.write("%s\n%s\n\n"%(align1[40*i:40*(i+1)], align2[40*i:40*(i+1)]))
    f.close()

def calc_expected_note_duration(tempo):
    '''
    :param tempo: song tempo
    :method: create dictionary of expected note duration
    :return: expected note dictionary
    '''
    expected_duration_dict = {
        "quarter": 0,
        "eighth": 0,
        "triplet_eighth": 0,
        "dotted_half": 0,
        "half": 0
    }
    sec_per_beat=60.00/tempo
    expected_duration_dict["quarter"] = sec_per_beat
    expected_duration_dict["eighth"] = (sec_per_beat)/2
    expected_duration_dict["triplet_eighth"] = (sec_per_beat) / 3
    expected_duration_dict["dotted_half"] = 3 * (sec_per_beat)
    expected_duration_dict["half"] = 2 * (sec_per_beat)
    return expected_duration_dict

def calc_note_type(idx, prediction_dict, expected_duration_dict):
    '''
    :param idx: note number
    :param prediction_dict: note duration prediction
    :param expected_duration_dict: expected note duration
    :method: calculate note duration of note within 100 milliseconds of expected note duration
    :return: predicted note duration
    '''
    note_type = None
    color = "black"
    comment = ''
    onset = prediction_dict[idx]['onset']
    offset = prediction_dict[idx]['offset']
    observed_duration = round(offset-onset,2)

    if (observed_duration >= round(float(expected_duration_dict["dotted_half"]) - 0.1, 2)) and (observed_duration <= round(float(expected_duration_dict["dotted_half"]) + 0.1, 2)):
        note_type= 'dotted_half'
    elif (observed_duration >= round(float(expected_duration_dict["quarter"]) - 0.1, 2)) and (observed_duration <= round(float(expected_duration_dict["quarter"]) + 0.1, 2)):
        note_type= 'quarter'
    elif (observed_duration >= round(float(expected_duration_dict["half"])-0.1, 2)) and (observed_duration<= round(float(expected_duration_dict["half"])+ 0.1, 2)):
        note_type= 'half'
    elif (observed_duration >= round(float(expected_duration_dict["eighth"])-0.1,2)) and (observed_duration<=round(float(expected_duration_dict["eighth"]) + 0.1, 2)):
        note_type = 'eighth'
    elif (observed_duration >= round(float(expected_duration_dict["triplet_eighth"]) - 0.1, 2)) and (observed_duration <= round(float(expected_duration_dict["triplet_eighth"] + 0.1), 2)):
        note_type= 'triplet_eighth'

    else:
        min_dist=99999999
        for k in expected_duration_dict.keys():
            dist = abs(expected_duration_dict[k]-observed_duration)
            if dist < min_dist:
                min_dist = dist
                note_type = k
                color = 'orange'
                print(note_type, color, dist, expected_duration_dict[k], observed_duration)
    return note_type, color, comment

def calc_note_type_for_predictions(prediction_dict, expected_duration_dict):
    '''
    :param prediction_dict: dictionary with note index, onset, offset, and midi number
    :param expected_duration_dict: expected note duration
    :method: adds note duration, color and comment to prediction_dict
    :return: prediction dictionary
    '''
    for i in range(len(prediction_dict)):
        print(i)
        note_type, color, comment = calc_note_type(i+1, prediction_dict, expected_duration_dict)
        prediction_dict[i+1]['note_type'] = note_type
        prediction_dict[i + 1]['color'] = color
        prediction_dict[i+1]['comment'] += comment
        print(note_type, color, comment)
    return prediction_dict

def write_prediction_dict(prediction_dict, output_file):
    '''
    :param prediction_dict: predicted  onset, midi, note duration, color, comment
    :param output_file: music 21 text file
    :method: write music21 text file
    :return: music 21 text file
    '''
    f = open(output_file, 'w')
    for i in range(len(prediction_dict)):
        onset = prediction_dict[i+1]['onset']
        offset = prediction_dict[i+1]['offset']
        midi = prediction_dict[i+1]['midi']
        note_type = prediction_dict[i+1]['note_type']
        color = prediction_dict[i+1]['color']
        comment = prediction_dict[i+1]['comment']
        articulation = prediction_dict[i + 1]['articulation']
        f.write('%d\t%0.2f\t%0.2f\t%s\t%s\t%s\t%s\t%s\n'%(i+1, onset, offset, midi, note_type, color, articulation, comment))
    f.close()

def create_color_note(align1_str, align2_str, prediction_dict):
    '''
    :param align1_str: reference string
    :param align2_str: predicted string
    :param prediction_dict: prediction dictionary of predicted  onset, midi, note duration, color, comment
    :method: color notes based on note error - red=wrong note, blue=extra note, orange=wrong duration, text=missing note
    :return: prediction dictionary
    '''
    cur_prediction_dict_idx = 1
    align1 = align1_str.strip().split(' ')
    align2 = align2_str.strip().split(' ')
    print(len(align1), len(align2))
    if len(align1) != len(align2):
        tmp1 = []
        for item in align1:
            if item != '':
                tmp1.append(item)
        align1 = tmp1

        tmp2 = []
        for item in align2:
            if item != '':
                tmp2.append(item)
        align2 = tmp2

    for i in range(len(align1)):
        print(len(align1), i, cur_prediction_dict_idx)
        if align1[i].startswith('---'):
            prediction_dict[cur_prediction_dict_idx]['color']= 'blue'
            cur_prediction_dict_idx+=1
        elif align2[i].startswith('---'):
            prediction_dict[cur_prediction_dict_idx]['comment'] += 'Missing %s'%(align1[i])
        elif align1[i] != align2[i]:
            prediction_dict[cur_prediction_dict_idx]['color']= 'red'
            cur_prediction_dict_idx += 1
        else:
            if prediction_dict[cur_prediction_dict_idx]['color'] == "":
                prediction_dict[cur_prediction_dict_idx]['color'] = 'black'
            cur_prediction_dict_idx += 1
    return prediction_dict

def get_distance_list(x):
    '''
    : DEPRECATED
    '''
    dist_list = []
    for i in range(len(x)-1):
        dist = abs(x[i]-x[i+1])
        dist_list.append(dist)
    return dist_list

def get_percent_above_noise(x):
    '''
        : DEPRECATED
    '''
    noise_threshold = 0.00030
    count = 1
    for i in range(len(x)):
        if x[i] > noise_threshold:
            count += 1
    percent = round(float(count * 100.0 / len(x)), 2)
    return percent

def get_articulation(x):
    '''
    : DEPRECATED
    '''
    articulation = ""
    legato_threshold = 0.00030
    dist_list = get_distance_list(x)
    med = statistics.median(dist_list)
    percent = get_percent_above_noise(x)
    if med > legato_threshold:
        articulation = "legato"
    elif percent > 40:
        articulation = "staccato"
    else:
        articulation = "regular"
    return articulation, med, percent

def wav2score():
    '''
    :method: convert audio into music score with corrected errors
    '''
    device = "cpu"
    # setup melspectrogram
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=256, n_mels=128)

    wavfiles = os.listdir(ORIGINAL_WAV_FOLDER)
    wav_id = 1
    duration_result_dict = {}
    for wfile in wavfiles:
        #preprocess audio
        if wfile.endswith('.wav') and (not ("noise_filtered" in wfile)):
            full_filename  = os.path.join(ORIGINAL_WAV_FOLDER, wfile)
            #ONSET DETECTION
            o_env, onset_times, tempo, sr, aud_duration, new_full_filename = get_onset_time(full_filename)
            timing_file = os.path.join(PREDICTION_FOLDER, wfile[:-4]+"_timing.txt")
            timing_dict = write_onset_file(timing_file, onset_times, aud_duration)
            #MAKE MINI WAV FILES
            generate_audios(new_full_filename, timing_file, MINI_OUTPUT_FOLDER, wav_id)
            wav_id += 1
            prediction_filename=os.path.join(PREDICTION_FOLDER, wfile[:-4]+"_prediction.txt")
            mini_filenames = os.listdir(MINI_OUTPUT_FOLDER)
            prediction_dict = {}
            #NOTE PREDICTION
            f = open(prediction_filename, 'w')
            for mini_filename in mini_filenames:
                if mini_filename.endswith('.wav') and mini_filename.startswith(wfile[:-4]):
                    fn = os.path.join(MINI_OUTPUT_FOLDER, mini_filename)
                    test_ds = SoundSet(fn, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device)
                    test_dl = create_data_loader(test_ds, BATCH_SIZE, shuffle=False)
                    final_model = CNNNetwork(10)
                    state_dict = torch.load(FINAL_MODEL)
                    final_model.load_state_dict(state_dict)
                    pred_list = test_model(test_dl, final_model, device)
                    prediction = ','.join(pred_list)
                    prediction_idx = mini_filename.split('-')[0].split('_')[-1]


                    list_of_midi=[]
                    for idx in range(len(pred_list)):
                        if pred_list[idx] == '1':
                            midi_value = INDEX2MIDI[idx]
                            list_of_midi.append(midi_value)
                    if list_of_midi == []:
                        midi = 0
                    else:
                        midi = ','.join(list_of_midi)

                    x, sr = librosa.load(fn)
                    articulation, med, percent = get_articulation(x)

                    print(mini_filename, prediction_idx, midi, articulation, med, percent, timing_dict[int(prediction_idx)]['onset'], timing_dict[int(prediction_idx)]['offset'])
                    prediction_dict[int(prediction_idx)] = { "articulation":articulation, "color": 'black', "comment": "", "midi": midi, "note_type":None,
                                                            "onset":timing_dict[int(prediction_idx)]['onset'], "offset": timing_dict[int(prediction_idx)]['offset']}
                    f.write('%s\t%s\t%s\n' % (mini_filename, prediction_idx, midi))
            f.close()
            #ALIGNMENT
            ref_list = get_ref_sequence(wfile)
            print(prediction_filename)
            print(ref_list)
            test_list = get_predicted_sequence(prediction_filename)
            print(test_list)
            gap_penalty = -1
            alignment_txt = os.path.join(PREDICTION_FOLDER, wfile[:-4]+"_alignment.txt")
            alignment_accuracy_txt = os.path.join(PREDICTION_FOLDER, wfile[:-4] + "_alignment_accuracy.txt")
            score_matrix, pointer_matrix = align(ref_list, test_list, gap_penalty)
            align1, align2 = get_alignment(test_list, ref_list, score_matrix, pointer_matrix)
            fh1 = open(alignment_accuracy_txt, 'w')
            fh1.write('%s\n%s\n'%(align1, align2))
            align1_list = align1.split(' ')
            align2_list = align2.split(' ')
            print(align1_list)
            print(align2_list)
            alignment_sum = 0
            total_aligned = 0

            for align_idx in range(len(align1_list)):
                if align1_list[align_idx] != '' :
                    if align1_list[align_idx] == align2_list[align_idx]:
                        alignment_sum += 1
                    if '---' not in align1_list[align_idx] or '*' not in align1_list[align_idx]:
                        total_aligned += 1

            fh1.write('Accuracy Count: %d\n'%(alignment_sum))
            fh1.write("Total Count: %d\n"%(total_aligned))
            fh1.close()
            print_alignment(align1, align2, alignment_txt)
            #DURATION
            expected_duration_dict = calc_expected_note_duration(tempo)
            print(wfile)
            prediction_dict = calc_note_type_for_predictions(prediction_dict, expected_duration_dict)
            prediction_dict = create_color_note(align1, align2, prediction_dict)

            duration_ref_fh = open(DURATION_REF_FILE, 'r')
            duration_lines = duration_ref_fh.readlines()
            duration_ref_fh.close()
            duration_ref_dict = {}
            duration_count = 1
            for duration_line in duration_lines:
                duration_tokens = duration_line.strip().split('\t')
                duration_ref_dict[duration_count] = {'midi': duration_tokens[0], 'note_duration': duration_tokens[1]}
                duration_count = duration_count + 1
            total_duration_tested = 0
            total_duration_tested_correct = 0

            ref_duration_cur_index = 1
            prediction_cur_index = 1

            for align_idx in range(len(align1_list)):
                if align1_list[align_idx] == align2_list[align_idx]:
                    #print("Matched", align_idx, ref_duration_cur_index, prediction_cur_index, align1_list[align_idx], align2_list[align_idx])
                    if prediction_cur_index > len(prediction_dict) or ref_duration_cur_index > len(duration_ref_dict):
                        break
                    total_duration_tested += 1
                    predicted_note_duration = prediction_dict[prediction_cur_index]['note_type']
                    ref_note_duration = duration_ref_dict[ref_duration_cur_index]['note_duration']
                    if predicted_note_duration == ref_note_duration:
                        total_duration_tested_correct += 1
                        if prediction_dict[prediction_cur_index]['color'] == 'orange':
                            prediction_dict[prediction_cur_index]['color'] = 'black'
                    else:
                        if prediction_dict[prediction_cur_index]['color'] == 'black':
                            prediction_dict[prediction_cur_index]['color'] = 'orange'
                    #print(predicted_note_duration, ref_note_duration, prediction_dict[prediction_cur_index]['color'])
                    prediction_cur_index += 1
                    ref_duration_cur_index += 1
                elif '---' not in align2_list[align_idx] and  '---' not in align1_list[align_idx] and align1_list[align_idx] != align2_list[align_idx]:
                    #print("Mismatched", align_idx, ref_duration_cur_index, prediction_cur_index, align1_list[align_idx], align2_list[align_idx])
                    prediction_cur_index += 1
                    ref_duration_cur_index += 1
                elif '---' in align1_list[align_idx] and align1_list[align_idx] != align2_list[align_idx]:
                    #print("Extra", align_idx, ref_duration_cur_index, prediction_cur_index, align1_list[align_idx], align2_list[align_idx])
                    prediction_cur_index += 1
                elif '---' in align2_list[align_idx] and align1_list[align_idx] != align2_list[align_idx]:
                    #print("Missing", align_idx, ref_duration_cur_index, prediction_cur_index, align1_list[align_idx], align2_list[align_idx])
                    ref_duration_cur_index += 1
            print("DURATION TEST RESULT: ", total_duration_tested, total_duration_tested_correct)
            duration_result_dict[wfile[:-4]] = {'total': total_duration_tested, 'correct': total_duration_tested_correct}
            write_prediction_dict(prediction_dict, os.path.join(PREDICTION_FOLDER, wfile[:-4]+"_music21.txt"))


    # write duration accuracy of each song within a dataset
    duration_result_fh = open(os.path.join(PREDICTION_FOLDER, "duration_result_summary.txt"), 'w')
    for mykey in duration_result_dict.keys():
        duration_result_fh.write('%s\t%d\t%s\n'%(mykey, duration_result_dict[mykey]['total'], duration_result_dict[mykey]['correct']))
    duration_result_fh.close()

if __name__ == "__main__":
    wav2score()