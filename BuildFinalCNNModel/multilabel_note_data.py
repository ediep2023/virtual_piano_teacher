import os
import re

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
CLASS2MIDI_DICT = {
    0: '62',
    1: '66',
    2: '67',
    3: '69',
    4: '71',
    5: '72',
    6: '74',
    7: '76',
    8: '78',
    9: '79',
    10: '66, 69, 74',
    11: '69, 74',
    12: '62, 66, 69',
    13: '67, 71',
    14: '69, 72',
    15: '71, 74',
    16: '72, 76',
    17: '66, 69',
    18: '62, 67, 71',
    19: '71, 74, 79',
    1000: '0'
 }

def write_csv(mydict, csv_file):
    '''
    :param mydict: my_dict where the key is a wav filename and the value contains class ID - string of 10 items
    :              seperated by comma
    :param csv_file: the name of the output file
    :method: generate a csv file with header where a row means a note in a song and 11 columns - first column is the
    :        wavfile name, and the rest are MIDI labels in CNN classification;
    :        0 means a MIDI is not played, and 1 means a MIDI is played
    '''
    header = 'wavfile,label1,label2,label3,label4,label5,label6,label7,label8,label9,label10\n'
    f = open(csv_file, 'w')
    f.write(header)
    for k, v in mydict.items():
        f.write('%s,%s\n' % (k, v))
    f.close()

def make_data_label_dict(folder):
    '''
    :param folder: directory of train or valid dataset
    :return: my_dict where the key is a wav filename and the value contains class ID - string of 10 items seperated by comma
    :method: for every wav file in the input folder, spilt each filename by '-' to find MIDI value.
    :        If MIDI value is chord, make a string of ten items where each item is either 0 or 1. This string is a note's class ID.
    :        Otherwise, search class ID in CLASS_DICT using MIDI value

    '''
    mydict = {}
    # for each .wav file in the input folder
    for wavfile in os.listdir(folder):
        filename_items = wavfile.split('-')
        note_class = filename_items[-2]
        midi = CLASS2MIDI_DICT[int(note_class)]
        classID = None
        # if midi is a chord
        if ',' in midi:
            # initialize the class ID
            midi_list = midi.split(',')
            label = '0,0,0,0,0,0,0,0,0,0'
            prev_label_list = label.split(',')
            # for every note in chord
            for i in midi_list:
                #find the class ID for each note
                my_midi = int(i)
                cur_label = CLASS_DICT[my_midi]
                cur_label_list = cur_label.split(',')
                merged_list = []
                # for every label in the class ID
                for j in range(len(cur_label_list)):
                    # merge labels for chord class ID
                    if cur_label_list[j]=='0' and prev_label_list[j]=='0':
                        merged_list.append('0')
                    else:
                        merged_list.append('1')
                prev_label_list=merged_list
            classID = ','.join(prev_label_list)
        else:
            classID = CLASS_DICT[int(midi)]
        mydict[wavfile] = classID
    return mydict


if __name__ == "__main__":

    mydict = make_data_label_dict("/home/parallels/PycharmProjects/PianoModelBuilder/data/Minuet/ScienceFair2023/NoiseFiltered/Minuet/train")
    write_csv(mydict, "/home/parallels/PycharmProjects/PianoModelBuilder/data/Minuet/ScienceFair2023/NoiseFiltered/Minuet/Minuet_training_multilabels.csv")
    mydict = make_data_label_dict("/home/parallels/PycharmProjects/PianoModelBuilder/data/Minuet/ScienceFair2023/NoiseFiltered/Minuet/valid")
    write_csv(mydict, "/home/parallels/PycharmProjects/PianoModelBuilder/data/Minuet/ScienceFair2023/NoiseFiltered/Minuet/Minuet_valid_multilabels.csv")