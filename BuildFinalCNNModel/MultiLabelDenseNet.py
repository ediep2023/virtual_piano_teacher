import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torchaudio
import pandas as pd
import os

import torch.nn as nn
import math
from torchsummary import summary
from torch.utils.data.dataloader import DataLoader

class SoundDataset(Dataset):
    def __init__(self, annotation_file, audio_dir, transformation, target_sample_rate, num_samples, device):
        '''
        :param annotation_file: csv file
        :param audio_dir: directory of train and valid audio
        :param transformation: function to transform audio into mel spectrogram
        :param target_sample_rate: number of samples * time
        :param num_samples: sample count
        :param device: CPU
        :method: initalize SoundDataset with input value
        '''
        self.annotations = pd.read_csv(annotation_file)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        '''
        :return: total number of notes
        '''
        return len(self.annotations)

    def __getitem__(self, index):
        '''
        :param index: a note's index within a song
        :method: adjust indesed chopped audio to one second and transform into mel spectorgram
        :return: digital signal of the indexed chopped audio, and its 10 corresponding labels.
        '''
        audio_sample_path = self._get_audio_sample_path(index)
        labels = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        #return signal, label
        label1 = torch.tensor(labels[0], dtype=torch.float32)
        label2 = torch.tensor(labels[1], dtype=torch.float32)
        label3 = torch.tensor(labels[2], dtype=torch.float32)
        label4 = torch.tensor(labels[3], dtype=torch.float32)
        label5 = torch.tensor(labels[4], dtype=torch.float32)
        label6 = torch.tensor(labels[5], dtype=torch.float32)
        label7 = torch.tensor(labels[6], dtype=torch.float32)
        label8 = torch.tensor(labels[7], dtype=torch.float32)
        label9 = torch.tensor(labels[8], dtype=torch.float32)
        label10 = torch.tensor(labels[9], dtype=torch.float32)
        return signal, {
            'label1': label1,
            'label2': label2,
            'label3': label3,
            'label4': label4,
            'label5': label5,
            'label6': label6,
            'label7': label7,
            'label8': label8,
            'label9': label9,
            'label10': label10
        }

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

    def _get_audio_sample_path(self, index):
        '''
        :param index: a note's index within a song
        :method: finds chopped wav file name under csv file
        :return: full path of the chopped audio file
        '''
        path = os.path.join(self.audio_dir, self.annotations.iloc[index, 0])
        return path

    def _get_audio_sample_label(self, index):
        '''
        :param index: a note's index within a song
        :method: finds csv file for columns 2 to 11 depending on note number
        :return: note labels
        '''
        return self.annotations.iloc[index, 1:11]

class Dense_Block(nn.Module):
    def __init__(self, in_channels):
        '''
        :param in_channels: input features
        :method: defines the building blocks for Dense Block
        '''
        super(Dense_Block, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(num_features=in_channels)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        '''
        :param x: signal
        :method: apply batch normalization->convolution->ReLU->feature1
        :        feature1->convolution->RELU->feature2
        :        combine feature1 and feature2 -> dense block
        :return: completed dense block
        '''
        bn = self.bn(x)
        conv1 = self.relu(self.conv1(bn))

        conv2 = self.relu(self.conv2(conv1))
        c2_dense = self.relu(torch.cat([conv1, conv2], 1))

        return c2_dense


class Transition_Layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        '''
        :param in_channels: input features
        :param out_channels: output features
        :method: defines building blocks for transition layer
        '''
        super(Transition_Layer, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        self.bn = nn.BatchNorm2d(num_features = out_channels)
        self.conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, bias = False)
        self.avg_pool = nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0)

    def forward(self, x):
        '''
        :param x: signal
        :method: signal->convolution->ReLU->batch normalization->average pooling->output of transition layer
        :return: output of transition layer
        '''
        bn = self.bn(self.relu(self.conv(x)))
        out = self.avg_pool(bn)
        return out

class DenseNet(nn.Module):
    def __init__(self, nr_classes):
        '''
        :param nr_classes: 10 notes that can be classified
        :method: 16 layers: 1 lowconv layer, 1 ReLU layer, 1 dense block, 1 transition layer, 1 batch normalization layer, 1 linear layer, 10 classifier layers
        :        512 outputs are classified into 10 classes
        '''
        super(DenseNet, self).__init__()
        self.lowconv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=3, bias=False)
        self.relu = nn.ReLU()

        # Make Dense Blocks
        self.denseblock1 = self._make_dense_block(Dense_Block, 32)

        # Make transition Layers
        self.transitionLayer1 = self._make_transition_layer(Transition_Layer, in_channels=64, out_channels=32)

        # Classifier
        self.bn = nn.BatchNorm2d(num_features=32)
        self.pre_classifier = nn.Linear(66560, 512)
        #self.classifier = nn.Linear(512, nr_classes)
        self.out1 = nn.Linear(512, 1)
        self.out2 = nn.Linear(512, 1)
        self.out3 = nn.Linear(512, 1)
        self.out4 = nn.Linear(512, 1)
        self.out5 = nn.Linear(512, 1)
        self.out6 = nn.Linear(512, 1)
        self.out7 = nn.Linear(512, 1)
        self.out8 = nn.Linear(512, 1)
        self.out9 = nn.Linear(512, 1)
        self.out10 = nn.Linear(512, 1)

    def _make_dense_block(self, block, in_channels):
        '''
        :param block: Dense_Block class
        :param in_channels: input features
        :method: create dense block
        :return: dense block
        '''
        layers = []
        layers.append(block(in_channels))
        return nn.Sequential(*layers)

    def _make_transition_layer(self, layer, in_channels, out_channels):
        '''
        :param layer: layers list
        :param in_channels: input features
        :param out_channels: output feathers
        :method: create transition layer
        :return: transition layer
        '''
        modules = []
        modules.append(layer(in_channels, out_channels))
        return nn.Sequential(*modules)

    def forward(self, x):
        '''
        :param x: signal
        :method: signal-> lowconv-> ReLU-> dense block-> transition layer-> batch normalization-> flatten->
        :        fully connected linear layer-> hidden (512 nodes)-> output layer (10 notes)-> sigmoidal function (per
        :        output)-> prediction for all 10 labels
        :return: note prediction
        '''
        out = self.relu(self.lowconv(x))

        out = self.denseblock1(out)
        out = self.transitionLayer1(out)

        out = self.bn(out)
        out = out.view(-1, 66560)

        out = self.pre_classifier(out)
        #out = self.classifier(out)
        out1 = torch.sigmoid(self.out1(out))
        out2 = torch.sigmoid(self.out2(out))
        out3 = torch.sigmoid(self.out3(out))
        out4 = torch.sigmoid(self.out4(out))
        out5 = torch.sigmoid(self.out5(out))
        out6 = torch.sigmoid(self.out6(out))
        out7 = torch.sigmoid(self.out7(out))
        out8 = torch.sigmoid(self.out8(out))
        out9 = torch.sigmoid(self.out9(out))
        out10 = torch.sigmoid(self.out10(out))
        return out1, out2, out3, out4, out5, out6, out7, out8, out9, out10


if __name__ == "__main__":
    ANNOTATIONS_FILE = "/home/parallels/PycharmProjects/PianoModelBuilder/jingle3_training_multilabels.csv"
    AUDIO_DIR = "/home/parallels/PycharmProjects/PianoModelBuilder/data/Minuet/train"
    SAMPLE_RATE = 16000
    NUM_SAMPLES = 16000
    device = "cpu"
    print(f"using device {device}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=256,
        n_mels=128
    )

    usd = SoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram,
                       SAMPLE_RATE, NUM_SAMPLES, device)
    print(f"There are {len(usd)} samples in the dataset.")
    data = usd[1]
    print(data[0].size())