import torch
import torchaudio
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.utils import save_image
import torch.optim as optim


from MultiLabelDenseNet import SoundDataset
from torchsummary import summary
from MultiLabelDenseNet import DenseNet as CNNNetwork

def transpose_list(src):
    '''
    :param src: a list
    :method: turns each item in dst list into torch.Tensor object
    :return: Tensor object dst (list)
    '''
    dst=[]
    for i in range(len(src)):
        x=src[i]
        dst.append([torch.tensor(x, dtype=torch.long)])
    return torch.Tensor(dst)

def loss_fn(outputs, targets):
    '''
    :param outputs: a list of ten predicted output labels
    :param targets: a list of ten expected output labels
    :method: calculate average error loss
    :return: average error loss
    '''
    o1, o2, o3, o4, o5, o6, o7, o8, o9, o10 = outputs
    target1, target2, target3, target4, target5, target6, target7, target8, target9, target10 = targets
    t1 = transpose_list(target1)
    t2 = transpose_list(target2)
    t3 = transpose_list(target3)
    t4 = transpose_list(target4)
    t5 = transpose_list(target5)
    t6 = transpose_list(target6)
    t7 = transpose_list(target7)
    t8 = transpose_list(target8)
    t9 = transpose_list(target9)
    t10 = transpose_list(target10)

    l1 = nn.BCELoss()(o1, t1)
    l2 = nn.BCELoss()(o2, t2)
    l3 = nn.BCELoss()(o3, t3)
    l4 = nn.BCELoss()(o4, t4)
    l5 = nn.BCELoss()(o5, t5)
    l6 = nn.BCELoss()(o6, t6)
    l7 = nn.BCELoss()(o7, t7)
    l8 = nn.BCELoss()(o8, t8)
    l9 = nn.BCELoss()(o9, t9)
    l10 = nn.BCELoss()(o10, t10)

    return (l1 + l2 + l3 + l4 + l5 + l6 + l7 + l8 + l9 + l10) / 10

def create_data_loader(dataset, batch_size, shuffle):
    '''
    :param dataset: dataset input
    :param batch_size: number of samples in a batch
    :param shuffle: shuffle samples
    :method: create data loader
    :return: data loader
    '''
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train(model, dataloader, optimizer, loss_fn, device):
    '''
    :param model: CNN Model
    :param dataloader: data loader object
    :param optimizer: model tuning
    :param loss_fn: function that calculates accuracy loss
    :param device: CPU
    :method: train and tune model
    :return: accuracy loss during training

    '''
    model.train()
    counter = 0
    train_running_loss = 0.0
    for batch_idx, (data, targets) in enumerate(train_dl):
        counter += 1
        print(counter)
        # extract the features and labels

        features = data.to(device)
        target1 = targets['label1'].to(device)
        target2 = targets['label2'].to(device)
        target3 = targets['label3'].to(device)
        target4 = targets['label4'].to(device)
        target5 = targets['label5'].to(device)
        target6 = targets['label6'].to(device)
        target7 = targets['label7'].to(device)
        target8 = targets['label8'].to(device)
        target9 = targets['label9'].to(device)
        target10 = targets['label10'].to(device)

        # zero-out the optimizer gradients
        optimizer.zero_grad()

        outputs = model(features)
        targets = (target1, target2, target3, target4, target5, target6, target7, target8, target9, target10)
        loss = loss_fn(outputs, targets)

        train_running_loss += loss.item()
        # backpropagation
        loss.backward()
        print(counter)
        # update optimizer parameters
        optimizer.step()
        print(counter)

    train_loss = train_running_loss / counter
    return train_loss

def training(train_dl, val_dl, model, learning_rate, epochs, device):
    '''
    :param train_dl: input training data loader
    :param val_dl: input validation data loader
    :param model: CNN Model
    :param learning_rate: learning rate
    :param epochs: number of loops
    :param device: CPU
    :method: train & tune model, and test model accuracy with valid dataset over a certain number of epochs
    '''
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
    model.to(device)
    train_loss = []
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        train_epoch_loss = train(model, train_dl, optimizer, loss_fn, device)
        train_loss.append(train_epoch_loss)
        print(f"Train Loss: {train_epoch_loss:.4f}")
        with torch.no_grad():
            num_correct = 0
            num_samples = 0
            for i, (data, targets) in enumerate(val_dl):
                print(f"SAMPLE {i}")
                features = data.to(device)
                target1 = targets['label1'].to(device)
                target2 = targets['label2'].to(device)
                target3 = targets['label3'].to(device)
                target4 = targets['label4'].to(device)
                target5 = targets['label5'].to(device)
                target6 = targets['label6'].to(device)
                target7 = targets['label7'].to(device)
                target8 = targets['label8'].to(device)
                target9 = targets['label9'].to(device)
                target10 = targets['label10'].to(device)

                ## Forward Pass
                outputs = model(data)

                # get all the labels
                all_labels = []
                for out in outputs:
                    if out >= 0.5:
                        all_labels.append(1)
                    else:
                        all_labels.append(0)
                targets = (target1, target2, target3, target4, target5, target6, target7, target8, target9, target10)

                # get all the targets in int format from tensor format
                all_targets = []
                for target in targets:
                    all_targets.append(int(target.squeeze(0).detach().cpu()))
                matched = True
                for idx in range(len(all_labels)):
                    if all_labels[idx]!=all_targets[idx]:
                        matched = False
                if matched:
                    num_correct += 1
                num_samples += 1
                print(f"ALL PREDICTIONS: {all_labels}")
                print(f"GROUND TRUTHS: {all_targets}")
            print(
                f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}"
            )


if __name__ == "__main__":
    '''
    train model based on batch size, learning rate, and epochs
    '''
    TRAIN_BATCH_SIZE = 50
    VALID_BATCH_SIZE = 1
    LEARNING_RATE = 0.001

    TRAINING_ANNOTATION_FILE = "/home/parallels/PycharmProjects/PianoModelBuilder/data/Minuet/ScienceFair2023/NoiseFiltered/Minuet/Minuet_training_multilabels.csv"
    TRAINING_AUDIO_DIR = "/home/parallels/PycharmProjects/PianoModelBuilder/data/Minuet/ScienceFair2023/NoiseFiltered/Minuet/train"

    VALID_ANNOTATION_FILE = "/home/parallels/PycharmProjects/PianoModelBuilder/data/Minuet/ScienceFair2023/NoiseFiltered/Minuet/Minuet_valid_multilabels.csv"
    VALID_AUDIO_DIR = "/home/parallels/PycharmProjects/PianoModelBuilder/data/Minuet/ScienceFair2023/NoiseFiltered/Minuet/valid"

    SAMPLE_RATE = 16000
    NUM_SAMPLES = 16000

    device = "cpu"
    print(f"Using {device}")

    # instantiating our dataset object and create data loader
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024,hop_length=256,n_mels=128)

    train_ds = SoundDataset(TRAINING_ANNOTATION_FILE, TRAINING_AUDIO_DIR, mel_spectrogram,SAMPLE_RATE,NUM_SAMPLES,device)
    valid_ds = SoundDataset(VALID_ANNOTATION_FILE, VALID_AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES,device)

    train_dl = create_data_loader(train_ds, TRAIN_BATCH_SIZE, shuffle=True)
    valid_dl = create_data_loader(valid_ds, VALID_BATCH_SIZE, shuffle=False)

    # construct model and assign it to device
    model = CNNNetwork(10)
    model = model.to(device=device)
    summary(model, (1, 128, 63))

    for i in range (10):
        EPOCHS=i+1
        training(train_dl, valid_dl, model, LEARNING_RATE, EPOCHS, device)
        # save model
        torch.save(model.state_dict(),"/home/parallels/PycharmProjects/PianoModelBuilder/data/Minuet/ScienceFair2023/NoiseFiltered/Minuet/FinalModels/NoiseFilteredMinuet_one_sec_multilabel_densenet_epoch%d.pth"%(EPOCHS))