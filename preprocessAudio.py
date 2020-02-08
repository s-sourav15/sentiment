import librosa
import numpy as np
import torch
import pandas as pd
import os
from torch.utils.data import DataLoader, TensorDataset
import torchaudio
from torchaudio import transforms
import re

def alignment(align_file):
    f = open(align_file, 'r')  ### alignment file
    lines = f.readlines()[1:]

    #
    # for line in lines:
    time_pairs = []
    #   print(line.split()[3], line.split()[0])
    for line in lines:
        if line.split()[0] != "Total":
            if (line.split()[3] != '<s>' and line.split()[3] != '</s>' and line.split()[3] != '<sil>'):
                tp0 = ((int(line.split()[0]) + 2) / 100)
                tp1 = ((int(line.split()[1]) + 2) / 100)
                time_pairs.append([tp0, tp1])
        else:
            break
    return time_pairs


def align_dict(align_file):
    path = align_file
    align_dict = {}
    for files in os.listdir(path):
        # new_path = str()
        if files.find('Ses') != -1:
            new_path = os.path.join(path + '/' + files)
            # print(new_path)
            for align_files in os.listdir(new_path):
                if align_files.endswith('.wdseg') == 1:
                    alignments = os.path.join(new_path + '/' + align_files)
                    time_pairs = alignment(alignments)
                    align_dict[align_files.split('.')[0]] = [align_files.split('.')[0], time_pairs]
    return align_dict


def wav2mfcc(wave, max_len=30):
    #     mfcc = librosa.feature.mfcc(wave, sr=16000)
    mfcc = librosa.feature.mfcc(wave, n_mfcc=20)

    # If maximum length exceeds mfcc lengths then pad the remaining ones
    if max_len > mfcc.shape[1]:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

    # Else cutoff the remaining parts
    else:
        mfcc = mfcc[:, :max_len]

    return mfcc


alignments = align_dict('forced_alignment')
# mfc=np.zeros((30,),dtype=float)
mfc = []



def pad_mfcc(mfcc):
    if np.shape(mfcc) != (30, 20, 30):
        zero = np.zeros((20, 30))
        for i in range(30 - np.shape(mfcc)[0]):
            mfcc.append(zero)
    return mfcc


def datasetAudio():
    emotion_dict = {'ang': 0, 'hap': 1, 'exc': 2, 'sad': 3, 'fru': 4, 'fea': 5, 'sur': 6, 'neu': 7, 'xxx': 8, 'oth': 8}

    features={}
    for items in alignments:
        audio_file = alignments[items][0]
        audios = os.path.join('Ses01F_impro01' + '/' + audio_file + '.wav')
        y, sample_rate = librosa.load(audios)
        time_pairs = alignments[items][1]
        feats = []
        for i in time_pairs:
            # print(i)
            diff = int(i[1] * sample_rate) - int(i[0] * sample_rate)
            difference = 10000 - diff
            audio = y[(int(i[0] * sample_rate)):(int(i[1] * sample_rate) + difference)]

            mf = wav2mfcc(audio)

            feats.append(mf)

        features[audio_file]=feats
    return features
    # with open('labels.txt', 'r') as f:
    #    content=f.read()
    # labels = []
    # # lines = f.readlines()
    # info=[]
    # useful_regex = re.compile(r'\[.+\]\n', re.IGNORECASE)
    # lines = re.findall(useful_regex, content)
    #
    # for line in lines[1:]:
    #     # print(line)
    #     info.append([line.strip().split('\t')[1],line.strip().split('\t')[2]])
    # for k in features:
    #     for line in lines:
    #         print(k)
    #         # exit()
    #         if k[0] == info[0]:
    #             # print(k[0])
    #             labels.append(info[1])
    #             k[1] = np.array(k[1])
    #             # print('i1',i[1].dtype)
    #
    #             mfc.append(k[1])
    # newest = np.zeros((30, 30, 600), dtype=float)
    # # print(newest.dtype)
    # # print(newest[0])  #todo remove the hardcoded values
    # for i in range(len(mfc)):
    #     mfc[i] = list(mfc[i])
    #     mfc[i] = pad_mfcc(mfc[i])
    #     mfc[i]=np.reshape(mfc[i],(30,600))
    #     # print(np.shape(mfc[i]))
    #     mfc[i] = np.array(mfc[i])
    #     newest[i] = mfc[i]

    # return newest,labels

def generateLowLevelAudio(path):
    y, sr = librosa.load(path)
    time_pairs = alignment('align.wdseg')
    features = []
    for i in time_pairs:
        feats = []
        y_new = y[int(i[0] * sr):int(i[1] * sr)]
        sig_mean = np.mean(np.abs(y_new))
        sig_std = np.std(y_new)
        # print(sig_mean, sig_std)
        rmse = librosa.feature.rmse(y_new + 0.0001)[0]

        rms_mean = np.mean(rmse)
        rmse_std = np.std(rmse)
        y_harmonic, y_percussive = librosa.effects.hpss(y_new)
        # print(np.mean(y_harmonic))
        autocorr = librosa.core.autocorrelate(y_new)
        # print(np.max(autocorr))
        feats.append([sig_mean, sig_std, rms_mean, rmse_std, y_harmonic, y_percussive, np.max(autocorr)])

        features.append(feats)

    return feats

