import numpy as np
from preprocessAudio import *
from preprocessText import *
from keras.preprocessing import text as text1, sequence


def pad_audioc(audioc):
    if np.shape(audioc) != (30, 20, 30):
        zero = np.zeros((20, 30))
        for i in range(30 - np.shape(audioc)[0]):
            audioc.append(zero)
    return audioc


def get_data():
    emotion_dict = {'ang': 0, 'hap': 1, 'exc': 2, 'sad': 3, 'fru': 4, 'fea': 5, 'sur': 6, 'neu': 7, 'xxx': 8, 'oth': 8}

    text_features, label_file = datasetText()
    audio_features = datasetAudio()

    audio = []
    text = []
    label = []

    data = {}
    for items in label_file:
        label.append(emotion_dict[label_file[items]])
        text.append(text_features[items])

        audio.append(audio_features[items])

    max_features = 12000
    tokenizer = text1.Tokenizer(lower=False, num_words=max_features)
    tokenizer.fit_on_texts(text)
    sorted_by_word_count = sorted(tokenizer.word_counts.items(), key=lambda x: x[1], reverse=True)
    tokenizer.word_index = {}
    i = 0
    for word, count in sorted_by_word_count:
        if i == max_features:
            break
        tokenizer.word_index[word] = i + 1  # <= because tokenizer is 1 indexed
        i += 1
    text_data = tokenizer.texts_to_sequences(text)
    text_data = sequence.pad_sequences(text_data, padding='post',maxlen=30)
    text_data = np.array(text_data)
    # print(np.shape(text_data))
    newest = np.zeros((30, 30, 600), dtype=float)
    # print(newest.dtype)
    # print(newest[0])  #todo remove the hardcoded values
    for i in range(len(audio)):
        audio[i] = list(audio[i])
        audio[i] = pad_audioc(audio[i])
        audio[i] = np.reshape(audio[i], (30, 600))
        # print(np.shape(audio[i]))
        audio[i] = np.array(audio[i])
        newest[i] = audio[i]

    audio_data = newest
    data = {'text': text_data, 'audio': audio, 'label': label}
    return data


class generator():
    def __init__(self):
        self.paths = 'text.txt'

    def __len__(self):
        return 30

    def __getitem__(self, idx):
        labels = get_data()['label']
        audio = get_data()['audio']
        text = get_data()['text']
        labels_new=labels[idx]
        audio_feats=audio[idx]
        text_feats=text[idx]
        # print(np.array(labels_new).dtype)
        sample = {'text': torch.from_numpy(text_feats), 'audio': torch.from_numpy(audio_feats), 'label': torch.from_numpy(np.asarray(labels_new,dtype=int))}

        return sample
