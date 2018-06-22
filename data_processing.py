import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
import os
from collections import defaultdict
import torch
import numpy as np
import glob
from layers import TacotronSTFT
from utils import load_wav_to_torch
MAX_WAV_VALUE = 32768.0
SAMPLING_RATE = 8000


def iterate_minibatches(dataset, lbl_id, batch_size, shuffle=False,
                        forever=True, length=128, to_numpy=False):

    if lbl_id is not None:
        batch_size = int(batch_size / 2)

    while True:
        data, labels = [], []
        if lbl_id is not None:
            # sample target speaker
            start_ids = np.random.choice(
                np.arange(dataset[lbl_id][0].shape[1] - length),
                batch_size,
                replace=True)
            data = [dataset[lbl_id][0][:, start_id:start_id+length]
                    for start_id in start_ids]
            labels = [lbl_id] * (batch_size)

        # sample other speakers
        other_ids = np.random.choice(
            [i for i in range(len(dataset)) if i != lbl_id],
            batch_size,
            replace=True)

        for i in other_ids:
            start_id = np.random.randint(dataset[i][0].shape[1] - length)
            data.append(dataset[i][0][:, start_id:start_id+length])
            labels.append(i)

        labels = np.eye(len(dataset))[labels]

        # convert to torch.FloatTensor
        data = torch.stack(data)
        labels = torch.from_numpy(labels)
        if to_numpy:
            data = data.data.cpu().numpy()
            labels = labels.numpy()

        if shuffle:
            rand_ids = np.random.permutation(np.arange(len(data)))
            yield data[rand_ids], labels[rand_ids]
        else:
            yield data, labels

        if not forever:
            break


def load_data(datapath, glob_file_str, scale=True, data_split=[0.8, 0.1]):
    data = defaultdict(list)
    stft = TacotronSTFT(
        filter_length=1024, hop_length=160,
        win_length=1024, sampling_rate=16000, n_mel_channels=64,
        mel_fmin=0, mel_fmax=None, representation='asrgen')

    for folderpath in sorted(glob.glob(os.path.join(datapath, '*/'))):
        label = os.path.basename(os.path.normpath(folderpath))
        filepaths = glob.glob(os.path.join(
            os.path.join(datapath, label), glob_file_str))
        for filepath in filepaths:
            audio = load_wav_to_torch(filepath, stft.sampling_rate)
            audio_norm = audio / MAX_WAV_VALUE
            audio_norm = audio_norm / torch.max(audio_norm.abs())
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            mel_spec = stft.mel_spectrogram(audio_norm)[0]
            mel_spec -= mel_spec.min()
            mel_spec = mel_spec / torch.max(mel_spec)
            mel_spec = (mel_spec * 2) - 1
            train_end = int(mel_spec.size(1)*data_split[0])
            val_end = int(mel_spec.size(1)*(data_split[0]+data_split[1]))
            data['train'].append([mel_spec[:, :train_end], label])
            data['valid'].append([mel_spec[:, train_end:val_end], label])
            data['test'].append([mel_spec[:, val_end:], label])
    return data
