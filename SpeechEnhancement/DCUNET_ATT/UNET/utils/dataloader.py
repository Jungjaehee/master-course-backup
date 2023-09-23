import os
import glob
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils.audio import Audio


def create_dataloader(hp, train):
    def train_collate_fn(batch):
        target_mag_list = list()
        mixed_mag_list = list()

        for target_mag, mixed_mag in batch:
            mixed_mag_list.append(mixed_mag)
            target_mag_list.append(target_mag)
        target_mag_list = torch.stack(target_mag_list, dim=0)
        mixed_mag_list = torch.stack(mixed_mag_list, dim=0)
        return target_mag_list, mixed_mag_list

    def test_collate_fn(batch):
        return batch

    if train:
        return DataLoader(dataset=VFDataset(hp, True),
                          batch_size=hp.train.batch_size,
                          shuffle=True,
                          num_workers=hp.train.num_workers,
                          collate_fn=train_collate_fn,
                          pin_memory=True,
                          drop_last=True,
                          sampler=None)
    else:
        return DataLoader(dataset=VFDataset(hp, False),
                          collate_fn=test_collate_fn,
                          batch_size=1, shuffle=False, num_workers=0)


class VFDataset(Dataset):
    def __init__(self, hp, train):
        def file_list_():
            lines = []
            with open(self.ctrl_dir, 'r') as f:
                for file in f:
                    lines.append(file.replace('\n', ''))

            new_target_wav = [(self.target_dir + ('/').join(x.split('/')[-2:]) + hp.form.target.wav) for x in lines]
            new_mixed_wav = [(self.data_dir + x + hp.form.mixed.wav) for x in lines]

            return [new_target_wav, new_mixed_wav]

        self.hp = hp
        self.train = train
        self.data_dir = hp.data.mixed_dir
        self.target_dir = hp.data.target_dir
        self.ctrl_dir = hp.data.train_ctrl_dir if train else hp.data.test_ctrl_dir

        new_lists = file_list_()
        self.target_wav_list = new_lists[0]
        self.mixed_wav_list = new_lists[1]

        assert len(self.target_wav_list) == len(self.mixed_wav_list), "number of training files must match"
        self.audio = Audio(hp)

    def __len__(self):
        return len(self.target_wav_list)

    def __getitem__(self, idx):
        if self.train: # need to be fast
            target_wav, _ = librosa.load(self.target_wav_list[idx], sr=self.hp.audio.sample_rate)
            mixed_wav, _ = librosa.load(self.mixed_wav_list[idx], sr=self.hp.audio.sample_rate)

            target_mag, _ = self.audio.wav2spec(np.concatenate([target_wav, np.zeros(self.hp.train.max_audio_len - target_wav.shape[0])],
                                        axis=0))
            mixed_mag, _ = self.audio.wav2spec(np.concatenate([mixed_wav,
                                                             np.zeros(
                                                                 self.hp.train.max_audio_len - mixed_wav.shape[0])],
                                                            axis=0))

            target_mag = torch.from_numpy(target_mag).float()
            mixed_mag = torch.from_numpy(mixed_mag).float()

            return target_mag, mixed_mag

        else:
            target_wav, _ = librosa.load(self.target_wav_list[idx], sr=self.hp.audio.sample_rate)
            mixed_wav, _ = librosa.load(self.mixed_wav_list[idx], sr=self.hp.audio.sample_rate)

            target_mag, _ = self.audio.wav2spec(target_wav)
            mixed_mag, mixed_phase = self.audio.wav2spec(mixed_wav)
            mixed_padding_mag, mixed_padding_phase = self.audio.wav2spec(np.concatenate([mixed_wav,
                                                             np.zeros(self.hp.train.max_audio_len - mixed_wav.shape[0])],
                                                            axis=0))
            target_mag = torch.from_numpy(target_mag).float()
            mixed_padding_mag = torch.from_numpy(mixed_padding_mag).float()

            return target_wav, mixed_wav, target_mag, mixed_mag, mixed_phase, mixed_padding_mag

