import os
import glob
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils.audio import Audio


def create_dataloader(hp, train):
    def train_collate_fn(batch):
        target_wav_list = list()
        mixed_spec_list = list()
        target_spec_list = list()

        for target_wav, mixed_spec, target_spec in batch:
            target_wav_list.append(target_wav)
            mixed_spec_list.append(mixed_spec)
            target_spec_list.append(target_spec)
        target_wav_list = torch.stack(target_wav_list, dim=0)
        mixed_spec_list = torch.stack(mixed_spec_list, dim=0)
        target_spec_list = torch.stack(target_spec_list, dim=0)
        return target_wav_list, mixed_spec_list, target_spec_list

    def test_collate_fn(batch):
        return batch

    if train:
        train_data = LoadDataset(hp, True)
        return DataLoader(dataset=train_data,
                          batch_size=hp.train.batch_size,
                          shuffle=False,
                          num_workers=4,
                          collate_fn=train_collate_fn,
                          pin_memory=True,
                          drop_last=True,
                          sampler=None)
    else:
        return DataLoader(dataset=LoadDataset(hp, False),
                          collate_fn=test_collate_fn,
                          batch_size=1, shuffle=False, num_workers=1, pin_memory=True)


class LoadDataset(Dataset):
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
        if self.train:  # Train set
            target_wav, _ = librosa.load(self.target_wav_list[idx], sr=self.hp.audio.sample_rate)
            mixed_wav, _ = librosa.load(self.mixed_wav_list[idx], sr=self.hp.audio.sample_rate)

            # Perform zero padding to match the length of the speeches
            target_wav = np.concatenate([target_wav, np.zeros(self.hp.train.max_audio_len - target_wav.shape[0])],
                                        axis=0)

            mixed_padding = np.concatenate([mixed_wav, np.zeros(self.hp.train.max_audio_len - mixed_wav.shape[0])],
                                           axis=0)

            mixed_spec = self.audio.wav2spec(mixed_padding)
            mixed_spec = torch.from_numpy(mixed_spec).float()

            target_spec = self.audio.wav2spec(target_wav)
            target_spec = torch.from_numpy(target_spec).float()

            target_wav = torch.from_numpy(target_wav).float()

            return target_wav, mixed_spec, target_spec

        else:  # Valid set
            target_wav, _ = librosa.load(self.target_wav_list[idx], sr=self.hp.audio.sample_rate)
            mixed_wav, _ = librosa.load(self.mixed_wav_list[idx], sr=self.hp.audio.sample_rate)

            target_spec = self.audio.wav2spec(target_wav)
            mixed_spec = self.audio.wav2spec(mixed_wav)

            mixed_wav_padding = np.concatenate([mixed_wav, np.zeros(self.hp.train.max_audio_len - mixed_wav.shape[0])],
                                               axis=0)
            mixed_spec_padding = self.audio.wav2spec(mixed_wav_padding)

            target_spec = torch.from_numpy(target_spec).float()
            mixed_spec = torch.from_numpy(mixed_spec).float()
            mixed_spec_padding = torch.from_numpy(mixed_spec_padding).float()

            return target_wav, mixed_wav, target_spec, mixed_spec, mixed_spec_padding

