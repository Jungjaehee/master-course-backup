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
        mixed_wav_list = list()
        mixed_spec_list = list()

        audio = Audio(hp)

        for target_wav, mixed_wav in batch:
            target_wav_list.append(target_wav)
            mixed_wav_list.append(mixed_wav)

        target_wav_list = torch.nn.utils.rnn.pad_sequence(target_wav_list, batch_first=True)
        mixed_wav_list = torch.nn.utils.rnn.pad_sequence(mixed_wav_list, batch_first=True)

        for mixed_wav in mixed_wav_list:
            mixed_spec = audio.wav2spec(np.array(mixed_wav))
            mixed_spec = torch.from_numpy(mixed_spec).float()
            mixed_spec_list.append(mixed_spec)

        # target_wav_list = torch.stack(target_wav_list, dim=0)
        mixed_spec_list = torch.stack(mixed_spec_list, dim=0)
        # target_spec_list = torch.stack(target_spec_list, dim=0)
        return target_wav_list, mixed_spec_list

    def test_collate_fn(batch):
        return batch

    if train:
        return DataLoader(dataset=LoadDataset(hp, True),
                          batch_size=hp.train.batch_size,
                          shuffle=True,
                          num_workers=hp.train.num_workers,
                          collate_fn=train_collate_fn,
                          pin_memory=True,
                          drop_last=True,
                          sampler=None)
    else:
        return DataLoader(dataset=LoadDataset(hp, False),
                          collate_fn=test_collate_fn,
                          batch_size=1, shuffle=False, num_workers=0)


class LoadDataset(Dataset):
    def __init__(self, hp, train):
        def file_list_():
            lines = []
            with open(self.ctrl_dir, 'r') as f:
                for file in f:
                    lines.append(file.strip())

            new_target_wav = [(self.target_dir + x + hp.form.target.wav) for x in lines]
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

            target_wav = torch.from_numpy(target_wav).float()
            mixed_wav = torch.from_numpy(mixed_wav).float()

            return target_wav, mixed_wav

        else:  # Valid set
            target_wav, _ = librosa.load(self.target_wav_list[idx], sr=self.hp.audio.sample_rate)
            mixed_wav, _ = librosa.load(self.mixed_wav_list[idx], sr=self.hp.audio.sample_rate)
            mixed_wav_padding = mixed_wav.copy()
            target_spec = self.audio.wav2spec(target_wav)

            if mixed_wav.shape[0] < self.hp.train.min_audio_len:
                mixed_wav_padding = np.concatenate([mixed_wav, np.zeros(self.hp.train.min_audio_len - mixed_wav.shape[0])], axis=0)

            mixed_spec = self.audio.wav2spec(mixed_wav)

            mixed_spec_padding = self.audio.wav2spec(mixed_wav_padding)


            target_spec = torch.from_numpy(target_spec).float()
            mixed_spec = torch.from_numpy(mixed_spec).float()

            mixed_wav = torch.from_numpy(mixed_wav).float()
            mixed_spec_padding = torch.from_numpy(mixed_spec_padding).float()

            return target_wav, mixed_wav, target_spec, mixed_spec, mixed_spec_padding

