# adapted from Keith Ito's tacotron implementation
# https://github.com/keithito/tacotron/blob/master/util/audio.py

import librosa
import numpy as np


class Audio():
    def __init__(self, hp):
        self.hp = hp

    def wav2spec(self, y):
        D = self.stft(y)
        S = self.amp_to_db(np.abs(D)) - self.hp.audio.ref_level_db
        S, D = self.normalize(S), np.angle(D)

        return S, D

    def spec2wav(self, spectrogram, phase):
        S = self.db_to_amp(self.denormalize(spectrogram) + self.hp.audio.ref_level_db)
        return self.istft(S, phase)

    def stft(self, y):
        return librosa.stft(y=y, n_fft=self.hp.audio.n_fft,
                            hop_length=self.hp.audio.hop_length,
                            win_length=self.hp.audio.win_length)

    def istft(self, mag, phase):
        stft_matrix = mag * np.exp(1j*phase)
        return librosa.istft(stft_matrix,
                             hop_length=self.hp.audio.hop_length,
                             win_length=self.hp.audio.win_length)

    def amp_to_db(self, x):
        return 20.0 * np.log10(np.maximum(1e-5, x))

    def db_to_amp(self, x):
        return np.power(10.0, x * 0.05)

    def normalize(self, S):
        return np.clip(S / -self.hp.audio.min_level_db, -1.0, 0.0) + 1.0

    def denormalize(self, S):
        return (np.clip(S, 0.0, 1.0) - 1.0) * -self.hp.audio.min_level_db