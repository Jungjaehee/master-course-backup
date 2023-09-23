import os
import torch
import librosa
import logging
from mir_eval.separation import bss_eval_sources
import argparse
import numpy as np

from scipy.io.wavfile import write
from utils.audio import Audio
from utils.hparams import HParam
from MODEL.model import UNET

def main(args, hp, ref):
    os.makedirs(args.output_data_dir, exist_ok=True)

    with torch.no_grad():
        model = UNET().cuda()

        chkpt_model = torch.load(args.checkpoint_path)['model']
        
        model.load_state_dict(chkpt_model)
        model.eval()

        audio = Audio(hp)
        for i in range(len(ref)):
            mix = ref[i]

            mixed_wav, _ = librosa.load(os.path.join(args.data_dir, mix), sr=hp.audio.sample_rate)

            mixed_padding_mag, mixed_padding_phase = audio.wav2spec(np.concatenate([mixed_wav, np.zeros(
                hp.train.max_audio_len - mixed_wav.shape[0])], axis=0))
            mag, phase = audio.wav2spec(mixed_wav)
            mixed_padding_mag = torch.from_numpy(mixed_padding_mag).float().unsqueeze(0).cuda()

            est_mask = model(mixed_padding_mag)

            est_mag = est_mask * mixed_padding_mag
            est_mag = est_mag[0]
            est_mag = est_mag[:, :mag.shape[1]].cpu().detach().numpy()

            est_wav = audio.spec2wav(est_mag, phase)

            target = ('/').join(ref[i].split('/')[-2:])
            target = "clean/" + target
            target_wav, _ = librosa.load(os.path.join(args.data_dir, target), sr=hp.audio.sample_rate)

            if len(est_wav) >= len(target_wav):
                est_wav = est_wav[0:len(target_wav)]
            else:
                target_wav = target_wav[0:len(est_wav)]

            out_path = os.path.join(args.output_data_dir, ref[i].replace('.wav', '_est.wav'))
            dir = out_path.split('/')
            dir = '/'.join(dir[:-1])
            os.makedirs(dir, exist_ok=True)
            scaled = np.int16(est_wav / np.max(np.abs(est_wav)) * 32767)
            print(out_path)
            write(out_path, rate=hp.audio.sample_rate, data=scaled)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_type', type=str, required=True,
                        help="model type")
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for configuration")
    parser.add_argument('-p', '--checkpoint_path', type=str, default=None,
                        help="path of checkpoint pt file")
    parser.add_argument('-d', '--data_dir', type=str, required=True,
                        help='path of mixed wav file')
    parser.add_argument('-l', '--ctrl_dir', type=str, required=True,
                        help='path of reference wav file')
    parser.add_argument('-o', '--output_data_dir', type=str, required=True,
                        help='')

    args = parser.parse_args()
    hp = HParam(args.config)

    ref_list = []
    with open(args.ctrl_dir, 'r') as f:
        for file in f:
            ref_list.append(file.strip() + '.wav')
    print('reference wav list num : %d\n' % len(ref_list))

    main(args, hp, ref_list)
