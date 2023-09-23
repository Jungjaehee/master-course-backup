import os
import torch
import librosa
import argparse
import numpy as np

from scipy.io.wavfile import write
from utils.audio import Audio
from utils.hparams import HParam
from MODEL.model import DCUNET_CA

def main(args, hp, ref):
    os.makedirs(args.output_data_dir, exist_ok=True)

    with torch.no_grad():
        if args.model_type == 'DCUNET':
            model = DCUNET_CA(isComplex=True, isAttention=False).cuda()
        elif args.model_type == 'ATTENTION':
            model = DCUNET_CA(isComplex=True, isAttention=True).cuda()
        else:
            raise Exception("Please check your model name %s \n"
                            "Choice within [DCUNET or ATTENTION]" % args.model_type)

        chkpt_model = torch.load(args.checkpoint_path)['model']
        
        model.load_state_dict(chkpt_model)
        model.eval()

        audio = Audio(hp)
        for i in range(len(ref)):
            mix = ref[i]
            mixed_wav, _ = librosa.load(os.path.join(args.data_dir, mix), sr=hp.audio.sample_rate)

            # Perform zero padding to match the length used during training
            mixed_spec_padding = audio.wav2spec(np.concatenate([mixed_wav, np.zeros(hp.train.max_audio_len - mixed_wav.shape[0])],
                                                                    axis=0))
            mixed_spec_padding = torch.from_numpy(mixed_spec_padding).float().unsqueeze(0).cuda()

            est_mask = model(mixed_spec_padding)

            mixed_mag, mixed_phase = audio.spec2magphase_torch(mixed_spec_padding)
            mask_mag, mask_phase = audio.spec2magphase_torch(est_mask)

            output_mag = mixed_mag * mask_mag
            output_phase = mixed_phase + mask_phase

            est_wav = audio.spec2wav(output_mag, output_phase)

            target = ('/').join(ref[i].split('/')[-2:])
            target = "clean/" + target
            target_wav, _ = librosa.load(os.path.join(args.data_dir, target), sr=hp.audio.sample_rate)

            if est_wav.shape[1] >= len(target_wav):
              est_wav = est_wav[:, 0:len(target_wav)]
            else:
              target_wav = target_wav[0:est_wav.shape]

            est_wav = est_wav[0].cpu().detach().numpy()

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
                        help="type of model used.")
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for configuration.")
    parser.add_argument('-p', '--checkpoint_path', type=str, default=None,
                        help="path of checkpoint pt file.")
    parser.add_argument('-d', '--data_dir', type=str, required=True,
                        help='path of mixed wav file')
    parser.add_argument('-l', '--ctrl_dir', type=str, required=True,
                        help='list of wav files to be tested; including noise folder and SNR folder path.')
    parser.add_argument('-o', '--output_data_dir', type=str, required=True,
                        help='path to store enhanced voice.')

    args = parser.parse_args()
    hp = HParam(args.config)

    ref_list = []
    with open(args.ctrl_dir, 'r') as f:
        for file in f:
            ref_list.append(file.strip() + '.wav')
    print('reference wav list num : %d\n' % len(ref_list))

    main(args, hp, ref_list)
