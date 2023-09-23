from scipy.io import wavfile
from pesq import pesq
import numpy as np
import os
import argparse
from args_to_list import arg_as_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--clean_wav_path', type=str, required=True,
                        help="path of clean wav files.")
    parser.add_argument('-e', '--est_wav_path', type=str, required=True,
                        help="path of enhanced wav files.")
    parser.add_argument('-c', '--ctrl_path', type=str, required=True,
                        help="list of wav files to be evaluated")
    parser.add_argument('-o', '--out_path', type=str, required=True,
                        help="path to save evaluation results.")
    parser.add_argument('-sr', '--sampling_rate', type=int, default=16000,
                        help="sampling rate of speech to be evaluated.")
    parser.add_argument('-n', '--noise_type', type=arg_as_list, default="[\'drm\', \'mus\', \'new\', \'spo\']",
                        help="list of noise data types.")
    parser.add_argument('-s', '--snr_type', type=arg_as_list, default=[5, 0, -5],
                        help="list of SNR types.")
    args = parser.parse_args()

    f = open(args.ctrl_path, 'r')
    lines = f.read().splitlines()
    f.close()

    os.makedirs(args.out_path, exist_ok=True)

    avg = []
    score = [[] for _ in range(len(args.noise_type) * len(args.snr_type))]
    index = 0
    j = 0
    snr_mean = [[] for _ in range(len(args.snr_type))]

    for n in args.noise_type:
        for s in args.snr_type:
            for name in lines:
                _, ref = wavfile.read(args.clean_wav_path + '/' + name + '.wav')
                _, deg = wavfile.read(args.est_wav_path + '/' + n + '/SNR' + str(s) + '/' + name + '_est.wav')
                score[index].append(pesq(args.sampling_rate, ref, deg, 'wb'))
            avg.append(float(np.mean(score[index])))
            snr_mean[j % len(args.snr_type)].append(float(np.mean(score[index])))
            with open(args.out_path + '/' + n + '_SNR' + str(s) + '.txt', 'w') as f:
                f.write(str(score[index]))
            with open(args.out_path + '/avg.txt', 'a') as f:
                f.write(n + '/SNR' + str(s) + " : " + str(avg[index]) + '\n')
            index += 1
            j += 1

    index = 0
    with open(args.out_path + '/avg.txt', 'a') as f:
        for s in args.snr_type:
            f.write('\nSNR' + str(s) + ' : ' + str(np.mean(snr_mean[index])))
            index += 1
        f.write('\nTotal avg : ' + str(np.mean(avg)))