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
    parser.add_argument('-b', '--base_path', type=str, default='pesq')

    args = parser.parse_args()

    f = open(args.ctrl_path, 'r')
    lines = f.read().splitlines()
    f.close()
    
    output_full = args.base_path + '/' + args.out_path.split('/')[-1] 
    os.makedirs(output_full, exist_ok=True)
 
    avg = []


    for name in lines:
        _, ref = wavfile.read(args.clean_wav_path + '/' + name)
        _, deg = wavfile.read(args.est_wav_path + '/' + name)
        pesq_res = pesq(args.sampling_rate, ref, deg, 'wb')
        avg.append(pesq_res)

        with open(output_full + '/indiv.txt', 'a') as f:
            f.write(name + " " + str(pesq_res) + "\n")

    with open(output_full + '/avg.txt', 'w') as f:
        f.write('Total avg : ' + str(np.mean(np.array(avg))))
