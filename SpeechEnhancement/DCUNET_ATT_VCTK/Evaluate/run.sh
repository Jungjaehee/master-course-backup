#!/bin/bash

data_dir_name=""
step=0

base_dir=~~/DB/VCTK

clean_wav_path=$base_dir/original_wav/clean_testset_wav_16KHz
est_wav_path=$base_dir/est_wavs/$data_dir_name

ctrl_path=$base_dir/CTRL/test.ctrl

# step 1. SDR
if [ $step -eq 1 -o $step -eq 0 ]; then
  echo "Calculate SDR"
  python3 calculate_sdr.py \
    -r $clean_wav_path \
    -e $est_wav_path \
    -c $ctrl_path \
    -o $data_dir_name || exit 1
fi


# step 2. PESQ
if [ $step -eq 2 -o $step -eq 0 ]; then
  echo "Calculate PESQ"
  python3 calculate_pesq.py \
    -r $clean_wav_path \
    -e $est_wav_path \
    -c $ctrl_path \
    -o $data_dir_name || exit 1
fi


# step 3. STOI
if [ $step -eq 3 -o $step -eq 0 ]; then
  echo "Calculate STOI"
  python3 calculate_stoi.py \
    -r $clean_wav_path \
    -e $est_wav_path \
    -c $ctrl_path \
    -o $data_dir_name || exit 1
fi
