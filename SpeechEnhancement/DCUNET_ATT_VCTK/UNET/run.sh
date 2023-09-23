#!/bin/bash

config="config/config.yaml"
cpu_num="24"
step=2
model_type=UNET_max_len

# step 1. train
if [ $step -eq 1 ]; then
  python3 trainer.py \
    -c $config \
    -m $model_type  || exit 1
fi


# step 2. inference
if [ $step -eq 2 ]; then
  test_ctrl=../ctrl/vctk_test.ctl
  data_dir=/media/jaehee/LargeDB/VCTK/original_wav/noisy_testset_wav_16KHz
  checkpoint_path=chkpt/$model_type/chkpt_68000.pt

  # sdr_res=sdr_res/$model_type
  output_data_dir=/media/jaehee/LargeDB/VCTK/est_wavs/${model_type}_68k
  python3 inference.py \
    -m $model_type \
    -c $config \
    -d $data_dir \
    -p $checkpoint_path \
    -l $test_ctrl \
    -o $output_data_dir || exit 1
fi

