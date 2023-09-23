#!/bin/bash

config="config/config.yaml"
step=1
model_type=DCUNET # [DCUNET or ATTENTION ]

output_model=''

# step 1. train
# If use the checkpoint model : option -p PATH
# EX) -p chkpt/$output_model/chkpt_1000.pt

if [ $step -eq 1 ]; then
  python3 trainer.py \
    -c $config \
    -m $model_type \
    -o $output_model || exit 1
fi

# step 2. inference
if [ $step -eq 2 ]; then
  test_ctrl=../ctrl/vctk_test.ctl
  data_dir=~~/VCTK/original_wav/noisy_testset_wav_16KHz
  checkpoint_path=chkpt/$output_model/chkpt_.pt

  output_data_dir=~~/VCTK/est_wavs/${output_model}
  python3 inference.py \
    -m $model_type \
    -c $config \
    -d $data_dir \
    -p $checkpoint_path \
    -l $test_ctrl \
    -o $output_data_dir || exit 1
fi
