#!/bin/bash

config="config/config.yaml"
cpu_num="24"
step=1
model_type=UNET

# step 1. train
if [ $step -eq 1 ]; then
  python3 trainer.py \
    -c $config \
    -m $model_type  || exit 1
fi


# step 2. inference
if [ $step -eq 2 ]; then
  test_ctrl=../ctrl/timit_test_full_path.fileids
  data_dir=~~/TIMIT16K/origin/TIMIT16K
  checkpoint_path=chkpt/$model_type/chkpt_.pt

  output_data_dir=~~/TIMIT16K/$model_type

  python3 inference.py \
    -m $model_type \
    -c $config \
    -d $data_dir \
    -p $checkpoint_path \
    -l $test_ctrl \
    -o $output_data_dir || exit 1
fi

