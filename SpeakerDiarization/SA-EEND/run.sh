#!/bin/bash

ulimit -S -n 4096
export PYTHONPATH=`pwd`:$PYTHONPATH

exp_dir=exp

# Config PATH
conf_dir=conf/base_16
train_conf=$conf_dir/train.yaml
adapt_conf=$conf_dir/adapt.yaml
infer_conf=$conf_dir/infer.yaml


# DATA PATH (wav.scp dir PATH)
train_dir=''
dev_dir=''
test_dir=''
train_adapt_dir=''
dev_adapt_dir=''
test_adapt_dir=''


# MODEL PATH
model_dir=$exp_dir/models
model_adapt_dir=$exp_dir/models_adapt

mkdir -p $model_dir
mkdir -p $model_adapt_dir


test_model=$model_dir/avg.th
test_adapt_model=$model_adapt_dir/avg.th

# OUTPUT PATH
infer_out_dir=$exp_dir/infer/simu_ns2_beta2
work=$infer_out_dir/.work
scoring_dir=$exp_dir/score/simu_ns2_beta2_avg

adapt_infer_out_dir=$exp_dir/infer/real_ns2
adapt_work=$adapt_infer_out_dir/.work
adapt_scoring_dir=$exp_dir/score/real_ns2_avg

stage=1

# Training
if [ $stage -le 1 ]; then
    echo "Start training"
    # SA-EEND
    python eend/bin/train.py -c $train_conf $train_dir $dev_dir $model_dir || exit 1

    # SA-EEND with speaker loss
    # python eend/bin/train_spk_loss.py -c $train_conf $train_dir $dev_dir $model_dir || exit 1
fi


# Model averaging
if [ $stage -le 2 ]; then
    echo "Start model averaging"
    ifiles=`eval echo $model_dir/transformer{91..100}.th`
    python eend/bin/model_averaging.py $test_model $ifiles
fi


# Inferring
if [ $stage -le 3 ]; then
    echo "Start inferring"
    echo $test_dir
    # SA-EEND
    python eend/bin/infer.py -c $infer_conf $test_dir $test_model $infer_out_dir || exit 1

    # SA-EEND with speaker loss
    # python eend/bin/infer_spk_loss.py -c $infer_conf $test_dir $test_model $infer_out_dir  || exit 1
fi

# Scoring
if [ $stage -le 4 ]; then
    echo "Start scoring"
    mkdir -p $work
    mkdir -p $scoring_dir
	  find $infer_out_dir -iname "*.h5" > $work/file_list
	  for med in 11 1; do   # med: median filter length
      for th in 0.3 0.4 0.5 0.6 0.7; do    # th: threshold
        python eend/bin/make_rttm.py --median=$med --threshold=$th \
          --frame_shift=160 --subsampling=10 --sampling_rate=16000 \
          $work/file_list $scoring_dir/hyp_${th}_$med.rttm
        ./md-eval.pl -c 0.25 \
          -r $test_dir/rttm \
          -s $scoring_dir/hyp_${th}_$med.rttm > $scoring_dir/result_th${th}_med${med}_collar0.25 2>/dev/null || exit
      done
	  done
fi

# INIT MODEL PATH
init_model=$model_dir/avg.th

# Adapting
if [ $stage -le 5 ]; then
    echo "Start adapting"
    # SA-EEND
    python eend/bin/train.py -c $adapt_conf $train_adapt_dir $dev_adapt_dir $model_adapt_dir --initmodel $init_model || exit 1

    # SA-EEND with speaker loss
    # python eend/bin/train_spk_loss.py -c $adapt_conf $train_adapt_dir $dev_adapt_dir $model_adapt_dir --initmodel $init_model || exit 1
fi

# Model averaging
if [ $stage -le 6 ]; then
   echo "Start model averaging"
   ifiles=`eval echo $model_adapt_dir/transformer{91..100}.th`
   python eend/bin/model_averaging.py $test_adapt_model $ifiles || exit 1
fi


# Inferring
if [ $stage -le 7 ]; then
    echo "Start inferring"

    # SA-EEND
    python eend/bin/infer.py -c $infer_conf $test_adapt_dir $test_adapt_model $adapt_infer_out_dir  || exit 1

    # SA-EEND with speaker loss
    # python eend/bin/infer_spk_loss.py -c $infer_conf $test_adapt_dir $test_adapt_model $adapt_infer_out_dir  || exit 1
fi

# Scoring
if [ $stage -le 8 ]; then
    echo "Start scoring"
    mkdir -p $adapt_work
    mkdir -p $adapt_scoring_dir
	find $adapt_infer_out_dir -iname "*.h5" > $adapt_work/file_list
	for med in 11 1; do
	for th in 0.3 0.4 0.5 0.6 0.7; do
	python eend/bin/make_rttm.py --median=$med --threshold=$th \
		--frame_shift=160 --subsampling=10 --sampling_rate=16000 \
		$adapt_work/file_list $adapt_scoring_dir/hyp_${th}_$med.rttm  || exit 1
	./md-eval.pl -c 0.25 \
		-r $test_adapt_dir/rttm \
		-s $adapt_scoring_dir/hyp_${th}_$med.rttm > $adapt_scoring_dir/result_th${th}_med${med}_collar0.25 2>/dev/null || exit
	done
	done
fi
