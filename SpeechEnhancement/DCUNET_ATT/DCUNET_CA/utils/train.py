import os
import math
import torch
import traceback
import torch.nn as nn
# from parallel import DataParallelModel, Data

from utils.audio import Audio
from utils.evaluation import validate
from MODEL.model import DCUNET_CA
from utils.loss_func import SI_SNR

def train(args, pt_dir, chkpt_path, trainloader, testloader, writer, logger, hp, hp_str):
    audio = Audio(hp)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_type == 'DCUNET':
        model = DCUNET_CA(isComplex=True, isAttention=False).cuda()
    elif args.model_type == 'ATTENTION':
        model = DCUNET_CA(isComplex=True, isAttention=True).cuda()
    else:
        raise Exception("Please check your model name %s \n"
                        "Choice within [DCUNET or ATTENTION]" % args.model_type)

    optimizer = torch.optim.Adam(model.parameters(), lr=hp.train.adam)

    step = 0
    if chkpt_path is not None:
        logger.info("Resuming from checkpoint: %s" % chkpt_path)
        checkpoint = torch.load(chkpt_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        step = checkpoint['step']

        # will use new given hparams.
        if hp_str != checkpoint['hp_str']:
            logger.warning("New hparams is different from checkpoint.")
    else:
        logger.info("Starting new training run")


    try:
        si_snr = SI_SNR()
        mse = nn.MSELoss()

        while True:
            model.train()
            for target_wav, mixed_spec, target_spec in trainloader:
                target_wav = target_wav.to(device)
                mixed_spec = mixed_spec.to(device)
                target_spec = target_spec.to(device)

                target_mag, target_phase = audio.spec2magphase_torch(target_spec)
                mixed_mag, mixed_phase = audio.spec2magphase_torch(mixed_spec)

                mask = model(mixed_spec)

                mask_mag, mask_phase = audio.spec2magphase_torch(mask)

                output_mag = mixed_mag * mask_mag
                output_phase = mixed_phase + mask_phase

                enhanced_wav = audio.spec2wav(output_mag, output_phase)

                if enhanced_wav.shape[1] >= target_wav.shape[1]:
                    enhanced_wav = enhanced_wav[:, 0:target_wav.shape[1]]
                else:
                    target_wav = target_wav[:, 0:enhanced_wav.shape[1]]

                loss = si_snr(target_wav, enhanced_wav)

                # Combine loss
                # loss = si_snr(target_wav, enhanced_wav) + mse(output_mag, target_mag)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1
                loss = loss.item()

                if loss > 1e8 or math.isnan(loss):
                    logger.error("Loss exploded to %.02f at step %d!" % (loss, step))
                    raise Exception("Loss exploded")

                # write loss to tensorboard
                if step % hp.train.summary_interval == 0:
                    writer.log_training(loss, step)
                    logger.info("Wrote summary at step %d" % step)

                # 1. save checkpoint file to resume training
                # 2. evaluate and save sample to tensorboard
                if step % hp.train.checkpoint_interval == 0:
                    save_path = os.path.join(pt_dir, 'chkpt_%d.pt' % step)
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'step': step,
                        'hp_str': hp_str,
                    }, save_path)
                    logger.info("Saved checkpoint to: %s" % save_path)
                    validate(hp, args.output_model, audio, model, testloader, writer, step, device)

    except Exception as e:
        logger.info("Exiting due to exception: %s" % e)
        traceback.print_exc()
