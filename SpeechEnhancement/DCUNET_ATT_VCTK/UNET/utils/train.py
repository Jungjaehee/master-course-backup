import os
import math
import torch
import torch.nn as nn
import traceback
from utils.audio import Audio
from utils.evaluation import validate
from MODEL.model import UNET

def train(args, pt_dir, chkpt_path, trainloader, testloader, writer, logger, hp, hp_str):
    audio = Audio(hp)
    model = UNET().cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=hp.train.adam)

    step = 0
    # chkpt_path = "./chkpt/UNET/chkpt_.pt"
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
        criterion = nn.MSELoss()
        while True:
            model.train()
            for target_mag, mixed_mag in trainloader:

                target_mag = target_mag.cuda()
                mixed_mag = mixed_mag.cuda()

                mask = model(mixed_mag)
                output = mixed_mag * mask

                loss = criterion(output, target_mag)

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
                    validate(hp, audio, model, testloader, writer, step)
    except Exception as e:
        logger.info("Exiting due to exception: %s" % e)
        traceback.print_exc()
