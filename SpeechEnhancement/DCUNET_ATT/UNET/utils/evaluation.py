import torch
import torch.nn as nn
from mir_eval.separation import bss_eval_sources
import numpy as np


def validate(hp, audio, model, testloader, writer, step):
    model.eval()
    test_list=[]
    sdr_list=[]
    criterion = nn.MSELoss()

    with torch.no_grad():
        for batch in testloader:
            target_wav, mixed_wav, target_mag, mixed_mag, mixed_phase, mixed_padding_mag = batch[0]

            mixed_padding_mag = mixed_padding_mag.unsqueeze(0).cuda()
            target_mag = target_mag.unsqueeze(0).cuda()
            est_mask = model(mixed_padding_mag)

            est_mag = est_mask * mixed_padding_mag

            est_mag = est_mag[:, :, :target_mag.shape[2]]

            test_loss = criterion(target_mag, est_mag).item()
            test_list.append(test_loss)

            target_mag = target_mag[0].cpu().detach().numpy()

            est_mag = est_mag[0].cpu().detach().numpy()
            est_wav = audio.spec2wav(est_mag, mixed_phase)

            est_mask = est_mask[0].cpu().detach().numpy()

            if len(est_wav) >= len(target_wav):
              est_wav = est_wav[0:len(target_wav)]
            else:
              target_wav = target_wav[0:len(est_wav)]

            sdr = bss_eval_sources(target_wav, est_wav, False)[0][0]
            sdr_list.append(sdr)

        test_loss=np.mean(test_list)
        sdr=np.mean(sdr_list)
        writer.log_evaluation(test_loss, sdr,
                                  mixed_wav, target_wav, est_wav,
                                  mixed_mag, target_mag, est_mag, est_mask,
                                  step)

    model.train()
