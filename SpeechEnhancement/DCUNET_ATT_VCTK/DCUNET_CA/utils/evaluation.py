import torch
import torch.nn as nn
from mir_eval.separation import bss_eval_sources
import numpy as np
from utils.loss_func import SI_SNR

def validate(hp, model_type, audio, model, testloader, writer, step):
    model.eval()
    test_list=[]
    sdr_list=[]
    si_snr = SI_SNR()
    with torch.no_grad():
        for batch in testloader:
            target_wav, mixed_wav, target_spec, mixed_spec, mixed_spec_padding = batch[0]

            mixed_spec_padding = mixed_spec_padding.unsqueeze(0).cuda()

            est_mask = model(mixed_spec_padding)

            mixed_mag, mixed_phase = audio.spec2magphase_torch(mixed_spec_padding)
            mask_mag, mask_phase = audio.spec2magphase_torch(est_mask)

            output_mag = mixed_mag * mask_mag
            output_phase = mixed_phase + mask_phase

            target_mag, _ = audio.spec2magphase_torch(target_spec.unsqueeze(0).cuda())

            enhanced_wav = audio.spec2wav(output_mag, output_phase)


            target_wav_torch = torch.from_numpy(target_wav).float().cuda().unsqueeze(0)

            if enhanced_wav.shape[1] >= target_wav_torch.shape[1]:
                enhanced_wav = enhanced_wav[:, 0:target_wav_torch.shape[1]]

            else:
                target_wav_torch = target_wav_torch[:, 0:enhanced_wav.shape[1]]

            test_loss = si_snr(target_wav_torch, enhanced_wav)

            test_loss = test_loss.item()
            test_list.append(test_loss)

            enhanced_wav = enhanced_wav[0].cpu().detach().numpy()

            if enhanced_wav.shape[0] >= target_wav.shape[0]:
                enhanced_wav = enhanced_wav[:target_wav.shape[0]]

            else:
                target_wav = target_wav[:enhanced_wav.shape[0]]

            sdr = bss_eval_sources(target_wav, enhanced_wav, False)[0][0]
            sdr_list.append(sdr)

        test_loss=np.mean(test_list)
        sdr=np.mean(sdr_list)
        writer.log_evaluation(test_loss, sdr, mixed_wav, target_wav, enhanced_wav,
                                  step)

    model.train()
    return test_loss
