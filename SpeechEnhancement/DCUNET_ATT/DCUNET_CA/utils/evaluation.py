import torch
import torch.nn as nn
from mir_eval.separation import bss_eval_sources
import numpy as np
from utils.loss_func import SI_SNR

def validate(hp, model_type, audio, model, testloader, writer, step, device):
    model.eval()
    test_list=[]
    sdr_list=[]
    mse = nn.MSELoss()
    si_snr = SI_SNR()

    with torch.no_grad():
        for batch in testloader:
            target_wav, mixed_wav, target_spec, mixed_spec, mixed_spec_padding = batch[0]
            mixed_spec_padding = mixed_spec_padding.unsqueeze(0).to(device)
            est_mask = model(mixed_spec_padding)

            mixed_mag, mixed_phase = audio.spec2magphase_torch(mixed_spec_padding)
            mask_mag, mask_phase = audio.spec2magphase_torch(est_mask)

            output_mag = mixed_mag * mask_mag
            output_phase = mixed_phase + mask_phase
            
            spec_real = output_mag * torch.cos(output_phase)
            spec_imag = output_mag * torch.sin(output_phase)
            est_spec = torch.stack([spec_real, spec_imag], dim=1)


            target_mag, _ = audio.spec2magphase_torch(target_spec.unsqueeze(0).to(device))

            enhanced_wav = audio.spec2wav(output_mag, output_phase)
            target_wav_torch = torch.from_numpy(target_wav).float().to(device).unsqueeze(0)

            if enhanced_wav.shape[1] >= target_wav_torch.shape[1]:
                enhanced_wav = enhanced_wav[:, 0:target_wav_torch.shape[1]]
                output_mag = output_mag[:, :, :target_mag.shape[2]]
            else:
                target_wav_torch = target_wav_torch[:, 0:enhanced_wav.shape[1]]
                target_mag = target_mag[:, :, :output_mag.shape[2]]

            test_loss = si_snr(target_wav_torch, enhanced_wav)
            # test_loss = si_snr(target_wav_torch, enhanced_wav) + mse(output_mag, target_mag.to(device))

            test_loss = test_loss.item()
            test_list.append(test_loss)

            mixed_mag, _ = audio.spec2magphase_torch(mixed_spec.to(device).unsqueeze(0))

            mixed_mag = mixed_mag[0].cpu().detach().numpy()

            target_mag = target_mag[0].cpu().detach().numpy()
            output_mag = output_mag[0, :, :target_mag.shape[1]].cpu().detach().numpy()

            mask_mag = mask_mag[0, :, :target_mag.shape[1]].cpu().detach().numpy()

            enhanced_wav = enhanced_wav[0].cpu().detach().numpy()
            sdr = bss_eval_sources(target_wav, enhanced_wav, False)[0][0]
            sdr_list.append(sdr)

        test_loss=np.mean(test_list)
        sdr=np.mean(sdr_list)
        writer.log_evaluation(test_loss, sdr, mixed_mag, target_mag, output_mag, mask_mag,
                                  mixed_wav, target_wav, enhanced_wav,
                                  step)

    model.train()
    return test_loss
