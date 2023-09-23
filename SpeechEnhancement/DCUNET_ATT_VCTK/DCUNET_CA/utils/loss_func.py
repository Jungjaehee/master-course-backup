import torch

class SI_SNR:
    def __init__(self):
        self.loss = self.si_snr

    def __call__(self, clean_wav, enhanced_wav):
        return self.loss(clean_wav, enhanced_wav)

    def si_snr(self, clean, enhanced):
        snr = torch.zeros(enhanced.shape[0]).cuda()
        for i in range(enhanced.shape[0]):
            target = (torch.dot(clean[i], enhanced[i])*clean[i])/torch.norm(clean[i])**2
            noise = enhanced[i] - target[i]
            snr[i] += 10*torch.log10((torch.norm(target)**2)/torch.norm(noise)**2)
        return -torch.mean(snr)


class Weighted_SDR:
    def __init__(self):
        self.loss = self.w_sdr

    def __call__(self, clean_wav, enhanced_wav):
        return self.loss(clean_wav, enhanced_wav)

    def w_sdr(self, clean, enhanced):
        snr = torch.zeros(enhanced.shape[0]).cuda()
        for i in range(enhanced.shape[0]):
            # z_hat = mixed[i] - enhanced[i]

            # alpha = torch.norm(clean[i])**2/(torch.norm(clean[i])**2 + torch.norm(z_hat)**2)

            # alpha * self.sdr(clean, enhanced) + (1-alpha)*self.sdr()

            snr[i] += self.sdr(clean[i], enhanced[i])
        return -torch.mean(snr)

    def sdr(self, clean_wav, enhanced_wav):
        denom = torch.dot(clean_wav, enhanced_wav)
        numer = torch.norm(clean_wav)*torch.norm(enhanced_wav)
        return -(denom**2/numer**2)