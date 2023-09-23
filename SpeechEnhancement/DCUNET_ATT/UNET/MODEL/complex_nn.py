import torch
import torch.nn as nn


class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super().__init__()

        ## Model components
        self.conv_re = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias, **kwargs)
        self.conv_im = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias, **kwargs)

    def forward(self, x):  # shpae of x : [batch,2,channel, Frequency,Frame]
        part_r = x[:, 0, ...]
        part_i = x[:, 1, ...]
        # print("part : ", part_r.shape)
        real = self.conv_re(part_r) - self.conv_im(part_i)
        imaginary = self.conv_re(part_i) + self.conv_im(part_r)
        output = torch.stack((real, imaginary), dim=1)
        return output


class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super().__init__()

        ## Model components
        self.tconv_re = nn.ConvTranspose2d(in_channels, out_channels,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           output_padding=output_padding,
                                           groups=groups,
                                           bias=bias,
                                           dilation=dilation,
                                           **kwargs)
        self.tconv_im = nn.ConvTranspose2d(in_channels, out_channels,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           output_padding=output_padding,
                                           groups=groups,
                                           bias=bias,
                                           dilation=dilation,
                                           **kwargs)

    def forward(self, x):  # shpae of x : [batch,2,channel, frequency, frame]
        real = self.tconv_re(x[:, 0, ...]) - self.tconv_im(x[:, 1, ...])
        imaginary = self.tconv_re(x[:, 1, ...]) + self.tconv_im(x[:, 0, ...])
        output = torch.stack((real, imaginary), dim=1)
        return output


class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, **kwargs):
        super().__init__()
        self.bn_re = nn.BatchNorm2d(num_features=num_features, momentum=momentum, affine=affine, eps=eps, track_running_stats=track_running_stats, **kwargs)
        self.bn_im = nn.BatchNorm2d(num_features=num_features, momentum=momentum, affine=affine, eps=eps, track_running_stats=track_running_stats, **kwargs)

    def forward(self, x):
        real = self.bn_re(x[:, 0, ...])
        imag = self.bn_im(x[:, 1, ...])
        output = torch.stack((real, imag), dim=1)
        return output


class ComplexActivation(nn.Module):
    def __init__(self, isLast=False):
        super().__init__()
        if isLast:
            self.act_real = nn.Tanh()
            self.act_imag = nn.Tanh()

        else:
            self.act_real = nn.LeakyReLU()
            self.act_imag = nn.LeakyReLU()

    def forward(self, x):
        real = self.act_real(x[:, 0, ...])
        imag = self.act_imag(x[:, 1, ...])
        output = torch.stack((real, imag), dim=1)
        return output