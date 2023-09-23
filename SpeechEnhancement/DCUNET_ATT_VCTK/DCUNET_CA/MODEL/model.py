import torch

import torch.nn as nn
import MODEL.complex_nn as complex_nn


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, isComplex=True, padding_mode="zeros"):
        super().__init__()

        if padding is None:
            padding = [(i - 1) // 2 for i in kernel_size]  # 'SAME' padding

        if isComplex:
            conv = complex_nn.ComplexConv2d
            bn = complex_nn.ComplexBatchNorm2d
            act = complex_nn.ComplexActivation
        else:
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
            act = nn.LeakyReLU

        self.conv = conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode)
        self.bn = bn(out_channels)
        self.act = act()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=(0, 0), output_padding=(0, 0), isComplex=True, isLast=False):
        super().__init__()

        if isComplex:
            tconv = complex_nn.ComplexConvTranspose2d
            bn = complex_nn.ComplexBatchNorm2d
            act = complex_nn.ComplexActivation
            self.act = act(isLast=isLast)

        else:
            tconv = nn.ConvTranspose2d
            bn = nn.BatchNorm2d
            act = nn.LeakyReLU
            self.act = act()

        self.transconv = tconv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bn = bn(out_channels)

    def forward(self, x):
        x = self.transconv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class ComplexAttention(nn.Module):
    def __init__(self, channel, feature_shape):
        super().__init__()
        # DCUNET-ATT
        self.att = complex_nn.SkipAttention(channel, feature_shape[0], feature_shape[1])

        # SkipConv
        # self.att = complex_nn.SkipConv(channel, feature_shape[0], feature_shape[1])

        # DCUNET TFSA DE
        # self.att = complex_nn.SelfAttention(channel, feature_shape[0], feature_shape[1])

        # SDAB (only consistent length)
        # self.att = complex_nn.SDAB(channel, feature_shape[0], feature_shape[1])

    def forward(self, P, Q):
        res = self.att(P, Q)  # [batch, 2, channel, frequency, time_frame]
        return res


class DCUNET_CA(nn.Module):
    def __init__(self, audio, isComplex=True, isAttention=True):
        super().__init__()
        self.set_param()
        self.audio = audio
        self.isAttention = isAttention

        self.encoders = []
        self.attentions = []
        self.model_len = len(self.enc_channels)-1

        for i in range(self.model_len):
            module = Encoder(in_channels=self.enc_channels[i], out_channels=self.enc_channels[i+1], kernel_size=self.enc_kernel[i],
                             isComplex=isComplex, stride=self.enc_stride[i], padding=self.enc_padding[i])
            self.add_module("encoder{}".format(i), module)
            self.encoders.append(module)

            if i == self.model_len-1:
                break

            if isAttention:
                att = ComplexAttention(self.enc_channels[self.model_len-i-1], self.feature_shape[i])
                self.add_module("attention{}".format(i), att)
                self.attentions.append(att)

        self.decoders = []
        for i in range(self.model_len-1):
            in_channel = self.dec_channels[i] + self.enc_channels[self.model_len - i]
            module = Decoder(in_channels=in_channel, out_channels=self.dec_channels[i+1], isComplex=isComplex,
                             kernel_size=self.dec_kernel[i], stride=self.dec_stride[i], padding=self.dec_padding[i],
                             output_padding=self.dec_output_padding[i])
            self.add_module("decoder{}".format(i), module)
            self.decoders.append(module)

        i = self.model_len-1
        module = Decoder(in_channels=self.dec_channels[i] + self.enc_channels[self.model_len - i],
                         out_channels=self.dec_channels[i + 1],
                         kernel_size=self.dec_kernel[i], stride=self.dec_stride[i], padding=self.dec_padding[i],
                         output_padding=self.dec_output_padding[i], isComplex=isComplex, isLast=True)
        self.add_module("decoder{}".format(i), module)
        self.decoders.append(module)

    def forward(self, spec):
        """
        Forward pass of generator.
        Args:
            x: input batch (signal)
        """
        # spec : [B, 2, frequency, time_frame]
        x = spec.unsqueeze(2)  # add channel : [B, 2, 1, frequency, time_frame]
        xs = []

        for i, encoder in enumerate(self.encoders):
            xs.append(x)
            x = encoder(x)

        p = x.clone()

        for i, decoder in enumerate(self.decoders):
            p = decoder(p)

            if i == self.model_len -1:   # Skip-connection is not applied to the last layer
                break

            if self.isAttention:  # Skip-connection is applied attention
                att = self.attentions[i](xs[self.model_len - i - 1], p)

                p = torch.cat([p, att], dim=2)

            else:  # Skip-connection (DCUNET)
                p = torch.cat([p, xs[self.model_len - i - 1]], dim=2)

        mask = p.squeeze(2)

        return mask

    def set_param(self, input_channels=1):
        self.enc_channels = [input_channels, 45, 45, 90, 90, 90, 90, 90, 90]
        self.enc_kernel = [(7, 5), (7, 5), (7, 5), (5, 3), (5, 3), (5, 3), (5, 3), (5, 3)]
        self.enc_stride = [(2, 2), (2, 1), (2, 2), (2, 1), (2, 2), (2, 1), (2, 2), (2, 1)]
        self.enc_padding = [(3, 0), (3, 0), (3, 0), (2, 0), (2, 0), (2, 0), (2, 0), (2, 0)]

        self.dec_channels = [0, 90, 90, 90, 90, 90, 45, 45, 1]
        self.dec_kernel = [(5, 3), (5, 3), (5, 3), (5, 3), (5, 3), (7, 5), (7, 5), (7, 5)]
        self.dec_stride = [(2, 1), (2, 2), (2, 1), (2, 2), (2, 1), (2, 2), (2, 1), (2, 2)]
        self.dec_padding = [(2, 0), (2, 0), (2, 0), (2, 0), (2, 0), (3, 0), (3, 0), (3, 0)]
        self.dec_output_padding = [(0, 0), (0, 1), (0, 0), (0, 1), (0, 0), (0, 0), (0, 0), (0, 0)]

        self.feature_shape = [(5, 55), (9, 112), (17, 114), (33, 230), (65, 232), (129, 467), (257, 471)]

