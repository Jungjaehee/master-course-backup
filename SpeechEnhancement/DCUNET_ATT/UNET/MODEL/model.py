import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, padding_mode="zeros"):
        super().__init__()

        if padding is None:
            padding = [(i - 1) // 2 for i in kernel_size]  # 'SAME' padding

        conv = nn.Conv2d
        bn = nn.BatchNorm2d
        act = nn.ReLU

        self.conv = conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode)
        self.bn = bn(out_channels)
        self.act = act()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=(0, 0), output_padding=(0, 0), isLast=False):
        super().__init__()

        tconv = nn.ConvTranspose2d
        bn = nn.BatchNorm2d

        if isLast:
            act = nn.Sigmoid
        else:
            act = nn.ReLU

        self.transconv = tconv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bn = bn(out_channels)
        self.act = act()

    def forward(self, x):
        x = self.transconv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        # encoder gets a noisy signal as input [B x 16 x 256 x 1]
        self.set_param()

        self.encoders = []
        self.model_len = len(self.enc_channels)-1

        for i in range(self.model_len):
            module = Encoder(in_channels=self.enc_channels[i], out_channels=self.enc_channels[i+1], kernel_size=self.enc_kernel[i],
                             stride=self.enc_stride[i], padding=self.enc_padding[i])
            self.add_module("encoder{}".format(i), module)
            self.encoders.append(module)

            if i == self.model_len-1:
                break

        self.decoders = []
        for i in range(self.model_len-1):
            in_channel = self.dec_channels[i] + self.enc_channels[self.model_len - i]
            module = Decoder(in_channels=in_channel, out_channels=self.dec_channels[i+1],
                             kernel_size=self.dec_kernel[i], stride=self.dec_stride[i], padding=self.dec_padding[i],
                             output_padding=self.dec_output_padding[i])
            self.add_module("decoder{}".format(i), module)
            self.decoders.append(module)

        i = self.model_len-1
        module = Decoder(in_channels=self.dec_channels[i] + self.enc_channels[self.model_len - i],
                         out_channels=self.dec_channels[i + 1],
                         kernel_size=self.dec_kernel[i], stride=self.dec_stride[i], padding=self.dec_padding[i],
                         output_padding=self.dec_output_padding[i], isLast=True)
        self.add_module("decoder{}".format(i), module)
        self.decoders.append(module)

    def forward(self, spec):
        """
        Forward pass of generator.
        Args:
            x: input batch (signal)
        """
        x = spec.unsqueeze(1)
        xs = []

        for i, encoder in enumerate(self.encoders):
            xs.append(x)
            x = encoder(x)
            # print(x.shape)

        p = x.clone()

        for i, decoder in enumerate(self.decoders):
            p = decoder(p)
            # print(p.shape)
            if i == self.model_len-1:
                break

            p = torch.cat([p, xs[self.model_len - i - 1]], dim=1)

        mask = p.squeeze(1)
        return mask

    def set_param(self, input_channels=1):
        self.enc_channels = [input_channels, 45, 45, 90, 90, 90, 90, 90, 90]
        self.enc_kernel = [(7, 5), (7, 5), (7, 5), (5, 3), (5, 3), (5, 3), (5, 3), (5, 3)]
        self.enc_stride = [(2, 2), (2, 1), (2, 2), (2, 1), (2, 2), (2, 1), (2, 2), (2, 1)]
        self.enc_padding = [(2, 1), (3, 2), (3, 2), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1)]

        self.dec_channels = [0, 90, 90, 90, 90, 90, 45, 45, 1]
        self.dec_kernel = [(5, 3), (5, 3), (5, 3), (5, 3), (5, 3), (7, 5), (7, 5), (7, 5)]
        self.dec_stride = [(2, 1), (2, 2), (2, 1), (2, 2), (2, 1), (2, 2), (2, 1), (2, 2)]
        self.dec_padding = [(2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (3, 2), (3, 2), (2, 1)]
        self.dec_output_padding = [(1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (0, 1)]
