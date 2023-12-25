import torch
import torch.nn as nn


def conv_plus_conv(in_channels, out_channels):
    """
    Makes UNet block
    :param in_channels: input channels
    :param out_channels: output channels
    :return: UNet block
    """
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        ),
        nn.BatchNorm2d(num_features=out_channels),
        nn.LeakyReLU(0.2),
        nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        ),
        nn.BatchNorm2d(num_features=out_channels),
        nn.LeakyReLU(0.2),
    )


class UNET(nn.Module):
    def __init__(self):
        super().__init__()

        base_channels = 16

        self.down1 = conv_plus_conv(3, base_channels)
        self.down2 = conv_plus_conv(base_channels, base_channels * 2)

        self.up1 = conv_plus_conv(base_channels * 2, base_channels)
        self.up2 = conv_plus_conv(base_channels * 4, base_channels)

        self.bottleneck = conv_plus_conv(base_channels * 2, base_channels * 2)

        self.out = nn.Conv2d(in_channels=base_channels, out_channels=1, kernel_size=1)

        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, inp):

        residual1 = self.down1(inp)
        out = self.downsample(residual1)

        residual2 = self.down2(out)
        out = self.downsample(residual2)

        out = self.bottleneck(out)

        out = nn.functional.interpolate(out, scale_factor=2)
        out = torch.cat((out, residual2), dim=1)
        out = self.up2(out)

        out = nn.functional.interpolate(out, scale_factor=2)
        out = torch.cat((out, residual1), dim=1)
        out = self.up1(out)

        out = self.out(out)

        return out
