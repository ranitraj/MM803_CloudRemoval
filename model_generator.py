import torch
import torch.nn as nn


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downward=True, activation="relu", dropout=False):
        """
         Creates the UNet Block used for the Generator in Pix2Pix.

        :param in_channel: Number of input channels supplied
        :param out_channels: Number of output channels supplied
        :param downward: Boolean flag which is True in case of Encoder, and False in case of Decoder
        :param activation: Activation function used
        :param dropout: Boolean flag stating if dropout is used
        """
        super().__init__()

        # Creating a Convolution layer
        if downward:
            # Encoder part of the UNet
            if activation == "relu":
                self.convolution = nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                        padding_mode="reflect"
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()
                )
            else:
                self.convolution = nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                        padding_mode="reflect"
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2)
                )
        else:
            # Decoder part of the UNet
            if activation == "relu":
                self.convolution = nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels,
                        out_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()
                )
            else:
                self.convolution = nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels,
                        out_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2)
                )
