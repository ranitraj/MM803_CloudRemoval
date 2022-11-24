import torch
import torch.nn as nn


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_downward=True, activation="relu", is_using_dropout=False):
        """
         Creates the UNet Block used for the Generator in Pix2Pix.

        :param in_channels: Number of input channels supplied
        :param out_channels: Number of output channels supplied
        :param is_downward: Boolean flag which is True in case of Encoder, and False in case of Decoder
        :param activation: Activation function used
        :param is_using_dropout: Boolean flag stating if dropout is used
        """
        super().__init__()

        # Creating a Convolution layer
        if is_downward:
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

        # Setting the Dropout
        self.is_using_dropout = is_using_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, _input):
        _input = self.convolution(_input)
        if self.is_using_dropout:
            return self.dropout(_input)
        else:
            return _input
