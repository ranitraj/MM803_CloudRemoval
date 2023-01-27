import torch
import torch.nn as nn


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_downward=True, activation="relu", is_dropout=False):
        """
         Creates the UNet Block used for the Generator in Pix2Pix.

        :param in_channels: Number of input channels supplied
        :param out_channels: Number of output channels supplied
        :param is_downward: Boolean flag which is True in case of Encoder, and False in case of Decoder
        :param activation: Activation function used
        :param is_dropout: Boolean flag stating if dropout is used
        """
        super().__init__()

        # Creating a Convolution layer
        if is_downward:
            # Encoder part of the UNet
            if activation == "relu":
                self.convolution = nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
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
                        in_channels=in_channels,
                        out_channels=out_channels,
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
                        in_channels=in_channels,
                        out_channels=out_channels,
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
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2)
                )

        # Setting the Dropout
        self.is_using_dropout = is_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, _input):
        _input = self.convolution(_input)
        if self.is_using_dropout:
            return self.dropout(_input)
        else:
            return _input


class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        """
        Creates the Discriminator model.
            1. Create the first Sequential layer which does not use BatchNorm2D using the first feature '64'
            2. Create remaining 6 downward UNet-Blocks
            3. Create the final UNet-Block
            4. Create total of 7 upward UNet-Blocks

        :param in_channels: Number of input channels supplied
        :param features: Convolution features
        """
        super().__init__()

        # First Sequential 'Downward' convolution layer without BatchNorm2D
        self.first_downward = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect"
            ),
            nn.LeakyReLU(0.2)
        )

        # Creating remaining downward layers
        self.second_downward = UNetBlock(features, features * 2, is_downward=True, activation="leaky",
                                         is_dropout=False)
        self.third_downward = UNetBlock(features * 2, features * 4, is_downward=True, activation="leaky",
                                        is_dropout=False)
        self.four_downward = UNetBlock(features * 4, features * 8, is_downward=True, activation="leaky",
                                       is_dropout=False)
        self.five_downward = UNetBlock(features * 8, features * 8, is_downward=True, activation="leaky",
                                       is_dropout=False)
        self.six_downward = UNetBlock(features * 8, features * 8, is_downward=True, activation="leaky",
                                      is_dropout=False)
        self.seven_downward = UNetBlock(features * 8, features * 8, is_downward=True, activation="leaky",
                                        is_dropout=False)

        # Last Downward Layer
        self.final_downward = nn.Sequential(
            nn.Conv2d(
                in_channels=features * 8,
                out_channels=features * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect"
            ),
            nn.ReLU()
        )

        # Creating Upward layers
        self.first_upward = UNetBlock(features * 8, features * 8, is_downward=False, activation="relu",
                                      is_dropout=True)
        self.second_upward = UNetBlock(features * 8 * 2, features * 8, is_downward=False, activation="relu",
                                       is_dropout=True)
        self.third_upward = UNetBlock(features * 8 * 2, features * 8, is_downward=False, activation="relu",
                                      is_dropout=True)
        self.four_upward = UNetBlock(features * 8 * 2, features * 8, is_downward=False, activation="relu",
                                     is_dropout=False)
        self.five_upward = UNetBlock(features * 8 * 2, features * 4, is_downward=False, activation="relu",
                                     is_dropout=False)
        self.six_upward = UNetBlock(features * 4 * 2, features * 2, is_downward=False, activation="relu",
                                    is_dropout=False)
        self.seven_upward = UNetBlock(features * 2 * 2, features, is_downward=False, activation="relu",
                                      is_dropout=False)

        # Last Upward Layer
        self.final_upward = nn.Sequential(
            nn.ConvTranspose2d(
                features * 2,
                in_channels,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.Tanh()  # Each pixel value should range between +1 to -1
        )

    def forward(self, _input):
        downward_1 = self.first_downward(_input)
        downward_2 = self.second_downward(downward_1)
        downward_3 = self.third_downward(downward_2)
        downward_4 = self.four_downward(downward_3)
        downward_5 = self.five_downward(downward_4)
        downward_6 = self.six_downward(downward_5)
        downward_7 = self.seven_downward(downward_6)

        final = self.final_downward(downward_7)

        upward_1 = self.first_upward(final)  # Same shape as downward_7
        upward_2 = self.second_upward(torch.cat([upward_1, downward_7], 1))  # Same shape as downward_6
        upward_3 = self.third_upward(torch.cat([upward_2, downward_6], 1))
        upward_4 = self.four_upward(torch.cat([upward_3, downward_5], 1))
        upward_5 = self.five_upward(torch.cat([upward_4, downward_4], 1))
        upward_6 = self.six_upward(torch.cat([upward_5, downward_3], 1))
        upward_7 = self.seven_upward(torch.cat([upward_6, downward_2], 1))

        return self.final_upward(torch.cat([upward_7, downward_1], 1))


# Unit-Test
def test():
    x = torch.randn((1, 3, 256, 256))
    model = Generator(in_channels=3, features=64)
    prediction = model(x)
    print(prediction.shape)


if __name__ == "__main__":
    test()
