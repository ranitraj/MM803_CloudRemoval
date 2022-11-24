import torch
import torch.nn as nn


# The discriminator in Pix2Pix is a PatchGAN, consisting of CNN Blocks
class CNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(CNNLayer, self).__init__()
        """
        Creates a CNN Block consisting of a 'convolution' defined by a Sequential Layer comprising:
            1. Convolution2D layer
            2. BatchNorm2D
            3. LeakyReLU

        :param in_channels: Number of input channels supplied
        :param out_channels: Number of output channels supplied
        :param stride: Convolution operation stride
        """
        self.convolution = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=stride,
                bias=False,
                padding_mode="reflect",
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, input_image):
        return self.convolution(input_image)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=None):
        super().__init__()
        """
        Creates the Discriminator model.
            1. Create the first Sequential layer which does not use BatchNorm2D using the first feature '64'
            2. Using the class 'CNNLayer', create the Sequential layers for all remaining features
            3. For feature with value '512', pass 'stride' as 1
            4. Unpack all the layers into the model

        :param in_channels: Number of input channels supplied
        :param features: Convolution features
        """
        # Initializations
        if features is None:
            # Convolution operations
            features = [64, 128, 256, 512]
        layers = []

        # First Sequential convolution layer without BatchNorm2D
        self.first = nn.Sequential(
            nn.Conv2d(
                in_channels*2,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )

        # Passing input-channel for second layer as the value of first feature
        in_channels = features[0]

        # Iterate through the features list (second feature onwards) and append the result on the 'layers' list
        for cur_feature in features[1:]:
            print(f"cur_feature = {cur_feature}")
            if cur_feature != 512:
                layers.append(
                    CNNLayer(in_channels, cur_feature, stride=2),
                ),
            else:
                # When cur_feature is '512' the stride value passed is 1
                layers.append(
                    CNNLayer(in_channels, cur_feature, stride=1),
                ),
            in_channels = cur_feature

        layers.append(
            nn.Conv2d(
                in_channels,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode="reflect",
            ),
        )

        # Unpack all features added to the 'layers' list into the model
        self.discriminator_model = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.first(x)
        x = self.discriminator_model(x)
        return x
