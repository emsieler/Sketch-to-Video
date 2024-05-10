import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, features=[64, 128, 256, 512]):
        super(Generator, self).__init__()

        self.encoder = nn.ModuleList()
        current_channels = in_channels
        for feature in features:
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(current_channels, feature, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.LeakyReLU(0.2)
                )
            )
            current_channels = feature

        self.decoder = nn.ModuleList()
        reversed_features = list(reversed(features))

        for i in range(len(reversed_features) - 1):
            self.decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        reversed_features[i] * 2 if i > 0 else reversed_features[i],
                        reversed_features[i + 1],
                        kernel_size=4,
                        stride=2,
                        padding=1
                    ),
                    nn.BatchNorm2d(reversed_features[i + 1]),
                    nn.ReLU()
                )
            )

        self.final_transpose = nn.Sequential(
            nn.ConvTranspose2d(
                reversed_features[-1] * 2,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(features[0]),
            nn.ReLU()
        )

        self.final_layer = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for layer in self.encoder:
            x = layer(x)
            skip_connections.append(x)

        skip_connections = skip_connections[::-1]

        for idx, layer in enumerate(self.decoder):
            x = layer(x)

            if idx < len(skip_connections) - 1:
                skip_feature = skip_connections[idx + 1]
                if x.shape[2:] == skip_feature.shape[2:]:
                    x = torch.cat([x, skip_feature], dim=1)

        x = self.final_transpose(x)
        return self.final_layer(x)
