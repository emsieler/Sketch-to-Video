class Discriminator(nn.Module):
    def __init__(self, in_channels=4, features=[64, 128, 256, 512]):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features[0], features[1], kernel_size=4, stride=2, padding=1),  #64 to 128
            nn.BatchNorm2d(features[1]),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features[1], features[2], kernel_size=4, stride=2, padding=1),  #128 to 256
            nn.BatchNorm2d(features[2]),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features[2], features[3], kernel_size=4, stride=2, padding=1),  #256 to 512
            nn.BatchNorm2d(features[3]),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features[3], 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()  #Ensure output is in the [0, 1] range
        )

    def forward(self, x):
        return self.layers(x)
