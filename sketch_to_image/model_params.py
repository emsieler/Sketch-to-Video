import os
import torch
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# Initialize the dataset
dataset = SketchToImageDataset(sketch_dir, photo_dir, transform=transform)
data_loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
)

# Initialize models
generator = Generator()
discriminator = Discriminator()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

# Loss functions
adversarial_loss = nn.BCELoss()
reconstruction_loss = nn.L1Loss()

num_epochs = 75
