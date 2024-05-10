import os
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

class SketchToImageDataset(Dataset):
    def __init__(self, sketch_dir, real_dir, transform=None, max_images=10000):
        self.sketch_dir = Path(sketch_dir)
        self.real_dir = Path(real_dir)
        self.transform = transform

        self.sketch_filenames = sorted(os.listdir(sketch_dir))
        self.real_filenames = sorted(os.listdir(real_dir))

        self.sketch_filenames = self.sketch_filenames[:max_images]
        self.real_filenames = self.real_filenames[:max_images]

    def __len__(self):
        return len(self.sketch_filenames)

    def __getitem__(self, index):
        sketch_filename = self.sketch_filenames[index]
        real_filename = self.real_filenames[index]

        sketch_path = self.sketch_dir / sketch_filename
        real_path = self.real_dir / real_filename

        sketch_image = Image.open(sketch_path).convert('L')
        real_image = Image.open(real_path).convert('RGB')

        if self.transform:
            sketch_image = self.transform(sketch_image)
            real_image = self.transform(real_image)

        return sketch_image, real_image
