import os
import random
from PIL import Image

# Function to randomly scale an image
def random_scale(image, min_scale=0.8, max_scale=1.2):
    scale_factor = random.uniform(min_scale, max_scale)
    width, height = image.size
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    scaled_image = image.resize((new_width, new_height), Image.ANTIALIAS)
    return scaled_image

# Function to randomly flip an image horizontally
def random_horizontal_flip(image):
    if random.random() > 0.5:
        return image.transpose(Image.FLIP_LEFT_RIGHT)  # Horizontal flip
    return image

# Function to randomly flip an image vertically
def random_vertical_flip(image):
    if random.random() > 0.5:
        return image.transpose(Image.FLIP_TOP_BOTTOM)  # Vertical flip
    return image

input_folder = 'photo'
output_folder = 'data_aug_photo'

os.makedirs(output_folder, exist_ok=True)

image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    image = Image.open(image_path)

    for i in range(724):
        random_h = random_horizontal_flip(image)
        random_v = random_vertical_flip(random_h)
        scaled_image = random_scale(random_v)
        augmented_image_path = os.path.join(output_folder, f'aug_{i}_{image_file}')
        scaled_image.save(augmented_image_path)

        print(f"Augmented image {i+1}/250 saved to {augmented_image_path}")
