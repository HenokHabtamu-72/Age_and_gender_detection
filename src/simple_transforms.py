import random

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps


class SimpleImageTransform:
    def __init__(self, image_size: int = 128, train: bool = False):
        self.image_size = int(image_size)
        self.train = bool(train)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        if self.train:
            if random.random() < 0.5:
                image = ImageOps.mirror(image)

            angle = random.uniform(-10.0, 10.0)
            image = image.rotate(angle, resample=Image.BILINEAR)

            brightness = random.uniform(0.85, 1.15)
            contrast = random.uniform(0.85, 1.15)
            saturation = random.uniform(0.90, 1.10)

            image = ImageEnhance.Brightness(image).enhance(brightness)
            image = ImageEnhance.Contrast(image).enhance(contrast)
            image = ImageEnhance.Color(image).enhance(saturation)

        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)
