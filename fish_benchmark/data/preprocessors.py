from PIL import Image
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode
import torch


class TorchVisionPreprocessor:
    def __init__(self, crop_size=(224, 224), resize_shortest=256,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                 interpolation=InterpolationMode.BICUBIC):

        self.resize = v2.Resize(resize_shortest, interpolation=interpolation, antialias=False)
        self.crop = v2.CenterCrop(crop_size)
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, image_tensor: torch.Tensor) -> torch.Tensor:
        image_tensor = self.resize(image_tensor)
        image_tensor = self.crop(image_tensor)
        image_tensor = (image_tensor - self.mean) / self.std
        return image_tensor