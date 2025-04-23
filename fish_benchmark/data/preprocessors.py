from PIL import Image
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode
import torch


class TorchVisionPreprocessor:
    def __init__(self,
                 crop_size=(224, 224),
                 resize_shortest=256,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 interpolation=InterpolationMode.BICUBIC):
        self.transform = v2.Compose([
            v2.Resize(resize_shortest, interpolation=interpolation, antialias=False),
            v2.CenterCrop(crop_size),
            v2.Normalize(mean=mean, std=std),
        ])

    def __call__(self, image_tensor: torch.Tensor) -> torch.Tensor:
        return self.transform(image_tensor)
