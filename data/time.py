import torch
import time
import timm
from torchvision import transforms
import torch
import torch.nn.functional as F
import time
from transformers import AutoImageProcessor, AutoModel
from torchvision.transforms import Resize, Normalize, Compose

# Load the large tensor from the .pt file

# if not isinstance(image, torch.Tensor):
#     raise ValueError("Expected a torch.Tensor in the .pt file.")

# if image.dim() != 3 or image.shape[0] != 3:
#     raise ValueError("Expected image tensor of shape [3, H, W]")

# Load DINOv2-base from timm
model = timm.create_model("vit_base_patch14_reg4_dinov2", pretrained=True)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
start = time.time()

image = torch.load("/share/j_sun/jth264/precomputed/mike_frames/test/SR_063023_GH010250/inputs/SR_063023_GH010250_00000000.pt")  # must be shape [3, H, W]

image = image.to(device)

# Transform: Normalize to match DINOv2 training setup
transform = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],  # ImageNet mean
    std=[0.229, 0.224, 0.225]    # ImageNet std
)

# Patch size and input tensor
patch_size = 518
_, H, W = image.shape

pad_h = 512- H%512 
pad_w = 512 - W%512

image = F.pad(image, (0, H-pad_w, 0, H-pad_h), mode="constant", value=0)


# Start timing
num_patches = 0

with torch.no_grad():
    for y in range(0, H , patch_size):
        for x in range(0, W , patch_size):
            patch = image[:, y:y+patch_size, x:x+patch_size]
            patch = transform(patch)
            patch = patch.unsqueeze(0).to(device)  
            embedding = model(patch) 
            num_patches += 1

end = time.time()

print(f"Processed {num_patches} patches in {end - start:.2f} seconds.")
