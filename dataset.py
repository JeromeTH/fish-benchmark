import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Video parameters
num_frames = 300
frame_height, frame_width = 224, 224
num_channels = 3
clip_length = 16  # Each model input has 16 frames
stride = 8  # Sliding window step size
batch_size = 32

# Generate a random video (dummy data) - This is the only memory allocation
video = np.random.randint(0, 256, (num_frames, num_channels, frame_height, frame_width), dtype=np.uint8)

# Custom dataset that references the original video without copying
class VideoDataset(Dataset):
    def __init__(self, video, clip_length, stride):
        self.video = video  # Reference to the original array
        self.clip_length = clip_length
        self.indices = list(range(0, len(video) - clip_length + 1, stride))  # Start indices of each clip

    def __len__(self):
        return len(self.indices)  # Number of clips

    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        clip = self.video[start_idx : start_idx + self.clip_length]  # View, not a copy
        clip = torch.tensor(clip, dtype=torch.float32) / 255.0  # Convert to tensor on-demand
        return clip

# Create dataset and dataloader
dataset = VideoDataset(video, clip_length, stride)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


