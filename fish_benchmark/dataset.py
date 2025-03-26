import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import av
import numpy as np
from fish_benchmark.utils import read_video_pyav, sample_frame_indices

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


class UCF101(Dataset):
    '''
    Each data entry should contain 16 frames and a label. The 16 frames are sampled from the video.
    Data is stored in a folder with the following structure:
    /path/to/data/
        train/
            ├── class1/
                |video1.avi
                |video2.avi
            ├── class2/
            │
    '''
    def __init__(self, data_path, train= True, clip_len = 16, transform=None, load_data = True):
        self.data_path = data_path
        self.train = train
        self.transform = transform
        self.clip_len = clip_len
        # Step 1: Precompute a fixed, sorted class list
        train_classes = sorted([d for d in os.listdir(os.path.join(data_path, 'train')) if os.path.isdir(os.path.join(data_path, 'train', d))])
        test_classes = sorted([d for d in os.listdir(os.path.join(data_path, 'test')) if os.path.isdir(os.path.join(data_path, 'test', d))])
        
        # Take only common classes to prevent index mismatch
        self.classes = sorted(set(train_classes) & set(test_classes))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}  # Fixed mapping

        self.data = []
        self.labels = []
        if load_data:
            self.load_data()

    def load_data(self):
        path = os.path.join(self.data_path, 'train' if self.train else 'test')
        for class_name in self.classes: 
            class_path = os.path.join(path, class_name)
            if not os.path.isdir(class_path): continue
            class_idx = self.class_to_idx[class_name]
            for video_name in os.listdir(class_path):
                video_path = os.path.join(class_path, video_name) 
                if not os.path.isfile(video_path): continue
                container = av.open(video_path)
                indices = sample_frame_indices(clip_len=self.clip_len, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
                video = read_video_pyav(container, indices)
                self.data.append(video)
                self.labels.append(class_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #preprocess the data using transform
        if self.transform:
            return self.transform(self.data[idx]), self.labels[idx]
        else: 
            return self.data[idx], self.labels[idx]
