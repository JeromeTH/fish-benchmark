import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import os
import av
import numpy as np
from fish_benchmark.utils import read_video_pyav, sample_frame_indices
from torchvision.datasets import Caltech101
from torch.utils.data import Subset
import webdataset as wds
import json

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

class CalTech101WithSplit(Dataset):
    def __init__(self, path, train=True, transform=None):
        dataset = Caltech101(root=path, target_type = "category", transform= transform, download=True)
        # print(dataset.categories)
        #generate random indices to be the training set * 0.8, will split using SubSet later to be the training set 
        random_perm = torch.randperm(len(dataset))
        training_indices = random_perm[:int(0.8 * len(dataset))]  # 80% for training
        testing_indices = random_perm[int(0.8 * len(dataset)):]  # 20% for testing
        if train: 
            res = Subset(dataset, training_indices)
        else: 
            res = Subset(dataset, testing_indices)
        res.categories = dataset.categories
        res.transform = transform
        self.data = res

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class HeinFishBehavior(IterableDataset):
    def __init__(self, tar_files, img_transform=None, label_transform=None):
        super().__init__()
        self.tar_files = tar_files
        self.data = wds.WebDataset(tar_files, shardshuffle=False).decode("pil").to_tuple("png", "json")
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.load_behavior_idx_map('behavior_categories.json')

    def load_behavior_idx_map(self, path):
        '''
        json file containing the list of all behaviors: 
        [
            "behavior1", 
            "behavior2",
            ...
        ]
        '''
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(cur_dir, path)
        with open(full_path, 'r') as f:
            behaviors = json.load(f)

        self.behavior_idx_map = {behavior['name']: idx for idx, behavior in enumerate(behaviors)}
        self.categories = [behavior['name'] for behavior in behaviors]

    def __iter__(self):
        for sample in self.data: 
            image, annotation = sample
            if self.img_transform:
                image = self.img_transform(image)

            behaviors = []
            for event in annotation['events']:
                behaviors.append(event['behavior']['name'])

            labels = [self.behavior_idx_map[behavior] for behavior in behaviors]
            yield image, labels

