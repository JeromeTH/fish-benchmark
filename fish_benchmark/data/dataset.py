import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset, TensorDataset
import os
import av
from fish_benchmark.utils import read_video_pyav, sample_frame_indices
from torchvision.datasets import Caltech101
from torch.utils.data import Subset
import webdataset as wds
import json
from abc import ABC, abstractmethod
from fish_benchmark.utils import get_files_of_type
from queue import Queue 
import time
from contextlib import contextmanager
from torchvision import transforms
import yaml
from fish_benchmark.debug import serialized_size
from typing import Iterator, Callable, Optional
from dataclasses import dataclass, asdict
from itertools import islice
from math import ceil
from tqdm import tqdm
from collections import deque
from fish_benchmark.debug import step_timer
from PIL import Image
from torchvision.transforms import ToTensor

to_tensor = ToTensor()

class BaseCategoricalDataset(Dataset):
    @property
    @abstractmethod
    def categories(self):
        pass

def onehot(num_total_classes, active_classes):
    """
    Convert a list of class indices to one-hot encoding.
    """
    one_hot = torch.zeros(num_total_classes, dtype=torch.float32)
    one_hot[active_classes] = 1
    return one_hot

class UCF101(BaseCategoricalDataset):
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
    def __init__(self, data_path, train= True, clip_len = 16, transform=None, load_data = True, label_type = "categorical"):
        super().__init__()
        self.data_path = data_path
        self.train = train
        self.transform = transform
        self.clip_len = clip_len
        self.label_type = label_type
        # Step 1: Precompute a fixed, sorted class list
        train_classes = sorted([d for d in os.listdir(os.path.join(data_path, 'train')) if os.path.isdir(os.path.join(data_path, 'train', d))])
        test_classes = sorted([d for d in os.listdir(os.path.join(data_path, 'test')) if os.path.isdir(os.path.join(data_path, 'test', d))])
        assert len(train_classes) == len(test_classes), "Train and test classes should be the same"

        # Take only common classes to prevent index mismatch
        self.classes = sorted(set(train_classes) & set(test_classes))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}  # Fixed mapping

        self.data = []
        self.labels = []
        if load_data:
            self.load_data()

    @property
    def categories(self):
        return self.classes
    
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
            d = self.transform(self.data[idx])
        
        if self.label_type == "onehot":
            return d, onehot(len(self.categories), self.labels[idx])
        elif self.label_type == "categorical":
            return d, self.labels[idx]
        else:
            raise ValueError(f"label_type {self.label_type} not recognized.")
        
class CalTech101WithSplit(Dataset):
    def __init__(self, path, train=True, transform=None, label_type = "categorical"):
        dataset = Caltech101(root=path, target_type = "category", transform= transform, download=True)
        '''
        label_type = "categorical" or "onehot"
        '''
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
        self.label_type = label_type

    @property
    def categories(self):
        return self.data.categories

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.label_type == "onehot":
            # Convert the label to one-hot encoding
            return self.data[idx][0], onehot(len(self.categories), self.data[idx][1])
        elif self.label_type == "categorical":
            return self.data[idx]
        else:
            raise ValueError(f"label_type {self.label_type} not recognized.")


def load_behavior_idx_map(path):
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

        behavior_idx_map = {behavior['name']: idx for idx, behavior in enumerate(behaviors)}
        return behavior_idx_map

def parse_annotation(annotation):
    behaviors = []
    for event in annotation['events']:
        behaviors.append(event['behavior']['name'])
    return behaviors

def get_sample_indices(lst_len, clip_len, sample_method = "even-spaced"):
    if sample_method == "even-spaced":
        indices = np.linspace(0, lst_len - 1, clip_len).astype(int)
    elif sample_method == "random":
        indices = np.random.choice(lst_len, clip_len, replace=False)
    else:
        raise ValueError(f"sample_method {sample_method} not recognized.")
    return indices

@dataclass
class BaseSlidingWindowDataset:
    '''
    Base Class for sliding through a video and getting a window of frames. Defines sampling, patching, and shuffling.
    The returned dataset would have items of size [samples_per_window * patch_per_sample, channels, height, width]
    The total number of items is (num_frames - window_size) // step_size
    The labels are one-hot encoded.
    '''
    input_transform: callable=None, 
    annotation_to_label: callable=None,
    label_type: str= "onehot", 
    window_size: int=16, 
    tolerance_region: int = 16, 
    samples_per_window: int= 16, 
    step_size: int = 1, 
    categories: list = None, 
    is_image_dataset: bool = False, 
    shuffle: bool = False,
    patch_grid_dim: int = 1, 
    temporal_sample_interval: int = 1,
    MAX_BUFFER_SIZE: int = 1000
    def __post_init__(self):
        assert self.window_size % self.samples_per_window == 0, f"window_size {self.window_size} should be a factor of samples_per_window {self.samples_per_window}"
        assert self.temporal_sample_interval > 0, f"temporal_sample_interval {self.temporal_sample_interval} should be greater than 0"
        assert self.tolerance_region <= (self.window_size - 1)//2, f"tolerance_region {self.tolerance_region} should be less than or equal to window_size {self.window_size//2}"
        assert self.label_type == "onehot", 'currently only onehot is supported'
        assert isinstance(self.patch_grid_dim, int), \
            f"patch_grid_dim should be an int, got {type(self.patch_grid_dim)}"
        assert self.MAX_BUFFER_SIZE > self.window_size, f"MAX_BUFFER_SIZE {self.MAX_BUFFER_SIZE} should be greater than window_size {self.window_size}, otherwise it will not be able to store enough frames"
        if self.is_image_dataset: assert self.samples_per_window ==1, "samples per window should be 1 for image datasets"
        self.image_window_queue = deque([], maxlen=self.window_size)
        self.annotations_window_queue = deque([], maxlen=self.window_size)
        self.clips = []
        self.labels = []

    def clear_window_queue(self):
        self.image_window_queue = deque([], maxlen=self.window_size)
        self.annotations_window_queue = deque([], maxlen=self.window_size)

    def clear_buffer(self):
        self.clips = []
        self.labels = []

    def flush(self):
        if len(self.clips) == 0: 
            self.clear_buffer()
            return
        clips = torch.stack(self.clips)
        labels = torch.stack(self.labels)
        if self.shuffle: 
            perm = torch.randperm(len(clips))
            clips = clips[perm]
            labels = labels[perm]
        
        dataset = TensorDataset(clips, labels)
        for image, label in dataset:
            yield image, label
        self.clear_buffer()

    def is_yielding_idx(self, idx):
        if idx - (self.window_size - 1) < 0: return False
        return (idx - (self.window_size - 1)) % self.step_size == 0
    
    def next_yielding_idx(self, idx):
        nearest_kth_yield = ceil((idx - (self.window_size - 1)) / self.step_size) #0 indexed 
        return nearest_kth_yield * self.step_size + (self.window_size - 1)

    def handle_item(self, ith_sample, image, annotation):
        '''
        depending on the context of the current seen images, this may return a clip or None
        '''
        
        if self.next_yielding_idx(ith_sample) - ith_sample > (self.window_size - 1):
            #if the next yielding index is more than window size away, then this frame would not be used
            return
        
        with step_timer("converting PIL image to numpy", verbose=True): 
            #images are converted to tensors from the start as we want to treat clip processing as batch processing using torch.stack
            self.image_window_queue.append(np.array(image.convert('RGB')))
            self.annotations_window_queue.append(annotation)
    
        if self.is_yielding_idx(ith_sample):
            #if the calculation of yielding index is correct, then we should have enough images in the window queue
            assert len(self.image_window_queue) >= self.window_size, f"image buffer should be at least {self.window_size} long"

            with step_timer("getting latest clip", verbose=True):
                self.clips.append(self.get_latest_clip())
                self.labels.append(self.get_latest_label())

        if len(self.clips) >= self.MAX_BUFFER_SIZE:
            yield from self.flush()
            

    def get_latest_label(self):
        last_idx = len(self.annotations_window_queue) - 1
        mid_idx = last_idx - int(self.window_size/2)
        relevant_annotations = list(self.annotations_window_queue)[mid_idx - self.tolerance_region: mid_idx + self.tolerance_region + 1]
        relevant_labels = torch.stack([self.annotation_to_label(annotation) for annotation in relevant_annotations]) 
        relevant_labels = relevant_labels[:, :len(self.categories)] #drop extra incomplete labels
        unioned_labels = torch.any(relevant_labels.bool(), dim=0).float()
        return unioned_labels

    def grid_patches(self, img):
        '''
        Given an image, return a list of patches based on the grid size.
        img is a numpy array of shape (H, W, C)
        
        '''
        H, W = img.shape[:2]

        # Compute patch size based on grid
        patch_h = H // self.patch_grid_dim
        patch_w = W // self.patch_grid_dim

        patches = []

        for i in range(self.patch_grid_dim):
            for j in range(self.patch_grid_dim):
                y1 = i * patch_h
                y2 = y1 + patch_h
                x1 = j * patch_w
                x2 = x1 + patch_w

                patch = img[y1:y2, x1:x2]
                patches.append(patch)

        return patches
    
    def numpy_to_tensor(self, clip):
        '''
        np.ndarray clip has shape (samples_per_window * patch_per_sample, height, width, channels)
        '''
        clip = torch.from_numpy(clip).permute(0, 3, 1, 2)
        clip = clip.float() / 255.0
        return clip

    def get_latest_clip(self):
        interval = int(self.window_size/self.samples_per_window)
        with step_timer("stacking patches", verbose=True):
            clip = np.stack([patch 
                            for img in list(self.image_window_queue)[-self.window_size::interval] 
                            for patch in self.grid_patches(img)])
        with step_timer("converting to tensor", verbose=True):
            tensor_clip = self.numpy_to_tensor(clip)
        with step_timer("applying vision transform", verbose=True):
            if self.input_transform:
                tensor_clip = self.input_transform(tensor_clip)

        if self.is_image_dataset: tensor_clip = tensor_clip.squeeze()
        return tensor_clip

    def scan(self, annotated_video_frames: Iterator):
        '''
        annotated_video_frames is a generator that yields (image, annotation) tuples
        image is a PIL image
        annotation is whatever the dataset annotation. One needs to implement the annotation_to_label function to convert it to a label
        '''
        self.clear_buffer()
        self.clear_window_queue()
        for i, (image, annotation) in enumerate(annotated_video_frames):
            if i % self.temporal_sample_interval == 0: 
                sample_idx = i // self.temporal_sample_interval
                yield from self.handle_item(sample_idx, image, annotation)

    def downsampled_length(self, video_frames_count):
        '''
        Returns give this configuration of sliding window, how many items will be generated given one video with 
        video_frames_count frames
        '''
        sampled_frames= video_frames_count // self.temporal_sample_interval
        items_count = max(0, (sampled_frames - self.window_size) // self.step_size)
        return items_count

        
class MikeDataset(IterableDataset, BaseSlidingWindowDataset):
    def __init__(self, 
                 path, 
                 train = True, 
                 transform=None, 
                 label_type = "onehot", 
                 window_size=16, 
                 tolerance_region = 16, 
                 samples_per_window = 16, 
                 step_size = 1, 
                 is_image_dataset = False, 
                 shuffle = False, 
                 patch_grid_dim = 2, 
                 temporal_sample_interval = 2):
        self.path = os.path.join(path, "train" if train else "test")
        self.behavior_idx_map = load_behavior_idx_map('behavior_categories.json')
        self.SHARD_SIZE = 1000
        BaseSlidingWindowDataset.__init__(
            self,
            input_transform=transform,
            annotation_to_label=lambda x: onehot(len(self.categories), [self.behavior_idx_map[behavior] for behavior in parse_annotation(x)]),
            label_type=label_type,
            window_size=window_size,
            tolerance_region=tolerance_region,
            samples_per_window=samples_per_window,
            step_size=step_size,
            categories=list(self.behavior_idx_map.keys()), 
            is_image_dataset=is_image_dataset,
            shuffle=shuffle,
            patch_grid_dim=patch_grid_dim, 
            temporal_sample_interval=temporal_sample_interval,
            MAX_BUFFER_SIZE = 100//window_size
        )

    def video_frames_shard_count_pairs(self):
        video_names = os.listdir(self.path)
        for video in video_names:
            tar_batches = os.listdir(os.path.join(""+self.path, video))
            tar_batches.sort()
            tar_file_paths = [os.path.join(self.path, video, tar_batch) for tar_batch in tar_batches]
            video_frames = wds.WebDataset(tar_file_paths, shardshuffle=False).decode("pil").to_tuple("png", "json")
            yield video_frames, len(tar_batches)

    def __len__(self):
        count = 0
        for _, shard_count in self.video_frames_shard_count_pairs():
            video_frames_count = shard_count * self.SHARD_SIZE
            count += self.downsampled_length(video_frames_count)
        return count

    def __iter__(self):
        for video_frames, _ in self.video_frames_shard_count_pairs():
            # print(f"iterating over {video} with {len(tar_batches)} tar files")
            yield from self.scan(video_frames)
            yield from self.flush()

class AbbyDataset(IterableDataset, BaseSlidingWindowDataset):
    def __init__(self, 
                 path, 
                 train = True, 
                 transform=None, 
                 label_type = "onehot", 
                 window_size=16, 
                 tolerance_region = 16, 
                 samples_per_window = 16, 
                 step_size = 1, 
                 is_image_dataset = False, 
                 shuffle = False, 
                 patch_grid_dim = 1,
                 temporal_sample_interval = 1):
        self.path = os.path.join(path, "train" if train else "test")
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(cur_dir, 'abby_dset_categories.json')
        self.categories = json.load(open(full_path, 'r'))
        BaseSlidingWindowDataset.__init__(
            self,
            input_transform=transform,
            annotation_to_label=lambda x: torch.tensor(x), 
            #label shouuld be a torch tensor of shape (num_classes,)
            label_type=label_type,
            window_size=window_size,
            tolerance_region=tolerance_region,
            samples_per_window=samples_per_window,
            step_size=step_size,
            categories=self.categories,    
            is_image_dataset=is_image_dataset, 
            shuffle=shuffle, 
            patch_grid_dim=patch_grid_dim,
            temporal_sample_interval=temporal_sample_interval,
            MAX_BUFFER_SIZE = 1000//window_size #can have more b/c resolution is lower
        )

    def container_label_pairs(self):
        for annotation_path in os.listdir(self.path):
            track_paths = sorted(get_files_of_type(os.path.join(self.path, annotation_path), ".mp4"))
            label_paths = sorted(get_files_of_type(os.path.join(self.path, annotation_path), ".txt"))
            label_dict = {
                os.path.splitext(os.path.basename(p))[0]: p
                for p in label_paths
            }
            for track_path in track_paths:
                track_name = os.path.splitext(os.path.basename(track_path))[0]
                label_path = label_dict.get(track_name)
                if label_path is None:
                    continue
                
                container = av.open(track_path)
                label = np.loadtxt(label_path, delimiter='\t', dtype=int)
                assert label.shape[0] == container.streams.video[0].frames, f"Number of frames in {track_path} does not match number of annotations in {label_path}"
                yield container, label

    def __len__(self):
        total_frames = 0
        for container, label in self.container_label_pairs():
            total_frames += self.downsampled_length(container.streams.video[0].frames) 
        return total_frames

    def __iter__(self):
        for container, label in self.container_label_pairs():
            def annotated_frame_iterator():
                for i, frame in enumerate(container.decode(video=0)):
                    yield frame.to_image(), label[i]

            yield from self.scan(annotated_frame_iterator())
            yield from self.flush()


class PrecomputedDataset(IterableDataset):
    def __init__(self, path, model_name, categories, transform=None, train=True):
        self.label_type = "onehot"
        TYPE = "train" if train else "test"
        self.path = os.path.join(path, model_name, TYPE)
        
        if not os.path.exists(self.path):
            print("Did not precompute for this specific model, falling back to precomputed dataset of the same input type")
            #fall back to default 
            config = yaml.safe_load(open('./config/models.yml', 'r'))
            model_type = config[model_name]['type']
            if model_type == 'video':
                self.path = os.path.join(path, "videomae", TYPE)
            elif model_type == 'image':
                self.path = os.path.join(path, "clip", TYPE)
            else:
                raise ValueError(f"Model type {model_type} not recognized.")
            
        self.transform = transform
        self.categories = categories

    def __len__(self):
        frames_path = os.path.join(self.path, "frames")
        frame_files = os.listdir(frames_path)
        return len(frame_files)
    
    def __iter__(self):
        '''
        frames/ contains the precomputed frames with <id>.pt
        labels/ contains the precomputed labels with <id>.pt
        '''
        frames_path = os.path.join(self.path, "frames")
        labels_path = os.path.join(self.path, "labels")
        frame_files = os.listdir(frames_path)
        for id in range(len(frame_files)):
            frame_path = os.path.join(frames_path, f'{id}.pt')
            label_path = os.path.join(labels_path, f'{id}.pt')
            if not os.path.isfile(frame_path): continue
            if not os.path.isfile(label_path): continue
            frame = torch.load(frame_path)
            label = torch.load(label_path)
            if self.transform:
                frame = self.transform(frame)
            yield frame, label

def get_summary(dataset):
    summary = {}
    summary['metadata'] = asdict(dataset)
    labels = torch.stack([
        label for _, label in tqdm(islice(dataset, 100))
    ])
    label_density = labels.sum(dim=0) / labels.shape[0]
    summary['label_density'] = label_density.tolist()
    summary['dataset_size'] = len(dataset)
    return summary

def get_dataset(dataset_name, path, augs=None, train=True, label_type = "onehot", model_name = None, shuffle = False):
    if dataset_name == 'UCF101':
        dataset = UCF101(path, train=train, transform=augs, label_type=label_type)
    elif dataset_name == 'Caltech101':
        dataset = CalTech101WithSplit(path, train=train, transform=augs, label_type=label_type)
    elif dataset_name == 'MikeFrames':
        # dataset = Mike(path, transform=augs, label_type=label_type, train=train)
        dataset = MikeDataset(
            path,
            train=train, 
            transform=augs,
            label_type=label_type,
            window_size = 1, 
            tolerance_region = 0,
            samples_per_window = 1,
            step_size = 10, 
            is_image_dataset=True, 
            shuffle = shuffle, 
            patch_grid_dim=1, 
            temporal_sample_interval=1
        )
    elif dataset_name == 'MikeFramesPatched':
        # dataset = Mike(path, transform=augs, label_type=label_type, train=train)
        dataset = MikeDataset(
            path,
            train=train, 
            transform=augs,
            label_type=label_type,
            window_size = 1, 
            tolerance_region = 0,
            samples_per_window = 1,
            step_size = 1, 
            is_image_dataset=False, 
            shuffle = shuffle, 
            patch_grid_dim=2, 
            temporal_sample_interval=1
        )
    elif dataset_name == 'MikeSlidingWindow':
        # dataset = MikeSlidingWindow(path, transform=augs, label_type=label_type, train=train)
        dataset = MikeDataset(
            path, 
            train=train,
            transform=augs,
            label_type=label_type,
            window_size = 16,
            tolerance_region = 7,
            samples_per_window = 16,
            step_size = 1, 
            is_image_dataset=False, 
            shuffle = shuffle,
            patch_grid_dim=2, 
            temporal_sample_interval=1
        )
    elif dataset_name == 'AbbyFrames':
        dataset = AbbyDataset(
            path, 
            train=train, 
            transform=augs, 
            label_type=label_type, 
            window_size=1, 
            tolerance_region = 0,
            samples_per_window = 1,
            step_size=1, 
            is_image_dataset=True, 
            shuffle = shuffle, 
            temporal_sample_interval=2
        )
    elif dataset_name == 'AbbySlidingWindow':
        dataset = AbbyDataset(
            path, 
            train=train, 
            transform=augs, 
            label_type=label_type, 
            window_size=16, 
            tolerance_region = 7,
            samples_per_window = 16,
            step_size=1, 
            is_image_dataset=False, 
            shuffle = shuffle, 
            temporal_sample_interval=2
        )
    elif dataset_name == 'MikePrecomputed': 
        behavior_idx_map = load_behavior_idx_map('behavior_categories.json')
        categories = list(behavior_idx_map.keys())
        dataset = PrecomputedDataset(path, model_name=model_name, categories=categories, transform=augs, train=train)
    elif dataset_name == 'MikeSlidingWindowPrecomputed':
        behavior_idx_map = load_behavior_idx_map('behavior_categories.json')
        categories = list(behavior_idx_map.keys())
        dataset = PrecomputedDataset(path, model_name=model_name, categories=categories, transform=augs, train=train)
    elif dataset_name == 'AbbySlidingWindowPrecomputed':
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(cur_dir, 'abby_dset_categories.json')
        categories = json.load(open(full_path, 'r'))
        dataset = PrecomputedDataset(path, model_name=model_name, categories=categories, transform=augs, train=train)
    else:
        raise ValueError(f"Dataset {dataset_name} not recognized.")
    return dataset