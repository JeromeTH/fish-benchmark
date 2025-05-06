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
from pydantic import BaseModel
from typing import Callable, Optional, List 
import logging
import torch.nn.functional as F
import math 

to_tensor = ToTensor()
PROFILE = False
PROFILE_LOADING = True
logger = logging.getLogger(__name__)

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

def load_from_cur_dir(path):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(cur_dir, path)
    with open(full_path, 'r') as f:
        data = json.load(f)
    return data

def load_behavior_idx_map(path):
        '''
        json file containing the list of all behaviors: 
        [
            "behavior1", 
            "behavior2",
            ...
        ]
        '''
        behaviors = load_from_cur_dir(path)
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

class BaseSource:
    def get_config(self):
        required_attrs = ['categories', 'label_type', 'total_frames']
        for attr in required_attrs:
            if not hasattr(self, attr):
                raise AttributeError(f"Missing required attribute '{attr}' in {self.__class__.__name__}")
        return {
            'categories': self.categories,
            'label_type': self.label_type,
            'total_frames': self.total_frames
        }
    
    def __iter__(self):
        raise NotImplementedError("Subclasses should implement this method")
    
    def labels(self) -> Iterator:
        raise NotImplementedError("Subclasses should implement this method")
    

class Patcher: 
    def __init__(self, patch_type, h, w):
        assert patch_type in ["absolute", "relative"], f"patch_type {patch_type} not recognized"
        # if patch_type == "absolute": h is the height of each patch, w is the width of each patch
        # if patch_type == "relative": the full height of the image would be cut into h patches, and the full width of the image would be cut into w patches
        self.patch_type = patch_type
        self.h = h
        self.w = w
    
    def pad_to_multiple_np(self, image: np.ndarray, h: int, w: int) -> np.ndarray:
        """
        Pads a HWC (Height-Width-Channel) image so that its height is a multiple of h
        and width is a multiple of w using 0-padding.

        Args:
            image (np.ndarray): Input image of shape (H, W, C)
            h (int): Desired height multiple
            w (int): Desired width multiple

        Returns:
            np.ndarray: Padded image
        """
        H, W, C = image.shape
        pad_h = (h - H % h) % h
        pad_w = (w - W % w) % w
        padding = ((0, pad_h), (0, pad_w), (0, 0))  # Pad bottom and right
        return np.pad(image, padding, mode='constant', constant_values=0)

    def patch(self, img, h, w):
        img = self.pad_to_multiple_np(img, h, w)
        return img.reshape(-1, h, w, img.shape[2])
    
    def __call__(self, img):
        if self.patch_type == "absolute":
            return self.patch(img, self.h, self.w)
        elif self.patch_type == "relative":
            # Split the image into patches
            h = math.ceil(img.shape[0] / self.h) #h is the 
            w = math.ceil(img.shape[1] / self.w)
            return self.patch(img, h, w)

@dataclass
class BaseSlidingWindowDataset(IterableDataset):
    '''
    Base Class for sliding through a video and getting a window of frames. Defines sampling, patching, and shuffling.
    The returned dataset would have items of size [samples_per_window * patch_per_sample, channels, height, width]
    The total number of items is (num_frames - window_size) // step_size
    The labels are one-hot encoded.
    '''
    input_transform: Callable=None, 
    label_type: str= "onehot", 
    window_size: int=16,  
    tolerance_region: int = 16, 
    samples_per_window: int= 16, 
    step_size: int = 1, 
    categories: list = None, 
    is_image_dataset: bool = False, 
    shuffle: bool = False,
    patch_type: str = 'relative', 
    patch_h: int = 1, 
    patch_w: int = 1,
    temporal_sample_interval: int = 1,
    MAX_BUFFER_SIZE: int = 1000
    total_frames: int = None
    def __post_init__(self):
        assert self.window_size % self.samples_per_window == 0, f"window_size {self.window_size} should be a factor of samples_per_window {self.samples_per_window}"
        assert self.temporal_sample_interval > 0, f"temporal_sample_interval {self.temporal_sample_interval} should be greater than 0"
        assert self.tolerance_region <= (self.window_size - 1)//2, f"tolerance_region {self.tolerance_region} should be less than or equal to window_size {self.window_size//2}"
        assert self.label_type == "onehot", 'currently only onehot is supported'
        assert self.MAX_BUFFER_SIZE > self.window_size, f"MAX_BUFFER_SIZE {self.MAX_BUFFER_SIZE} should be greater than window_size {self.window_size}, otherwise it will not be able to store enough frames"
        if self.is_image_dataset: assert self.samples_per_window ==1, "samples per window should be 1 for image datasets"
        self.image_window_queue = deque([], maxlen=self.window_size)
        self.labels_window_queue = deque([], maxlen=self.window_size)
        self.patcher = Patcher(self.patch_type, self.patch_h, self.patch_w)
        self.clips = []
        self.labels = []
        self.source = None

    def set_source(self, source: BaseSource):
        '''
        Source iterator should yield (image, label) tuples with images being PIL images and label 
        being a pytorch tensor of shape (num_classes,)
        '''
        self.source = source

    def __len__(self):
        return self.downsampled_length(self.total_frames) if self.total_frames else None

    def clear_window_queue(self):
        self.image_window_queue = deque([], maxlen=self.window_size)
        self.labels_window_queue = deque([], maxlen=self.window_size)

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

    def handle_item(self, ith_sample, image, label):
        '''
        depending on the context of the current seen images, this may return a clip or None
        '''
        
        if self.next_yielding_idx(ith_sample) - ith_sample > (self.window_size - 1):
            #if the next yielding index is more than window size away, then this frame would not be used
            return
        
        with step_timer("converting PIL image to numpy", verbose=PROFILE): 
            #images are converted to tensors from the start as we want to treat clip processing as batch processing using torch.stack
            self.image_window_queue.append(np.array(image.convert('RGB')))
            self.labels_window_queue.append(label)
    
        if self.is_yielding_idx(ith_sample):
            #if the calculation of yielding index is correct, then we should have enough images in the window queue
            assert len(self.image_window_queue) >= self.window_size, f"image buffer should be at least {self.window_size} long"

            with step_timer("getting latest clip", verbose=PROFILE):
                self.clips.append(self.get_latest_clip())
                self.labels.append(self.get_latest_label())

        if len(self.clips) >= self.MAX_BUFFER_SIZE:
            yield from self.flush()
            

    def get_latest_label(self):
        last_idx = len(self.labels_window_queue) - 1
        mid_idx = last_idx - int(self.window_size/2)
        relevant_labels = torch.stack(list(self.labels_window_queue)[mid_idx - self.tolerance_region: mid_idx + self.tolerance_region + 1]) 
        relevant_labels = relevant_labels[:, :len(self.categories)] #drop extra incomplete labels
        unioned_labels = torch.any(relevant_labels.bool(), dim=0)
        return unioned_labels
    
    def numpy_to_tensor(self, clip):
        '''
        np.ndarray clip has shape (samples_per_window * patch_per_sample, height, width, channels)
        '''
        clip = torch.from_numpy(clip).permute(0, 3, 1, 2).to(torch.uint8)
        return clip

    def get_latest_clip(self):
        interval = int(self.window_size/self.samples_per_window)
        with step_timer("stacking patches", verbose=PROFILE):
            clip = np.stack([patch 
                            for img in list(self.image_window_queue)[-self.window_size::interval] 
                            for patch in self.patcher(img)])
        with step_timer("converting to tensor", verbose=PROFILE):
            tensor_clip = self.numpy_to_tensor(clip)
        with step_timer("applying vision transform", verbose=PROFILE):
            if self.input_transform:
                tensor_clip = self.input_transform(tensor_clip)

        if self.is_image_dataset: tensor_clip = tensor_clip.squeeze()
        return tensor_clip

    def scan(self, annotated_video_frames: Iterator):
        '''
        annotated_video_frames is a generator that yields (image, annotation) tuples
        image is a PIL image
        label is whatever the dataset label. One needs to implement the annotation_to_label function to convert it to a label
        '''
        self.clear_buffer()
        self.clear_window_queue()
        for i, (image, label) in enumerate(annotated_video_frames):
            if i % self.temporal_sample_interval == 0: 
                sample_idx = i // self.temporal_sample_interval
                yield from self.handle_item(sample_idx, image, label)

    def downsampled_length(self, video_frames_count):
        '''
        Returns give this configuration of sliding window, how many items will be generated given one video with 
        video_frames_count frames
        '''
        sampled_frames= video_frames_count // self.temporal_sample_interval
        items_count = max(0, (sampled_frames - self.window_size) // self.step_size)
        return items_count

    def __iter__(self):
        assert self.source is not None, "source is not set. Please set the source using set_source()"
        yield from self.scan(self.source)
        yield from self.flush()

    def get_summary(self):
        summary = {}
        summary['metadata'] = asdict(self)
        label_count = torch.zeros(len(self.categories))
        for label in tqdm(self.source.labels()): 
            assert label.shape == (len(self.categories),), f"label shape {label.shape} does not match categories {self.categories}"
            label_count += label
        summary['label_count'] = label_count.tolist()
        #summary['dataset_size'] = len(self)
        return summary

def get_categories(dataset_name):
    if dataset_name == 'ucf101':
        return load_from_cur_dir('ucf101_categories.json')
    elif dataset_name == 'mike':
        return load_behavior_idx_map('behavior_categories.json').keys()
    elif dataset_name == 'abby':
        return load_from_cur_dir('abby_dset_categories.json')
    else:
        raise ValueError(f"dataset_name {dataset_name} not recognized")
         

class MikeSource(BaseSource):
    def __init__(self, path):
        self.path = path
        self.categories = get_categories('mike')
        self.label_type = "onehot"
        self.annotation_to_label = lambda x: onehot(len(self.behavior_idx_map), [self.behavior_idx_map[behavior] for behavior in parse_annotation(x)])
        self.SHARD_SIZE = 1000  
        self.tar_file_paths = get_files_of_type(self.path, ".tar")
        self.source = wds.WebDataset(self.tar_file_paths, shardshuffle=False).decode("pil").to_tuple("png", "json")
        self.total_frames = self.SHARD_SIZE * len(self.tar_file_paths)

    def __iter__(self):
        for frame, annotation in self.source: 
            yield frame, self.annotation_to_label(annotation)

class MikeSourceV2(BaseSource):
    def __init__(self, path):
        self.path = path
        self.categories = get_categories('mike')
        self.label_type = "onehot"
        self.annotation_to_label = lambda x: torch.tensor(x)
        self.video_paths = get_files_of_type(self.path, ".mp4")
        self.label_paths = get_files_of_type(self.path, ".tsv")
        self.label_dict = {
            os.path.splitext(os.path.basename(p))[0]: p
            for p in self.label_paths
        }
        self.total_frames = self.calculate_total_frames()

    def calculate_total_frames(self):
        total_frames = 0
        for track_path in self.video_paths:
            track_name = os.path.splitext(os.path.basename(track_path))[0]
            label_path = self.label_dict.get(track_name)
            if label_path is None:
                continue
            
            container = av.open(track_path)
            total_frames += container.streams.video[0].frames
        return total_frames
    
    def labels(self):
        for label_path in self.label_paths:
            label = np.loadtxt(label_path, delimiter='\t', dtype=int)
            for i in range(label.shape[0]):
                yield label[i]

    def __iter__(self):
        for video_path in self.video_paths:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            label_path = self.label_dict.get(video_name)
            if label_path is None:
                continue
            
            container = av.open(video_path)
            label = np.loadtxt(label_path, delimiter='\t', dtype=int)
            assert label.shape[0] == container.streams.video[0].frames, f"Number of frames in {video_path} does not match number of annotations in {label_path}"
            for i, frame in enumerate(container.decode(video=0)):
                yield frame.to_image(), self.annotation_to_label(label[i])


class AbbySource(BaseSource):
    def __init__(self, path):
        self.path = path
        self.categories = get_categories('abby')
        self.label_type = "onehot"
        self.annotation_to_label = lambda x: torch.tensor(x)
        self.track_paths = sorted(get_files_of_type(self.path, ".mp4"))
        self.label_paths = sorted(get_files_of_type(self.path, ".txt"))
        self.label_dict = {
            os.path.splitext(os.path.basename(p))[0]: p
            for p in self.label_paths
        }
        self.total_frames = self.calculate_total_frames()

    def calculate_total_frames(self):
        total_frames = 0
        for track_path in self.track_paths:
            track_name = os.path.splitext(os.path.basename(track_path))[0]
            label_path = self.label_dict.get(track_name)
            if label_path is None:
                continue
            
            container = av.open(track_path)
            total_frames += container.streams.video[0].frames
        return total_frames
    
    def labels(self):
        for label_path in self.label_paths:
            label = np.loadtxt(label_path, delimiter='\t', dtype=int)
            for i in range(label.shape[0]):
                yield label[i]
    
    def __iter__(self):
        for track_path in self.track_paths:
            track_name = os.path.splitext(os.path.basename(track_path))[0]
            label_path = self.label_dict.get(track_name)
            if label_path is None:
                continue
            
            container = av.open(track_path)
            label = np.loadtxt(label_path, delimiter='\t', dtype=int)
            if label.shape[0] != container.streams.video[0].frames:
                logger.warning(f"Number of frames in {track_path} does not match number of annotations in {label_path}, skipping")
                continue
            for i, frame in enumerate(container.decode(video=0)):
                yield frame.to_image(), self.annotation_to_label(label[i])

def get_dicts_and_common_keys(list1, list2):
    '''
    lists are file paths and keys are file names
    '''
    dict1 = {os.path.basename(p).split('.')[0]: p for p in list1}
    dict2 = {os.path.basename(p).split('.')[0]: p for p in list2}
    common_keys = list(dict1.keys() & dict2.keys())
    return dict1, dict2, common_keys

class UCF101Source(BaseSource):
    def __init__(self, path):
        self.path = path
        self.categories = get_categories('ucf101')
        self.label_type = "onehot"  
        self.annotation_to_label=lambda x: onehot(len(self.categories), [x])
        self.video_paths = sorted(get_files_of_type(self.path, ".avi"))
        self.annotation_paths = sorted(get_files_of_type(self.path, ".txt"))
        self.video_dict, self.annotation_dict, self.keys = get_dicts_and_common_keys(self.video_paths, self.annotation_paths)
        self.total_frames = self.calculate_total_frames()

    def calculate_total_frames(self):
        total_frames = 0
        for key in self.keys:
            video_path = self.video_dict[key]
            container = av.open(video_path)
            total_frames += container.streams.video[0].frames
        return total_frames

    def labels(self):
        for key in self.keys:
            annotation_path = self.annotation_dict[key]
            with open(annotation_path, 'r') as f:
                annotation_idx = int(f.read().strip())
            container = av.open(self.video_dict[key])
            for i in range(container.streams.video[0].frames):
                yield self.annotation_to_label(annotation_idx)

    def __iter__(self):
        for key in self.keys:
            video_path = self.video_dict[key]
            annotation_path = self.annotation_dict[key]
            with open(annotation_path, 'r') as f:
                annotation_idx = int(f.read().strip())
            container = av.open(video_path)
            for i, frame in enumerate(container.decode(video=0)):
                yield frame.to_image(), self.annotation_to_label(annotation_idx)

def get_source(path, dataset_name):

    if dataset_name == 'ucf101':
        return UCF101Source(path)
    elif dataset_name == 'mike':
        return MikeSourceV2(path)
    elif dataset_name == 'abby':
        return AbbySource(path)

class PrecomputedDatasetV2(Dataset):
    '''
    Dataset mounted on precomputed sliding window clips and labels. 
    Corresponding clips and labels have the same name but live in different folders
    '''
    def __init__(self, path, categories, transform=None, feature_model=None, min_ctime=None):
        '''
        path should be contain 2 subfolders: frames and labels
        '''
        self.label_type = "onehot"
        self.path = path
        self.transform = transform
        self.categories = categories
        with step_timer("fetching files", verbose=PROFILE_LOADING):
            file_paths = get_files_of_type(self.path, ".npy", min_ctime=min_ctime)
        print(f"found {len(file_paths)} files")
        with step_timer("filtering files", verbose=PROFILE_LOADING):
            self.label_paths = [p for p in file_paths if "labels" in p]
            INPUT_TYPE = "inputs" if feature_model is None else f"{feature_model}_features"
            self.input_paths = [p for p in file_paths if INPUT_TYPE in p]
        print(f"found {len(self.input_paths)} input files")
        print(f"found {len(self.label_paths)} label files")
        print(self.input_paths[0])
        print(self.label_paths[0])
        with step_timer("getting dicts", verbose=PROFILE_LOADING):
            self.input_dict, self.label_dict, self.keys = get_dicts_and_common_keys(self.input_paths, self.label_paths)
        print(f"found {len(self.keys)} common files")

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        with step_timer(f"loading {key}", verbose=False):
            input = torch.from_numpy(np.load(self.input_dict[key])).float()
            label = torch.from_numpy(np.load(self.label_dict[key]))
        if self.transform:
            input = self.transform(input)
        return input, label

    def get_summary(self):
        summary = {}
        summary['metadata'] = {
            'path': self.path,
            'categories': self.categories,
            'label_type': self.label_type
        }
        label_count = torch.zeros(len(self.categories))
        for key in self.keys:
            label = torch.from_numpy(np.load(self.label_dict[key]))
            assert label.shape == (len(self.categories),), f"label shape {label.shape} does not match categories {self.categories}"
            label_count += label
    
        summary['label_count'] = label_count.tolist()
        summary['dataset_size'] = len(self)
        return summary

class SlidingWindowConfig(BaseModel):
    window_size: int
    tolerance_region: int 
    samples_per_window: int
    step_size: int
    is_image_dataset: bool  
    shuffle:  bool
    patch_type: str
    patch_h: int
    patch_w: int
    temporal_sample_interval: int
    MAX_BUFFER_SIZE: int

class DatasetConfig(BaseModel):
    categories: list
    total_frames: int
    label_type: str

class DatasetBuilder():
    def __init__(self, path, dataset_name, style, transform=None, precomputed=False, feature_model=None, min_ctime=None):
        self.path = path
        self.set_sliding_window_config(yaml.safe_load(open("config/sliding_style.yml", "r"))[style])
        self.source = get_source(path, dataset_name)
        self.set_dataset_config(self.source.get_config())
        self.input_transform = None
        self.transform = transform
        self.precomputed = precomputed
        if feature_model: assert transform is None, "cannot transform extracted features"
        self.feature_model = feature_model
        self.min_ctime = min_ctime
        if self.min_ctime: assert precomputed, "min_ctime only works with precomputed datasets"

    def set_sliding_window_config(self, config_dict):
        self.swconfig = SlidingWindowConfig(**config_dict)

    def set_transform(self, transform):
        self.transform = transform

    def set_dataset_config(self, config_dict):
        self.dsconfig = DatasetConfig(**config_dict)

    def set_dsconfig_attr(self, attr_name, value):
        if not hasattr(self.dsconfig, attr_name):
            raise AttributeError(f"Config has no attribute '{attr_name}'")
        setattr(self.dsconfig, attr_name, value)

    def set_swconfig_attr(self, attr_name, value):
        if not hasattr(self.swconfig, attr_name):
            raise AttributeError(f"Config has no attribute '{attr_name}'")
        setattr(self.swconfig, attr_name, value)

    def build(self):
        if self.precomputed: 
            #if precomputed is true, then the sliding style information should be contained in the path
            return PrecomputedDatasetV2(
                self.path, 
                self.dsconfig.categories, 
                self.transform, 
                self.feature_model,
                self.min_ctime
            )
        else: 
            config = {
                **self.swconfig.model_dump(), 
                **self.dsconfig.model_dump(), 
                'input_transform': self.transform,
            }
            dataset = BaseSlidingWindowDataset(**config)
            dataset.set_source(self.source)
            return dataset   