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
from abc import ABC, abstractmethod
from fish_benchmark.utils import get_files_of_type
from queue import Queue 
import time
from contextlib import contextmanager
from torchvision import transforms
import yaml

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

class HeinFishBehavior(IterableDataset):
    def __init__(self, path, transform=None, label_type = "onehot", train=True):
        super().__init__()
        self.path = path
        tar_files = get_files_of_type(path, ".tar")
        self.data = wds.WebDataset(tar_files, shardshuffle=False).decode("pil").to_tuple("png", "json")
        self.transform = transform
        self.label_type = label_type
        self.behavior_idx_map = load_behavior_idx_map('behavior_categories.json')
        self.categories = list(self.behavior_idx_map.keys())

    def __iter__(self):
        for sample in self.data: 
            image, annotation = sample
            if self.transform:
                image = self.transform(image)

            labels = [self.behavior_idx_map[behavior] for behavior in parse_annotation(annotation)]
            if self.label_type == 'onehot':
                yield image, onehot(len(self.categories), labels)
            elif self.label_type == 'categorical': 
                yield image, torch.tensor(labels)
            else:
                raise ValueError(f"label_type {self.label_type} not recognized.")

# @contextmanager
# def step_timer(name):
#     start = time.time()
#     yield
#     end = time.time()
#     print(f"[{name}] took {end - start:.6f} seconds")

class HeinFishBehaviorSlidingWindow(IterableDataset):
    def __init__(self, path, transform=None, label_type = "onehot", window_size=16):
        super().__init__()
        self.path = path
        self.transform = transform
        self.label_type = label_type
        self.window_size = window_size
        self.behavior_idx_map = load_behavior_idx_map('behavior_categories.json')
        self.categories = list(self.behavior_idx_map.keys())

    def __iter__(self):
        video_names = os.listdir(self.path)
        for video in video_names:
            tar_batches = os.listdir(os.path.join(self.path, video))
            tar_batches.sort()
            tar_file_paths = [os.path.join(self.path, video, tar_batch) for tar_batch in tar_batches]
            dataset = wds.WebDataset(tar_file_paths).decode("pil").to_tuple("png", "json")
            seen_images = []
            seen_annotations = []
            for i, (image, annotation) in enumerate(dataset):
                seen_images.append(image)
                seen_annotations.append(annotation)
                if i < self.window_size:
                    continue

                # with step_timer("sliding_window"):
                clip = np.stack([np.array(img.convert('RGB')) for img in seen_images[-self.window_size:]])
                mid_annotation = seen_annotations[-int((self.window_size)/2)]

                # with step_timer("transform"):
                if self.transform:
                    clip = self.transform(clip)
                        
                labels = [self.behavior_idx_map[behavior] for behavior in parse_annotation(mid_annotation)]
                if self.label_type == 'onehot':
                    yield clip, onehot(len(self.categories), labels)
                elif self.label_type == 'categorical': 
                    yield clip, torch.tensor(labels)
                else:
                    raise ValueError(f"label_type {self.label_type} not recognized.")

class AbbyDataset(IterableDataset):
    def __init__(self, path, transform=None, label_type = "onehot", train=True):
        super().__init__()
        self.path = path
        self.transform = transform
        self.label_type = label_type
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(cur_dir, 'abby_dset_categories.json')
        self.categories = json.load(open(full_path, 'r'))
        self.train = train
    
    def __iter__(self):
        for annotation_path in os.listdir(self.path):
            track_paths = sorted(get_files_of_type(os.path.join(self.path, annotation_path), ".mp4"))
            annotation_paths = sorted(get_files_of_type(os.path.join(self.path, annotation_path), ".txt"))
            assert len(track_paths) == len(annotation_paths), f"Number of tracks and annotations do not match in {annotation_path}"
            for track_path, annotation_path in zip(track_paths, annotation_paths):
                container = av.open(track_path)
                annotation = np.loadtxt(annotation_path, delimiter='\t', dtype=str)
                assert annotation.shape[0] == container.streams.video[0].frames, f"Number of frames in {track_path} does not match number of annotations in {annotation_path}"
                for i, frame in enumerate(container.decode(video=0)):
                    yield frame.to_image(), torch.tensor(annotation[i])

class PrecomputedDataset(IterableDataset):
    def __init__(self, path, model_name, transform=None, train=True):
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
        self.behavior_idx_map = load_behavior_idx_map('behavior_categories.json')
        self.categories = list(self.behavior_idx_map.keys())

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

def get_dataset(dataset_name, path, augs=None, train=True, label_type = "onehot", model_name = None):
    if dataset_name == 'UCF101':
        dataset = UCF101(path, train=train, transform=augs, label_type=label_type)
    elif dataset_name == 'Caltech101':
        dataset = CalTech101WithSplit(path, train=train, transform=augs, label_type=label_type)
    elif dataset_name == 'HeinFishBehavior':
        dataset = HeinFishBehavior(path, transform=augs, label_type=label_type, train=train)
    elif dataset_name == 'HeinFishBehaviorSlidingWindow':
        dataset = HeinFishBehaviorSlidingWindow(path, transform=augs, label_type=label_type, train=train)
    elif dataset_name == 'HeinFishBehaviorPrecomputed': 
        dataset = PrecomputedDataset(path, model_name=model_name, transform=augs, train=train)
    elif dataset_name == 'HeinFishBehaviorSlidingWindowPrecomputed':
        dataset = PrecomputedDataset(path, model_name=model_name, transform=augs, train=train)
    else:
        raise ValueError(f"Dataset {dataset_name} not recognized.")
    return dataset