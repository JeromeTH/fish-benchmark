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



class BaseSlidingWindowDataset():
    def __init__(self, 
                 input_transform: callable=None, 
                 annotation_to_label: callable=None,
                 label_type: str= "onehot", 
                 window_size: int=16, 
                 tolerance_region: int = 16, 
                 samples_per_window: int= 16, 
                 step_size = 1, 
                 categories: list = None, 
                 is_image_dataset = False, 
                 shuffle = False,
                 patch = False):
        self.input_transform = input_transform
        self.annotation_to_label = annotation_to_label
        self.label_type = label_type
        self.window_size = window_size
        assert window_size % samples_per_window == 0, f"window_size {window_size} should be a factor of samples_per_window {samples_per_window}"
        assert tolerance_region <= (window_size - 1)//2, f"tolerance_region {tolerance_region} should be less than or equal to window_size {window_size//2}"
        assert label_type == "onehot", 'currently only onehot is supported'
        self.tolerance_region = tolerance_region
        self.step_size = step_size
        self.categories = categories
        self.samples_per_window = samples_per_window
        self.is_image_dataset = is_image_dataset
        if self.is_image_dataset: assert self.samples_per_window ==1, "samples per window should be 1 for image datasets"
        self.shuffle = shuffle
        self.patch = patch


    def create_video_dataset(self, annotated_video_frames):
        clips = []
        labels = []

        seen_images = []
        seen_annotations = []
        for i, (image, annotation) in enumerate(annotated_video_frames):
            #image is of type PIL image
            seen_images.append(np.array(image.convert('RGB')))
            seen_annotations.append(annotation)
            if i + 1 < self.window_size:
                #currently there is i + 1 frames in the buffer
                continue

            if i % self.step_size != 0:
                continue
            
            gap = int(self.window_size/self.samples_per_window)
            clip = np.stack([img for img in seen_images[-self.window_size::gap]])
            mid_idx = i - int(self.window_size/2)
            relevant_annotations = seen_annotations[mid_idx - self.tolerance_region: mid_idx + self.tolerance_region + 1]
            relevant_labels = torch.stack([self.annotation_to_label(annotation) for annotation in relevant_annotations]) 
            relevant_labels = relevant_labels[:, :len(self.categories)] #drop extra incomplete labels
            assert relevant_labels.shape == (len(relevant_annotations), len(self.categories)), f"relevant_labels shape {relevant_labels.shape} does not match relevant_annotations shape [{len(relevant_annotations)}, {len(self.categories)}]"
            
            #turn clip into a tensor
            if self.input_transform:
                clip = self.input_transform(clip)
            else: 
                clip = torch.from_numpy(clip)

            if self.is_image_dataset: clip = clip.squeeze()

            unioned_labels = torch.any(relevant_labels.bool(), dim=0).float()
            clips.append(clip)
            labels.append(unioned_labels)

        if len(clips) == 0:
            #if the clips is less than the window size
            return TensorDataset(torch.empty(0), torch.empty(0))
        
        clips = torch.stack(clips)
        labels = torch.stack(labels)
        
        if self.shuffle: 
            perm = torch.randperm(len(clips))
            clips = clips[perm]
            labels = labels[perm]

        return TensorDataset(clips, labels)
        
class MikeDataset(IterableDataset, BaseSlidingWindowDataset):
    def __init__(self, path, train = True, transform=None, label_type = "onehot", window_size=16, tolerance_region = 16, samples_per_window = 16, step_size = 1, is_image_dataset = False, shuffle = False, patch=False):
        self.path = os.path.join(path, "train" if train else "test")
        self.behavior_idx_map = load_behavior_idx_map('behavior_categories.json')
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
            patch = patch
        )

    def __iter__(self):
        video_names = os.listdir(self.path)
        for video in video_names:
            tar_batches = os.listdir(os.path.join(".."+self.path, video))
            tar_batches.sort()
            tar_file_paths = [os.path.join(self.path, video, tar_batch) for tar_batch in tar_batches]
            video_frames = wds.WebDataset(tar_file_paths).decode("pil").to_tuple("png", "json")
            dataset = self.create_video_dataset(video_frames)
            for clip, label in dataset: 
                yield clip, label


class AbbyDataset(IterableDataset, BaseSlidingWindowDataset):
    def __init__(self, path, train = True, transform=None, label_type = "onehot", window_size=16, tolerance_region = 16, samples_per_window = 16, step_size = 1, is_image_dataset = False, shuffle = False):
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
            shuffle=shuffle       
        )

    def __iter__(self):
        for annotation_path in os.listdir(self.path):
            # print(annotation_path)
            # print(get_files_of_type(os.path.join(self.path, annotation_path), ".mp4"))
            track_paths = sorted(get_files_of_type(os.path.join(self.path, annotation_path), ".mp4"))
            label_paths = sorted(get_files_of_type(os.path.join(self.path, annotation_path), ".txt"))
            label_dict = {
                os.path.splitext(os.path.basename(p))[0]: p
                for p in label_paths
            }
            # print(track_paths)
            for track_path in track_paths:
                # print(f"Processing {track_path}")
                track_name = os.path.splitext(os.path.basename(track_path))[0]
                # print(f"Processing {track_name}")
                label_path = label_dict.get(track_name)
                if label_path is None:
                    # print(f"Warning: No label file found for {track_path}. Skipping.")
                    continue
                
                container = av.open(track_path)
                label = np.loadtxt(label_path, delimiter='\t', dtype=int)
                assert label.shape[0] == container.streams.video[0].frames, f"Number of frames in {track_path} does not match number of annotations in {label_path}"
                # print(f"iterating over {track_path} with {label.shape[0]} frames")
                def annotated_frame_iterator():
                    for i, frame in enumerate(container.decode(video=0)):
                        yield frame.to_image(), label[i]

                dataset = self.create_video_dataset(annotated_frame_iterator())
                for clip, label in dataset:
                    yield clip, label


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

def get_dataset(dataset_name, path, augs=None, train=True, label_type = "onehot", model_name = None, shuffle = False):
    if dataset_name == 'UCF101':
        dataset = UCF101(path, train=train, transform=augs, label_type=label_type)
    elif dataset_name == 'Caltech101':
        dataset = CalTech101WithSplit(path, train=train, transform=augs, label_type=label_type)
    elif dataset_name == 'HeinFishBehavior':
        # dataset = HeinFishBehavior(path, transform=augs, label_type=label_type, train=train)
        dataset = MikeDataset(
            path,
            train=train, 
            transform=augs,
            label_type=label_type,
            window_size = 1, 
            tolerance_region = 0,
            samples_per_window = 1,
            step_size = 1, 
            is_image_dataset=True, 
            shuffle = shuffle
        )
    elif dataset_name == 'HeinFishBehaviorSlidingWindow':
        # dataset = HeinFishBehaviorSlidingWindow(path, transform=augs, label_type=label_type, train=train)
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
            shuffle = shuffle
            patch = patch
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
            shuffle = shuffle
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
            shuffle = shuffle
        )
    elif dataset_name == 'HeinFishBehaviorPrecomputed': 
        behavior_idx_map = load_behavior_idx_map('behavior_categories.json')
        categories = list(behavior_idx_map.keys())
        dataset = PrecomputedDataset(path, model_name=model_name, categories=categories, transform=augs, train=train)
    elif dataset_name == 'HeinFishBehaviorSlidingWindowPrecomputed':
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