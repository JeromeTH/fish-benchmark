import numpy as np
from torchvision.datasets import Caltech101
from torch.utils.data import Subset
import torch
def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    
    Example:
        container = av.open(video_path)
        frames = read_video_pyav(container, indices)
    '''
    frames = []
    start_index = indices[0]
    end_index = indices[-1]
    container.seek(0)
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

def load_caltech101(train_augs, test_augs):
    dataset = Caltech101(root='.', target_type = "category", transform= train_augs, download=True)
    # print(dataset.categories)
    #generate random indices to be the training set * 0.8, will split using SubSet later to be the training set 
    random_perm = torch.randperm(len(dataset))
    training_indices = random_perm[:int(0.8 * len(dataset))]  # 80% for training
    testing_indices = random_perm[int(0.8 * len(dataset)):]  # 20% for testing
    train_dataset = Subset(dataset, training_indices)
    test_dataset = Subset(dataset, testing_indices) 
     # Reattach `categories` attribute
    train_dataset.categories = dataset.categories
    test_dataset.categories = dataset.categories
    train_dataset.transform = train_augs
    test_dataset.transform = test_augs
    return train_dataset, test_dataset
