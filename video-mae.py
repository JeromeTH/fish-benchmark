'''
In this file, we want to train the video MAE model for video classification with pytorch lighning module
'''
import torch
from torch.utils.data.dataset import Dataset
import lightning as L
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from transformers import VideoMAEForVideoClassification
from transformers import AutoImageProcessor, VideoMAEForPreTraining, VideoMAEConfig
import av
import os
import numpy as np  

data_path = '/share/j_sun/jth264/UCF101_subset'
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
    def __init__(self, data_path, train= True, clip_len = 16, transform=None):
        self.data_path = data_path
        self.train = train
        self.transform = transform
        self.data = []
        self.labels = []
        self.classes = []
        self.load_data()
    
    def read_video_pyav(self, container, indices):
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
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        return np.stack([x.to_ndarray(format="rgb24") for x in frames])

    def sample_frame_indices(self, clip_len, frame_sample_rate, seg_len):
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

    def load_data(self):
        path = os.path.join(self.data_path, 'train' if self.train else 'test')
        for class_name in os.listdir(path):
            print(f"Loading {class_name}...")
            class_path = os.path.join(path, class_name)
            if not os.path.isdir(class_path): continue
            self.classes.append(class_name)
            class_idx = len(self.classes) - 1
            for video_name in os.listdir(class_path):
                #print(f"Loading {video_name}...")
                video_path = os.path.join(class_path, video_name) 
                if not os.path.isfile(video_path): continue
                container = av.open(video_path)
                indices = self.sample_frame_indices(clip_len=16, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
                video = self.read_video_pyav(container, indices)
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

class VideoMAEModel(L.LightningModule):
    def __init__(self, num_classes):
        super(VideoMAEModel, self).__init__()
        self.model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

    def forward(self, x):
        return self.model(x).logits
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('test_loss', loss)
        return loss
    
    def configure_optimizers(self):
        #only update the classifier
        optimizer = torch.optim.Adam(self.model.classifier.parameters(), lr=1e-4)
        return optimizer


if __name__ == '__main__':
    print("Loading data...")
    image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    video_transform = lambda video: image_processor(list(video), return_tensors="pt", use_fast = True).pixel_values.squeeze(0)
    train_dataset = UCF101(data_path, transform=video_transform)
    test_dataset = UCF101(data_path, train=False, transform=video_transform)
    print("Data loaded.")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    #Question: How does the image preprocessing stage come into the pipeline if we want to make each entity as decoupled as possible. 
    #Naturally, we make dataset only responsible for loading the data, and the model for training. but we also don't want to preprocess everytime we input the data to the model.
    #Answer: We can preprocess the data in the __getitem__ method of the dataset class, and store the preprocessed data in the dataset.
   
    trainer = L.Trainer(max_epochs=10)
    model = VideoMAEModel(num_classes=len(train_dataset.classes))
    #train the model
    trainer.fit(model, train_dataloader)
    #evaluate the model
    trainer.test(model, test_dataloader)
