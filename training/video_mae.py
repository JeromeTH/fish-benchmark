'''
In this file, we want to train the video MAE model for video classification with pytorch lighning module
'''
import torch
from torch.utils.data.dataset import Dataset
import lightning as L
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from transformers import AutoImageProcessor, VideoMAEForPreTraining, VideoMAEConfig
from fish_benchmark.dataset import UCF101
from fish_benchmark.models import VideoMAEModel

data_path = '/share/j_sun/jth264/UCF101_subset'

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
    print("classes:", train_dataset.classes)
    trainer = L.Trainer(max_epochs=1)
    model = VideoMAEModel(classes=train_dataset.classes)
    #train the model
    trainer.fit(model, train_dataloader)
    #evaluate the model
    trainer.test(model, test_dataloader)
    #save the model
    trainer.save_checkpoint("video_mae_model.ckpt")