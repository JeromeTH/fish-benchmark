'''
In this file, we want to train the video MAE model for video classification with pytorch lighning module
'''
import torch
import lightning as L
from transformers import AutoImageProcessor
from fish_benchmark.models import DINOImageClassifier
from torchvision.datasets import Caltech101
from torch.utils.data import Subset
from fish_benchmark.utils import load_caltech101
from pytorch_lightning.loggers import WandbLogger

data_path = '/share/j_sun/jth264/UCF101_subset'

wandb_logger = WandbLogger(
    project="dino_training",    
    save_dir="./logs",
    log_model=True
)

if __name__ == '__main__':
    print("Loading data...")
    processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    image_transform = lambda img: processor(img, return_tensors="pt").pixel_values.squeeze(0)
    train_dataset, test_dataset = load_caltech101(train_augs=image_transform, test_augs=image_transform)
    print("Data loaded.")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = DINOImageClassifier(num_classes=len(train_dataset.categories))
    trainer = L.Trainer(max_epochs=1, logger=wandb_logger)
    #train the model
    trainer.fit(model, train_dataloader)
    #evaluate the model
    trainer.test(model, test_dataloader)
    #save the model
    trainer.save_checkpoint("dino_model.ckpt")
