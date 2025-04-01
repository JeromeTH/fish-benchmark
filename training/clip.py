'''
In this file, we want to train the video MAE model for video classification with pytorch lighning module
'''
import torch
import lightning as L
from transformers import AutoProcessor
from fish_benchmark.models import CLIPImageClassifier
from fish_benchmark.utils import load_caltech101
from pytorch_lightning.loggers import WandbLogger
import wandb

data_path = '/share/j_sun/jth264/UCF101_subset'
available_gpus = torch.cuda.device_count()
print(f"Available GPUs: {available_gpus}")
project = "clip_training"

if __name__ == '__main__':
    with wandb.init(
        project=project,
        notes="Freezing the model parameters and only tuning the classifier head",
        tags=["dino", "first_stage", "clip_classification"],
        config={"epochs": 10, "learning_rate": 0.001, "batch_size": 32},
        dir="./logs"
    ) as run:
        wandb_logger = WandbLogger(
            project=run.project,    
            save_dir="./logs",
            log_model=True
        )
        print("Loading data...")
        processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        image_transform = lambda img: processor(images = img, return_tensors="pt").pixel_values.squeeze(0)
        train_dataset = load_caltech101(augs =image_transform, train = True)
        test_dataset = load_caltech101(augs = image_transform, train=False)
        print("Data loaded.")
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=run.config['batch_size'], shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=run.config['batch_size'], shuffle=False)
        
        trainer = L.Trainer(max_epochs=10, logger=wandb_logger)
        model = CLIPImageClassifier(
            num_classes=len(train_dataset.categories), 
            learning_rate=run.config['learning_rate']
        )
        #train the model
        trainer.fit(model, train_dataloader)
        #evaluate the model
        trainer.test(model, test_dataloader)
        #save the model
        trainer.save_checkpoint("clip_model.ckpt")