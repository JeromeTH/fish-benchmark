'''
In this file, we want to train the video MAE model for video classification with pytorch lighning module
'''
import torch
import lightning as L
from transformers import AutoImageProcessor
from fish_benchmark.models import DINOImageClassifier
from fish_benchmark.utils import load_caltech101
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

data_path = '/share/j_sun/jth264/UCF101_subset'

wandb_logger = WandbLogger(
    project="dino_training",    
    save_dir="./logs",
    log_model=True
)
# You can set a description for the experiment
wandb_logger.experiment.config.update({
    "experiment_description": 
    "This is training DINO only tuning the classifier head by setting model.parameters() requires_grad to False",
})

# Define a ModelCheckpoint callback to save checkpoints
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",  # Monitor the validation loss
    dirpath=f"./logs/{wandb_logger.experiment.id}",  # Save checkpoints in a folder named by the WandB run ID
    filename="best_checkpoint",  # Save the best model with this filename
    save_top_k=1,  # Keep only the best checkpoint
    mode="min",  # Minimize validation loss
    every_n_epochs=1  # Save checkpoints every epoch
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
    trainer = L.Trainer(max_epochs=5, logger=wandb_logger) #removed callbacks=[checkpoint_callback]
    #train the model
    trainer.fit(model, train_dataloader)
    #evaluate the model
    trainer.test(model, test_dataloader)
    #save the model
    trainer.save_checkpoint("dino_model.ckpt")
