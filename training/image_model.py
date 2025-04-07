'''
In this file, we want to train the video MAE model for video classification with pytorch lighning module
'''
import torch
import lightning as L
from fish_benchmark.models import get_processor, ImageClassifier
from fish_benchmark.data.dataset import get_dataset
from fish_benchmark.litmodule import LitClassifierModule
from pytorch_lightning.loggers import WandbLogger
import wandb

PRETRAINED_MODEL = 'clip'
CLASSIFIER = 'linear'
DATA_PATH = '.'
DATASET = 'Caltech101'

available_gpus = torch.cuda.device_count()
print(f"Available GPUs: {available_gpus}")
project = f"{PRETRAINED_MODEL}_training"

if __name__ == '__main__':
    with wandb.init(
        project=project,
        notes="Freezing the model parameters and only tuning the classifier head",
        tags=[PRETRAINED_MODEL],
        config={"epochs": 10, "learning_rate": 0.001, "batch_size": 32},
        dir="./logs"
    ) as run:
        wandb_logger = WandbLogger(
            project=run.project,    
            save_dir="./logs",
            log_model=True
        )
        print("Loading data...")
        processor = get_processor(PRETRAINED_MODEL)
        image_transform = lambda img: processor(images = img, return_tensors="pt").pixel_values.squeeze(0)
        train_dataset = get_dataset(DATASET, DATA_PATH, augs =image_transform, train=True, label_type="onehot")
        test_dataset = get_dataset(DATASET, DATA_PATH, augs = image_transform, train=False, label_type="onehot")
        print("Data loaded.")
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=run.config['batch_size'])
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=run.config['batch_size'])
        
        model = ImageClassifier(
            num_classes=len(train_dataset.categories), 
            pretrained_model=PRETRAINED_MODEL,
            classifier_type=CLASSIFIER, 
            freeze_pretrained=True,
        )

        lit_module = LitClassifierModule(model, learning_rate=run.config['learning_rate'])
        trainer = L.Trainer(max_epochs=10, logger=wandb_logger)
        #train the model
        trainer.fit(lit_module, train_dataloader)
        #evaluate the model
        trainer.test(lit_module, test_dataloader)
        #save the model
        trainer.save_checkpoint(f"{PRETRAINED_MODEL}_model.ckpt")