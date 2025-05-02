import torch
import lightning as L
from fish_benchmark.models import get_processor, VideoClassifier
from fish_benchmark.data.dataset import get_dataset
from fish_benchmark.litmodule import get_lit_module
from pytorch_lightning.loggers import WandbLogger
import wandb


PRETRAINED_MODEL = 'videomae'
CLASSIFIER = 'linear'
DATA_PATH = '/share/j_sun/jth264/UCF101_subset'
DATASET = 'UCF101'
LABEL_TYPE = 'categorical'

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
        video_transform = lambda video: processor(list(video), return_tensors="pt").pixel_values.squeeze(0)
        train_dataset = get_dataset(DATASET, DATA_PATH, augs =video_transform, train=True, label_type=LABEL_TYPE)
        test_dataset = get_dataset(DATASET, DATA_PATH, augs = video_transform, train=False, label_type=LABEL_TYPE)
        print("Data loaded.")
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=run.config['batch_size'])
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=run.config['batch_size'])
        
        model = VideoClassifier(
            num_classes=len(train_dataset.categories), 
            pretrained_model=PRETRAINED_MODEL,
            classifier_type=CLASSIFIER, 
            freeze_pretrained=True,
        )

        lit_module = get_lit_module(model, learning_rate=run.config['learning_rate'], label_type=LABEL_TYPE)
        trainer = L.Trainer(max_epochs=5, logger=wandb_logger)
        trainer.fit(lit_module, train_dataloader)
        trainer.test(lit_module, test_dataloader)
        trainer.save_checkpoint(f"{PRETRAINED_MODEL}_model.ckpt")