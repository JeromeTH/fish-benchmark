'''
In this file, we want to train the video MAE model for video classification with pytorch lighning module
'''
import torch
import lightning as L
from fish_benchmark.models import get_input_transform, MediaClassifier
from fish_benchmark.data.dataset import get_dataset
from fish_benchmark.litmodule import get_lit_module
from pytorch_lightning.loggers import WandbLogger
import wandb
import yaml

PRETRAINED_MODEL = 'clip'
CLASSIFIER = 'mlp'
DATASET = 'HeinFishBehaviorPrecomputed'

dataset_config = yaml.safe_load(open('config/datasets.yml', 'r'))
model_config = yaml.safe_load(open('config/models.yml', 'r'))
assert(dataset_config[DATASET]['type'] == model_config[PRETRAINED_MODEL]['type']), "Dataset and model type mismatch"
LABEL_TYPE = dataset_config[DATASET]['label_types'][0]
available_gpus = torch.cuda.device_count()
print(f"Available GPUs: {available_gpus}")
project = f"{PRETRAINED_MODEL}_training"

if __name__ == '__main__':
    with wandb.init(
        project=project,
        notes="Freezing the model parameters and only tuning the classifier head",
        tags=[PRETRAINED_MODEL, CLASSIFIER, DATASET, LABEL_TYPE],
        config={"epochs": 20, "learning_rate": 0.001, "batch_size": 32},
        dir="./logs"
    ) as run:
        wandb_logger = WandbLogger(
            project=run.project,    
            save_dir="./logs",
            log_model=True
        )
        print("Loading data...")
        train_dataset = get_dataset(DATASET, 
                                    dataset_config[DATASET]['path'], 
                                    augs = get_input_transform(PRETRAINED_MODEL) if not dataset_config[DATASET]['preprocessed'] else None, 
                                    train=True, 
                                    label_type=LABEL_TYPE, 
                                    model_name=PRETRAINED_MODEL)
        test_dataset = get_dataset(DATASET, 
                                   dataset_config[DATASET]['path'], 
                                   augs = get_input_transform(PRETRAINED_MODEL) if not dataset_config[DATASET]['preprocessed'] else None, 
                                   train=False, 
                                   label_type=LABEL_TYPE, 
                                   model_name=PRETRAINED_MODEL)
        print("Data loaded.")
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=run.config['batch_size'])
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=run.config['batch_size'])
        
        model = MediaClassifier(
            num_classes=len(train_dataset.categories), 
            pretrained_model=PRETRAINED_MODEL,
            classifier_type=CLASSIFIER, 
            freeze_pretrained=True,
        )

        lit_module = get_lit_module(model, learning_rate=run.config['learning_rate'], label_type=LABEL_TYPE)
        trainer = L.Trainer(max_epochs=run.config['epochs'], logger=wandb_logger)
        trainer.fit(lit_module, train_dataloader)
        trainer.test(lit_module, test_dataloader)
        trainer.save_checkpoint(f"{PRETRAINED_MODEL}_model.ckpt")