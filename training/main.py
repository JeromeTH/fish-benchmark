'''
In this file, we want to train the video MAE model for video classification with pytorch lighning module
'''
import torch
import lightning as L
from fish_benchmark.models import get_input_transform, MediaClassifier, MediaClassifier 
from fish_benchmark.data.dataset import get_dataset
from fish_benchmark.litmodule import get_lit_module
from pytorch_lightning.loggers import WandbLogger
import wandb
import yaml
import argparse
from lightning.pytorch.callbacks import ModelCheckpoint
import sys

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--classifier", default="mlp")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--label_type", default='onehot')
    parser.add_argument("--epochs", default=1)
    parser.add_argument("--lr", default=.00005)
    parser.add_argument("--batch_size", default=32)
    parser.add_argument("--shuffle", default=True)

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    PRETRAINED_MODEL = args.model
    CLASSIFIER = args.classifier
    DATASET = args.dataset
    LABEL_TYPE = args.label_type
    EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    BATCH_SIZE = args.batch_size
    SHUFFLE = args.shuffle

    dataset_config = yaml.safe_load(open('config/datasets.yml', 'r'))
    model_config = yaml.safe_load(open('config/models.yml', 'r'))
    assert(dataset_config[DATASET]['type'] == model_config[PRETRAINED_MODEL]['type']), "Dataset and model type mismatch"
    assert(LABEL_TYPE in dataset_config[DATASET]['label_types']), f"Label type {LABEL_TYPE} not supported for dataset {DATASET}"
    available_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {available_gpus}")

    PROJECT = f"{dataset_config[DATASET]['training_project']}_training"
    print(type(dataset_config[DATASET]['preprocessed']))

    with wandb.init(
        project=PROJECT,
        entity = "fish-benchmark",
        notes="Freezing the model parameters and only tuning the classifier head",
        tags=[PRETRAINED_MODEL, CLASSIFIER, DATASET, LABEL_TYPE],
        config={"epochs": EPOCHS, "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE, "optimizer": "adam", "classifier": CLASSIFIER, "dataset": DATASET, "label_type": LABEL_TYPE, "shuffle": SHUFFLE},
        dir="./logs"
    ) as run:
        wandb_logger = WandbLogger(
            project=run.project,    
            save_dir="./logs",
            log_model="best"
        )
        print("Loading data...")
        train_dataset = get_dataset(DATASET, 
                                    dataset_config[DATASET]['path'], 
                                    augs = get_input_transform(PRETRAINED_MODEL) if not dataset_config[DATASET]['preprocessed'] else None, 
                                    train=True, 
                                    label_type=LABEL_TYPE, 
                                    model_name=PRETRAINED_MODEL, 
                                    shuffle=SHUFFLE)
        test_dataset = get_dataset(DATASET, 
                                   dataset_config[DATASET]['path'], 
                                   augs = get_input_transform(PRETRAINED_MODEL) if not dataset_config[DATASET]['preprocessed'] else None, 
                                   train=False, 
                                   label_type=LABEL_TYPE, 
                                   model_name=PRETRAINED_MODEL, 
                                   shuffle=SHUFFLE)
    
        print("Data loaded.")
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=run.config['batch_size'], num_workers=7)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=run.config['batch_size'], num_workers=7)
        
        model = MediaClassifier(
            num_classes=len(train_dataset.categories), 
            pretrained_model=PRETRAINED_MODEL,
            classifier_type=CLASSIFIER, 
            freeze_pretrained=True,
        )

        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            save_top_k=1,
            mode="min",
            dirpath=f"./checkpoints/{run.id}",
            filename="best-{epoch:02d}-{val_loss:.2f}",
        )

        lit_module = get_lit_module(model, learning_rate=run.config['learning_rate'], label_type=LABEL_TYPE)
        tqdm_disable = not sys.stdout.isatty()
        print(f"Are we in an interactive terminal? {not tqdm_disable}")
        trainer = L.Trainer(max_epochs=run.config['epochs'], 
                            logger=wandb_logger, 
                            log_every_n_steps= 10, 
                            callbacks=[checkpoint_callback], 
                            val_check_interval=100, 
                            limit_val_batches=1)
        
        trainer.fit(lit_module, train_dataloader, val_dataloaders=test_dataloader)

        if checkpoint_callback.best_model_path:
            artifact = wandb.Artifact(
                name=f"model-{run.id}",
                type="model",
                metadata={
                    "tags": run.tags,
                    "config": dict(run.config),
                    "notes": run.notes
                }
            )
            artifact.add_file(checkpoint_callback.best_model_path)
            run.log_artifact(artifact)