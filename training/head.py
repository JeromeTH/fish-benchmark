'''
In this file, we want to train the video MAE model for video classification with pytorch lighning module
'''
import torch
import lightning as L
from fish_benchmark.models import get_input_transform, get_pretrained_model, ModelBuilder
from fish_benchmark.data.dataset import DatasetBuilder, MultiLabelBalancedSampler
from fish_benchmark.litmodule import get_lit_module
from pytorch_lightning.loggers import WandbLogger
import wandb
import yaml
import argparse
from lightning.pytorch.callbacks import ModelCheckpoint
import sys
from artifact import log_best_model, log_dataset_summary
import os
import json
# python training/head.py --classifier mlp --dataset abby --sliding_style frames --model dino
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", required=True)
    parser.add_argument("--pooling", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sliding_style", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--epochs", default=50)
    parser.add_argument("--lr", default=.00005)
    parser.add_argument("--batch_size", default=32)
    parser.add_argument("--shuffle", default=True)
    parser.add_argument("--label_type", default='onehot')
    parser.add_argument("--min_ctime", default = '1746331200.0')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    CLASSIFIER = args.classifier
    POOLING = args.pooling
    DATASET = args.dataset
    EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    BATCH_SIZE = args.batch_size
    SLIDING_STYLE = args.sliding_style
    SHUFFLE = args.shuffle
    MODEL = args.model
    LABEL_TYPE = args.label_type
    MIN_CTIME = float(args.min_ctime)

    dataset_config = yaml.safe_load(open('config/datasetsv2.yml', 'r'))
    available_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {available_gpus}")
    with wandb.init(
        project=DATASET,
        entity = "fish-benchmark",
        notes="Using precomputed embeddings",
        tags=[CLASSIFIER, POOLING, DATASET, SLIDING_STYLE, MODEL],
        config={"epochs": EPOCHS, 
                "learning_rate": LEARNING_RATE, 
                "batch_size": BATCH_SIZE, 
                "optimizer": "adam", 
                "classifier": CLASSIFIER, 
                "pooling": POOLING,
                "model": MODEL, 
                "dataset": DATASET, 
                "sliding_style": SLIDING_STYLE, 
                "shuffle": SHUFFLE},
        dir="./logs"
    ) as run:
        wandb_logger = WandbLogger(
            project=run.project,    
            save_dir="./logs",
            log_model="best"
        )
        print("Loading data...")
        train_dataset = DatasetBuilder(
            path = os.path.join(dataset_config[DATASET]['precomputed_path'], SLIDING_STYLE, 'train'), 
            dataset_name = DATASET,
            style=SLIDING_STYLE,
            transform=None, 
            precomputed=True, 
            feature_model=MODEL,
            min_ctime=MIN_CTIME,
        ).build()

        test_dataset = DatasetBuilder(
            path = os.path.join(dataset_config[DATASET]['precomputed_path'], SLIDING_STYLE, 'test'), 
            dataset_name = DATASET,
            style=SLIDING_STYLE,
            transform=None, 
            precomputed=True, 
            feature_model=MODEL,
        ).build()
        
        print("Data loaded.")
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=run.config['batch_size'], 
            num_workers=7, 
            shuffle=run.config['shuffle']
        )

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=run.config['batch_size'], 
            num_workers=7, 
            shuffle=False
        )
        
        # to get hidden size
        hidden_size = ModelBuilder().set_model(MODEL).get_hidden_size()
        classifier = (ModelBuilder()
                      .set_hidden_size(hidden_size)
                      .set_pooling(POOLING)
                      .set_classifier(CLASSIFIER, input_dim=hidden_size, output_dim=len(train_dataset.categories))
                      .build())
        
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            save_top_k=1,
            mode="min",
            dirpath=f"./checkpoints/{run.id}",
            filename="best-{epoch:02d}-{val_loss:.2f}",
        )

        lit_module = get_lit_module(classifier, learning_rate=run.config['learning_rate'], label_type=LABEL_TYPE)
        tqdm_disable = not sys.stdout.isatty()
        print(f"Are we in an interactive terminal? {not tqdm_disable}")
        trainer = L.Trainer(max_epochs=run.config['epochs'], 
                            logger=wandb_logger, 
                            log_every_n_steps= 50, 
                            callbacks=[checkpoint_callback])
        
        trainer.fit(lit_module, train_dataloader, test_dataloader)
        log_best_model(checkpoint_callback, run)
        #log_dataset_summary(train_dataset, run)