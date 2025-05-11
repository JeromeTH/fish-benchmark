'''
run oriented evaluation
'''
import os
import yaml
import torch
import wandb
import pytz
from datetime import datetime
from lightning.pytorch import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from fish_benchmark.models import ModelBuilder
from fish_benchmark.litmodule import LitBinaryClassifierModule
from fish_benchmark.data.dataset import DatasetBuilder
from wandb.apis.public.runs import Run

# ==== CONFIG ====
TRAIN_PROJECT = "fish-benchmark/abby"  # <- CHANGE THIS
EVAL_PROJECT = "fish-benchmark/abby-eval"  # <- CHANGE THIS
CUTOFF_DATE = datetime(2024, 12, 1, tzinfo=pytz.UTC)
BATCH_SIZE = 32
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
filt = {
    "dataset": "abby",
    "sliding_style": "frames",
    "backbone": "dino_large",
    "pooling": "mean",
    "classifier": "mlp",
    # "optimizer": "adam",
    # "model": ,
    # "epochs": 10,
    # "sampler": "balanced",
    # "shuffle": False,
    # "batch_size": 32,
    # "weight_decay": "0.01",
    # "learning_rate": 5e-5,
    # "max_samples_per_class": 1000
}
def match_config(config, filt):
    return all(config.get(k) == v for k, v in filt.items() if v is not None)

def main():
    api = wandb.Api()
    dataset_config = yaml.safe_load(open("config/datasetsv2.yml", "r"))
    runs = api.runs(TRAIN_PROJECT, filters={"state": "finished"})
    print(runs)
    for run in runs: 
        # config = {k: v["value"] for k, v in run.json_config.items()}
        # print(run)
        # print(run.json_config)
        if match_config(run.config, filt):
            print(run)
            print(run.config)
    
if __name__ == "__main__":
    # run = Run()
    main()
