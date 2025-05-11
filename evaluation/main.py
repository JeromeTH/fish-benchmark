import argparse
import wandb
from fish_benchmark.litmodule import LitBinaryClassifierModule
from fish_benchmark.data.dataset import DatasetBuilder
import os
import yaml
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
import lightning as L
import json
import glob

dataset_config = yaml.safe_load(open('config/datasetsv2.yml', 'r'))
eval_config = yaml.safe_load(open('config/eval.yml', 'r'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
TEST_METRIC_DIR = os.path.join('logs', 'test_metrics')


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate a model on a dataset")
    parser.add_argument("--entity", type=str, required=True, help="WandB entity name")
    parser.add_argument("--project", type=str, required=True, help="WandB project name")
    parser.add_argument("--run", type=str, required=True, help="WandB run ID")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    #load the artifact
    api = wandb.Api()
    artifact = api.artifact(f"{args.entity}/{args.project}/model-{args.run}:latest", type="model")
    artifact_dir = artifact.download()
    ckpt_file = glob.glob(os.path.join(artifact_dir, "*.ckpt"))
    assert len(ckpt_file) == 1, f"Expected exactly one .ckpt file in {artifact_dir}"
    print(f"Artifact downloaded to {artifact_dir}")
    train_config = artifact.logged_by().config

    #define testing config
    DATASET = train_config.get("dataset", None)
    SLIDING_STYLE = train_config.get("sliding_style", None)
    BACKBONE = train_config.get("backbone", None)
    POOLING = train_config.get("pooling", None)
    CLASSIFIER = train_config.get("classifier", None)
    AGGREGATOR = train_config.get("aggregator", None)
    TEST_SLIDING_STYLE = eval_config[SLIDING_STYLE]

    #setup wandb logger for testing run
    config_dict = {
        "batch_size": 32,
        "dataset": DATASET,
        "backbone": BACKBONE,
        "pooling": POOLING,
        "classifier": CLASSIFIER,
        "aggregator": AGGREGATOR,
        "train_sliding_style": SLIDING_STYLE,
        "test_sliding_style": TEST_SLIDING_STYLE,
        "shuffle": False,
    }

    wandb_logger = WandbLogger(
        project=f'{DATASET}_eval',    
        entity="fish-benchmark",
        save_dir="./logs",
        tags=[DATASET, SLIDING_STYLE, TEST_SLIDING_STYLE, BACKBONE, POOLING, CLASSIFIER],
        config=config_dict,
    )

    test_data_dir = os.path.join(dataset_config[DATASET]['precomputed_path'], TEST_SLIDING_STYLE, 'test')
    test_dataset = DatasetBuilder(
        path=test_data_dir, 
        dataset_name=DATASET,
        style=TEST_SLIDING_STYLE,
        transform=None,
        precomputed=True,
        feature_model=train_config['backbone']
    ).build()
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=wandb_logger.experiment.config["batch_size"],
        shuffle=wandb_logger.experiment.config["shuffle"],
        num_workers=7,
        pin_memory=True,
    )
    lit_module = LitBinaryClassifierModule.load_from_checkpoint(ckpt_file[0])
    lit_module.freeze()
    lit_module.eval()
    lit_module.to(device)
    trainer = L.Trainer(logger=wandb_logger, log_every_n_steps= 50)
    results = trainer.test(lit_module, test_dataloader)
    test_metrics = results[0]
    metric_file_path = os.path.join(TEST_METRIC_DIR, f'test_metrics-{wandb_logger.experiment.id}.json')
    with open(metric_file_path, "w") as f:
        json.dump(test_metrics, f, indent=4)
    artifact = wandb.Artifact(
        f"test_metrics-{wandb_logger.experiment.id}", 
        type="metrics", 
        description=f"Test metrics for model trained in run {args.run}"
    )
    artifact.add_file(metric_file_path)
    wandb_logger.experiment.log_artifact(artifact)