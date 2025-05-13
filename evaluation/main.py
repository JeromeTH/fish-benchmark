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
os.makedirs(TEST_METRIC_DIR, exist_ok=True)


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
    config = artifact.logged_by().config
    config['test_sliding_style'] = eval_config[config['sliding_style']]
    config['training_run_id'] = args.run
    config['training_entity'] = args.entity
    config['training_project'] = args.project
    #define testing config
    tags_keys = [
        'dataset', 
        'sliding_style',
        'backbone',
        'pooling',
        'classifier',
        'sampler',
        'test_sliding_style',
    ]

    wandb_logger = WandbLogger(
        project=f'{config['dataset']}_eval',    
        entity="fish-benchmark",
        save_dir="./logs",
        tags=[v for k, v in config.items() if k in tags_keys],
        config=config,
    )

    test_data_dir = os.path.join(dataset_config[config['dataset']]['precomputed_path'], config['test_sliding_style'], 'test')
    test_dataset = DatasetBuilder(
        path=test_data_dir, 
        dataset_name=config['dataset'],
        style=config['test_sliding_style'],
        transform=None,
        precomputed=True,
        feature_model=config['backbone']
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
#python evaluation/main.py --entity fish-benchmark --project abby --run g5hc3uqy