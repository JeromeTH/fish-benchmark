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

SOURCE = 'fish-benchmark/AbbySlidingWindow_training/model-h834d782:v0'
DATASET = 'AbbySlidingWindow'
PROJECT = f'{DATASET}_evaluation'

if __name__ == '__main__':
    dataset_config = yaml.safe_load(open('config/datasets.yml', 'r'))
    model_config = yaml.safe_load(open('config/models.yml', 'r'))
    available_gpus = torch.cuda.device_count()
    LABEL_TYPE = 'onehot'
    print(f"Available GPUs: {available_gpus}")
    with wandb.init(
        project=PROJECT,
        entity = "fish-benchmark",
        notes="Freezing the model parameters and only tuning the classifier head",
        config={"source": SOURCE},
        dir="./logs"
    ) as run:
        artifact = wandb.use_artifact('fish-benchmark/AbbySlidingWindow_training/model-h834d782:v0', type='model')
        artifact_dir = artifact.download()

        # wandb_logger = WandbLogger(
        #     project=run.project,    
        #     save_dir="./logs",
        #     log_model=True
        # )
        # test_dataset = get_dataset(DATASET, 
        #                            dataset_config[DATASET]['path'], 
        #                            augs = get_input_transform(PRETRAINED_MODEL) if not dataset_config[DATASET]['preprocessed'] else None, 
        #                            train=False, 
        #                            label_type=LABEL_TYPE, 
        #                            model_name=PRETRAINED_MODEL, 
        #                            shuffle=False)
    
        # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=run.config['batch_size'], num_workers=7)
        # trainer = L.Trainer(max_epochs=run.config['epochs'], logger=wandb_logger, log_every_n_steps= 10)
        # trainer.test(lit_module, test_dataloader)