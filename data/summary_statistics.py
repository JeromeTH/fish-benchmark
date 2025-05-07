import yaml
from tqdm import tqdm
from fish_benchmark.data.dataset import DatasetBuilder
from dataclasses import asdict
import torch
import os 
dataset_config = yaml.safe_load(open('config/datasetsv2.yml', 'r'))
DATASET = 'abby'
SLIDING_STYLE = 'frames'
if __name__ == '__main__':
    dataset = DatasetBuilder(
        path = dataset_config[DATASET]['path'], 
        dataset_name = DATASET,
        style='frames',
        transform=None, 
        precomputed=False, 
        feature_model=None,
    ).build()
    summary = dataset.get_summary()
    with open(f'data/summary_statistics/{DATASET}_{SLIDING_STYLE}_summary.yaml', 'w') as f:
        yaml.dump(summary, f)