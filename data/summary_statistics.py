import yaml
from tqdm import tqdm
from fish_benchmark.data.dataset import get_dataset_builder
from dataclasses import asdict
import torch
import os 
dataset_config = yaml.safe_load(open('config/datasets.yml', 'r'))
SPECIFIC_DATASETS = ['MikeSlidingWindow']
if __name__ == '__main__':
    for dataset_name in dataset_config:
        print(dataset_name)
        if dataset_config[dataset_name]['preprocessed']:continue
        if SPECIFIC_DATASETS and dataset_name not in SPECIFIC_DATASETS: continue
        if not dataset_config[dataset_name]['ours']: continue
        print(f"Processing {dataset_name}")
        builder = get_dataset_builder(
            dataset_name,
            os.path.join(dataset_config[dataset_name]['path'], 'train')
        )
        dataset = builder.build()
        summary = dataset.get_summary()
        with open(f'data/summary_statistics/{dataset_name}_summary.yaml', 'w') as f:
            yaml.dump(summary, f)