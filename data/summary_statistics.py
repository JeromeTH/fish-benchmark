import yaml
from tqdm import tqdm
from fish_benchmark.data.dataset import get_dataset, get_summary
from dataclasses import asdict
import torch
from itertools import islice
dataset_config = yaml.safe_load(open('config/datasets.yml', 'r'))
SPECIFIC_DATASETS = ['UCF101Frames']
if __name__ == '__main__':
    for dataset_name in dataset_config:
        print(dataset_name)
        if dataset_config[dataset_name]['preprocessed']:continue
        if SPECIFIC_DATASETS and dataset_name not in SPECIFIC_DATASETS: continue
        # if not dataset_config[dataset_name]['ours']: continue
        print(f"Processing {dataset_name}")
        dataset = get_dataset(dataset_name, 
                              dataset_config[dataset_name]['path'] + '/train', 
                              augs=None, 
                              label_type='onehot', 
                              shuffle=True)
        summary = get_summary(dataset)
        with open(f'data/summary_statistics/{dataset_name}_summary.yaml', 'w') as f:
            yaml.dump(summary, f)