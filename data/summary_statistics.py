import yaml
from tqdm import tqdm
from fish_benchmark.data.dataset import get_dataset
from dataclasses import asdict
import torch
from itertools import islice
dataset_config = yaml.safe_load(open('config/datasets.yml', 'r'))
SPECIFIC_DATASETS = None # ['AbbyFrames']
if __name__ == '__main__':
    for dataset_name in dataset_config:
        if dataset_config[dataset_name]['preprocessed']:continue
        if SPECIFIC_DATASETS and dataset_name not in SPECIFIC_DATASETS: continue
        if not dataset_config[dataset_name]['ours']: continue
        print(f"Processing {dataset_name}")
        dataset = get_dataset(dataset_name, 
                              dataset_config[dataset_name]['path'], 
                              augs=None, 
                              train=True, 
                              label_type='onehot', 
                              shuffle=True)
        summary = {}
        summary['metadata'] = asdict(dataset)
        labels = torch.stack([
            label for _, label in islice(dataset, 100)
        ])
        label_density = labels.sum(dim=0) / labels.shape[0]
        summary['label_density'] = label_density.tolist()

        #store summary
        with open(f'data/summary_statistics/{dataset_name}_summary.yaml', 'w') as f:
            yaml.dump(summary, f)