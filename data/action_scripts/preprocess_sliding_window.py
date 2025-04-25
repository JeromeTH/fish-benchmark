import argparse
import os
import yaml
import torch
from fish_benchmark.data.dataset import get_dataset
from fish_benchmark.utils import frame_id_with_padding, setup_logger
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--dataset", required=True)
    return parser.parse_args()

logger = setup_logger(
    'precompute_sliding_window'
)

if __name__ == '__main__':
    args = get_args()
    SOURCE = args.source
    DATASET = args.dataset
    DEST = args.dest
    ID = args.id
    dataset_config = yaml.safe_load(open("config/datasets.yml", "r"))
    
    # Check if the path exists
    if not os.path.exists(SOURCE):
        raise FileNotFoundError(f"The specified path does not exist: {SOURCE}")
    
    # Check if the dataset is valid
    if DATASET not in dataset_config:
        raise ValueError(f"The specified dataset is not valid: {DATASET}")
    
    dataset = get_dataset(
        DATASET, 
        path=SOURCE, 
        augs=None,
        shuffle=False
    )
    os.makedirs(os.path.join(DEST, 'inputs'), exist_ok=True)
    os.makedirs(os.path.join(DEST, 'labels'), exist_ok=True)
    logger.info(f"Saving to {DEST}")
    TOTAL = len(dataset)
    for i, (clip, label) in enumerate(dataset):
        torch.save(clip.clone(), os.path.join(DEST, 'inputs', f'{ID}_{frame_id_with_padding(i)}.pt'))
        torch.save(label.clone(), os.path.join(DEST, 'labels', f'{ID}_{frame_id_with_padding(i)}.pt'))
        if i % 100 == 0:
            logger.info(f"Processed {i}/{TOTAL} clips")