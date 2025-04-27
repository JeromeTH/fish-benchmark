import argparse
import os
import yaml
import torch
from fish_benchmark.data.dataset import get_dataset
from fish_benchmark.utils import frame_id_with_padding, setup_logger
from tqdm import tqdm
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--input_dest", required=True)
    parser.add_argument("--label_dest", required=True)
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
    INPUT_DEST = args.input_dest
    LABEL_DEST = args.label_dest
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
    os.makedirs(INPUT_DEST, exist_ok=True)
    os.makedirs(LABEL_DEST, exist_ok=True)
    logger.info(f"Saving input to {INPUT_DEST}, label to {LABEL_DEST}")
    TOTAL = len(dataset)
    print(len(dataset))
    for i, (clip, label) in enumerate(dataset):
        clip_np = clip.clone().cpu().numpy()
        label_np = label.clone().cpu().numpy()
        np.save(os.path.join(INPUT_DEST, f'{ID}_{frame_id_with_padding(i)}.npy'), clip_np)
        np.save(os.path.join(LABEL_DEST, f'{ID}_{frame_id_with_padding(i)}.npy'), label_np)
        if i % 100 == 0:
            logger.info(f"Processed {i}/{TOTAL} clips")

# python data/action_scripts/preprocess_sliding_window.py --source "/share/j_sun/jth264/UCF101_subset/test/Archery" --input_dest "/share/j_sun/jth264/precomputed/UCF101_sliding_window/test/inputs/Archery" --label_dest "/share/j_sun/jth264/precomputed/UCF101_sliding_window/test/labels/Archery" --id "Archery" --dataset "UCF101SlidingWindow"