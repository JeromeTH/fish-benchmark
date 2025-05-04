import argparse
import os
import yaml
import torch
from fish_benchmark.data.dataset import DatasetBuilder
from fish_benchmark.utils import frame_id_with_padding, setup_logger
from tqdm import tqdm
import numpy as np
import shutil


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--input_dest", required=True)
    parser.add_argument("--label_dest", required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sliding_style", required=True)
    parser.add_argument("--type", default="train")
    parser.add_argument("--save_input", default=True)
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
    SLIDING_STYLE = args.sliding_style
    SAVE_INPUT = True if args.save_input == 'True' else False
    ID = args.id
    dataset_config = yaml.safe_load(open("config/datasetsv2.yml", "r"))
    
    # Check if the path exists
    if not os.path.exists(SOURCE):
        raise FileNotFoundError(f"The specified path does not exist: {SOURCE}")
    
    # Check if the dataset is valid
    if DATASET not in dataset_config:
        raise ValueError(f"The specified dataset is not valid: {DATASET}")
    
    builder = DatasetBuilder(
        path = SOURCE, 
        dataset_name = DATASET,
        style= SLIDING_STYLE
    )
    dataset = builder.build()

    # Delete old folders if they exist
    if SAVE_INPUT and os.path.exists(INPUT_DEST): shutil.rmtree(INPUT_DEST)
    if os.path.exists(LABEL_DEST): shutil.rmtree(LABEL_DEST)
            
    if SAVE_INPUT: os.makedirs(INPUT_DEST, exist_ok=True)
    os.makedirs(LABEL_DEST, exist_ok=True)
    logger.info(f"Saving input to {INPUT_DEST}, label to {LABEL_DEST}")
    TOTAL = len(dataset)
    print(len(dataset))
    for i, (clip, label) in tqdm(enumerate(dataset)):
        clip_np = clip.clone().cpu().numpy()
        label_np = label.clone().cpu().numpy()
        if SAVE_INPUT: np.save(os.path.join(INPUT_DEST, f'{ID}_{frame_id_with_padding(i)}.npy'), clip_np)
        np.save(os.path.join(LABEL_DEST, f'{ID}_{frame_id_with_padding(i)}.npy'), label_np)
        # if i % 100 == 0:
        #     logger.info(f"Processed {i}/{TOTAL} clips")