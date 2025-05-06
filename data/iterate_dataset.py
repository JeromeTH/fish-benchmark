import argparse
from fish_benchmark.data.dataset import DatasetBuilder
import yaml
from fish_benchmark.models import get_input_transform
from tqdm import tqdm
from dataclasses import asdict
from fish_benchmark.utils import setup_logger 
import os

logger = setup_logger("iterate_dataset")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--style", required=True)
    parser.add_argument("--precomputed", default=False)
    parser.add_argument("--model", required=False)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    DATASET = args.dataset
    STYLE = args.style
    MODEL = args.model #nullable
    PRECOMPUTED = True if args.precomputed == 'True' else False
    SPLIT = 'train'
    config = yaml.safe_load(open("config/datasetsv2.yml", "r"))
    input_transform = None
    PATH = os.path.join(config[DATASET]['precomputed_path'], STYLE, SPLIT) if PRECOMPUTED else os.path.join(config[DATASET]['path'], SPLIT)
    builder = DatasetBuilder(
        path= PATH,
        dataset_name=DATASET,
        style=STYLE,
        precomputed=PRECOMPUTED, 
        feature_model=MODEL,
    )
    dataset = builder.build()
    # print(dataset.get_summary())
    frame_0, label_0 = next(iter(dataset))
    print(f"frame shape {frame_0.shape}, label shape {label_0.shape}")
    for frame, label in tqdm(dataset):
        pass
    