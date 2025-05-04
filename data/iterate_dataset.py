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
    parser.add_argument("--model", required=False)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    DATASET = args.dataset
    STYLE = args.style
    MODEL = args.model #nullable
    config = yaml.safe_load(open("config/datasetsv2.yml", "r"))
    input_transform = None
    builder = DatasetBuilder(
        path= os.path.join(config[DATASET]['precomputed_path'], STYLE, 'train'),
        dataset_name=DATASET,
        style=STYLE,
        precomputed=True, 
        feature_model=MODEL,
    )
    dataset = builder.build()
    # print(dataset.get_summary())
    for frame, label in tqdm(dataset):
        pass
    