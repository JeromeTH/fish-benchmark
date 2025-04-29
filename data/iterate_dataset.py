import argparse
from fish_benchmark.data.dataset import get_dataset
import yaml
from fish_benchmark.models import get_input_transform
from tqdm import tqdm
from dataclasses import asdict

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=False)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    DATASET = args.dataset
    MODEL = args.model #nullable
    config = yaml.safe_load(open("config/datasets.yml", "r"))
    input_transform = None
    dataset = get_dataset(
        DATASET, 
        path = config[DATASET]['path'] + '/train', 
        augs=input_transform,
        shuffle=True
    )
    if config[DATASET]['ours']: print(asdict(dataset))
    for frame, label in tqdm(dataset):
        pass
    