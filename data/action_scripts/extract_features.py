from fish_benchmark.models import get_pretrained_model, get_input_transform
from fish_benchmark.data.dataset import PrecomputedDatasetV2
import yaml 
import argparse
# python data/action_scripts/extract_features.py --model "multipatch_dino" --dataset "MikeFramesPatchedPrecomputed"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    return parser.parse_args()

dataset_config = yaml.safe_load(open("config/datasets.yml", "r"))
model_config = yaml.safe_load(open("config/models.yml", "r"))

if __name__ == '__main__':
    args = get_args()
    MODEL = args.model
    DATASET = args.dataset
    assert dataset_config[DATASET]['type'] == model_config[MODEL]['type'], f"Model {MODEL} and dataset {DATASET} do not match"
    # Load the model
    model = get_pretrained_model(MODEL)

    input_transform = get_input_transform(MODEL)
    print("loading data...")
    dataset = PrecomputedDatasetV2(
        path=dataset_config[DATASET]['path'],
        categories=None,
        transform=input_transform
    )
    print("data loaded")
    clip, label = dataset[0]
    
    print(clip.shape)
    print(model(clip.unsqueeze(0)).shape)
    
