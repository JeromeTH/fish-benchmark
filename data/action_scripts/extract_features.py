from fish_benchmark.models import get_pretrained_model, get_input_transform
from fish_benchmark.data.dataset import PrecomputedDatasetV2
import yaml 
import argparse
from torch.utils.data import DataLoader
import torch
from concurrent.futures import ThreadPoolExecutor
import os
from tqdm import tqdm
from fish_benchmark.utils import frame_id_with_padding
from fish_benchmark.debug import step_timer
import numpy as np

# python data/action_scripts/extract_features.py --model "multipatch_dino" --dataset "MikeFramesPatchedPrecomputed"
BATCH_SIZE = 32
PROFILE = False 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--train", default = True)
    return parser.parse_args()

dataset_config = yaml.safe_load(open("config/datasets.yml", "r"))
model_config = yaml.safe_load(open("config/models.yml", "r"))

def save_feature(dest, save_name, feature_tensor):
    path = os.path.join(dest, f"{save_name}.npy")
    np.save(path, feature_tensor.cpu().numpy())

def avg_pool(tensor):
    return torch.mean(tensor, dim=0)

def parallel_save_features(outputs, dest_path, video_id, start_frame_id):
    futures = []
    os.makedirs(dest_path, exist_ok=True)
    with ThreadPoolExecutor(max_workers=8) as executor:
        for j in range(outputs.shape[0]):
            last_hidden_state = outputs[j]
            feature_tensor = avg_pool(last_hidden_state)
            #print(feature_tensor.shape)
            save_name = f"{video_id}_{frame_id_with_padding(start_frame_id + j)}"
            f = executor.submit(save_feature, dest_path, save_name, feature_tensor)
            futures.append(f)

        # Wait for all jobs to complete
        for f in futures:
            f.result()
    
def calculate_feature(subset_path, dest_path, video_id, model, input_transform):
    print(f"loading data mounted at {subset_path}")
    dataset = PrecomputedDatasetV2(
                input_path=subset_path,
                label_path = os.path.join(PATH, 'labels'),  
                categories=None,
                transform=input_transform
            )
    print("loaded data")
    print(len(dataset))
    # for item in tqdm(dataset):
    #     pass
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    print("loaded dataloader")
    for i, (batch_clip, _) in tqdm(enumerate(dataloader)):
        #print(f"Processing batch {i + 1}/{len(dataloader)}")
        batch_clip = batch_clip.to(device)
        with step_timer("Feature Extraction", verbose=PROFILE), torch.no_grad():
            outputs = model(batch_clip)
        with step_timer("Saving Features", verbose=PROFILE):
            parallel_save_features(outputs, dest_path, video_id, i * BATCH_SIZE)

if __name__ == '__main__':
    args = get_args()
    MODEL = args.model
    DATASET = args.dataset
    TRAIN = args.train
    PATH = os.path.join(dataset_config[DATASET]['path'], 'train') if TRAIN else os.path.join(dataset_config[DATASET]['path'], 'test')
    DEST = os.path.join(PATH, f'{MODEL}_features')
    assert dataset_config[DATASET]['type'] == model_config[MODEL]['type'], f"Model {MODEL} and dataset {DATASET} do not match"
    # Load the model
    model = get_pretrained_model(MODEL)
    model = model.to(device)
    input_transform = get_input_transform(MODEL)
    INPUT_PATH = os.path.join(PATH, 'inputs')
    for root, dirs, files in os.walk(INPUT_PATH):
        #check if there is a .pt file
        if any(file.endswith('.npy') for file in files):
            #This is a subset of the dataset
            rel_path = os.path.relpath(root, INPUT_PATH)
            dest_path = os.path.join(DEST, rel_path)
            video_id = os.path.basename(root)
            calculate_feature(root, dest_path, video_id, model, input_transform)  
            break 