import yaml
import os 
import torch
from fish_benchmark.utils import setup_logger
import subprocess

TARGET_MODELS = ['multipatch_dino']
TARGET_DATASETS = ['MikeFramesPatchedPrecomputed']
model_config = yaml.safe_load(open("config/models.yml", "r"))
dataset_config = yaml.safe_load(open("config/datasets.yml", "r"))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT_ROOT = os.path.join('logs', 'extract_features')

logger = setup_logger('extract_features', os.path.join(OUT_ROOT, 'extract_features.log'))

if __name__ == '__main__':
    for TYPE in ['train', 'test']:
        for DATASET in TARGET_DATASETS: 
            for MODEL in TARGET_MODELS:
            # check if the model and dataset are compatible
                if model_config[MODEL]['type'] != dataset_config[DATASET]['type']: continue
                PATH = os.path.join(dataset_config[DATASET]['path'], TYPE)
                DEST_ROOT = os.path.join(PATH, f'{MODEL}_features')
                # Load the model
                
                INPUT_PATH = os.path.join(PATH, 'inputs')
                LABEL_PATH = os.path.join(PATH, 'labels')
                for root, dirs, files in os.walk(INPUT_PATH):
                    #check if there is a .pt file
                    if any(file.endswith('.npy') for file in files):
                        #This is a subset of the dataset
                        rel_path = os.path.relpath(root, INPUT_PATH)
                        DEST_PATH = os.path.join(DEST_ROOT, rel_path)
                        SUBSET_ID = os.path.basename(root)
                        SUBSET_PATH = root
                        #we need to call this function
                        # def calculate_feature(subset_path, label_path, dest_path, video_id, model, input_transform):
                        wrap_cmd = (
                            f'python data/action_scripts/extract_features.py '
                            f'--subset_path "{SUBSET_PATH}" --label_path "{LABEL_PATH}" --dest_path "{DEST_PATH}" --model {MODEL} --id "{SUBSET_ID}" '
                        )
                        output_dir = os.path.join(OUT_ROOT, DATASET, MODEL, TYPE, SUBSET_ID)
                        os.makedirs(output_dir, exist_ok=True)
                        out = os.path.join(output_dir, f"{DATASET}_{TYPE}_{SUBSET_ID}.out")
                        err = os.path.join(output_dir, f"{DATASET}_{TYPE}_{SUBSET_ID}.err")
                        command = (
                            f"sbatch -J {DATASET}_{TYPE}_{SUBSET_ID} "
                            f"-o {out} "
                            f"-e {err} "
                            f"-N 1 -n 1 --get-user-env --requeue --time=infinite "
                            f"--cpus-per-task=4 --mem=256G --partition=gpu "
                            f'--wrap="{wrap_cmd}"'
                        )
                        logger.info(f"Submitted job for {DATASET}_{TYPE}_{SUBSET_ID} with command: {command}")
                        subprocess.run(command, shell=True, check=True)
                                

