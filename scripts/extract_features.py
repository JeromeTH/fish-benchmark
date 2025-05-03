'''
extracts features from precomputed inputs
'''

import yaml
import os 
import torch
from fish_benchmark.utils import setup_logger
import subprocess
import argparse

TARGET_MODELS = ['dino']
TARGET_DATASETS = ['abby']
SLIDING_STYLES = ['frames']
PRECOMPUTED = True
PARALLEL = False

model_config = yaml.safe_load(open("config/models.yml", "r"))
dataset_config = yaml.safe_load(open("config/datasetsv2.yml", "r"))
sliding_style_config = yaml.safe_load(open("config/sliding_style.yml", "r"))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT_ROOT = os.path.join('logs', 'extract_features')
logger = setup_logger('extract_features', os.path.join(OUT_ROOT, 'extract_features.log'))

def get_wrap_command(source, dataset, sliding_style, dest_path, video_id, model, precomputed):
    '''
    SOURCE = args.source
    DATASET = args.dataset
    SLIDING_STYLE = args.sliding_style
    DEST_PATH = args.dest_path
    VIDEO_ID = args.id
    MODEL = args.model
    PRECOMPUTED = args.precomputed
    '''
    return (
        f'python data/action_scripts/extract_features.py '
        f'--source "{source}" --dest_path "{dest_path}" --id "{video_id}" --sliding_style {sliding_style} '
        f'--dataset {dataset} --model {model} --precomputed {precomputed} '
    )
def get_slurm_submission_command(model, subset_id, dataset, type, wrap_cmd):
    output_dir = os.path.join(OUT_ROOT, dataset, model, type, subset_id)
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f"{dataset}_{type}_{subset_id}.out")
    err = os.path.join(output_dir, f"{dataset}_{type}_{subset_id}.err")
    command = (
        f"sbatch -J {dataset}_{type}_{subset_id} "
        f"-o {out} "
        f"-e {err} "
        f"-N 1 -n 1 --get-user-env --requeue --time=infinite "
        f"--cpus-per-task=4 --mem=256G --partition=gpu "
        f"--gres=gpu:1 "
        f'--wrap="{wrap_cmd}"'
    )
    logger.info(f"Submitted job for {dataset}_{type}_{subset_id} with command: {command}")
    return command 

def check_match(sliding_style, model):
    if sliding_style_config[sliding_style]['is_image_dataset']: 
        return model_config[model]['type'] == 'image'
    else: 
        return model_config[model]['type'] == 'video'

def main():
    for DATASET in TARGET_DATASETS:
        for SLIDING_STYLE in SLIDING_STYLES:
            for TYPE in ['train', 'test']:
                for MODEL in TARGET_MODELS:
                    if not check_match(SLIDING_STYLE, MODEL): continue
                    PATH = (os.path.join(dataset_config[DATASET]['precomputed_path'], SLIDING_STYLE, TYPE) 
                            if PRECOMPUTED 
                            else os.path.join(dataset_config[DATASET]['path'], TYPE))
                    for SUBSET in os.listdir(PATH):
                        SUBSET_PATH = os.path.join(PATH, SUBSET)
                        SOURCE = os.path.join(SUBSET_PATH, 'inputs')
                        DEST_PATH = os.path.join(SUBSET_PATH, f'{MODEL}_features')
                        wrap_cmp = get_wrap_command(
                            SOURCE, DATASET, SLIDING_STYLE, DEST_PATH, SUBSET, MODEL, PRECOMPUTED
                        )
                        command = get_slurm_submission_command(MODEL, SUBSET, DATASET, TYPE, wrap_cmp) if PARALLEL else wrap_cmp    
                        subprocess.run(command, shell=True, check=True)

if __name__ == '__main__':
    main()
    

