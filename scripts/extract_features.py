'''
extracts features from precomputed inputs
'''

import yaml
import os 
import torch
from fish_benchmark.utils import setup_logger
import subprocess
import argparse
import shutil

TARGET_MODELS = ['dino', 'videomae']
TARGET_DATASETS = ['abby']
SLIDING_STYLES = ['frames', 'frames_w_temp', 'sliding_window', 'sliding_window_w_temp', 'sliding_window_w_stride']
PRECOMPUTED = False
PARALLEL = True

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
def get_slurm_submission_command(name, output_dir, wrap_cmd):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, "%j.out")
    err = os.path.join(output_dir, "%j.err")
    command = (
        f"sbatch -J {name} "
        f"-o {out} "
        f"-e {err} "
        f"-N 1 -n 1 --get-user-env --requeue --time=infinite "
        f"--cpus-per-task=4 --mem=256G --partition=gpu "
        f"--gres=gpu:1 "
        f'--wrap="{wrap_cmd}"'
    )
    logger.info(f"Submitted job for {name} with command: {command}")
    return command 

def check_match(sliding_style, model):
    if sliding_style_config[sliding_style]['is_image_dataset']: 
        return model_config[model]['input_ndims'] == 4
    else: 
        return model_config[model]['input_ndims'] == 5

def main():
    for DATASET in TARGET_DATASETS:
        for SLIDING_STYLE in SLIDING_STYLES:
            for TYPE in ['train', 'test']:
                for MODEL in TARGET_MODELS:
                    if not check_match(SLIDING_STYLE, MODEL): continue
                    SOURCE_PATH = (os.path.join(dataset_config[DATASET]['precomputed_path'], SLIDING_STYLE, TYPE) 
                            if PRECOMPUTED 
                            else os.path.join(dataset_config[DATASET]['path'], TYPE))
                    
                    DEST_PATH = os.path.join(dataset_config[DATASET]['precomputed_path'], SLIDING_STYLE, TYPE)
                    for SUBSET in os.listdir(SOURCE_PATH):
                        SUBSET_PATH = os.path.join(SOURCE_PATH, SUBSET)
                        SUBSET_DEST_PATH = os.path.join(DEST_PATH, SUBSET)
                        SUBSET_SOURCE = os.path.join(SUBSET_PATH, 'inputs') if PRECOMPUTED else SUBSET_PATH
                        FEATURE_DEST = os.path.join(SUBSET_DEST_PATH, f'{MODEL}_features')
                        wrap_cmp = get_wrap_command(
                            SUBSET_SOURCE, DATASET, SLIDING_STYLE, FEATURE_DEST, SUBSET, MODEL, PRECOMPUTED
                        )
                        output_dir = os.path.join(OUT_ROOT, DATASET, SLIDING_STYLE, TYPE, SUBSET, MODEL)
                        submission_name = f'{DATASET}_{SLIDING_STYLE}_{TYPE}_{SUBSET}_{MODEL}'
                        command = (get_slurm_submission_command(submission_name, output_dir, wrap_cmp) 
                                   if PARALLEL 
                                   else wrap_cmp)   
                        subprocess.run(command, shell=True, check=True)

if __name__ == '__main__':
    main()
    

