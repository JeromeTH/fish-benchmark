import subprocess
import os
import yaml
from submission import get_slurm_submission_command

#arguments of the file to run
# python training/head.py --classifier mlp --dataset abby --sliding_style frames --model dino

MODELS = [
    'dino', 
    'dino_large',
    'videomae'
]
CLASSIFIERS = [
    'mlp'
]
POOLINGS = [
    'mean', 
    'attention'
]
DATASETS = [
    'abby', 
    'mike'
]
SLIDING_STYLES = [
    'frames', 
    'frames_w_temp', 
    'sliding_window', 
    'sliding_window_w_temp', 
    'sliding_window_w_stride'
]
OUTPUT_BASE = os.path.join('logs', 'train')
PARALLEL = False
model_config = yaml.safe_load(open("config/models.yml", "r"))
dataset_config = yaml.safe_load(open("config/datasetsv2.yml", "r"))

def get_wrap_cmd(model, classifier, pooling, dataset, sliding_style):
    return (
        f'python training/head.py '
        f'--classifier {classifier} --pooling {pooling} --dataset {dataset} --sliding_style {sliding_style} '
        f'--model {model}' 
    )

def main():
    for MODEL in MODELS:
        for CLASSIFIER in CLASSIFIERS:
            for POOLING in POOLINGS: 
                for DATASET in DATASETS:
                    for SLIDING_STYLE in SLIDING_STYLES:
                        if not SLIDING_STYLE in dataset_config[DATASET]['sliding_styles']: continue
                        if not SLIDING_STYLE in model_config[MODEL]['sliding_styles']: continue
                        wrap_cmd = get_wrap_cmd(MODEL, CLASSIFIER, POOLING, DATASET, SLIDING_STYLE)
                        OUTPUT_DIR = os.path.join(OUTPUT_BASE, DATASET, SLIDING_STYLE, MODEL, CLASSIFIER)
                        command = get_slurm_submission_command(
                            f"{MODEL}_{CLASSIFIER}_{POOLING}_{DATASET}_{SLIDING_STYLE}",
                            OUTPUT_DIR,
                            wrap_cmd,
                            gpu=1
                        ) if PARALLEL else wrap_cmd
                        print(f"Running command: {command}")
                        subprocess.run(command, shell=True, check=True)
        
if __name__ == "__main__":
    main()