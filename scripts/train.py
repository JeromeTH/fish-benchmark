import subprocess
import os

#arguments of the file to run
# python training/head.py --classifier mlp --dataset abby --sliding_style frames --model dino

MODELS = ['videomae']
CLASSIFIERS = ['mlp']
DATASET = ['abby']
SLIDING_STYLES = ['sliding_window', 'sliding_window_w_temp', 'sliding_window_w_stride']
OUTPUT_BASE = os.path.join('logs', 'train')
PARALLEL = True

def get_wrap_cmd(model, classifier, dataset, sliding_style):
    return (
        f'python training/head.py '
        f'--classifier {classifier} --dataset {dataset} --sliding_style {sliding_style} '
        f'--model {model}' 
    )

def get_slurm_submission_command(dataset, sliding_style, model, classifier, wrap_cmd):
    OUTPUT_DIR = os.path.join(OUTPUT_BASE, dataset, sliding_style, model, classifier)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, f"train.out")
    err = os.path.join(OUTPUT_DIR, f"train.err")
    command = (
        f"sbatch -J {dataset}_{sliding_style}_{model}_{classifier} "
        f"-o {out} "
        f"-e {err} "
        f"-N 1 -n 1 --get-user-env --requeue --time=infinite "
        f"--cpus-per-task=8 --mem=256G --partition=gpu "
        f"--gres=gpu:1 "
        f'--wrap="{wrap_cmd}"'
    )
    return command 

if __name__ == "__main__":
    for MODEL in MODELS:
        for CLASSIFIER in CLASSIFIERS:
            for DATASET in DATASET:
                for SLIDING_STYLE in SLIDING_STYLES:
                    wrap_cmd = get_wrap_cmd(MODEL, CLASSIFIER, DATASET, SLIDING_STYLE)
                    command = get_slurm_submission_command(
                        DATASET, SLIDING_STYLE, MODEL, CLASSIFIER, wrap_cmd
                    ) if PARALLEL else wrap_cmd
                    print(f"Running command: {command}")
                    subprocess.run(command, shell=True, check=True)