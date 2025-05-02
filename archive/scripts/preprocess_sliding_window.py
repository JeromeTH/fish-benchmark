import os
import subprocess
import yaml
from fish_benchmark.utils import setup_logger

# Example config values (replace with loading from a file if needed)
TARGETS = ["mike"]
PARALLEL = False

config = yaml.safe_load(open("config/datasetsv2.yml", "r"))
# logger = setup_logger(
#     'precompute_sliding_window', 'logs/output/precompute_sliding_window.log'
# )
logger = setup_logger('precompute_sliding_window')

def get_wrap_cmd(source, input_dest, label_dest, subset, dataset, sliding_style):
    return (
        f'python data/action_scripts/preprocess_sliding_window.py '
        f'--source "{source}" --input_dest "{input_dest}" --label_dest "{label_dest}" --id "{subset}" --dataset "{dataset}" --sliding_style "{sliding_style}"'
    )

def get_slurm_submission_command(dataset, sliding_style, type, subset, output_dir, wrap_cmd):
    out = os.path.join(output_dir, f"{dataset}_{sliding_style}_{type}_{subset}.out")
    err = os.path.join(output_dir, f"{dataset}_{sliding_style}_{type}_{subset}.err")
    command = (
        f"sbatch -J {dataset}_{sliding_style}_{type}_{subset} "
        f"-o {out} "
        f"-e {err} "
        f"-N 1 -n 1 --get-user-env --requeue --time=infinite "
        f"--cpus-per-task=4 --mem=256G --partition=gpu "
        f"--gres=gpu:1 "
        f'--wrap="{wrap_cmd}"'
    )
    logger.info(f"Submitted job for {dataset}_{sliding_style}_{type}_{subset} with command: {command}")
    return command

def main():
    for DATASET in TARGETS:
        for SLIDING_STYLE in config[DATASET]['sliding_styles']:
            for TYPE in ['train', 'test']:
                root_dir = os.path.join(config[DATASET]['path'], TYPE)
                dest_root_dir = os.path.join(config[DATASET]['precomputed_path'], SLIDING_STYLE, TYPE)
                for SUBSET in os.listdir(root_dir):
                    assert(os.path.isdir(os.path.join(root_dir, SUBSET))), f"Subset path {SUBSET} is not a directory"
                    SOURCE = os.path.join(root_dir, SUBSET)
                    INPUT_DEST = os.path.join(dest_root_dir, 'inputs', SUBSET)
                    LABEL_DEST = os.path.join(dest_root_dir, 'labels', SUBSET)
                    output_dir = os.path.join('logs', 'precompute_sliding_window', DATASET, SLIDING_STYLE, TYPE, SUBSET)
                    os.makedirs(output_dir, exist_ok=True)
                    # submit a job for each subset
                    if PARALLEL:
                        wrap_cmd = get_wrap_cmd(SOURCE, INPUT_DEST, LABEL_DEST, SUBSET, DATASET, SLIDING_STYLE)
                        command = get_slurm_submission_command(
                            DATASET, SLIDING_STYLE, TYPE, SUBSET, output_dir, wrap_cmd
                        )
                    else:
                        command = get_wrap_cmd(SOURCE, INPUT_DEST, LABEL_DEST, SUBSET, DATASET, SLIDING_STYLE)
                    logger.info(f"Running command for {DATASET}_{SLIDING_STYLE}_{TYPE}_{SUBSET} with command: {command}")
                    subprocess.run(command, shell=True, check=True)
                
if __name__ == '__main__':
    main()

'''
example: 
python data/action_scripts/preprocess_sliding_window.py --source "/share/j_sun/jth264/mike/train/AT_070523_GH010367" --dataset "MikeFramesPatched" --dest "/share/j_sun/jth264/precomputed/mike_frames_patched/train/AT_070523_GH010367" --id "AT_070523_GH010367"
'''