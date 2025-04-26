import os
import subprocess
import yaml
from fish_benchmark.utils import setup_logger

# Example config values (replace with loading from a file if needed)
TARGETS = ["MikeFramesPatched"]

config = yaml.safe_load(open("config/datasets.yml", "r"))
logger = setup_logger(
    'precompute_sliding_window', 'logs/output/precompute_sliding_window.log'
)
def main():
    for DATASET in TARGETS:
        for TYPE in ['train', 'test']:
            root_dir = os.path.join(config[DATASET]['path'], TYPE)
            dest_root_dir = os.path.join(config[DATASET]['precomputed_path'], TYPE)
            for SUBSET in os.listdir(root_dir):
                assert(os.path.isdir(os.path.join(root_dir, SUBSET))), f"Subset path {SUBSET} is not a directory"
                SOURCE = os.path.join(root_dir, SUBSET)
                INPUT_DEST = os.path.join(dest_root_dir, 'inputs', SUBSET)
                LABEL_DEST = os.path.join(dest_root_dir, 'labels', SUBSET)
                output_dir = os.path.join('logs', 'precompute_sliding_window', TYPE, DATASET, SUBSET)
                os.makedirs(output_dir, exist_ok=True)
                # submit a job for each subset
                wrap_cmd = (
                    f'python data/action_scripts/preprocess_sliding_window.py '
                    f'--source "{SOURCE}" --input_dest "{INPUT_DEST}" --label_dest "{LABEL_DEST}" --id "{SUBSET}" --dataset "{DATASET}"'
                )
                out = os.path.join(output_dir, f"{DATASET}_{TYPE}_{SUBSET}.out")
                err = os.path.join(output_dir, f"{DATASET}_{TYPE}_{SUBSET}.err")
                command = (
                    f"sbatch -J {DATASET}_{TYPE}_{SUBSET} "
                    f"-o {out} "
                    f"-e {err} "
                    f"-N 1 -n 1 --get-user-env --requeue --time=infinite "
                    f"--cpus-per-task=4 --mem=256G --partition=gpu "
                    f"--gres=gpu:1 "
                    f'--wrap="{wrap_cmd}"'
                )
                logger.info(f"Submitted job for {DATASET}_{TYPE}_{SUBSET} with command: {command}")
                subprocess.run(command, shell=True, check=True)
                
if __name__ == '__main__':
    main()

'''
example: 
python data/action_scripts/preprocess_sliding_window.py --source "/share/j_sun/jth264/mike/train/AT_070523_GH010367" --dataset "MikeFramesPatched" --dest "/share/j_sun/jth264/precomputed/mike_frames_patched/train/AT_070523_GH010367" --id "AT_070523_GH010367"
'''