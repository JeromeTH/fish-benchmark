import os
import subprocess
import yaml
from fish_benchmark.utils import setup_logger

# Example config values (replace with loading from a file if needed)
TARGETS = ["MikeFrames", "MikeFramesPatched", "MikeSlidingWindow"]

config = yaml.safe_load(open("config/datasets.yml", "r"))
logger = setup_logger(
    'precompute_sliding_window', 'logs/output/precompute_sliding_window.log'
)

for DATASET in TARGETS:
    for TYPE in ['train', 'test']:
        root_dir = os.path.join(config[DATASET]['path'], TYPE)
        dest_root_dir = os.path.join(config[DATASET]['precomputed_path'], TYPE)
        for SUBSET in os.listdir(root_dir):
            assert(os.path.isdir(os.path.join(root_dir, SUBSET))), f"Subset path {SUBSET} is not a directory"
            SOURCE = os.path.join(root_dir, SUBSET)
            DEST = os.path.join(dest_root_dir, SUBSET)
            # submit a job for each subset
            command = f"""sbatch -J {DATASET}_{TYPE}_{SUBSET} \
                -o {DATASET}_{TYPE}_{SUBSET}.out \
                -e {DATASET}_{TYPE}_{SUBSET}.err \
                -N 1 -n 1 --get-user-env --requeue --time=infinite \
                --cpus-per-task=4 --mem=256G --partition=gpu \
                --wrap="export SOURCE={SOURCE}; 
                    export DEST={DEST}; 
                    export SUBSET={SUBSET}; \
                    python data/action_scripts/preprocess_sliding_window.py --source $SOURCE \
                    --dest $DEST --id $SUBSET --dataset $DATASET" 
                """
            subprocess.run(command, shell=True, check=True)


'''
example: 
python data/action_scripts/preprocess_sliding_window.py --source "/share/j_sun/jth264/mike/train/AT_070523_GH010367" --dataset "MikeFramesPatched" --dest "/share/j_sun/jth264/precomputed/mike_frames_patched/train/AT_070523_GH010367" --id "AT_070523_GH010367"
'''