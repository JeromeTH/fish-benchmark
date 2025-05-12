'''
run oriented evaluation
'''
import os 
import yaml
import torch
import wandb
from submission import get_slurm_submission_command
import subprocess

# ==== CONFIG ====
ENTITY = "fish-benchmark"
PROJECT = "abby"
PARALLEL = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
filt = {
    "dataset": "abby",
    "sliding_style": "*",
    "backbone": "*",
    "pooling": "*",
    "classifier": "mlp",
    "sampler": "*",
}
def match_config(config, filt):
    return all(config.get(k) == v or (v == '*' and k in config) for k, v in filt.items())

def get_wrap_cmd(entity, project, run_id):
    return (
        f'python evaluation/main.py '
        f'--entity {entity} --project {project} --run {run_id} '
    )

def main():
    api = wandb.Api()
    runs = api.runs(f'{ENTITY}/{PROJECT}', filters={"state": "finished"})
    filtered_runs = filter(lambda run: match_config(run.config, filt), runs)
    for run in filtered_runs:
        wrap_cmd = get_wrap_cmd(ENTITY, PROJECT, run.id)
        cmd = (get_slurm_submission_command(f"{run.id}", os.path.join('logs', 'test', run.id), wrap_cmd, gpu=1)
               if PARALLEL else wrap_cmd)
        print(f"Running command for {run.id} with command: {cmd}")
        subprocess.run(cmd, shell=True, check=True)
if __name__ == "__main__":
    main()