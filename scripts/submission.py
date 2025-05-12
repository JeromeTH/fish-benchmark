import os
import logging
import shutil
logger = logging.getLogger(__name__)  # This will inherit the global configuration


def get_slurm_submission_command(name, output_dir, wrap_cmd, gpu_types = ":a6000,6000ada,H100,a100", gpu_count=0):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, "%j.out")
    err = os.path.join(output_dir, "%j.err")
    gres = f"--gres=gpu{gpu_types}:{gpu_count} " if gpu_count > 0 else ""
    command = (
        f"sbatch -J {name} "
        f"-o {out} "
        f"-e {err} "
        f"-N 1 -n 1 --get-user-env --requeue --time=infinite "
        f"--cpus-per-task=8 --mem=256G --partition=jjs533,gpu "
        f"{gres}"
        f'--wrap="{wrap_cmd}"'
    )
    #srun --pty --ntasks=1 --cpus-per-task=4 --mem=128G --time=48:00:00 --gres=gpu:a6000:1 --partition=jjs533-interactive,gpu-interactive bash
    logger.info(f"Submitted job for {name} with command: {command}")
    return command 