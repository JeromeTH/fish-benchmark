# Fish Behavior Classification Benchmark

## Overview
This project implements video classification benchmarks for fish behavior using **VideoMAE**, **DINO**, and **CLIP**. It includes model definitions, training scripts, and a demo pipeline to evaluate different models on fish behavior datasets.

## Project Structure
```
./fish_benchmark   # Model, Dataset definitions and demo pipeline
./training         # Scripts for training each model
./demo            # Scripts for running model demos
```

## Installation
To set up the project, clone the repository and install the required dependencies:

```bash
conda env create -f environment.yml
conda activate benchmark
```

## Usage

### 1. Training Models
Each model can be trained using the corresponding script in the `./training` directory. For example:

```bash
python ./training/video_mae.py  # Train VideoMAE
python ./training/dino.py      # Train DINO
python ./training/clip.py      # Train CLIP
```
Modify hyperparameters in the script

### 2. Running Model Demos
To test the models on sample videos, use the scripts in the `./demo` directory:

```bash
python ./demo/video_mae_demo.py  # Run VideoMAE on a video
python ./demo/clip_demo.py      # Run CLIP on a video
```

## Logging and Experiment Tracking
This project uses **Weights & Biases (wandb)** for experiment tracking. Ensure you log in to `wandb` before running training:

```bash
wandb login
```

Logs and results will be saved in the specified `wandb` directory. Modify `wandb.init()` in scripts to set a custom logging directory.

## License
This project is licensed under [MIT License](LICENSE).

## Acknowledgments
- [VideoMAE](https://github.com/MCG-NJU/VideoMAE)
- [DINO](https://github.com/facebookresearch/dino)
- [CLIP](https://github.com/openai/CLIP)
