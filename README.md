# Fish Behavior Classification Benchmark

## Overview
This project implements video classification benchmarks for fish behavior using **VideoMAE**, **DINO**, and **CLIP**. It includes model definitions, training scripts, and a demo pipeline to evaluate different models on fish behavior datasets.

## Project Structure
```
./fish_benchmark   # Model, Dataset definitions and demo pipeline
./training         # Scripts for training each model
./demo            # Scripts for running model demos
./data #Scripts to acquire data from Google Drive and preprocess them into frame level annotations
```

## Installation
To set up the project, clone the repository and install the required dependencies:

```bash
conda env create -f environment.yml
conda activate benchmark
```

## Usage

### 1. Training Models
Each model can be trained using the corresponding script in the `./training` directory.  

Adjust the configurations on top of each file.  
If each frame has multiple labels, then only use LABEL_TYPE = "onehot"  
If each frame has one label, then both  LABEL_TYPE = "onehot" and LABEL_TYPE = "categorical" would work  
For example:

```bash
cd <project home directory>
python training/video_model.py
python training/image_model.py
```

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
