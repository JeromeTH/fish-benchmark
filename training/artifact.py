import wandb
from fish_benchmark.data.dataset import get_summary

def log_best_model(checkpoint_callback, run):
    if checkpoint_callback.best_model_path:
        artifact = wandb.Artifact(
            name=f"model-{run.id}",
            type="model",
            metadata={
                "tags": run.tags,
                "config": dict(run.config),
                "notes": run.notes
            }
        )
        artifact.add_file(checkpoint_callback.best_model_path)
        run.log_artifact(artifact)

def log_dataset_summary(dataset, run):
    #store summary
    summary = get_summary(dataset)
    artifact = wandb.Artifact(
        name=f"dataset-{run.id}",
        type="dataset",
        metadata={
            "tags": run.tags,
            "config": dict(run.config),
            "notes": run.notes
        }
    )
    artifact.add_file(summary)
    run.log_artifact(artifact)