#https://wandb.ai/fish-benchmark/abby_eval/runs/1w3vt7yx

'''
run oriented evaluation
'''
import os 
import yaml
import wandb
import csv
import json
import torch 
from typing import Callable, Dict, List, Union
from dataclasses import dataclass
from torchmetrics.functional.classification import (
    multilabel_precision,
    multilabel_recall,
    multilabel_f1_score,
    multilabel_average_precision
)

# ==== CONFIG ====
ENTITY = "fish-benchmark"
PROJECT = "mike_eval"
PARALLEL = False
OUTPUT_PATH = os.path.join('results', f"{PROJECT}.csv")
DOWNLOAD_DIR = "test_metrics"

# cutoff = datetime(2024, 5, 12, 17, 44, tzinfo=timezone.utc)

filt = {
    "dataset": "*",
    "sliding_style": "*",
    "backbone": "*",
    "pooling": "mean",
    "classifier": "mlp",
    "sampler": "*",
}
def match_config(config, filt):
    return all(config.get(k) == v or (v == '*' and k in config) for k, v in filt.items())

def get_group_key(config):
    '''
    assumes config contains the keys in filt, otherwise it would've been filtered out
    '''
    return tuple(config.get(k) for k in filt.keys())

@dataclass
class Metric:
    name: str
    fn: Callable  # should accept (preds, targets, **kwargs)
    kwargs: Dict  # fixed args like average='micro'

class MetricCalculator: 
    def __init__(self, probs: torch.Tensor, targets: torch.Tensor):
        '''
        probs and targets should have the same shape of (n, d), where n is the number of samples and d is the number of classes 
        '''
        self.probs = probs
        self.targets = targets

    def compute(self, metrics: List[Metric]) -> Dict[str, Union[float, List[float]]]:
        results = {}
        for metric in metrics: 
            output = metric.fn(self.probs, self.targets, **metric.kwargs)
            assert isinstance(output, torch.Tensor), f"Output of {metric.name} should be a tensor, but got {type(output)}"
            results[metric.name] = output.tolist() if output.ndim > 0 else output.item()            
        return results
    
def main():
    api = wandb.Api()
    runs = api.runs(f'{ENTITY}/{PROJECT}', filters={"state": "finished"})
    filtered_runs = filter(lambda run: match_config(run.config, filt), runs)
    latest_runs_by_group = {}
    for run in filtered_runs:
        key = get_group_key(run.config)
        if key not in latest_runs_by_group or run.created_at > latest_runs_by_group[key].created_at:
            latest_runs_by_group[key] = run
    
    #sort by jey alphabetical order
    latest_runs_by_group = dict(sorted(latest_runs_by_group.items()))
    rows = []
    for key, run in latest_runs_by_group.items():
        artifact_name = f"test_metrics_{run.id}.json"
        artifact_path = f"{ENTITY}/{PROJECT}/{artifact_name}:v0"
        artifact = api.artifact(artifact_path)
        file_path = artifact.download(root=DOWNLOAD_DIR)
        #download artifact named artifact_name would give us a file named aftifact_file_name
        artifact_file_name = f"{run.id}.json"
        local_path = os.path.join(file_path, artifact_file_name)
        with open(local_path, "r") as f:
            data = json.load(f)  # <== Loaded into dictionary here
            print(f"Loaded JSON for {run.id}: keys = {list(data.keys())}")
        probs = torch.tensor(data["probs"])
        targets = torch.tensor(data["targets"])
        calc = MetricCalculator(
            probs=probs,
            targets=targets
        )
        num_classes = probs.shape[1]
        metrics = [
            Metric("f1_micro", multilabel_f1_score, {"num_labels": num_classes, "average": "micro"}),
            Metric("f1_macro", multilabel_f1_score, {"num_labels": num_classes, "average": "macro"}),
            Metric("precision_micro", multilabel_precision, {"num_labels": num_classes, "average": "micro"}),
            Metric("precision_macro", multilabel_precision, {"num_labels": num_classes, "average": "macro"}),
            Metric("recall_micro", multilabel_recall, {"num_labels": num_classes, "average": "micro"}),
            Metric("recall_macro", multilabel_recall, {"num_labels": num_classes, "average": "macro"}),
            Metric("mAP", multilabel_average_precision, {"num_labels": num_classes, "average": "macro"}),
            Metric("acc", lambda x,y: (x == y).float().mean(), {}),
            Metric("f1_per_class", multilabel_f1_score, {"num_labels": num_classes, "average": None}),
            Metric("mAP_per_class", multilabel_average_precision, {"num_labels": num_classes, "average": None}),
        ]
        results = calc.compute(metrics)
        row = run.config | {"run_id": run.id} | results
        rows.append(row)
        
    #write to csv
    with open(OUTPUT_PATH, "w", newline='') as csvfile:
        fieldnames = rows[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows)} rows to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()