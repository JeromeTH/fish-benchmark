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
import numpy as np
from functools import partial
from typing import Callable, Dict, List, Union
from dataclasses import dataclass
from torchmetrics.functional.classification import (
    multilabel_precision,
    multilabel_recall,
    multilabel_f1_score,
    multilabel_average_precision
)
from functools import reduce


# ==== CONFIG ====
ENTITY = "fish-benchmark"
DATASET = 'mike'
PROJECT = f"{DATASET}_eval"
LABEL_TOLERANCES = [0, 1, 3, 5, 7]
PARALLEL = False
DOWNLOAD_DIR = "test_metrics"

subgroup_mappings = {
    "abby": {
        'biting': [0, 1], 
        'aggression': [3, 4]
    }, 
    "mike":{
        'habitat': [16, 17, 19], 
        'biting': [1, 2, 6, 9], 
        'movement': [3, 4], 
        'foraging': [8, 11, 14], 
        'interactions': [5, 7, 13], 
        'other': [0, 10, 12, 15, 18]
    }
}
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

    def flood_1d(self, bits, dis):
        res = np.zeros_like(bits)
        last = None
        for i in range(len(bits)):
            if bits[i] == 1:
                last = i
                res[i] = 1
            elif last is not None and i - last <= dis:
                res[i] = 1
        return res

    def flood(self, bits, dis):
        left = self.flood_1d(bits, dis)
        right = self.flood_1d(bits[::-1], dis)[::-1]
        return np.logical_or(left, right)

    def flood_all_columns(self, targets, dis):
        # targets: (n, d), apply flood along axis 0 per column
        flood_func = partial(self.flood, dis=dis)
        return np.apply_along_axis(flood_func, axis=0, arr=targets)
    
    def compute(self, metrics: List[Metric], label_tolerance = 0, column_subset = None, prefix = None) -> Dict[str, Union[float, List[float]]]:
        results = {}
        probs = self.probs if column_subset is None else self.probs[:, column_subset]
        targets = self.targets if column_subset is None else self.targets[:, column_subset]
        #print(f"flooding with label tolerance {label_tolerance} on {targets.shape[0]} samples and {targets.shape[1]} classes")
        transformed_targets = torch.from_numpy(self.flood_all_columns(targets.cpu().numpy(), label_tolerance)) 
        assert probs.shape == transformed_targets.shape, f"probs shape {probs.shape} and targets shape {transformed_targets.shape} should match"
        num_classes = probs.shape[1]
        #print(f"calculating {len(metrics)} metrics")
        for metric in metrics: 
            if "num_labels" in metric.kwargs.keys(): metric.kwargs["num_labels"] = num_classes
            output = metric.fn(probs, transformed_targets, **metric.kwargs)
            assert isinstance(output, torch.Tensor), f"Output of {metric.name} should be a tensor, but got {type(output)}"
            result_name = f'{prefix}_{metric.name}' if prefix else metric.name
            results[result_name] = output.tolist() if output.ndim > 0 else output.item()            
        return results
    
def get_results(runs):
    results = {}
    api = wandb.Api()
    for run in runs: 
        try: 
            with open(f'logs/test_metrics/{run.id}.json', "r") as f:
                data = json.load(f)
                print(f"Loaded locally {run.id}: config = {run.config}")
        except Exception as e:
            print(f"Failed to find local file {run.id}: {e}")
            print(f"Downloading artifact for {run.id}")
            artifact_name = f"test_metrics_{run.id}.json"
            artifact_path = f"{ENTITY}/{PROJECT}/{artifact_name}:v0"
            artifact = api.artifact(artifact_path)
            file_path = artifact.download(root=DOWNLOAD_DIR)
            #download artifact named artifact_name would give us a file named aftifact_file_name
            artifact_file_name = f"{run.id}.json"
            local_path = os.path.join(file_path, artifact_file_name)
            with open(local_path, "r") as f:
                data = json.load(f)  # <== Loaded into dictionary here
                print(f"Loaded JSON for {run.id}: config = {run.config}")
        results[run.id] = data
    return results

def compute_with_label_tolerance(results, label_tolerance, output_path):
    api = wandb.Api()
    rows = []
    for run_id, data in results.items():
        run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
        probs = torch.tensor(data["probs"])
        targets = torch.tensor(data["targets"])
        calc = MetricCalculator(
            probs=probs,
            targets=targets
        )
        num_classes = probs.shape[1]
        metrics = [
            #num_labels would be dynamically determined by the shape of probs
            Metric("f1_micro", multilabel_f1_score, {"num_labels": None, "average": "micro"}),
            Metric("f1_macro", multilabel_f1_score, {"num_labels": None, "average": "macro"}),
            Metric("precision_micro", multilabel_precision, {"num_labels": None, "average": "micro"}),
            Metric("precision_macro", multilabel_precision, {"num_labels": None, "average": "macro"}),
            Metric("recall_micro", multilabel_recall, {"num_labels": None, "average": "micro"}),
            Metric("recall_macro", multilabel_recall, {"num_labels": None, "average": "macro"}),
            Metric("mAP", multilabel_average_precision, {"num_labels": None, "average": "macro"}),
            Metric("acc", lambda x,y: ((x > 0.5) == y).float().mean(), {}),
            Metric("f1_per_class", multilabel_f1_score, {"num_labels": None, "average": None}),
            Metric("mAP_per_class", multilabel_average_precision, {"num_labels": None, "average": None}),
            Metric("precision_per_class", multilabel_precision, {"num_labels": None, "average": None}),
            Metric("recall_per_class", multilabel_recall, {"num_labels": None, "average": None}),
            Metric("positive_per_class", lambda _,y: (y.sum(dim=0)).float(), {}),
        ]
        
        aggregate_results = calc.compute(metrics, label_tolerance=label_tolerance)
        per_group_results = reduce(lambda a, b: a | b, 
                                   [calc.compute(metrics, 
                                                 label_tolerance=label_tolerance, 
                                                 column_subset=subgroup_mappings[DATASET][k], 
                                                 prefix=k) 
                                                 for k in subgroup_mappings[DATASET].keys()]) 
        row = run.config | {"run_id": run.id} | aggregate_results | per_group_results
        rows.append(row)
            
    #write to csv
    with open(output_path, "w", newline='') as csvfile:
        fieldnames = rows[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote {len(rows)} rows to {output_path}")

def main():
    api = wandb.Api()
    print(f"Fetching runs for {ENTITY}/{PROJECT}")
    runs = api.runs(f'{ENTITY}/{PROJECT}', filters={"state": "finished"})
    filtered_runs = filter(lambda run: match_config(run.config, filt), runs)
    latest_runs_by_group = {}
    for run in filtered_runs:
        key = get_group_key(run.config)
        if key not in latest_runs_by_group or run.created_at > latest_runs_by_group[key].created_at:
            latest_runs_by_group[key] = run
    #sort by jey alphabetical order
    latest_runs_by_group = dict(sorted(latest_runs_by_group.items()))
    print("fetching data")
    results = get_results(latest_runs_by_group.values())
    for label_tolerance in LABEL_TOLERANCES:
        output_path = os.path.join('results', f"{PROJECT}_tol={label_tolerance}_w_subgroup_metrics.csv")
        compute_with_label_tolerance(results, label_tolerance, output_path)
    print("Done")
if __name__ == "__main__":
    main()