#https://wandb.ai/fish-benchmark/abby_eval/runs/1w3vt7yx

'''
run oriented evaluation
'''
import os 
import yaml
import wandb
import csv

# ==== CONFIG ====
ENTITY = "fish-benchmark"
PROJECT = "abby_eval"
PARALLEL = False
OUTPUT_PATH = os.path.join('results', f"{PROJECT}.csv")
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

def get_wrap_cmd(entity, project, run_id):
    pass
    # return (
    #     f'python evaluation/main.py '
    #     f'--entity {entity} --project {project} --run {run_id} '
    # )

def get_group_key(config):
    '''
    assumes config contains the keys in filt, otherwise it would've been filtered out
    '''
    return tuple(config.get(k) for k in filt.keys())

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
    for key, run in latest_runs_by_group.items():
        print(key)
        print(run.id)
        print(run.created_at)

    all_summary_keys = set()
    for run in latest_runs_by_group.values():
        all_summary_keys.update(run.summary.keys())
    all_summary_keys = sorted(all_summary_keys)

    print(f"Summary keys: {all_summary_keys}")

    with open(OUTPUT_PATH, 'w', newline='') as f: 
        writer = csv.writer(f)
        # Write header
        header = list(filt.keys()) + all_summary_keys
        writer.writerow(header)

        # Write one row per run
        for group_key, run in latest_runs_by_group.items():
            summary = run.summary
            row = list(group_key)
            row += [summary.get(k, "") for k in all_summary_keys]
            writer.writerow(row)
    #make a chart with run.summary 

if __name__ == "__main__":
    main()