import pandas as pd
DATASETS = ['mike', 'abby']
TOLS = [0, 1, 3, 5, 7]
relevant_columns = {
    'abby': [
        'backbone', 
        'sliding_style', 
        'sampler', 
        'mAP', 
        'f1_macro', 
        'acc', 
        'biting_f1_macro', 
        'biting_mAP',
        'biting_acc',
        'aggression_f1_macro',
        'aggression_mAP',
        'aggression_acc',
    ], 
    'mike': [
        'backbone',
        'sliding_style',
        'sampler',
        'mAP',
        'f1_macro',
        'acc',
        'movement_f1_macro',
        'movement_mAP',
        'movement_acc',
        'biting_f1_macro',
        'biting_mAP',
        'biting_acc',
        'foraging_f1_macro',
        'foraging_mAP',
        'foraging_acc',
        'interactions_f1_macro',
        'interactions_mAP',
        'interactions_acc',
        'habitat_f1_macro',
        'habitat_mAP',
        'habitat_acc',
        'other_f1_macro',
        'other_mAP',
        'other_acc',
    ]
}
for DATASET in DATASETS:
    for TOL in TOLS:
        PATH = f'results/{DATASET}_eval_tol={TOL}_w_subgroup_metrics.csv'
        df = pd.read_csv(PATH)
        cleaned_df = df[relevant_columns[DATASET]].copy()
        cleaned_df.to_csv(f'results/{DATASET}_eval_tol={TOL}_cleaned.csv', index=False)