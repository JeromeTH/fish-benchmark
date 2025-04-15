from fish_benchmark.models import get_input_transform
from fish_benchmark.data.dataset import get_dataset
import yaml
import torch
import os
from fish_benchmark.utils import setup_logger
from tqdm import tqdm
dataset_config = yaml.safe_load(open('./config/datasets.yml', 'r'))
model_config = yaml.safe_load(open('./config/models.yml', 'r'))
SPECIFIC_DATASET_MODEL_PAIRS = [
    # ('HeinFishBehaviorSlidingWindow', 'videomae')
]

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

logger = setup_logger('precompute_transform', 'logs/output/precompute_transform.log')

if __name__ == "__main__":
    for DSET in dataset_config.keys():
        for MODEL in model_config.keys():
            for TYPE in ['train', 'test']:
                # image datasets are preprocessed for image models etc. 
                if(dataset_config[DSET]['type'] != model_config[MODEL]['type']): continue
                if len(SPECIFIC_DATASET_MODEL_PAIRS) > 0 and (DSET, MODEL) not in SPECIFIC_DATASET_MODEL_PAIRS: continue
                logger.info(f"Precomputing input transformation of dataset {DSET} for model {MODEL}")
                try: 
                    LABEL_TYPE = dataset_config[DSET]['label_types'][0]
                    dataset = get_dataset(DSET, dataset_config[DSET]['path'], augs=get_input_transform(MODEL), train=(TYPE == 'train'), label_type=LABEL_TYPE)
                    save_path = mkdir(os.path.join(dataset_config[DSET]['precomputed_path'], MODEL, TYPE))
                    frame_save_path = mkdir(os.path.join(save_path, 'frames'))
                    label_save_path = mkdir(os.path.join(save_path, 'labels'))
                    #wrap with tqdm 
                    for id, (image, label) in tqdm(enumerate(dataset), desc=f"Precomputing {DSET} for {MODEL}"):
                        torch.save(image, os.path.join(frame_save_path, f'{id}.pt'))
                        torch.save(label, os.path.join(label_save_path, f'{id}.pt'))

                except Exception as e:
                    logger.exception(f"Error precomputing dataset {DSET} for model {MODEL}: {e}")