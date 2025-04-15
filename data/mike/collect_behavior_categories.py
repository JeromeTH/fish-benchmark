from fish_benchmark.data.boris import BorisAnnotation
import json
import os
from fish_benchmark.utils import get_files_of_type
DATA_PATH = '/share/j_sun/jth264/bites_training_data'
annotation_paths = get_files_of_type(DATA_PATH, ".csv")
output_path = './fish_benchmark/data'
if __name__ == '__main__':
    behaviors = []
    for path in annotation_paths:
        try: 
            annotation = BorisAnnotation(path)
            behaviors.extend(annotation.behaviors)
        except Exception as e:
            print(f"Error parsing {path}: {e}")
            continue
    
    unique_behaviors = list(set(behaviors))
    unique_behaviors_dict = [behavior.model_dump() for behavior in unique_behaviors]
    #save the behaviors to a json file
    output_path = os.path.join(output_path, 'behavior_categories.json')
    with open(output_path, 'w') as f:
        json.dump(unique_behaviors_dict, f, indent=4)