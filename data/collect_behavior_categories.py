from fish_benchmark.data.boris import BorisAnnotation
import json
import os
annotation_paths = ['/share/j_sun/jth264/sample_fish_data/JGL_DaaiBoui_SR_070723_GH030275.csv']
output_path = './fish_benchmark/data'
if __name__ == '__main__':
    behaviors = []
    for path in annotation_paths:
        annotation = BorisAnnotation(path)
        behaviors.extend(annotation.behaviors)
    
    unique_behaviors = list(set(behaviors))
    unique_behaviors_dict = [behavior.model_dump() for behavior in unique_behaviors]
    #save the behaviors to a json file
    output_path = os.path.join(output_path, 'behavior_categories.json')
    with open(output_path, 'w') as f:
        json.dump(unique_behaviors_dict, f, indent=4)