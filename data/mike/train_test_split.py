from fish_benchmark.utils import get_files_of_type, extract_video_identifier, extract_annotation_identifier, setup_logger
import os
import json
DATA_PATH = "/share/j_sun/jth264/bites_frame_annotation"
TARGET_PATH = "/share/j_sun/jth264/bites_frame_annotation_splitted"
def extract_identifier(dirname):
    """
    Extracts the identifier from the filename.
    The identifier is everything before the first underscore.
    """
    return dirname.split('_')[-1]

def detemine_split():
    video_names = []
    for dir_name in os.listdir(DATA_PATH):
        video_names.append(extract_identifier(dir_name))
    print(f"Video names: {video_names}")
    train_test_split = {}
    #sample 80 percent of the video names for training
    train_test_split['train'] = video_names[:int(len(video_names)*0.8)]
    train_test_split['test'] = video_names[int(len(video_names)*0.8):]
    #write the video names to a json file
    os.makedirs(TARGET_PATH, exist_ok=True)
    with open(os.path.join(TARGET_PATH, 'train_test_split.json'), 'w') as f:
        json.dump(train_test_split, f)

def move_directories():
    with open(os.path.join(TARGET_PATH, 'train_test_split.json'), 'r') as f:
        train_test_split = json.load(f)

    train_dir = os.path.join(TARGET_PATH, 'train')
    test_dir = os.path.join(TARGET_PATH, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    print(train_test_split)
    for video_name in os.listdir(DATA_PATH):
        if extract_identifier(video_name) not in train_test_split['train'] and extract_identifier(video_name) not in train_test_split['test']:
            print(f"Video name {video_name} not in train_test_split.json")
            continue
        
        train = True if extract_identifier(video_name) in train_test_split['train'] else False
        #move the directories to the split directory
        src_dir = os.path.join(DATA_PATH, video_name)
        dst_dir = os.path.join(train_dir if train else test_dir, video_name)
        if os.path.exists(src_dir):
            print(f"Moving {src_dir} to {dst_dir}")
            os.rename(src_dir, dst_dir)

if __name__ == "__main__":
    detemine_split()
    move_directories()

# Video name ER_062623_2_GH013726 not in train_test_split.json
# Video name EH_070623_GH010105 not in train_test_split.json
# Video name ER_070323_GH010130 not in train_test_split.json
# Video name EH_063023_GH010090 not in train_test_split.json
# Video name SR_063023_GH030249 not in train_test_split.json
# Video name SR_063023_GH010249 not in train_test_split.json
# Video name EH_062523_GH023724 not in train_test_split.json
# Video name EH_070623_GH010106 not in train_test_split.json
# Video name EH_063023_GH020093 not in train_test_split.json
# Video name EH_063023_GH030093 not in train_test_split.json
# Video name EH_063023_GH010093 not in train_test_split.json
# Video name SR_063023_GH010250 not in train_test_split.json
# Video name ER_062723_GH010111 not in train_test_split.json
# Video name ER_062623_2_GH023726 not in train_test_split.json
# Video name ER_061323_GX027108 not in train_test_split.json
# Video name ER_061323_GX017108 not in train_test_split.json