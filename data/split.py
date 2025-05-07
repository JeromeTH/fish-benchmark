import os
from fish_benchmark.utils import get_files_of_type
import json
import shutil
PATH = '/share/j_sun/jth264/mikev2'
DEST = '/share/j_sun/jth264/mikev3'

split = json.load(open('data/mike/split.json', 'r'))
if __name__ == '__main__':
    # videos = get_files_of_type(PATH, '.mp4')
    # for video in videos: 
    #     video_name = os.path.basename(video).split('.')[0]
    #     print(video_name)
    #     print(split[video_name])
    #     video_dest = os.path.join(DEST, split[video_name], video_name, f'{video_name}.mp4')
    #     os.makedirs(os.path.dirname(video_dest), exist_ok=True)
    #     #copy the video 
    #     shutil.copy(video, video_dest)

    labels = get_files_of_type(PATH, '.tsv')
    for label in labels:
        label_name = os.path.basename(label).split('.')[0]
        print(label_name)
        print(split[label_name])
        label_dest = os.path.join(DEST, split[label_name], label_name, f'{label_name}.tsv')
        os.makedirs(os.path.dirname(label_dest), exist_ok=True)
        shutil.copy(label, label_dest)
