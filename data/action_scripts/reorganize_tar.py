import re
import os
import shutil


SOURCE = "/share/j_sun/jth264/bites_frame_annotation"
TARGET = "/share/j_sun/jth264/bites_frame_annotation_reorganized"
# match = re.match(r'^(.*?)-', filename)

if __name__ == '__main__':
    for file_name in os.listdir(SOURCE):
        video_id = re.match(r'^(.*?)-', file_name).group(1)
        target_dir = os.path.join(TARGET, video_id)
        if not os.path.exists(target_dir):
            print("Creating directory:", target_dir)
            os.makedirs(target_dir)
        source_path = os.path.join(SOURCE, file_name)
        target_path = os.path.join(target_dir, file_name)
        os.rename(source_path, target_path)