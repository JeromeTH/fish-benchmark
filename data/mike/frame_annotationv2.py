'''
Preprocess the annotations to frame and label pairs. frames are stored as pngs, and labels are stored as json. 
every 1000 frames are stored in as a shard using a tar file. This is then loaded using the webdataset module. 
'''
import os
from fish_benchmark.data.boris import BorisAnnotation
import json
from tqdm import tqdm
import tarfile
import io
from PIL import Image
from pydantic import BaseModel
from typing import List
from concurrent.futures import ThreadPoolExecutor
import time
import threading
import psutil
from multiprocessing import Pool, cpu_count
from fish_benchmark.utils import get_files_of_type, extract_video_identifier, extract_annotation_identifier, setup_logger, frame_id_with_padding
import re
import logging


DATA_PATH = "/share/j_sun/jth264/bites_training_data"
PREDEFINED_TARGET_IDS = None# 'missing_annotations.txt'
OUTPUT_PATH = "/share/j_sun/jth264/mikev2"
SHARD_SIZE = 1000

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# logger = setup_logger('frame_annotation', 'logs/output/frame_annotation.log')
logger = setup_logger('frame_annotation')
def get_png_bytes(image: Image) -> bytes:
    img_buffer = io.BytesIO()
    image.save(img_buffer, format="PNG")
    return img_buffer.getvalue()

def get_jpeg_bytes(image: Image) -> bytes:
    img_buffer = io.BytesIO()
    image.save(img_buffer, format="JPEG", quality=100)
    return img_buffer.getvalue()

def convert_to_bytes(frame, label):
    # thread_id = threading.get_ident()  # Get thread ID
    # current_process = psutil.Process(os.getpid())  # Get the current process
    # cpu_affinity = current_process.cpu_affinity()  # CPUs the process can run on
    # print(f"Thread ID: {thread_id} (Task is running on CPUs: {cpu_affinity}")
    
    # Convert frame to bytes (PNG) and label to bytes (JSON)
    frame_bytes = get_jpeg_bytes(frame)  # Convert to PNG bytes
    #print(label)
    label_bytes = json.dumps(label).encode("utf-8")  # Convert label to JSON bytes
    #print(len(label_bytes))
    return frame_bytes, label_bytes

def load_special_ids(file_path):
    if file_path is None:
        return None
    with open(file_path, 'r') as f:
        lines = f.readlines()
    special_ids = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):  # Ignore empty lines and comments
            special_ids.append(line)
        
    return special_ids

if __name__ == '__main__':
    video_paths = get_files_of_type(DATA_PATH, ".mp4")
    annotation_paths = get_files_of_type(DATA_PATH, ".csv")
    id_video_map = {extract_video_identifier(video_path): video_path for video_path in video_paths}
    id_annotation_map = {extract_annotation_identifier(annotation_path): annotation_path for annotation_path in annotation_paths}
    logger.info(f"Found {len(id_video_map)} video files and {len(id_annotation_map)} annotation files.")
    logger.info(f"Video files: {list(id_video_map.keys())}")
    logger.info(f"Annotation files: {list(id_annotation_map.keys())}")
    special_ids = load_special_ids(PREDEFINED_TARGET_IDS)
    for id in id_video_map.keys():
        if special_ids is not None and id not in special_ids:
            continue
        if id not in id_annotation_map:
            logger.warning(f"Annotation file not found for video {id}. Skipping.")
            continue
        try:
            logger.info(f"Processing video {id} with annotation file {id_annotation_map[id]}")
            #video_path = id_video_map[id]
            annotation_path = id_annotation_map[id]
            annotation = BorisAnnotation(annotation_path)
            #annotation.load_video(video_path)
            file_output_path = os.path.join(OUTPUT_PATH, id)
            os.makedirs(file_output_path, exist_ok=True)

            # filename = os.path.basename(annotation_path)
            # destination = os.path.join(file_output_path, filename)
            # os.rename(annotation_path, destination)

            with open(os.path.join(file_output_path, f'{id}.tsv'), 'w') as f:
                for label in tqdm(annotation.stream_annotations()):
                    row = "\t".join(str(int(x)) for x in label)
                    f.write(row + "\n")

        
                # print(label)
            # video_name = annotation.metadata.observation_id
            # video_output_path = os.path.join(OUTPUT_PATH, video_name)
            # shard_writer = BatchShardWriter(video_name, video_output_path, batch_size=SHARD_SIZE, save_as_tar=True)
            # for frame, label in tqdm(annotation.stream_frame_annotations(), total=annotation.frame_count, desc=f"Processing {video_name}", leave=False):
            #     #save PIL image frame as png 
            #     frame_id = frame_id_with_padding(label['frame_id'])
            #     base_name = f"{video_name}_{frame_id}"
            #     shard_writer.add(frame_id, f"{base_name}.png", frame, f"{base_name}.json", label)
            # shard_writer.flush()
        except Exception as e:
            logger.exception(f"An unexpected error occurred during processing of video {id}")
