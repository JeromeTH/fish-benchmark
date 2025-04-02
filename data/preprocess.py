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
from fish_benchmark.utils import get_files_of_type
import re
import logging


data_path = "/share/j_sun/jth264/bites_training_data"
OUTPUT_PATH = "/share/j_sun/jth264/bites_frame_annotation"
video_paths = get_files_of_type(data_path, ".mp4")
annotation_paths = get_files_of_type(data_path, ".csv")
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# Set up logging
logger = logging.getLogger('preprocess_logger')
logger.setLevel(logging.DEBUG) 
file_handler = logging.FileHandler('./logs/output/preprocess.log')
file_handler.setLevel(logging.INFO)  # Set the level for file handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def extract_annotation_identifier(filename):
    """Extracts the annotation identifier from the filename."""
    match = re.search(r'.*_([^_]*)\.csv$', filename)
    return match.group(1) if match else None

def extract_video_identifier(file_path):
    # Extract the filename (part after the last slash)
    filename = os.path.basename(file_path)
    
    # If there's an underscore, extract everything before it
    if '_' in filename:
        return filename.split('_')[0]
    
    # If there's no underscore, return the filename without extension
    return os.path.splitext(filename)[0]

def frame_id_with_padding(id):
    # Pad the frame ID with leading zeros to make it 6 digits
    return str(id).zfill(6)

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


class Batch(BaseModel):
    frame_ids: List[int]
    data: List[tuple]
    def __len__(self):
        return len(self.frame_ids)

class BatchShardWriter: 
    def __init__(self, video_name, output_path, batch_size=1000, save_as_tar=True):
        self.video_name = video_name
        self.output_path = output_path
        self.batch_size = batch_size
        self.batch = Batch(frame_ids=[], data=[])
        self.save_as_tar = save_as_tar

    def add(self, frame_id, frame_name, frame, label_name, label):
        self.batch.frame_ids.append(frame_id)
        self.batch.data.append((frame_name, frame, label_name, label))
        if len(self.batch) >= self.batch_size:
            self.create_shard()

    def flush(self):
        if self.batch.data:
            self.create_shard()

    def create_shard(self):
        #shard name shows the frame id ranges in the shard
        #shard name example: publaynet-train-{000000..000009}.tar
        shard_name = f"{self.video_name}-{self.batch.frame_ids[0]}..{self.batch.frame_ids[-1]}"
        if self.save_as_tar: 
            shard_path = os.path.join(self.output_path, f"{shard_name}.tar")
            with tarfile.open(shard_path, "w") as tar:

                total_frames = len(self.batch.data)
                for frame_name, frame, label_name, label in tqdm(self.batch.data, desc="Saving frames and labels", total=total_frames, disable=True):
                    frame_bytes, label_bytes = convert_to_bytes(frame, label)
                    frame_info = tarfile.TarInfo(frame_name)
                    frame_info.size = len(frame_bytes)
                    label_info = tarfile.TarInfo(label_name)
                    label_info.size = len(label_bytes)
                    tar.addfile(frame_info, io.BytesIO(frame_bytes))
                    tar.addfile(label_info, io.BytesIO(label_bytes))
        else:
            shard_path = os.path.join(self.output_path, shard_name)
            #shard path is the directory where the pngs and jsons are saved
            os.makedirs(shard_path, exist_ok=True)
            total_frames = len(self.batch.data)
            for frame_name, frame, label_name, label in tqdm(self.batch.data, desc="Saving frames and labels", total=total_frames, disable=True):
                # Save the frame and label to the shard directory
                frame.save(os.path.join(shard_path, frame_name), format="JPEG", quality=95)
                #print(frame.size)
                with open(os.path.join(shard_path, label_name), "w") as f:
                    json.dump(label, f)


        self.batch.data.clear()
        self.batch.frame_ids.clear()


if __name__ == '__main__':
    id_video_map = {extract_video_identifier(video_path): video_path for video_path in video_paths}
    id_annotation_map = {extract_annotation_identifier(annotation_path): annotation_path for annotation_path in annotation_paths}

    for id in id_video_map.keys():
        if id not in id_annotation_map:
            logger.warning(f"Video {id} does not have a corresponding annotation file.")
            continue
        logger.info(f"Processing video {id} with annotation file {id_annotation_map[id]}")
        video_path = id_video_map[id]
        annotation_path = id_annotation_map[id]

        annotation = BorisAnnotation(annotation_path)
        annotation.load_video(video_path)
        video_name = annotation.metadata.observation_id
        shard_writer = BatchShardWriter(video_name, OUTPUT_PATH, batch_size=100, save_as_tar=True)
        for frame, label in tqdm(annotation.stream_frame_annotations(), total=annotation.frame_count, desc=f"Processing {video_name}", leave=False):
            #save PIL image frame as png 
            frame_id = frame_id_with_padding(label['frame_id'])
            base_name = f"{video_name}_{frame_id}"
            shard_writer.add(frame_id, f"{base_name}.png", frame, f"{base_name}.json", label)
        shard_writer.flush()