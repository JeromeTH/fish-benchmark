from fish_benchmark.utils import get_files_of_type, extract_video_identifier, extract_annotation_identifier
from fish_benchmark.data.schemas import Behavior, Event, Metadata
from fish_benchmark.data.boris import BORIS_NAME_TO_BEHAVIOR, BORIS_NAME_TO_METADATA, BORIS_NAME_TO_EVENT, BorisAnnotation, read_df
DATA_PATH = "/share/j_sun/jth264/bites_training_data"

if __name__ == '__main__':
    video_paths = get_files_of_type(DATA_PATH, ".mp4")
    annotation_paths = get_files_of_type(DATA_PATH, ".csv")
    id_video_map = {extract_video_identifier(video_path): video_path for video_path in video_paths}
    id_annotation_map = {extract_annotation_identifier(annotation_path): annotation_path for annotation_path in annotation_paths}
    missing_annotations = []

    for id in id_video_map.keys():
        if id not in id_annotation_map:
            print(f"Missing annotation for video {id}")
            continue
        annotation_file = read_df(id_annotation_map[id])
        for col in BORIS_NAME_TO_METADATA.keys():
            if col not in annotation_file.columns:
                print(f"Missing column {col} in annotation file {id_annotation_map[id]}")
                missing_annotations.append(id)
                break
    
    #write the ids missing annotations to a file
    with open("missing_annotations.txt", "w") as f:
        for id in missing_annotations:
            f.write(f"{id}\n")
    
                
                