import webdataset as wds
import os
path = "/share/j_sun/jth264/sample_fish_data/training" 
tar_files = [os.path.join(path, tarfile) for tarfile in os.listdir(path)]


if __name__ == '__main__':
    dataset = wds.WebDataset(tar_files).decode("pil").to_tuple("png", "json")
