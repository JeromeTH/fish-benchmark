import numpy as np

data = np.load("/share/j_sun/jth264/precomputed/abby_sliding_window/test/inputs/GX017042_9365/GX017042_9365_00000377.npy")
print("Shape:", data.shape)
print("Data type:", data.dtype)
print("Total size (in bytes):", data.nbytes)


# import torch

# data = torch.load("/share/j_sun/jth264/precomputed/mike_frames/test/SR_063023_GH010250/inputs/SR_063023_GH010250_00000000.pt")

# print("Shape:", data.shape)
# print("Data type:", data.dtype)
# print("Size in memory:", data.element_size() * data.nelement(), "bytes")

# import os

# def count_files_in_subdirectories(root_dir):
#     # Loop through all directories and subdirectories
#     for subdir, dirs, files in os.walk(root_dir):
#         # Count number of files in the current subdirectory
#         num_files = len(files)
#         if num_files > 0:  # If there are files in this subdirectory
#             print(f"{subdir} has {num_files} files")

# # Specify the root directory
# root_directory = "/share/j_sun/jth264/precomputed/mike_frames_patched_v3/test/labels"

# # Run the function
# count_files_in_subdirectories(root_directory)
