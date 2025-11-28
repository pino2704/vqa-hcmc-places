import os
import shutil
import re

source_dir = "data/places"
dest_base = "section"

subfolders = [os.path.join(f"s{i}", "data") for i in range(1, 6)]

all_files = os.listdir(source_dir)

def extract_number(filename):
    match = re.search(r'id_(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        return float('inf')  

all_files = sorted(all_files, key=extract_number)

total_files = len(all_files)
folders_count = len(subfolders)

if total_files != 4200:
    print(f"Warning: Expected 4200 images but found {total_files}.")

images_per_folder = total_files // folders_count


for idx, subfolder in enumerate(subfolders):
    dest_dir = os.path.join(dest_base, subfolder)
    os.makedirs(dest_dir, exist_ok=True)
    
    start_index = idx * images_per_folder
    end_index = (idx + 1) * images_per_folder
    
    for filename in all_files[start_index:end_index]:
        src_path = os.path.join(source_dir, filename)
        dst_path = os.path.join(dest_dir, filename)
        shutil.copy(src_path, dst_path)
    
    print(f"Copied {images_per_folder} images to {dest_dir}")

print("Image splitting completed.")
