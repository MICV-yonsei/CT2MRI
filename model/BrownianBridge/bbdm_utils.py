import os
import glob

def find_latest_file(pa_id, path):
    files = glob.glob(os.path.join(path, f"{pa_id}_*.png"))
    if not files:
        return None
    
    latest_number = max(int(file.split('_')[-1].split('.')[0]) for file in files)
    return latest_number

def file_path(pa_id, path):
    latest_number = find_latest_file(pa_id, path)
    next_number = latest_number + 1 if latest_number is not None else 1
    
    new_file_name = f"{pa_id}_{next_number}.png"
    new_file_path = os.path.join(path, new_file_name)
    
    return new_file_path

