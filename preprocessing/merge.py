import os
import json

INDEX_SRC = "sections/test/sI/file.json"

def process_json(file_path, is_read=True, data=None):
    if is_read:
        with open(file_path, 'r', encoding="utf-8") as file:
            return json.load(file)
    if not is_read:
        with open(file_path, 'w', encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
    

def load_index(file_name):
    merged = []
    for i in range(1, 6):
        file_path = INDEX_SRC.replace("sI", f"s{i}")
        file_path = file_path.replace("file", file_name)
        if not os.path.exists(file_path):
            print(f"File {file_path} không tồn tại!")
            continue
        data = process_json(file_path)[f"{file_name}"]
        merged.extend(data)
    print(len(merged))
    merged = {f"{file_name}": merged}
    process_json(f"{file_name}.json", False, merged)

if __name__ == "__main__":
    lst = ["annotations", "questions"]
    load_index(lst[1])
    load_index(lst[0])