import json
from copy import deepcopy
def process_json(file_path, is_read=True, data=None):
    if is_read:
        with open(file_path, 'r', encoding="utf-8") as file:
            return json.load(file)
    if not is_read:
        with open(file_path, 'w', encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

def main():
    data = process_json("questions.json")
    fixed_data = deepcopy(data)[f"{"questions"}"]
    data_anno = process_json("annotations.json")
    fixed_data_anno = deepcopy(data_anno)[f"{"annotations"}"]

    for i in range(1, 1200*12+1):
        fixed_data[i-1]["question_id"] = i 
        fixed_data_anno[i-1]["question_id"] = i   

    data["questions"] = fixed_data
    data_anno["annotations"] = fixed_data_anno

    process_json("test_questions.json", False, data)
    process_json("test_annotations.json", False, data_anno)

if __name__ == "__main__":
    main()