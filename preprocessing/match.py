import json

with open("data/raw_train_annotations.json", "r", encoding="utf-8") as f:
    data = json.load(f)

def get_fixed_location(image_id):
    folder_index = (image_id - 1) // 210
    mapping = {
        0: "Bảo tàng Chứng tích Chiến tranh",
        1: "Bảo tàng Lịch sử",
        2: "Bảo tàng Mỹ thuật",
        3: "Bảo tàng Thành phố",
        4: "Bến Nhà Rồng",
        5: "Bitexco",
        6: "Bưu điện Thành phố Hồ Chí Minh",
        7: "Chợ Bến Thành",
        8: "Chợ Bình Tây",
        9: "Chùa Bà Thiên Hậu",
        10: "Chùa Bửu Long",
        11: "Chùa Ngọc Hoàng",
        12: "Chùa Pháp Hoa",
        13: "Chùa Vĩnh Nghiêm",
        14: "Dinh Độc Lập",
        15: "Hồ Con Rùa",
        16: "Landmark 81",
        17: "Nhà hát Thành Phố",
        18: "Nhà thờ Đức Bà",
        19: "Thảo Cầm Viên Sài Gòn",
    }
    return mapping.get(folder_index, "Không xác định")


for annotation in data["annotations"]:
    question_id = annotation["question_id"]
    image_id = annotation["image_id"]

    if question_id % 12 == 1 or question_id % 12 == 2 or question_id % 12 == 3:
        fixed_location = get_fixed_location(image_id)

        annotation["answers"] = [{"answer": fixed_location, "answer_confidence": "yes"}]
        annotation["multiple_choice_answer"] = fixed_location
        annotation["answer_type"] = "other"

        print(f"✅ Set cứng địa điểm | question_id={question_id}, image_id={image_id}: {fixed_location}")

with open("annotations_final.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("🎯 Hoàn thành! Đã lưu vào annotations_final.json.")
