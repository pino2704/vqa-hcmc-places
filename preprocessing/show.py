import cv2
import json
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont


questions_file = "data/questions.json"
annotations_file = "data/annotations.json"
images_folder = "data/sample"


font_path = "C:/Windows/Fonts/Arial.ttf"
font_size = 20


with open(questions_file, "r", encoding="utf-8") as f_q:
    questions_data = json.load(f_q)


with open(annotations_file, "r", encoding="utf-8") as f_a:
    annotations_data = json.load(f_a)


answers_dict = {}
for ann in annotations_data["annotations"]:
    qid = ann["question_id"]
    if ann["answers"]:
        answer_text = ann["answers"][0]["answer"]
    else:
        answer_text = "Không có câu trả lời"
    answers_dict[qid] = answer_text


questions_by_image_id = {}
for q in questions_data["questions"]:
    img_id = q["image_id"]
    q_id = q["question_id"]
    question_text = q["question"]

    ans = answers_dict.get(q_id, "Không tìm thấy đáp án")
    if img_id not in questions_by_image_id:
        questions_by_image_id[img_id] = []
    questions_by_image_id[img_id].append((question_text, ans))


font = ImageFont.truetype(font_path, font_size)


for image_id, qa_list in questions_by_image_id.items():

    possible_exts = [".png", ".jpg", ".jpeg"]
    image_path = None
    for ext in possible_exts:
        candidate_path = os.path.join(images_folder, f"id_{image_id}{ext}")
        if os.path.exists(candidate_path):
            image_path = candidate_path
            break

    if image_path is None:
        print(f"Không tìm thấy file ảnh hợp lệ cho image_id = {image_id}")
        continue


    image = cv2.imread(image_path)
    if image is None:
        print(f"Không mở được file ảnh: {image_path}")
        continue
    image = cv2.resize(image, (500, 500), interpolation=cv2.INTER_AREA)



    qa_height, qa_width = 800, 600
    qa_display = np.zeros((qa_height, qa_width, 3), dtype=np.uint8)


    pil_image = Image.fromarray(qa_display)
    draw = ImageDraw.Draw(pil_image)


    line_x, line_y = 7, 7
    line_spacing = 20

    draw.text((line_x, line_y),
              f"Image ID: {image_id}",
              font=font,
              fill=(255, 255, 255))
    line_y += 2 * line_spacing


    for idx, (q_text, a_text) in enumerate(qa_list, start=1):
        question_str = f"{idx}. Câu hỏi: {q_text}"
        draw.text((line_x, line_y), question_str, font=font, fill=(255, 255, 255))
        line_y += line_spacing

        answer_str = f"   Trả lời: {a_text}"
        draw.text((line_x, line_y), answer_str, font=font, fill=(255, 255, 255))
        line_y += 2 * line_spacing


    qa_display = np.array(pil_image)


    cv2.imshow("Q&A Window", qa_display)
    cv2.imshow("Image Window", image)


    key = cv2.waitKey(0)
    if key == 27:
        break

cv2.destroyAllWindows()