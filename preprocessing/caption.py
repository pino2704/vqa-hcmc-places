import base64
import json
import os
import re
import shutil
import time
import uuid
import google.generativeai as genai
from google.api_core import exceptions
import multiprocessing


API_KEYS = ["YOUR_GEMINI_API_KEY_1", "YOUR_GEMINI_API_KEY_2", ...]

IMAGE_FOLDER = "main_data/sample"
OUTPUT_FOLDER = "main_data/"
MAX_RETRIES = 3
RETRY_DELAY = 10
REQUEST_DELAY = 3


NUM_SECTIONS = 5
SECTION_FOLDER = "sections/"
MAX_WAIT_TIME = 60


def extract_number(filename):
    match = re.search(r'id_(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        return float('inf')

def create_sections(is_train=True):
    if is_train:
        SECTION_FOLDER = "sections/train/"
        SRC_FOLDER = "main_data/train"
        no_section_images = 4200/NUM_SECTIONS
        all_files = os.listdir("main_data/train")
    else:
        SECTION_FOLDER = "sections/test/"
        SRC_FOLDER = "main_data/test"
        no_section_images = 1200/NUM_SECTIONS
        all_files = os.listdir("main_data/test")
    
    for i in range(NUM_SECTIONS):
        section_folder = os.path.join(SECTION_FOLDER, f"s{int(i + 1)}/data")
        os.makedirs(section_folder, exist_ok=True)
    
    
    all_files = sorted(all_files, key=extract_number)

    
    for idx, filename in enumerate(all_files):
        src_path = os.path.join(SRC_FOLDER, filename)
        section_idx = idx // no_section_images
        dst_path = os.path.join(SECTION_FOLDER, f"s{int(section_idx + 1)}/data", filename)
        shutil.copy(src_path, dst_path)
        print(f"Đã copy ảnh {filename} vào section s{int(section_idx + 1)}/data")

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def generate_prompt(location=None):
    return """
    Hãy phân tích hình ảnh và tạo đúng 4 câu hỏi VQA (Visual Question Answering) kèm câu trả lời.

    Các câu hỏi phải thuộc các loại sau (mỗi loại đúng 3 câu hỏi):
    1. Nhận dạng địa điểm (ví dụ: "Đây là địa điểm nào?", "Địa điểm trong hình tên là gì?", "Tên của nơi này là gì?", ...)
    2. Phát hiện con người (ví dụ: "Có người xuất hiện trong hình không?", "Ảnh này có người không?",...)
    3. Nhận dạng thời gian trong ngày (ví dụ: "Ảnh được chụp vào ban ngày hay ban đêm?", "Thời điểm trong ngày của ảnh là gì?", ...)
    4. Nhận dạng màu sắc chủ đạo của kiến trúc là gì? (ví dụ: "Màu sắc chủ đạo trong hình là gì?", "Kiến trúc chủ đạo trong hình có màu sắc gì?", ...)
    
    Bạn được phép diễn đạt các câu hỏi theo nhiều cách khác nhau miễn sao giữ nguyên ý nghĩa thuộc đúng từng loại câu hỏi trên. Tuy nhiên nhiên 3 câu hỏi với mỗi loại đều phải trả về 1 đáp án giống nhau và chính xác.


    Yêu cầu đặc biệt:
    - Câu trả lời phải bằng tiếng Việt.
    - Đối với câu hỏi nhận dạng địa điểm, tôi sẽ truyền vào chính xác địa điểm của hình ảnh. Câu trả lời phải trả về tên địa điểm đó và ở dạng "Địa điểm trong hình là ..." hoặc "Vị trí trong hình là ..." hoặc "Bức hình này được chụp ở ...".
    - Riêng câu hỏi về màu sắc chủ đạo, câu trả lời phải trả về 1 trong 8 màu sắc (đỏ, vàng, cam, lục, lam, tím, trắng, đen). Không được trả lời nhiều hơn một màu hoặc các từ ghép như "xanh và trắng", "đỏ gạch", "vàng cam". Câu trả lời phải ở dạng "Màu chủ đạo trong hình là ..." hoặc "Màu chính trong ảnh là ..." hoặc "Kiến trúc chủ đạo trong hình có màu sắc là ...".
    - Câu trả lời đối với phát hiện con người phải trả về "Bức ảnh này có người xuất hiện" hoặc "Có người trong hình" hoặc "Trong hình có người" hoặc "Có người ở đây" hoặc "Có người ở đâu đó trong ảnh". Nếu không có người, trả về "Không có người trong hình" hoặc "Không có người xuất hiện" hoặc "Không có người ở đây" hoặc "Bức ảnh này không có sự xuất hiện của con người".
    - Câu trả lời đối với thời gian trong ngày phải trả về "Ảnh này được chụp vào ban ngày" hoặc "Thời điểm trong hình là ban ngày" hoặc "Ảnh này được chụp vào ban đêm" hoặc "Thời điểm trong hình là ban đêm".
    - Trả về 3 cách diễn đạt câu trả lời khác nhau nhưng chung một ý nghĩa đối với từng loại câu hỏi đồng thời cố gắng trả lời ngắn nhất có thể: dưới 10 từ cho câu trả lời, dưới 15 từ cho câu hỏi.

    Trả JSON thuần:
    {
        "questions": [{"question": "...", "answer": "..."}]    
    }

    *Hướng dẫn nghiêm ngặt:*
        - Luôn đảm bảo JSON trả về chứa đúng 12 câu hỏi với mỗi hình ảnh.
        - Không thêm bất kỳ văn bản nào khác ngoài JSON thuần.
        - Không sử dụng Personal pronoun trong bất kỳ câu hỏi hay câu trả lời nào.
        - Nếu không thể xác định, trả {"questions": []}
    """ + f"\n\n Câu trả lời cho địa điểm trong ảnh: {location}"

def analyze_image(encoded_image, retries=MAX_RETRIES, location=None):
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = generate_prompt(location=location)

    for attempt in range(retries):
        try:
            response = model.generate_content(
                contents=[{"role": "user", "parts": [{"text": prompt}, {
                    "inline_data": {"mime_type": "image/jpeg", "data": encoded_image}}]}]
            )
            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:-3].strip()
            return json.loads(response_text)

        except (exceptions.PermissionDenied, exceptions.ResourceExhausted, json.JSONDecodeError):
            if attempt < retries - 1:
                print(f"Thử lại sau {RETRY_DELAY} giây (lần thử {attempt + 1}/{retries})...")
                time.sleep(RETRY_DELAY)
            else:
                return {"questions": []}

def multi_process_images(is_train=True):
    if is_train:
        api_lst = API_KEYS[5:]
    else:
        api_lst = API_KEYS[:5]
    processes = []
    for i in range(NUM_SECTIONS):
        process = multiprocessing.Process(target=process_images, args=(i, is_train, api_lst[i]))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    print("Tất cả các tiến trình đã hoàn tất.")

def get_location(image_id, is_train=True):
    if is_train:
        folder_index = (image_id - 1) // 210
    else:
        folder_index = (image_id - 1) // 60
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

def process_images(section, is_train=True, api_key=None):
    genai.configure(api_key=api_key)

    if is_train:
        question_id_counter = section * 840 + 1
        section_folder = f"sections/train/s{section + 1}/data"
        out_section_folder = f"sections/train/s{section + 1}"
    else:
        question_id_counter = section * 240 + 1
        section_folder = f"sections/test/s{section + 1}/data"
        out_section_folder = f"sections/test/s{section + 1}"

    images = [f for f in os.listdir(section_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    
    images.sort(key=lambda x: int(os.path.splitext(x)[0].split("_")[-1]))
    questions_data = {"questions": []}
    annotations_data = {"annotations": []}

    for image_name in images:
        image_id = int(os.path.splitext(image_name)[0].split("_")[-1])
        print(f"Xử lý ảnh: {image_name} (id: {image_id})")


        encoded_image = encode_image(os.path.join(section_folder, image_name))
        result = analyze_image(encoded_image=encoded_image, location=get_location(image_id, is_train=is_train))

        if result["questions"]:
            for qa_pair in result["questions"]:
                question_text = qa_pair["question"]
                answer_text = qa_pair["answer"]

                
                questions_data["questions"].append({
                    "question_id": question_id_counter,
                    "image_id": image_id,
                    "question": question_text
                })

                
                annotations_data["annotations"].append({
                    "question_id": question_id_counter,
                    "image_id": image_id,
                    "answers": [{"answer": answer_text, "answer_confidence": "yes"}],
                    "multiple_choice_answer": answer_text,
                    "answer_type": "other"
                })

                question_id_counter += 1
        else:
            print(f"⚠️ Không thể sinh câu hỏi cho ảnh {image_name}")

        time.sleep(REQUEST_DELAY)

    
    with open(os.path.join(out_section_folder, "questions.json"), "w", encoding="utf-8") as fq:
        json.dump(questions_data, fq, ensure_ascii=False, indent=4)

    with open(os.path.join(out_section_folder, "annotations.json"), "w", encoding="utf-8") as fa:
        json.dump(annotations_data, fa, ensure_ascii=False, indent=4)

    print(f"\n🎉 Xử lý hoàn tất! {question_id_counter - 1} câu hỏi được tạo.")

if __name__ == "__main__":
    multi_process_images(is_train=False)