# Visual Question Answering on Ho Chi Minh City 

This repository contains the **midterm project for the Deep Learning course (DL_N01, 2024–2025)** at **Ton Duc Thang University**.

We build an end-to-end **Visual Question Answering (VQA)** system for images of **20 famous places in Ho Chi Minh City**.  
Given an image and a Vietnamese question, the model predicts a textual answer.

The project includes:

- A **custom VQA dataset** built from images and question–answer pairs generated with Gemini API.
- A **preprocessing pipeline** to create JSON datasets and vocabularies.
- Several **CNN–LSTM based VQA architectures** with and without Attention.
- Both **classification** and **sequence-to-sequence** (Seq2Seq) VQA settings.
- Experiments comparing **MobileNetV2, ResNet50, EfficientNet-B3** and a **CNN built from scratch**, with different fusion strategies.

---

## Table of Contents

- [Project structure](#project-structure)
- [Dataset](#dataset)
- [Preprocessing pipeline](#preprocessing-pipeline)
- [Models](#models)
  - [Visual encoder](#visual-encoder)
  - [Question encoder](#question-encoder)
  - [Answer heads](#answer-heads)
- [Requirements](#requirements)
- [How to run](#how-to-run)
  - [Classification models](#classification-models)
  - [Seq2Seq models](#seq2seq-models)
- [Experimental results (summary)](#experimental-results-summary)
- [Pretrained weights and demo](#pretrained-weights-and-demo)
- [Team](#team)
- [Acknowledgements](#acknowledgements)

---

## Project structure

```text
.
├── data/
│   ├── train_questions.json
│   ├── train_annotations.json
│   ├── test_questions.json
│   └── test_annotations.json
├── preprocessing/
│   ├── caption.py          # Call Gemini API to generate raw Q&A data
│   ├── fixed.py            # Fix indices / IDs in questions & annotations
│   ├── match.py            # Manually correct labels if Gemini is wrong
│   ├── merge.py            # Merge JSON sections into final dataset
│   ├── processing.py       # Build vocabularies for questions / answers
│   ├── show.py             # Visualize images with questions and answers
│   └── split.py            # Split images into multiple chunks for parallel generation
└── main_source/
    ├── dl-mt-class-concat.ipynb            # Classification VQA – feature concatenation
    ├── dl-mt-class-elementwise.ipynb       # Classification VQA – element-wise (Hadamard) fusion
    ├── dl-mt-seqtoseq-concat.ipynb         # Seq2Seq VQA – feature concatenation
    └── dl-mt-seqtoseq-elementwise.ipynb    # Seq2Seq VQA – element-wise fusion
```

---

## Dataset

### Image collection

The raw dataset contains **5,400 images** (4,200 train, 1,200 test) of **20 in Ho Chi Minh City**:

* War Remnants Museum
* History Museum
* Museum of Fine Arts
* Ho Chi Minh City Museum
* Ben Nha Rong
* Bitexco Financial Tower
* Central Post Office
* Ben Thanh Market
* Binh Tay Market
* Thien Hau Temple
* Buu Long Pagoda
* Ngoc Hoang Pagoda
* Phap Hoa Pagoda
* Vinh Nghiem Pagoda
* Independence Palace
* Turtle Lake
* Landmark 81
* City Opera House
* Notre Dame Cathedral
* Saigon Zoo and Botanical Garden

Images are collected manually from Google Images, YouTube and photos taken by team members.

### VQA question–answer pairs

For VQA, we generate question–answer pairs using **Gemini API**:

* For each image we create **4 questions** of different types:

  1. **Place recognition**

     > “Đây là địa điểm nào?”, “Địa điểm trong hình tên là gì?”, …
  2. **Human presence**

     > “Có người xuất hiện trong hình không?”, …
  3. **Time of day**

     > “Ảnh được chụp vào ban ngày hay ban đêm?”, …
  4. **Dominant colors**

     > “Màu sắc chủ đạo trong hình là gì?”, …

* The raw output is saved to JSON files with the following structure:

**`*_questions.json`**

```json
{
  "question_id": ...,
  "image_id": ...,
  "question": "...",
  "question_type": "...",
  "answer_type": "other"
}
```

**`*_annotations.json`**

```json
{
  "question_id": ...,
  "image_id": ...,
  "answer": "...",
  "answer_confidence": "yes",
  "multiple_choice_answer": "...",
  "answer_type": "other"
}
```

Final train/test splits are stored in the `data/` directory.

---

## Preprocessing pipeline

The scripts in `preprocessing/` support building and inspecting the dataset:

* `caption.py`
  Calls Gemini API on the images and writes out temporary JSON files.

  > **Important:** API keys are not included in this public version. Please add your own keys via environment variables or a config file before running.

* `split.py`
  Splits the raw image folder into multiple sections to parallelize data generation.

* `merge.py`
  Merges JSON outputs from multiple sections into a single `questions.json` / `annotations.json`.

* `match.py`
  Post-processes Gemini predictions to **ensure the correct label** for each image.

* `processing.py`
  Builds vocabularies for questions and answers (token → index mapping).

* `fixed.py`
  Fixes `question_id` indices to be consistent across JSON files.

* `show.py`
  Utility to visualize an image together with its generated question and answer.

If you only want to train / evaluate the models, you can use the **ready-made JSON files** in `data/` without running the preprocessing scripts.

---

## Models

### Visual encoder

We experiment with several CNN backbones:

* **MobileNetV2**
* **ResNet50**
* **EfficientNet-B3**
* A **custom CNN built from scratch**

Pretrained models are initialized from **ImageNet** weights; the custom CNN is trained from scratch on our places dataset.

### Question encoder

Questions are tokenized and encoded with:

* An **LSTM** network
* Optionally followed by an **Attention mechanism** to focus on the most relevant words in the question.

Attention helps the model emphasize informative tokens (e.g., “ban đêm”, “màu chủ đạo”, “có người không”) when combining with visual features.

### Answer heads

We consider two VQA settings:

1. **Classification VQA**

   * The answer is chosen from a **fixed vocabulary** of possible answers
     (e.g., places names, “có/không”, “ban ngày/ban đêm”, color phrases).
   * The combined image–question features are passed through fully connected layers to predict a class.

2. **Seq2Seq VQA**

   * The model generates the answer as a **sequence of tokens**.
   * An LSTM decoder with Attention produces the answer word by word.
   * Evaluation uses **BLEU score**.

### Fusion strategies

We evaluate two ways to fuse image and question representations:

* **Element-wise multiplication** (Hadamard product)
* **Concatenation** of visual and textual feature vectors

Both approaches are implemented for classification and Seq2Seq models.

---

## Requirements

Main libraries:

* Python 3.8+
* NumPy, Pandas
* Matplotlib
* scikit-learn
* PyTorch
* Torchvision
* Pillow (PIL)

Optional (only needed if you want to regenerate Q&A with Gemini):

* `google-generativeai`
* `google-api-core`

You can install the core dependencies with:

```bash
pip install numpy pandas matplotlib scikit-learn torch torchvision pillow
```

---

## How to run

### 1. Clone the repository

```bash
git clone https://github.com/pino2704/vqa-hcmc-places.git
cd vqa-hcmc-places
```

(Optional) create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate
```

### 2. Start Jupyter

```bash
pip install jupyter
jupyter notebook
```

Then open one of the notebooks in `main_source/`.

---

### Classification models

Use:

* `dl-mt-class-elementwise.ipynb`
* `dl-mt-class-concat.ipynb`

Each notebook:

1. Loads train/test JSON files from `data/`.
2. Builds vocabularies and the VQADataset.
3. Chooses a CNN backbone (MobileNetV2 / ResNet50 / EfficientNet-B3 / custom CNN).
4. Trains the model and prints **Accuracy** and **F1-score**, plus sample predictions.

You can switch:

* Fusion: element-wise vs concat
* Attention: `with_attention = True/False`
* Backbone: change the model constructor.

---

### Seq2Seq models

Use:

* `dl-mt-seqtoseq-elementwise.ipynb`
* `dl-mt-seqtoseq-concat.ipynb`

The workflow is similar:

1. Load dataset and vocabularies.
2. Build CNN + LSTM + Attention + decoder.
3. Train the model to generate answer sequences.
4. Evaluate using **BLEU score** and visualize some examples.

---

## Experimental results (summary)

### Classification VQA

* Pretrained CNNs (MobileNetV2, ResNet50, EfficientNet-B3) with element-wise fusion achieve around
  **81–83% Accuracy** and **F1-score ≈ 0.80–0.81**.
* The best configuration reaches **Accuracy 82.54%** and **F1 0.8111** (MobileNetV2 without Attention, element-wise fusion).
* The custom CNN trained from scratch performs significantly worse
  (**~60.9% Accuracy, F1 ~0.59**), confirming the benefit of ImageNet pretraining.
* Overall, **element-wise fusion** slightly outperforms concatenation on this dataset.

### Seq2Seq VQA

* With pretrained CNN backbones, Seq2Seq models reach **BLEU ≈ 0.67–0.68**.
* The custom CNN from scratch lags behind with **BLEU ≈ 0.53–0.54**.
* Attention provides small but consistent gains in several configurations.

The results show that even with a relatively small, custom dataset,
**pretrained CNNs + LSTM (with Attention)** can achieve competitive performance on a localized VQA task.

---

## Pretrained weights and demo

Trained models and a small demo are provided here:

* Google Drive: [https://drive.google.com/drive/folders/1DS31V7WF-zOzQuhEiMB8IIpNRsysOo9T?usp=sharing](https://drive.google.com/drive/folders/1DS31V7WF-zOzQuhEiMB8IIpNRsysOo9T?usp=sharing)

> Note: Access and content of the Drive folder may change over time.

---

## Team

Deep Learning Midterm – Group 03 (DL_N01):

* **Nguyen Thanh Phong – 52200198**
* **Nguyen Vu Gia Phuong – 52200205**
* **Cao Thi Thanh Hoa – 52200137**

---

## Acknowledgements

We would like to thank **Assoc. Prof. Dr. Le Anh Cuong** and the
**Faculty of Information Technology – Ton Duc Thang University** for their guidance, lectures and support throughout the course.

```

