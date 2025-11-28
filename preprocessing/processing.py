import json

def make_ans_vocab(annotation_file, save_path="answer_vocabs.txt"):
    with open(annotation_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)['annotations']

    answers = set()
    for ann in annotations:
        for ans in ann['answers']:
            answers.add(ans['answer'])

    answers = sorted(answers) 
    answers.insert(0, '<unk>')  
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.writelines([ans + '\n' for ans in answers])

    print(f"Generated answer vocab with {len(answers)} unique answers.")

make_ans_vocab(annotation_file="data/test_annotations.json")
