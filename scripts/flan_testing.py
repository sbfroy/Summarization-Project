import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from tqdm import tqdm
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

def format_examples():
    pass

def main():

    base_dir = Path(os.getcwd())

    # Model and tokenizer
    model_name = 'flan-ul2'
    model_path = f'google/{model_name}'

    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)

    with open(base_dir / 'data/val_set.json', 'r') as f:
        data = json.load(f)
    
    results = []

    data_sample = data[:1]

    for item in tqdm(data_sample):

        version_1 = item['version_1']
        version_2 = item['version_2']
        ref_summary = item['ref_summary']

        input_text = f"""
    You are a helpful assistant. Your task is to summarize the changes between two versions of a text.

    The first version is: {version_1}
    
    The second version is: {version_2}
    
    Please provide a summary of the changes between the two versions.
    """

        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
        outputs = model.generate(input_ids)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        results.append({
            'id': item['id'],
            'ref_summary': ref_summary,
            'model_summary': generated_text
        })
   
    output_file = base_dir / f'results/{model_name}_ZEROSHOT.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    main()
