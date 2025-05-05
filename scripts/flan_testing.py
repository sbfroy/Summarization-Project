import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from tqdm import tqdm
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

def format_examples(example_subset): 
    # Formats the examples into a string for later prompt
    formatted = []
    for i, ex in enumerate(example_subset):
        formatted.append(
            f"Eksempel {i+1}:\n"
            f"Versjon 1:\n{ex['version_1']}\n\n"
            f"Versjon 2:\n{ex['version_2']}\n\n"
            f"Oppsummering:\n{ex['ref_summary']}\n##\n"
        )
    
    return "\n".join(formatted)

def main():

    base_dir = Path(os.getcwd())

    # Model and tokenizer
    model_name = 'flan-ul2'
    model_path = f'google/{model_name}'

    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)

    with open(base_dir / 'data/val_set.json', 'r') as f:
        data = json.load(f)

    #ids = [27]
    ids = [27, 39, 23]

    examples = [next(ex for ex in data if ex["id"] == id) for id in ids]
    
    formatted_examples = format_examples(examples)
        
    results = []

    data_sample = data # data[:1]

    for item in tqdm(data_sample):

        version_1 = item['version_1']
        version_2 = item['version_2']
        ref_summary = item['ref_summary']

        input_text = f"""\
    You are an urban planning and regulatory documentation expert, specializing in Norwegian zoning plans. Your role is to assist case workers by identifying and clearly summarizing the differences between two versions of a zoning plan text. The summary should be as brief as possible, yet clear and informative. Write full sentences, avoid unnecessary details, and summarize in Norwegian. If version 1 contains text, but not version 2, it means the text has been removed. If it's the other way around, the text has been added. Do not refer to version 1 or version 2 in the summary. Only describe what has been removed, added, or changed.
    
    Use the examples to guide word choice.

    {formatted_examples}
    
    Summarize the differences between the following two versions:

    Version 1:
    {version_1}
    
    Version 2:
    {version_2}
    
    Summary:
    """

        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
        outputs = model.generate(input_ids)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        results.append({
            'id': item['id'],
            'ref_summary': ref_summary,
            'model_summary': generated_text
        })
   
    output_file = base_dir / f'results/{model_name}_ENG_FEWSHOT_3.jsonl'
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            json.dump(result, f, ensure_ascii=False)
            f.write('\n')

if __name__ == '__main__':
    main()
