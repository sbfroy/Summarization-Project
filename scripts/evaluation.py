from pathlib import Path
import os 
import json
from rouge_score import rouge_scorer
import sacrebleu

def load_results(jsonl_path):
    results = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line))
    return results

def evaluate_summaries(results):

    predictions = [r['model_summary'] for r in results]
    references = [r['ref_summary'] for r in results]

    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    total_scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0} 

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for key in total_scores:
            total_scores[key] += scores[key].fmeasure

    rouge_scores = {k: v / len(predictions) for k, v in total_scores.items()}

    # BLEU
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    bleu_score = bleu.score

    return rouge_scores, {'bleu': bleu_score}


if __name__ == "__main__":

    base_dir = Path(os.getcwd())
    file_name = "gpt-4o-mini_ZEROSHOT_test_5_NOR.jsonl"
    jsonl_path = base_dir / 'results' / file_name

    results = load_results(jsonl_path)

    rouge_scores, bleu_scores = evaluate_summaries(results)

    scores_path = base_dir / 'results' / file_name.replace('.jsonl', '_scores.json')

    """with open(scores_path, 'w', encoding='utf-8') as f:
        json.dump({'rouge': rouge_scores, 'bleu': bleu_scores}, f, indent=4)"""

    print('ROUGE Scores:')
    for key, value in rouge_scores.items():
        print(f'{key}: {value:.4f}')

    print('\n BLEU Score:')
    print(f"BLEU: {bleu_scores['bleu']:.4f}")
