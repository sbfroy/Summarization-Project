import sys
import os
import random, numpy as np
import random
import json
import time

from pathlib import Path
from src.utils.config_loader import load_config
from rouge_score import rouge_scorer
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import (SystemMessage, HumanMessage)
from openai import RateLimitError, BadRequestError
from langchain_core.exceptions import OutputParserException
from deap import base, creator, tools, algorithms

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

base_dir = Path(os.getcwd())
config = load_config(base_dir / 'secrets.yaml')

random.seed(42)
np.random.seed(42)

os.environ['OPENAI_API_VERSION'] = config['OPENAI_API_VERSION']
os.environ['AZURE_OPENAI_ENDPOINT'] = config['OPENAI_API_BASE']
os.environ['AZURE_OPENAI_API_KEY'] = config['OPENAI_API_KEY']

llm = AzureChatOpenAI(
    deployment_name=config['OPENAI_DEPLOYMENT_NAME'],
    temperature=0.0
)

with open(base_dir / 'data/val_set.json', 'r') as f:
    example_bank = json.load(f)

SCORER = rouge_scorer.RougeScorer(
    ['rouge1', 'rouge2', 'rougeL'],
    use_stemmer=True
)

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

def evaluate_example_subset(examples, version_1, version_2, ref_summary):
    formatted_examples = format_examples(examples)
    msg = [
        SystemMessage(
            content=(
                "Du er en fagperson med ekspertise innen arealplanlegging og "
                "reguleringsplaner. Din oppgave er å bistå saksbehandlere med å "
                "identifisere og oppsummere forskjellene mellom to versjoner av "
                "tekst hentet fra en reguleringsplan. Oppsummeringen skal være så "
                "kort som mulig, men fortsatt tydelig og informativ. Unngå unødvendige "
                "detaljer. Dersom versjon 1 inneholder tekst, men ikke versjon 2, betyr "
                "det at teksten har blitt fjernet. Hvis det er omvendt har teksten blitt "
                "lagt til. Ikke referer til versjon 1 eller versjon 2 i oppsummeringen. "
                "Beskriv kun hva som er fjernet, lagt til eller endret."
            )
        ),
        HumanMessage(
            content=f"""\
Bruk eksemplene som hjelp til å velge riktig ord.

{formatted_examples}
           
Oppsummer forskjellene mellom følgende to versjoner:

Versjon 1:
{version_1}

Versjon 2:
{version_2}

Oppsummering:
"""
        )
    ]

    max_retries = 5
    retry_delay = 10
    response = None

    for attempt in range(max_retries):
        try:
            response = llm.invoke(msg)
            break
        except BadRequestError as e:
            print(f"BadRequestError: {e}. Skipping this prompt.")
            return 0.0
        except ValueError as e:
            print(f"ValueError (possibly content filter): {e}. Skipping this prompt.")
            return 0.0
        except (RateLimitError, OutputParserException) as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                print(f"Retryable error: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise

    if response is None or not hasattr(response, "content"):
        print("Response was None or invalid. Skipping.")
        return 0.0

    scores = SCORER.score(ref_summary, response.content.strip())
    f1_vals = [scores[m].fmeasure for m in ('rouge1', 'rouge2', 'rougeL')]
    return sum(f1_vals) / len(f1_vals)

NUM_EXAMPLES = len(example_bank)
SUBSET_SIZE = 3
POP_SIZE = 6
NUM_GEN = 15
CXPB = 0.5
MUTPB = 0.2
TOURNSIZE = 3

ga_params = {
    "population_size": POP_SIZE,
    "num_generations": NUM_GEN,
    "crossover_probability": CXPB,
    "mutation_probability": MUTPB,
    "selection_tournament_size": TOURNSIZE,
    "subset_size": SUBSET_SIZE
}

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_sample", lambda: random.sample(range(NUM_EXAMPLES), SUBSET_SIZE))
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_sample)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def mutate(individual):
    idx_to_replace = random.randint(0, SUBSET_SIZE - 1)
    available_examples = list(set(range(NUM_EXAMPLES)) - set(individual))
    if available_examples:
        new_example = random.choice(available_examples)
        individual[idx_to_replace] = new_example
    print("After mutation:", individual)
    return (individual,)

def evaluate_fitness(individual):
    examples = [example_bank[i] for i in individual]
    scores = []
    for item in example_bank:
        version_1 = item['version_1']
        version_2 = item['version_2']
        ref_summary = item['ref_summary']
        score = evaluate_example_subset(examples, version_1, version_2, ref_summary)
        scores.append(score)
    avg_score = sum(scores) / len(scores)
    return (avg_score,)

toolbox.register("evaluate", evaluate_fitness)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutate)
toolbox.register("select", tools.selTournament, tournsize=TOURNSIZE)

stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

pop = toolbox.population(n=POP_SIZE)
pop, logbook = algorithms.eaSimple(
    pop, toolbox,
    cxpb=CXPB, mutpb=MUTPB, ngen=NUM_GEN,
    stats=stats, verbose=True
)

best_individual = tools.selBest(pop, 1)[0]
best_examples = [example_bank[i] for i in best_individual]
print("Best example subset:", best_examples)

gen = logbook.select("gen")
avg = logbook.select("avg")
std = logbook.select("std")
min_ = logbook.select("min")
max_ = logbook.select("max")

log_data = {
    "parameters": ga_params,
    "logbook": {
        "gen": gen,
        "avg": avg,
        "std": std,
        "min": min_,
        "max": max_
    }
}

with open("genetic_algorithm_summarization_results.json", "w") as f:
    json.dump(log_data, f, indent=4)

plt.plot(gen, avg, label='avg')
plt.fill_between(gen, np.array(avg) - np.array(std), np.array(avg) + np.array(std), alpha=0.2)
plt.plot(gen, min_, label='min')
plt.plot(gen, max_, label='max')
plt.legend()
plt.savefig("genetic_algorithm_summarization.png")
plt.show()
