# !usr/bin/env python
# -*- coding:utf-8 _*-

import numpy as np
import utils
from datasets import load_dataset
from tqdm import tqdm
from unsloth import FastLanguageModel
from utils import BleuScore

global noise

def initialize(args):
    """Initialize training arguments."""
    args.data_test = utils.data_dir + args.data_test

    global noise
    if (args.noise_type == 'awgn'):
        noise = utils.awgn
    else:
        noise = utils.rayleigh

def load_model():
    """Load the pre-trained model and tokenizer."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = utils.local_model_path,
        max_seq_length = utils.max_seq_length,
        dtype = utils.dtype,
        load_in_4bit = utils.load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def format_prompts(data):
    """Format prompts for the model."""
    global noise
    texts = [utils.llama_prompt.format(noise, noise, input) for input in data["input"]]
    return {"formatted_prompt": texts}

def generate_responses(model, tokenizer, batch):
    """Generate responses from the model."""
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=64, use_cache=False)
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return {"generated_text": decoded_outputs}

def extract_response(output):
    """Extract the response from the generated output."""
    response_label = "assistant"
    start_index = output.find(response_label)
    if start_index != -1:
        start_index += len(response_label)
        response = output[start_index:].strip()
        return response
    return None

def generate_answers(model, tokenizer, batch_size=16):
    """Generate answers from the dataset."""
    sentences = []

    dataset = load_dataset("json", data_files=args.data_test, split="train")
    dataset = dataset.map(format_prompts, batched = True)
    prompts = dataset["formatted_prompt"]

    model.generation_config.pad_token_id = tokenizer.pad_token_id

    j = 0
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating Responses"):
        j += 1
        if (j == 10):
            break
        batch = prompts[i:i+batch_size]
        generated_texts = generate_responses(model, tokenizer, batch)["generated_text"]
        sentences.extend(map(extract_response, generated_texts))

    return dataset['output'], sentences

def compute_bleu_score(sent1, sent2):
    """Compute BLEU scores for the given sentences."""
    bleu_scores = []
    weights = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]
    
    for weight in weights:
        bleu_scorer = utils.BleuScore(*weight)
        bleu_scores.append(np.mean([bleu_scorer.compute_bleu_score([s1], [s2]) for s1, s2 in zip(sent1, sent2)]))
    
    return bleu_scores

def main(args):
    initialize(args)

    model, tokenizer = load_model()

    sent1, sent2 = generate_answers(model, tokenizer)

    score = compute_bleu_score(sent1, sent2)

    print(sent1[0])
    print(sent2[0])
    print(score)

if __name__ == '__main__':
    parser = utils.initialize_argparse()
    args = parser.parse_args()
    main(args)