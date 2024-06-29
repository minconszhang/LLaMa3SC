import argparse
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Configuration parameters
data_dir = '/home/cons/NR/'
local_model_path = 'model_min'
# local_model_path = '/home/cons/NR/model_rayleigh/0snr-100k'
max_seq_length = 512  # Maximum sequence length
dtype = None  # Data type for model training
load_in_4bit = True  # 4-bit quantization flag
random_seed = 3407
awgn = 'Additive White Gaussian Noise'
rayleigh = 'Rayleigh Fading'

# Alpaca prompt template
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Reconstruct the original, noise-free text from the given input text distorted by AWGN. Ensure that:
1. The reconstructed text exactly matches the input text in both total length and word length.
2. Distorted characters represented in escaped format (e.g., \\xf4) are treated as single characters.
3. The original text consists only of lowercase letters and periods.

### Input:
{}

### Response:
{}"""

llama_training_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Below is an instruction describing a task, paired with an input text distorted by {}. Your job is to reconstruct the original, noise-free text from the given input text distorted by {}. Ensure:
1. The reconstructed text matches the input text exactly in total length and the length of each word.
2. Treat distorted characters in escaped format (e.g., \\xf4) as single characters.
3. The original text contains only lowercase letters and periods.<|eot_id|><|start_header_id|>user<|end_header_id|>

{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{}<|eot_id|>"""

llama_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Below is an instruction describing a task, paired with an input text distorted by {}. Your job is to reconstruct the original, noise-free text from the given input text distorted by {}. Ensure:
1. The reconstructed text matches the input text exactly in total length and the length of each word.
2. Treat distorted characters in escaped format (e.g., \\xf4) as single characters.
3. The original text contains only lowercase letters and periods.<|eot_id|><|start_header_id|>user<|end_header_id|>

{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

def initialize_argparse():
    """
    Initialize argument parser for command line arguments.
    """
    parser = argparse.ArgumentParser(description="Utility script for model training.")
    parser.add_argument('--input-data-dir', default='europarl/en', type=str, help='Directory containing input data.')
    parser.add_argument('--data-train', default='cleaned_data/train.json', type=str, help='Path to training data.')
    parser.add_argument('--data-val', default='cleaned_data/val.json', type=str, help='Path to validation data.')
    parser.add_argument('--data-test', default='cleaned_data/test.json', type=str, help='Path to test data.')
    parser.add_argument('--noise_type', default='rayleigh', type=str, choices=['awgn', 'rayleigh'], help='Type of noise to apply')
    return parser

class BleuScore:
    """
    Class for computing BLEU scores.
    """
    def __init__(self, w1, w2, w3, w4):
        """
        Initialize BLEU score calculator with n-gram weights.
        """
        self.weights = (w1, w2, w3, w4)
    
    def compute_bleu_score(self, real, predicted):
        """
        Compute BLEU score for a list of real and predicted sentences.
        """
        scores = []
        for ref, pred in zip(real, predicted):
            ref_tokens = ref.split()
            pred_tokens = pred.split()
            score = sentence_bleu([ref_tokens], pred_tokens, 
                                  weights=self.weights, 
                                  smoothing_function=SmoothingFunction().method1)
            scores.append(score)
        return scores