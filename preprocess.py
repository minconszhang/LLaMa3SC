# !usr/bin/env python
# -*- coding:utf-8 _*-

import json
import nltk
import os
import random
import re
import unicodedata
import utils
import numpy as np
from multiprocessing import Pool, cpu_count
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from w3lib.html import remove_tags

def initialize(args):
    nltk.download('punkt')

    args.input_data_dir = utils.data_dir + args.input_data_dir
    args.data_train = utils.data_dir + args.data_train
    args.data_val = utils.data_dir + args.data_val
    args.data_test = utils.data_dir + args.data_test

def unicode_to_ascii(s):
    """Convert Unicode string to ASCII."""
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def normalize_string(s):
    """Normalize a string by removing tags, non-alphabet characters, and extra spaces."""
    s = unicode_to_ascii(s)
    s = remove_tags(s)
    s = re.sub(r'([.])', r'\1 ', s)
    s = re.sub(r'[^a-zA-Z.]+', r' ', s)
    s = re.sub(r'\s+', r' ', s).strip()
    s = s.lower()
    return s

def cutted_data(cleaned, MIN_LENGTH=4, MAX_LENGTH=30):
    """Filter sentences by length."""
    return [line for line in cleaned if MIN_LENGTH < len(line.split()) < MAX_LENGTH]

def process_file(text_path):
    """Process a text file and return cleaned sentences."""
    try:
        with open(text_path, 'r', encoding='utf8') as f:
            raw_data = f.read()
        sentences = sent_tokenize(raw_data)
        cleaned_data = [normalize_string(sentence) for sentence in sentences]
        return cutted_data(cleaned_data)
    except Exception as e:
        print(f"Error processing {text_path}: {e}")
        return []

def string_to_binary(s):
    """Convert a string to its binary representation."""
    return ''.join(format(ord(c), '08b') for c in s)

def binary_to_string(b):
    """Convert binary representation back to a string."""
    chars = [b[i:i+8] for i in range(0, len(b), 8)]
    return ''.join(chr(int(char, 2)) for char in chars)

def escape_non_printable_chars(s):
    """Escape non-printable characters in a string."""
    return ''.join(c if 32 <= ord(c) <= 126 else '\\x{0:02x}'.format(ord(c)) for c in s)

def awgn_channel(binary_signal, snr_db):
    """Simulate an AWGN channel for a binary signal."""
    signal = np.array([1 if bit == '1' else -1 for bit in binary_signal])
    signal_power = np.mean(signal**2)
    snr_linear = 10**(snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power / 2) * np.random.randn(len(signal))

    # AWGN
    received_signal = signal + noise
    received_binary_signal = ''.join('1' if bit >= 0 else '0' for bit in received_signal)
    return received_binary_signal

def rayleigh_fading_channel(binary_signal, snr_db):
    """Simulate a Rayleigh fading channel for a binary signal."""
    signal = np.array([1 if bit == '1' else -1 for bit in binary_signal])
    signal_power = np.mean(signal**2)
    snr_linear = 10**(snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power / 2) * np.random.randn(len(signal))
    
    # Rayleigh fading
    rayleigh_fading = np.sqrt(np.random.exponential(scale=2, size=len(signal)))
    faded_signal = signal * rayleigh_fading + noise
    received_binary_signal = ''.join('1' if bit >= 0 else '0' for bit in faded_signal)
    return received_binary_signal

def save_to_json(original_dataset, noisy_dataset, file_path):
    """Save the original and noisy datasets to a JSON file."""
    data = [{"input": noisy, "output": orig} for orig, noisy in zip(original_dataset, noisy_dataset)]
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, ensure_ascii=False)

def process_and_apply_noise(sentence, snr_db=3, noise_type='awgn'):
    """Process a sentence: convert to binary, apply noise, convert back to string, escape non-printable characters."""
    binary_sentence = string_to_binary(sentence)
    if noise_type == 'awgn':
        received_binary_sentence = awgn_channel(binary_sentence, snr_db)
    elif noise_type == 'rayleigh':
        received_binary_sentence = rayleigh_fading_channel(binary_sentence, snr_db)
    else:
        raise ValueError("Unsupported noise type")
    received_sentence = binary_to_string(received_binary_sentence)
    escaped_sentence = escape_non_printable_chars(received_sentence)
    return escaped_sentence

def batch_process_and_apply_noise(sentences, snr_db=3, noise_type='awgn'):
    """Process a batch of sentences."""
    return [process_and_apply_noise(sentence, snr_db, noise_type) for sentence in sentences]

def save_to_file(dataset, file_path, batch_size=100, noise_type='awgn'):
    """Save the dataset to a file after applying noise in parallel."""
    snr_db = 0
    num_batches = len(dataset)

    with Pool(processes=cpu_count()) as pool:
        results = []
        for i in tqdm(range(num_batches), desc="Processing Dataset"):
            batch = dataset[i*batch_size:(i+1)*batch_size]
            batch_results = pool.apply_async(batch_process_and_apply_noise, (batch, snr_db, noise_type))
            results.append(batch_results)

        processed_batches = [batch_result.get() for batch_result in tqdm(results, desc='Processing Batches')]

    flattened_results = [sentence for batch in processed_batches for sentence in batch]

    print('----Saving to {}.----'.format(file_path))
    save_to_json(dataset, flattened_results, file_path)


def main(args):
    initialize(args)

    print('----Start Processing Raw Data----')

    sentences = []

    files = [os.path.join(args.input_data_dir, fn) for fn in os.listdir(args.input_data_dir) if fn.endswith('.txt')]

    with Pool(processes=cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(process_file, files), total=len(files), desc="Reading Files"):
            sentences.extend(result)
    
    sentences = list(set(sentences))

    print('----Number of sentences: {}----'.format(len(sentences)))

    print('----Start Applying Noise----')

    random.seed(utils.random_seed)
    random.shuffle(sentences)

    train_data = sentences[: round(len(sentences) * 0.068)]
    val_data = sentences[round(len(sentences) * 0.5):round(len(sentences) * 0.5007)]
    test_data = sentences[round(len(sentences) * 0.9995):]

    save_to_file(train_data, args.data_train, noise_type=args.noise_type)
    save_to_file(val_data, args.data_val, noise_type=args.noise_type)
    save_to_file(test_data, args.data_test, noise_type=args.noise_type)

if __name__ == '__main__':
    parser = utils.initialize_argparse()
    args = parser.parse_args()
    main(args)