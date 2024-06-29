# Noise Reduction Based on LLaMa3

<center>Min Zhang</center>

This is the implementation of Noise Reduction Base on LLaMa3.

## Requirements

```shell
conda create -n NR-llama3 python=3.10
conda activate NR-llama3
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install w3lib
conda install nltk
conda intall tqdm
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes
```

## Bibtex

N/A

## Preprocess

```shell
wget http://www.statmt.org/europarl/v7/europarl.tgz
tar zxvf europarl.tgz
python preprocess.py
```

## Train

```shell
python train.py
```

## Evaluation

```shell
python evaluation.py
```

### Notes

N/A
