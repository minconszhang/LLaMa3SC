# !usr/bin/env python
# -*- coding:utf-8 _*-

import json
import torch
import utils
from datasets import load_dataset
from trl import SFTTrainer
from transformers import AutoTokenizer, TrainingArguments, TrainerCallback
from unsloth import FastLanguageModel, is_bfloat16_supported

training_losses = []
validation_losses = []
global noise

class LossLoggerCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Log the losses
        if logs is not None:
            training_loss = logs.get("loss")
            validation_loss = logs.get("eval_loss")
            if training_loss is not None:
                training_losses.append(training_loss)
            if validation_loss is not None:
                validation_losses.append(validation_loss)

def initialize(args):
    """Initialize training arguments."""
    args.data_train = utils.data_dir + args.data_train
    args.data_val = utils.data_dir + args.data_val

    global noise
    if (args.noise_type == 'awgn'):
        noise = utils.awgn
    else:
        noise = utils.rayleigh

def load_model():
    """Load and prepare the model and tokenizer."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit",
        max_seq_length = utils.max_seq_length,
        dtype = utils.dtype,
        load_in_4bit = utils.load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 64, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 128,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    return model, tokenizer

def formatting_prompts_func(examples):
    """Format prompts with EOS token for training."""
    global noise
    EOS_TOKEN = '<|end_of_text|>'
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = [
        utils.llama_training_prompt.format(noise, noise, input_text, output_text) + EOS_TOKEN
        for input_text, output_text in zip(inputs, outputs)
    ]
    return { "text" : texts }
pass

def generate_dataset(data_path):
    """Load and prepare the dataset."""
    dataset = load_dataset("json", data_files=data_path, split="train")
    dataset = dataset.map(formatting_prompts_func, batched = True, remove_columns=["input", "output"])
    return dataset

def generate_trainer(model, tokenizer, train_dataset, val_dataset, num_epochs):
    """Create and configure the trainer."""
    batch_size = 4 * 1  # Assume 1 GPU, change if using more GPUs
    gradient_accumulation_steps = 4
    steps_per_epoch = len(train_dataset) / batch_size / gradient_accumulation_steps
    print('Steps per epoch: {}'.format(steps_per_epoch))

    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=0.1,
        num_train_epochs=num_epochs,
        # max_steps=60,
        learning_rate=1e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=int(steps_per_epoch / 100),
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=utils.random_seed,
        output_dir="outputs",
        save_steps=int(steps_per_epoch / 3),
        save_total_limit=7,
        eval_strategy="steps",
        eval_steps=int(steps_per_epoch / 3),
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field = "text",
        max_seq_length = utils.max_seq_length,
        dataset_num_proc = 20,
        packing = False, # Can make training 5x faster for short sequences.
        args = training_args,
        callbacks=[LossLoggerCallback()]
    )

    return trainer

def main(args):
    initialize(args)

    model, tokenizer = load_model()

    train_dataset = generate_dataset(args.data_train)
    val_dataset = generate_dataset(args.data_val)

    print(train_dataset[2])

    trainer = generate_trainer(model, tokenizer, train_dataset, val_dataset, 2)

    trainer_stats = trainer.train()

    model.save_pretrained(utils.local_model_path)
    tokenizer.save_pretrained(utils.local_model_path)

    # Print the training and validation losses
    print("Training Losses:", training_losses)
    print("Validation Losses:", validation_losses)

if __name__ == '__main__':
    parser = utils.initialize_argparse()
    args = parser.parse_args()
    main(args)