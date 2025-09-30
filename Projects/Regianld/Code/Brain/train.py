#!/usr/bin/env python3
# filename: finetune_chatgpt2_from_jsonl.py

import os
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

def main():
    # 1) Prepare tokenizer & model
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    # Just set pad_token if needed
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    # 2) Load your JSONL data
    ds = load_dataset("json", data_files="data.jsonl", split="train")

    # 3) Merge prompt & completion (assuming EOS is already in the data)
    def concat_fn(example):
        example["text"] = example["prompt"].strip() + example["completion"].strip()
        return example

    ds = ds.map(concat_fn, remove_columns=["prompt", "completion"])

    # 4) Tokenize
    max_len = 128
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_len,
        )
    
    tokenized = ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"]
    )

    # 5) Data collator for causal LM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # 6) Setup Trainer
    output_dir = "Regie"
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        logging_steps=50,
        save_steps=200,
        save_total_limit=2,
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    # 7) Fine‐tune & save
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main()
