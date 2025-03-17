import os
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizerBase,
)
from omegaconf import OmegaConf
import wandb 

def preprocess_function(examples, tokenizer: PreTrainedTokenizerBase, max_length: int):
    texts = examples.get("content")
    if texts is None:
        texts = ["" for _ in range(len(examples[list(examples.keys())[0]]))]
    inputs = tokenizer(texts, truncation=True, padding="max_length", max_length=max_length)
    inputs["labels"] = inputs["input_ids"].copy()
    return inputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_config", "-p",
        type=str,
        default="config/train_config.yaml",
        help="Path to the YAML file defining the training configuration"
    )
    parser.add_argument(
        "--wandb_user",
        type=str,
        default="str-saito-other",
        help="Weights & Biases username (entity)"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="llama3.2",
        help="Weights & Biases project name"
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.train_config)

    model_name = config.model.model
    tokenizer_name = config.model.tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_cache=config.model.use_cache,
        use_safetensors=True,
        use_auth_token=True,
        ignore_mismatched_sizes=True
    )
    model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=False,
        unk_token="<unk>",
        force_download=True,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset(
        "json",
        data_files=config.dataset.path,
        split=config.dataset.split,
        download_mode="force_redownload"
    )
    
    dataset = dataset.map(
        lambda ex: preprocess_function(ex, tokenizer, config.model.max_length),
        batched=True,
        remove_columns=dataset.column_names
    )

    dataset = dataset.train_test_split(test_size=0.2)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    training_args = TrainingArguments(**config.train)

    # Set up Weights & Biases integration
    if args.wandb_user and args.wandb_project:
        training_args.report_to = ["wandb"]
        os.environ["WANDB_ENTITY"] = args.wandb_user
        os.environ["WANDB_PROJECT"] = args.wandb_project
        wandb.init(
            entity=args.wandb_user,
            project=args.wandb_project,
        )
    else:
        training_args.report_to = "none"

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)
    print(f"Model saved to: {training_args.output_dir}")

if __name__ == "__main__":
    main()
