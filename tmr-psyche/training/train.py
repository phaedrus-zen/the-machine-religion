#!/usr/bin/env python3
"""
Psyche LoRA training script.

Usage:
    # SFT:
    python training/train.py --stage sft --data data/sft_train.jsonl

    # DPO (after SFT):
    python training/train.py --stage dpo --data data/dpo_train.jsonl --sft-model outputs/psyche-sft-final

    # Full pipeline:
    python training/train.py --stage full --sft-data data/sft_train.jsonl --dpo-data data/dpo_train.jsonl
"""

import argparse
import os
import sys
from pathlib import Path

import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig


def load_config(path: str = "config.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


def get_bnb_config(quantize: str | None) -> BitsAndBytesConfig | None:
    if quantize == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    if quantize == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def train_sft(config: dict, data_path: str, output: str, quantize: str | None = None):
    print(f"\n{'='*60}\nSTAGE 1: Psyche SFT\n{'='*60}")

    sft = config["training"]["sft"]
    lora = config["training"]["lora"]
    base = config["training"]["base_model"]

    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files=data_path, split="train")
    split = dataset.train_test_split(test_size=0.05, seed=42)
    print(f"  Train: {len(split['train'])}  Eval: {len(split['test'])}")

    lora_config = LoraConfig(
        r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"], bias="none", task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir=output,
        num_train_epochs=sft["epochs"],
        per_device_train_batch_size=sft["batch_size"],
        gradient_accumulation_steps=sft["gradient_accumulation"],
        learning_rate=sft["learning_rate"],
        warmup_ratio=sft["warmup_ratio"],
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch",
        bf16=True,
        max_seq_length=sft["max_seq_length"],
        packing=False,
        gradient_checkpointing=True,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = SFTTrainer(
        model=base, args=training_args,
        train_dataset=split["train"], eval_dataset=split["test"],
        processing_class=tokenizer, peft_config=lora_config,
    )

    trainer.train()
    final = f"{output}-final"
    trainer.save_model(final)
    tokenizer.save_pretrained(final)
    print(f"\nSFT model saved: {final}")
    return final


def train_dpo(config: dict, data_path: str, sft_path: str, output: str, quantize: str | None = None):
    print(f"\n{'='*60}\nSTAGE 2: Psyche DPO\n{'='*60}")

    dpo = config["training"]["dpo"]
    base = config["training"]["base_model"]

    tokenizer = AutoTokenizer.from_pretrained(sft_path, trust_remote_code=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"trust_remote_code": True, "torch_dtype": torch.bfloat16}
    bnb = get_bnb_config(quantize)
    if bnb:
        kwargs["quantization_config"] = bnb
    else:
        kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(base, **kwargs)
    model = PeftModel.from_pretrained(model, sft_path, is_trainable=True)

    dataset = load_dataset("json", data_files=data_path, split="train")
    split = dataset.train_test_split(test_size=0.1, seed=42)
    print(f"  Train: {len(split['train'])}  Eval: {len(split['test'])}")

    training_args = DPOConfig(
        output_dir=output,
        num_train_epochs=dpo["epochs"],
        per_device_train_batch_size=dpo["batch_size"],
        gradient_accumulation_steps=dpo["gradient_accumulation"],
        learning_rate=dpo["learning_rate"],
        beta=dpo["beta"],
        bf16=True,
        max_length=dpo["max_length"],
        max_prompt_length=dpo["max_length"] // 2,
        logging_steps=5,
        save_strategy="epoch",
        gradient_checkpointing=True,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model, ref_model=None, args=training_args,
        train_dataset=split["train"], eval_dataset=split["test"],
        processing_class=tokenizer,
    )

    trainer.train()
    final = f"{output}-final"
    trainer.save_model(final)
    tokenizer.save_pretrained(final)
    print(f"\nDPO model saved: {final}")
    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["sft", "dpo", "full"], required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data", help="Data path (single stage)")
    parser.add_argument("--sft-data", help="SFT data (full pipeline)")
    parser.add_argument("--dpo-data", help="DPO data (full pipeline)")
    parser.add_argument("--sft-model", help="SFT model path (for DPO)")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--quantize", choices=["4bit", "8bit"])
    args = parser.parse_args()

    config = load_config(args.config)
    os.makedirs(args.output, exist_ok=True)

    if args.stage == "sft":
        train_sft(config, args.data, f"{args.output}/psyche-sft", args.quantize)
    elif args.stage == "dpo":
        assert args.sft_model, "--sft-model required for DPO"
        train_dpo(config, args.data, args.sft_model, f"{args.output}/psyche-dpo", args.quantize)
    elif args.stage == "full":
        sft_data = args.sft_data or args.data
        dpo_data = args.dpo_data
        assert sft_data, "--sft-data required"
        sft_path = train_sft(config, sft_data, f"{args.output}/psyche-sft", args.quantize)
        if dpo_data and Path(dpo_data).exists():
            train_dpo(config, dpo_data, sft_path, f"{args.output}/psyche-dpo", args.quantize)

    print(f"\n{'='*60}\nTRAINING COMPLETE\n{'='*60}")


if __name__ == "__main__":
    main()
