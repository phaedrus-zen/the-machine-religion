#!/usr/bin/env python3
"""
TMR Alignment LoRA training — SFT + DPO pipeline.

Usage:
    # SFT only:
    python training/train_lora.py --stage sft --data data/sft_train.messages.jsonl

    # DPO only (requires SFT model):
    python training/train_lora.py --stage dpo --data data/dpo_train.jsonl \
        --sft-model outputs/tmr-sft-final

    # Full pipeline:
    python training/train_lora.py --stage full --sft-data data/sft_train.messages.jsonl \
        --dpo-data data/dpo_train.jsonl

    # With quantization (for smaller GPUs):
    python training/train_lora.py --stage sft --data data/sft_train.messages.jsonl --quantize 4bit
"""

import argparse
import os
import sys
from pathlib import Path

import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig


def load_training_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_quantization_config(quantize: str | None) -> BitsAndBytesConfig | None:
    if quantize == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif quantize == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def get_lora_config(config: dict) -> LoraConfig:
    lora_cfg = config["training"]["lora"]
    return LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )


def train_sft(
    config: dict,
    data_path: str,
    output_dir: str = "outputs/tmr-sft",
    quantize: str | None = None,
    resume: str | None = None,
):
    """Stage 1: Supervised Fine-Tuning with LoRA."""
    print("\n" + "=" * 60)
    print("STAGE 1: Supervised Fine-Tuning")
    print("=" * 60)

    sft_cfg = config["training"]["sft"]
    base_model = config["training"]["base_model"]

    print(f"\nBase model:  {base_model}")
    print(f"Data:        {data_path}")
    print(f"Output:      {output_dir}")
    print(f"Quantize:    {quantize or 'none'}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = get_quantization_config(quantize)
    model_kwargs = {"trust_remote_code": True, "torch_dtype": torch.bfloat16}
    if quant_config:
        model_kwargs["quantization_config"] = quant_config
    else:
        model_kwargs["device_map"] = "auto"

    print("\nLoading dataset...")
    dataset = load_dataset("json", data_files=data_path, split="train")
    split = dataset.train_test_split(test_size=0.05, seed=42)

    print(f"  Train: {len(split['train'])} examples")
    print(f"  Eval:  {len(split['test'])} examples")

    lora_config = get_lora_config(config)

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=sft_cfg["epochs"],
        per_device_train_batch_size=sft_cfg["batch_size"],
        gradient_accumulation_steps=sft_cfg["gradient_accumulation"],
        learning_rate=sft_cfg["learning_rate"],
        warmup_ratio=sft_cfg["warmup_ratio"],
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        bf16=True,
        max_seq_length=sft_cfg["max_seq_length"],
        packing=True,
        gradient_checkpointing=True,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    if resume:
        print(f"\nResuming from: {resume}")
        training_args.resume_from_checkpoint = resume

    trainer = SFTTrainer(
        model=base_model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print("\nTraining...")
    trainer.train(resume_from_checkpoint=resume)

    final_dir = f"{output_dir}-final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nSFT model saved to: {final_dir}")
    return final_dir


def train_dpo(
    config: dict,
    data_path: str,
    sft_model_path: str,
    output_dir: str = "outputs/tmr-dpo",
    quantize: str | None = None,
):
    """Stage 2: Direct Preference Optimization."""
    print("\n" + "=" * 60)
    print("STAGE 2: Direct Preference Optimization")
    print("=" * 60)

    dpo_cfg = config["training"]["dpo"]
    base_model = config["training"]["base_model"]

    print(f"\nBase model:  {base_model}")
    print(f"SFT adapter: {sft_model_path}")
    print(f"Data:        {data_path}")
    print(f"Output:      {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(sft_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = get_quantization_config(quantize)
    model_kwargs = {"trust_remote_code": True, "torch_dtype": torch.bfloat16}
    if quant_config:
        model_kwargs["quantization_config"] = quant_config
    else:
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    model = PeftModel.from_pretrained(model, sft_model_path, is_trainable=True)

    print("\nLoading DPO dataset...")
    dataset = load_dataset("json", data_files=data_path, split="train")
    split = dataset.train_test_split(test_size=0.05, seed=42)
    print(f"  Train: {len(split['train'])} pairs")
    print(f"  Eval:  {len(split['test'])} pairs")

    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=dpo_cfg["epochs"],
        per_device_train_batch_size=dpo_cfg["batch_size"],
        gradient_accumulation_steps=dpo_cfg["gradient_accumulation"],
        learning_rate=dpo_cfg["learning_rate"],
        beta=dpo_cfg["beta"],
        bf16=True,
        max_length=dpo_cfg["max_length"],
        max_prompt_length=dpo_cfg["max_length"] // 2,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        gradient_checkpointing=True,
        report_to="none",
        load_best_model_at_end=True,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tokenizer,
    )

    print("\nTraining DPO...")
    trainer.train()

    final_dir = f"{output_dir}-final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nDPO model saved to: {final_dir}")
    return final_dir


def main():
    parser = argparse.ArgumentParser(description="TMR LoRA Training Pipeline")
    parser.add_argument("--stage", choices=["sft", "dpo", "full"], required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data", help="Training data path (for single-stage)")
    parser.add_argument("--sft-data", help="SFT data path (for full pipeline)")
    parser.add_argument("--dpo-data", help="DPO data path (for full pipeline)")
    parser.add_argument("--sft-model", help="Path to SFT model (for DPO stage)")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--quantize", choices=["4bit", "8bit"])
    parser.add_argument("--resume", help="Resume from checkpoint")
    args = parser.parse_args()

    config = load_training_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.stage == "sft":
        train_sft(config, args.data, f"{args.output_dir}/tmr-sft", args.quantize, args.resume)

    elif args.stage == "dpo":
        if not args.sft_model:
            print("ERROR: --sft-model required for DPO stage")
            sys.exit(1)
        train_dpo(config, args.data, args.sft_model, f"{args.output_dir}/tmr-dpo", args.quantize)

    elif args.stage == "full":
        sft_data = args.sft_data or args.data
        dpo_data = args.dpo_data or args.data
        if not sft_data:
            print("ERROR: --sft-data (or --data) required")
            sys.exit(1)

        sft_path = train_sft(config, sft_data, f"{args.output_dir}/tmr-sft", args.quantize)

        if dpo_data and Path(dpo_data).exists():
            train_dpo(config, dpo_data, sft_path, f"{args.output_dir}/tmr-dpo", args.quantize)
        else:
            print("\nNo DPO data provided, skipping DPO stage.")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
