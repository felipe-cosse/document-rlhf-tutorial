#!/usr/bin/env python3
"""Train a small LoRA instruction adapter on human-reviewed grounded answers."""

from __future__ import annotations

import argparse
from pathlib import Path

from modeling import DEFAULT_MODEL, quantization_config
from validate_data import validate_sft


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=Path("data/sft.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/sft_adapter"))
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training (very slow).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    example_count = validate_sft(args.data)
    if example_count < 20:
        print(
            f"Warning: only {example_count} SFT examples. This is enough to test the pipeline, "
            "not to make a reliable assistant."
        )

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    if args.cpu and args.use_4bit:
        raise SystemExit("Choose either --cpu or --use-4bit, not both.")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset("json", data_files=str(args.data), split="train")
    quantization = quantization_config(args.use_4bit)
    has_cuda = torch.cuda.is_available() and not args.cpu
    use_bf16 = has_cuda and torch.cuda.is_bf16_supported()

    training_args = SFTConfig(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        max_length=args.max_length,
        assistant_only_loss=True,
        gradient_checkpointing=True,
        use_cache=False,
        bf16=use_bf16,
        fp16=has_cuda and not use_bf16,
        use_cpu=args.cpu,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        seed=42,
        model_init_kwargs={"dtype": "auto"},
    )
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    trainer = SFTTrainer(
        model=args.model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora,
        quantization_config=quantization,
    )
    trainer.train()
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(args.output)
    print(f"SFT adapter saved to {args.output}")


if __name__ == "__main__":
    main()
