#!/usr/bin/env python3
"""Continue the SFT LoRA adapter with Direct Preference Optimization."""

from __future__ import annotations

import argparse
from pathlib import Path

from modeling import quantization_config
from validate_data import validate_preferences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-adapter", type=Path, default=Path("outputs/sft_adapter"))
    parser.add_argument("--data", type=Path, default=Path("data/preferences.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/dpo_adapter"))
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training (extremely slow).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pair_count, demo_count = validate_preferences(args.data)
    if demo_count:
        print(
            f"Warning: {demo_count} preference pair(s) are demos. Replace them with your own "
            "human choices for real alignment."
        )
    if pair_count < 20:
        print(
            f"Warning: only {pair_count} preference pairs. This can test the code, "
            "but a useful run normally needs many more reviewed choices."
        )
    if not args.sft_adapter.exists():
        raise SystemExit(f"SFT adapter does not exist: {args.sft_adapter}")
    if args.cpu and args.use_4bit:
        raise SystemExit("Choose either --cpu or --use-4bit, not both.")

    import torch
    from datasets import load_dataset
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    quantization = quantization_config(args.use_4bit)
    model_kwargs = {"is_trainable": True, "device_map": "auto", "dtype": "auto"}
    if quantization is not None:
        model_kwargs["quantization_config"] = quantization
    model = AutoPeftModelForCausalLM.from_pretrained(args.sft_adapter, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.sft_adapter)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset("json", data_files=str(args.data), split="train")

    has_cuda = torch.cuda.is_available() and not args.cpu
    use_bf16 = has_cuda and torch.cuda.is_bf16_supported()
    training_args = DPOConfig(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        beta=args.beta,
        max_length=args.max_length,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=True,
        use_cache=False,
        bf16=use_bf16,
        fp16=has_cuda and not use_bf16,
        use_cpu=args.cpu,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        seed=43,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(args.output)
    print(f"DPO adapter saved to {args.output}")


if __name__ == "__main__":
    main()
