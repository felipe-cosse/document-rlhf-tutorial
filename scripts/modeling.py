"""Model-loading and text-generation helpers for the tutorial."""

from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_MODEL = "Qwen/Qwen3-0.6B-Base"


def quantization_config(use_4bit: bool) -> Any | None:
    if not use_4bit:
        return None
    import torch
    from transformers import BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("--use-4bit requires a CUDA-capable GPU in this tutorial.")
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )


def load_for_inference(
    model_id: str = DEFAULT_MODEL,
    adapter: Path | None = None,
    use_4bit: bool = False,
) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    quantization = quantization_config(use_4bit)
    load_kwargs: dict[str, Any] = {"device_map": "auto", "dtype": "auto"}
    if quantization is not None:
        load_kwargs["quantization_config"] = quantization

    if adapter is not None:
        if not adapter.exists():
            raise FileNotFoundError(f"Adapter does not exist: {adapter}")
        from peft import AutoPeftModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(adapter)
        model = AutoPeftModelForCausalLM.from_pretrained(adapter, **load_kwargs)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def render_chat(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def generate_answer(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int = 220,
    temperature: float = 0.0,
    seed: int = 42,
) -> str:
    import torch

    text = render_chat(tokenizer, messages)
    inputs = tokenizer(text, return_tensors="pt")
    input_device = next(parameter for parameter in model.parameters() if parameter.device.type != "meta").device
    inputs = {key: value.to(input_device) for key, value in inputs.items()}
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation_kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
    else:
        generation_kwargs.update(do_sample=False)

    with torch.inference_mode():
        output = model.generate(**inputs, **generation_kwargs)
    generated = output[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()
