#!/usr/bin/env python3
"""LoRA SFT training and merge helper for messages-format datasets."""

from __future__ import annotations

import argparse
import inspect
import json
import os
from dataclasses import dataclass
from typing import Any

import torch
from datasets import Dataset, DatasetDict, load_from_disk
from huggingface_hub import snapshot_download
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from safetensors import safe_open
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors_file
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, Trainer, TrainingArguments
from transformers.pytorch_utils import Conv1D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--merge_only", action="store_true")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=float, default=3)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument(
        "--gradient_checkpointing_use_reentrant",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use reentrant activation checkpointing. This avoids PyTorch metadata "
            "mismatch checks that can fail with DeepSpeed ZeRO-3 partitioned weights."
        ),
    )
    parser.add_argument("--save_safetensors", action="store_true")
    parser.add_argument("--save_only_model", action="store_true")
    parser.add_argument("--report_to", default="none")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated LoRA target module names.",
    )
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype | str:
    normalized = str(name).lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    if normalized == "auto":
        return "auto"
    raise ValueError(f"Unsupported torch dtype: {name}")


def training_precision_flags(dtype_name: str) -> dict[str, bool]:
    normalized = str(dtype_name).lower()
    if normalized in {"bf16", "bfloat16"}:
        return {"bf16": True, "fp16": False}
    if normalized in {"fp16", "float16"}:
        return {"bf16": False, "fp16": True}
    return {}


def load_sft_dataset(path: str, split: str) -> Dataset:
    if os.path.isdir(path):
        dataset = load_from_disk(path)
        if isinstance(dataset, DatasetDict):
            if split not in dataset:
                raise ValueError(f"Split '{split}' not found in {path}; available={list(dataset.keys())}")
            dataset = dataset[split]
        if not isinstance(dataset, Dataset):
            raise ValueError(f"Unsupported dataset object from {path}: {type(dataset).__name__}")
        return dataset

    if not path.endswith(".json"):
        raise ValueError(f"Expected a Hugging Face dataset directory or .json file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"Expected JSON list in {path}")
    return Dataset.from_list([row for row in rows if isinstance(row, dict)])


def render_messages(tokenizer: AutoTokenizer, messages: list[dict[str, Any]], add_generation_prompt: bool) -> str:
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    return "\n".join(str(message.get("content", "")) for message in messages)


def tokenize_record(tokenizer: AutoTokenizer, max_length: int, row: dict[str, Any]) -> dict[str, list[int]]:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("SFT rows must contain at least two messages.")

    prompt_text = render_messages(tokenizer, messages[:-1], add_generation_prompt=True)
    full_text = render_messages(tokenizer, messages, add_generation_prompt=False)

    full = tokenizer(full_text, add_special_tokens=False)
    prompt = tokenizer(prompt_text, add_special_tokens=False)
    input_ids = list(full["input_ids"])
    attention_mask = list(full["attention_mask"])
    labels = list(input_ids)

    prompt_len = min(len(prompt["input_ids"]), len(labels))
    labels[:prompt_len] = [-100] * prompt_len

    if len(input_ids) > max_length:
        input_ids = input_ids[-max_length:]
        attention_mask = attention_mask[-max_length:]
        labels = labels[-max_length:]

    if all(label == -100 for label in labels):
        labels[-1] = input_ids[-1]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


@dataclass
class CausalLMCollator:
    tokenizer: AutoTokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        max_len = max(len(feature["input_ids"]) for feature in features)
        batch: dict[str, list[list[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [pad_id] * pad_len)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_len)
            batch["labels"].append(feature["labels"] + [-100] * pad_len)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def load_tokenizer(model_path: str, trust_remote_code: bool) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def processor_source_candidates(model_path: str, adapter_path: str | None) -> list[str]:
    candidates = [model_path]
    if adapter_path:
        config_path = os.path.join(adapter_path, "adapter_config.json")
        if os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            base_model = str(config.get("base_model_name_or_path") or "").strip()
            if base_model and base_model not in candidates:
                candidates.append(base_model)
    return candidates


def save_processor_if_available(
    model_path: str,
    output_dir: str,
    trust_remote_code: bool,
    adapter_path: str | None = None,
) -> None:
    errors = []
    for source_path in processor_source_candidates(model_path, adapter_path):
        try:
            processor = AutoProcessor.from_pretrained(source_path, trust_remote_code=trust_remote_code)
        except Exception as exc:
            errors.append(f"{source_path}: {type(exc).__name__}: {exc}")
            continue

        processor.save_pretrained(output_dir)
        print(f"[processor] saved processor metadata from {source_path} to {output_dir}")
        return

    print("[processor] no processor metadata saved: " + " | ".join(errors))


def restore_missing_safetensor_keys(
    output_dir: str,
    source_paths: list[str],
    suffixes: tuple[str, ...] = ("self_attn.k_norm.weight",),
) -> None:
    safetensor_files = [
        os.path.join(output_dir, name)
        for name in os.listdir(output_dir)
        if name.endswith(".safetensors")
    ]
    if len(safetensor_files) != 1:
        return

    path = safetensor_files[0]
    saved_tensors = load_safetensors_file(path, device="cpu")
    source_suffix_keys: set[str] = set()
    missing_tensors: dict[str, torch.Tensor] = {}
    for source_path in source_paths:
        for source_file in resolve_safetensor_files(source_path):
            with safe_open(source_file, framework="pt", device="cpu") as source_tensors:
                source_keys = [
                    key for key in source_tensors.keys()
                    if any(key.endswith(suffix) for suffix in suffixes)
                ]
                source_suffix_keys.update(source_keys)
                for key in source_keys:
                    if key in saved_tensors or key in missing_tensors:
                        continue
                    missing_tensors[key] = source_tensors.get_tensor(key).clone()

    if missing_tensors:
        saved_tensors.update(missing_tensors)
        tmp_path = f"{path}.tmp"
        save_safetensors_file(saved_tensors, tmp_path, metadata={"format": "pt"})
        os.replace(tmp_path, path)
        print(f"[safetensors] restored {len(missing_tensors)} missing checkpoint keys in {path}")

    missing_after_restore = sorted(key for key in source_suffix_keys if key not in saved_tensors)
    if missing_after_restore:
        sample = ", ".join(missing_after_restore[:10])
        raise ValueError(
            "Merged checkpoint is missing required safetensor keys after restore: "
            f"{sample}"
        )
    if source_suffix_keys and not missing_tensors:
        print(f"[safetensors] verified {len(source_suffix_keys)} checkpoint keys in {path}")


def resolve_safetensor_files(path_or_repo: str) -> list[str]:
    if os.path.isdir(path_or_repo):
        root = path_or_repo
    else:
        try:
            root = snapshot_download(
                repo_id=path_or_repo,
                allow_patterns=["*.safetensors"],
                local_files_only=True,
            )
        except Exception:
            return []
    return sorted(
        os.path.join(root, name)
        for name in os.listdir(root)
        if name.endswith(".safetensors")
    )


def lora_target_matches(module_name: str, target_modules: list[str]) -> bool:
    return module_name in target_modules or any(
        module_name.endswith(f".{target_module}") for target_module in target_modules
    )


def is_peft_supported_target(module: torch.nn.Module) -> bool:
    return isinstance(
        module,
        (
            torch.nn.Linear,
            torch.nn.Embedding,
            torch.nn.Conv1d,
            torch.nn.Conv2d,
            torch.nn.Conv3d,
            torch.nn.MultiheadAttention,
            Conv1D,
        ),
    )


def resolve_lora_target_modules(model: torch.nn.Module, target_modules: list[str]) -> list[str]:
    matched_supported: list[str] = []
    matched_unsupported: list[str] = []
    for module_name, module in model.named_modules():
        if not lora_target_matches(module_name, target_modules):
            continue
        if is_peft_supported_target(module):
            matched_supported.append(module_name)
        else:
            matched_unsupported.append(f"{module_name}:{type(module).__name__}")

    if matched_unsupported:
        if not matched_supported:
            sample = ", ".join(matched_unsupported[:5])
            raise ValueError(
                "LoRA target modules matched only unsupported module types. "
                f"targets={target_modules}; unsupported_sample={sample}"
            )
        sample = ", ".join(matched_unsupported[:5])
        print(
            "[LoRA] resolved target_modules to exact supported module names "
            f"({len(matched_supported)} matched, skipped unsupported: {sample})"
        )
        return matched_supported

    return target_modules


def merge_adapter(args: argparse.Namespace) -> None:
    if not args.adapter_path:
        raise ValueError("--adapter_path is required with --merge_only")
    tokenizer = load_tokenizer(args.model_path, args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype(args.torch_dtype),
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter_path)
    merged = model.merge_and_unload()
    os.makedirs(args.output_dir, exist_ok=True)
    merged.save_pretrained(args.output_dir, safe_serialization=args.save_safetensors)
    if args.save_safetensors:
        restore_missing_safetensor_keys(
            args.output_dir,
            processor_source_candidates(args.model_path, args.adapter_path),
        )
    save_processor_if_available(args.model_path, args.output_dir, args.trust_remote_code, args.adapter_path)
    tokenizer.save_pretrained(args.output_dir)
    print(f"merged_model_path={args.output_dir}")


def train(args: argparse.Namespace) -> None:
    if not args.dataset_path:
        raise ValueError("--dataset_path is required for training")

    tokenizer = load_tokenizer(args.model_path, args.trust_remote_code)
    raw_dataset = load_sft_dataset(args.dataset_path, args.dataset_split)
    if len(raw_dataset) == 0:
        raise ValueError(f"Empty SFT dataset: {args.dataset_path}")

    tokenized = raw_dataset.map(
        lambda row: tokenize_record(tokenizer, args.max_length, row),
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing SFT records",
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype(args.torch_dtype),
        trust_remote_code=args.trust_remote_code,
    )
    gradient_checkpointing_kwargs = None
    if args.gradient_checkpointing:
        gradient_checkpointing_kwargs = {"use_reentrant": bool(args.gradient_checkpointing_use_reentrant)}
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        model.config.use_cache = False
    model.enable_input_require_grads()

    target_modules = [name.strip() for name in args.target_modules.split(",") if name.strip()]
    target_modules = resolve_lora_target_modules(model, target_modules)
    lora_alpha = args.lora_alpha if args.lora_alpha is not None else args.lora_r * 2
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    report_to = [] if args.report_to in {"", "none", "None"} else [args.report_to]
    training_args_kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "gradient_checkpointing": args.gradient_checkpointing,
        "report_to": report_to,
        "run_name": args.run_name,
        "optim": args.optim,
        "remove_unused_columns": False,
    }
    if (
        args.gradient_checkpointing
        and gradient_checkpointing_kwargs is not None
        and "gradient_checkpointing_kwargs" in inspect.signature(TrainingArguments.__init__).parameters
    ):
        training_args_kwargs["gradient_checkpointing_kwargs"] = gradient_checkpointing_kwargs
    if "save_safetensors" in inspect.signature(TrainingArguments.__init__).parameters:
        training_args_kwargs["save_safetensors"] = args.save_safetensors
    if "save_only_model" in inspect.signature(TrainingArguments.__init__).parameters:
        training_args_kwargs["save_only_model"] = args.save_only_model
    for precision_key, precision_value in training_precision_flags(args.torch_dtype).items():
        if precision_key in inspect.signature(TrainingArguments.__init__).parameters:
            training_args_kwargs[precision_key] = precision_value
    training_args = TrainingArguments(**training_args_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=CausalLMCollator(tokenizer),
    )
    trainer.train()

    adapter_dir = os.path.join(args.output_dir, "adapter_model")
    trainer.save_model(adapter_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(adapter_dir)
        print(f"adapter_path={adapter_dir}")


def main() -> None:
    args = parse_args()
    if args.merge_only:
        merge_adapter(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
