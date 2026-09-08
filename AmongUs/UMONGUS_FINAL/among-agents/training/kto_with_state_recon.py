"""
KTO training with auxiliary SFT(CE) supervision for rewards_with_belief datasets.

This script is tailored to the current Among Us dataset layout:

- KTO dataset:
    rewards_with_belief/data/kto_dataset
    columns: prompt, completion, label, phase

- Optional auxiliary SFT(CE) role-prediction dataset:
    rewards_with_belief/data/role_prediction_sft_dataset
    columns: prompt, completion, phase, actor_role, label_policy,
             state_point, state_timing, ...

Training objective:

- task_action / meeting_vote / meeting_speech rows use standard KTO loss.
- state_recon rows use auxiliary causal-LM CE loss only.

Batching rule:

- Each local batch contains exactly one training group.
- All distributed ranks see the same training group at the same step.
- Group sequence follows dataset proportions.
- KTO groups keep the target True/False ratio cumulatively across group batches.
"""

import argparse
import inspect
import json
import os
import re
import sys
from collections import Counter, defaultdict
from itertools import takewhile
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from torch.utils.data import BatchSampler, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
try:
    from trl.experimental.kto import KTOConfig, KTOTrainer
except ImportError:
    from trl import KTOConfig, KTOTrainer


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from kto_weighting import resolve_kto_loss_weights


GROUP_SEQUENCE_ORDER = [
    "meeting_speech",
    "meeting_vote",
    "task_action",
    "state_recon",
]

ACTION_PREFIXES = (
    "COMPLETE FAKE TASK",
    "COMPLETE TASK",
    "REPORT DEAD BODY",
    "CALL MEETING",
    "VIEW MONITOR",
    "KILL",
    "VENT",
    "MOVE",
    "VOTE",
    "SPEAK",
)
ACTION_BLOCK_RE = re.compile(r"\[Action\]\s*(.*)", re.IGNORECASE | re.DOTALL)
ACTION_PREFIX_RE = re.compile(r"\b(" + "|".join(re.escape(prefix) for prefix in ACTION_PREFIXES) + r")\b", re.IGNORECASE)


def _rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def _message_text(value: Any) -> str:
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value or "")


def _label_is_true(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _first_line(text: str) -> str:
    return text.strip().splitlines()[0].strip() if text else ""


def extract_completion_action(completion: Any) -> str:
    text = _message_text(completion)
    match = ACTION_BLOCK_RE.search(text)
    candidate = _first_line(match.group(1)) if match else _first_line(text)
    candidate_upper = candidate.upper()
    for prefix in ACTION_PREFIXES:
        if candidate_upper.startswith(prefix):
            return prefix

    fallback = ACTION_PREFIX_RE.search(candidate_upper)
    if fallback:
        return fallback.group(1).upper()
    return ""


def infer_training_group(example: Dict[str, Any]) -> str:
    explicit = str(example.get("phase_group", "") or "").strip()
    if explicit:
        return explicit

    phase = str(example.get("phase", "") or "").strip().lower()
    if "state_reconstruction" in phase or "state_recon" in phase or "role_prediction" in phase:
        return "state_recon"

    action = extract_completion_action(example.get("completion"))

    if "task" in phase:
        return "task_action"
    if "meeting" in phase:
        if action == "VOTE":
            return "meeting_vote"
        if action == "SPEAK":
            return "meeting_speech"
        return "meeting_speech"
    if action == "VOTE":
        return "meeting_vote"
    if action == "SPEAK":
        return "meeting_speech"
    if action or "label" in example:
        return "task_action"
    return "other"


def display_group(group: str) -> str:
    return "role_pred" if group == "state_recon" else group


class GroupBalancedBatchSampler(BatchSampler):
    """
    Sequence-based group sampler.

    Each local batch is homogeneous in training group. This matters because
    state_recon rows take the aux CE branch while KTO rows take the KTO branch.
    """

    def __init__(
        self,
        dataset,
        per_device_batch_size: int,
        seed: int = 42,
        target_true_ratio: float = 0.6,
        group_cycle_length: int = 16,
        debug_batch_ratio: bool = False,
        debug_batch_ratio_steps: int = 20,
    ) -> None:
        self.per_device_bs = per_device_batch_size
        self.seed = seed
        self.epoch = 0
        self.target_true_ratio = min(max(target_true_ratio, 0.0), 1.0)
        self.group_cycle_length = max(1, group_cycle_length)
        self.debug_batch_ratio = debug_batch_ratio
        self.debug_batch_ratio_steps = max(0, debug_batch_ratio_steps)

        self.groups: Dict[str, Dict[bool, List[int]]] = defaultdict(lambda: defaultdict(list))
        for idx in range(len(dataset)):
            sample = dataset[idx]
            group = infer_training_group(sample)
            if group == "other":
                continue
            label = _label_is_true(sample.get("label", True))
            self.groups[group][label].append(idx)

        self.group_total = {
            group: len(label_dict[True]) + len(label_dict[False])
            for group, label_dict in self.groups.items()
            if len(label_dict[True]) + len(label_dict[False]) > 0
        }
        self.total = sum(self.group_total.values())
        if self.total == 0:
            raise ValueError("No valid samples for GroupBalancedBatchSampler.")

        self.group_sequence = [
            group for group in GROUP_SEQUENCE_ORDER if self.group_total.get(group, 0) > 0
        ]
        if not self.group_sequence:
            self.group_sequence = sorted(self.group_total)
        self.group_cycle = self._build_group_cycle()

    def _build_group_cycle(self) -> List[str]:
        groups = self.group_sequence
        cycle_len = self.group_cycle_length

        if cycle_len >= len(groups):
            counts = {group: 1 for group in groups}
            remaining_slots = cycle_len - len(groups)
        else:
            counts = {group: 0 for group in groups}
            remaining_slots = cycle_len

        raw = {
            group: remaining_slots * self.group_total[group] / self.total
            for group in groups
        }
        for group in groups:
            counts[group] += int(np.floor(raw[group]))
        remaining = cycle_len - sum(counts.values())
        if remaining > 0:
            by_remainder = sorted(
                groups,
                key=lambda group: (raw[group] - np.floor(raw[group]), self.group_total[group]),
                reverse=True,
            )
            for group in by_remainder[:remaining]:
                counts[group] += 1

        used = {group: 0 for group in groups}
        sequence: List[str] = []
        for step in range(cycle_len):
            candidates = [group for group in groups if used[group] < counts[group]]
            if not candidates:
                break
            chosen = max(
                candidates,
                key=lambda group: (
                    ((step + 1) * counts[group] / cycle_len) - used[group],
                    -used[group],
                    counts[group],
                    -groups.index(group),
                ),
            )
            sequence.append(chosen)
            used[chosen] += 1
        return sequence or [groups[0]]

    def _compute_label_alloc_for_group(
        self,
        group: str,
        batch_size: int,
        emitted_true: int = 0,
        emitted_total: int = 0,
    ) -> tuple[int, int]:
        true_cnt = len(self.groups[group].get(True, []))
        false_cnt = len(self.groups[group].get(False, []))
        if batch_size <= 0:
            return 0, 0
        if true_cnt == 0:
            return 0, batch_size
        if false_cnt == 0:
            return batch_size, 0

        target_true_after_this_batch = (emitted_total + batch_size) * self.target_true_ratio
        n_true = int(round(target_true_after_this_batch - emitted_true))
        n_true = min(max(n_true, 0), batch_size)
        return n_true, batch_size - n_true

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rank = _rank()
        world_size = _world_size()
        global_bs = self.per_device_bs * world_size
        rng = np.random.default_rng(self.seed + self.epoch)

        shuffled: Dict[str, Dict[bool, List[int]]] = {}
        ptrs: Dict[str, Dict[bool, int]] = {}
        for group, label_dict in self.groups.items():
            shuffled[group] = {}
            ptrs[group] = {}
            for label, indices in label_dict.items():
                if not indices:
                    continue
                shuffled[group][label] = list(rng.permutation(indices))
                ptrs[group][label] = 0

        def next_idx(group: str, label: bool) -> int:
            pool = shuffled[group].get(label, [])
            if not pool:
                raise ValueError(f"No samples for group={group} label={label}")
            ptr = ptrs[group][label]
            if ptr >= len(pool):
                shuffled[group][label] = list(rng.permutation(self.groups[group][label]))
                ptrs[group][label] = 0
                ptr = 0
            idx = shuffled[group][label][ptr]
            ptrs[group][label] += 1
            return idx

        n_batches = self.total // global_bs
        try:
            emitted_true_by_group = {group: 0 for group in self.group_sequence}
            emitted_total_by_group = {group: 0 for group in self.group_sequence}
            for batch_idx in range(n_batches):
                target_group = self.group_cycle[(batch_idx + self.epoch) % len(self.group_cycle)]
                n_true, n_false = self._compute_label_alloc_for_group(
                    target_group,
                    self.per_device_bs,
                    emitted_true=emitted_true_by_group[target_group],
                    emitted_total=emitted_total_by_group[target_group],
                )
                emitted_true_by_group[target_group] += n_true
                emitted_total_by_group[target_group] += self.per_device_bs

                global_batch: List[int] = []
                global_batch_labels: List[bool] = []
                global_batch_groups: List[str] = []

                for _rank_slot in range(world_size):
                    local_batch: List[int] = []
                    local_labels: List[bool] = []
                    local_groups: List[str] = []

                    for _ in range(n_true):
                        local_batch.append(next_idx(target_group, True))
                        local_labels.append(True)
                        local_groups.append(target_group)
                    for _ in range(n_false):
                        local_batch.append(next_idx(target_group, False))
                        local_labels.append(False)
                        local_groups.append(target_group)

                    if local_batch:
                        perm = rng.permutation(len(local_batch))
                        local_batch = [local_batch[i] for i in perm]
                        local_labels = [local_labels[i] for i in perm]
                        local_groups = [local_groups[i] for i in perm]

                    global_batch.extend(local_batch)
                    global_batch_labels.extend(local_labels)
                    global_batch_groups.extend(local_groups)

                start = rank * self.per_device_bs
                end = start + self.per_device_bs

                if self.debug_batch_ratio and batch_idx < self.debug_batch_ratio_steps:
                    global_size = len(global_batch_labels)
                    global_true = sum(global_batch_labels)
                    global_false = global_size - global_true
                    local_labels = global_batch_labels[start:end]
                    local_groups = global_batch_groups[start:end]
                    local_size = len(local_labels)
                    local_true = sum(local_labels)
                    local_false = local_size - local_true
                    print(
                        "[DEBUG][batch_ratio] "
                        f"epoch={self.epoch} batch={batch_idx + 1}/{n_batches} "
                        f"target_group={display_group(target_group)} "
                        f"global(T/F)={global_true}/{global_false} "
                        f"local_rank{rank}(T/F)={local_true}/{local_false} "
                        f"local_groups={dict(Counter(map(display_group, local_groups)))}",
                        flush=True,
                    )

                yield global_batch[start:end]
        finally:
            self.epoch += 1

    def __len__(self) -> int:
        return self.total // (self.per_device_bs * _world_size())


class GroupBalancedKTOTrainer(KTOTrainer):
    def __init__(
        self,
        *args,
        target_true_ratio: float = 0.6,
        group_cycle_length: int = 16,
        debug_batch_ratio: bool = False,
        debug_batch_ratio_steps: int = 20,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.target_true_ratio = target_true_ratio
        self.group_cycle_length = max(1, group_cycle_length)
        self.debug_batch_ratio = debug_batch_ratio
        self.debug_batch_ratio_steps = debug_batch_ratio_steps

    def get_train_dataloader(self) -> DataLoader:
        sampler = GroupBalancedBatchSampler(
            dataset=self.train_dataset,
            per_device_batch_size=self.args.per_device_train_batch_size,
            seed=getattr(self.args, "seed", 42),
            target_true_ratio=self.target_true_ratio,
            group_cycle_length=self.group_cycle_length,
            debug_batch_ratio=self.debug_batch_ratio,
            debug_batch_ratio_steps=self.debug_batch_ratio_steps,
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            collate_fn=self._collate_with_metadata,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def _collate_with_metadata(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        metadata_keys = (
            "phase_group",
            "phase",
            "sample_source",
            "state_point",
            "state_timing",
            "source_phase",
            "source_model",
            "kto_bucket",
        )
        metadata = {
            key: [feature.get(key, "") for feature in features]
            for key in metadata_keys
            if any(key in feature for feature in features)
        }
        collator_features = [
            {key: value for key, value in feature.items() if key not in metadata_keys}
            for feature in features
        ]
        batch = self.data_collator(collator_features)
        batch.update(metadata)

        groups = batch.get("phase_group")
        if groups and any(group == "state_recon" for group in groups):
            batch.update(self._build_aux_sft_batch(features))

        return batch

    def _build_aux_sft_batch(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        tokenizer = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("Cannot build state_recon SFT batch without a tokenizer/processing_class.")

        max_length = getattr(self.args, "max_length", None)
        if max_length is None:
            max_length = max(
                getattr(self.args, "max_prompt_length", 0) or 0,
                1,
            )
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id or eos_token_id for aux SFT padding.")

        input_rows: List[List[int]] = []
        label_rows: List[List[int]] = []
        for feature in features:
            prompt_ids, completion_ids = self._extract_aux_token_ids(tokenizer, feature)
            input_ids = prompt_ids + completion_ids
            labels = [-100] * len(prompt_ids) + completion_ids

            if max_length and len(input_ids) > max_length:
                overflow = len(input_ids) - max_length
                input_ids = input_ids[overflow:]
                labels = labels[overflow:]

            input_rows.append(input_ids)
            label_rows.append(labels)

        batch_max_length = max(len(row) for row in input_rows)
        aux_input_ids = []
        aux_attention_mask = []
        aux_labels = []
        for input_ids, labels in zip(input_rows, label_rows):
            pad_length = batch_max_length - len(input_ids)
            aux_input_ids.append(input_ids + [pad_token_id] * pad_length)
            aux_attention_mask.append([1] * len(input_ids) + [0] * pad_length)
            aux_labels.append(labels + [-100] * pad_length)

        return {
            "aux_input_ids": torch.tensor(aux_input_ids, dtype=torch.long),
            "aux_attention_mask": torch.tensor(aux_attention_mask, dtype=torch.long),
            "aux_labels": torch.tensor(aux_labels, dtype=torch.long),
        }

    @staticmethod
    def _render_chat_field(tokenizer, value: Any, *, add_generation_prompt: bool) -> str:
        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
            return tokenizer.apply_chat_template(
                value,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)
        return str(value or "")

    def _extract_aux_token_ids(self, tokenizer, feature: Dict[str, Any]) -> tuple[List[int], List[int]]:
        if "prompt_input_ids" in feature and "completion_input_ids" in feature:
            return list(feature["prompt_input_ids"]), list(feature["completion_input_ids"])

        prompt, completion = render_prompt_completion_fields(
            tokenizer,
            feature.get("prompt"),
            feature.get("completion"),
        )
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
        return prompt_ids, completion_ids


class KTOWithStateReconTrainer(GroupBalancedKTOTrainer):
    def __init__(self, *args, aux_loss_weight: float = 0.2, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.aux_loss_weight = aux_loss_weight

    def get_batch_loss_metrics(self, model, batch):
        groups = batch.get("phase_group")
        phases = batch.get("phase")
        if groups is None and phases is None:
            return super().get_batch_loss_metrics(model, batch)

        if groups is None:
            groups = [infer_training_group({"phase": phase}) for phase in phases]

        sr_mask = torch.tensor(
            [group == "state_recon" for group in groups],
            dtype=torch.bool,
        )
        kto_mask = ~sr_mask

        total_loss = torch.tensor(0.0, device=self.accelerator.device)
        metrics: Dict[str, Any] = {}

        if kto_mask.any():
            kto_batch = self._filter_batch(batch, kto_mask)
            kto_loss, kto_metrics = super().get_batch_loss_metrics(model, kto_batch)
            total_loss = total_loss + kto_loss
            metrics.update(kto_metrics)

        if sr_mask.any():
            aux_loss = self._compute_aux_ce_loss(model, batch, sr_mask)
            total_loss = total_loss + self.aux_loss_weight * aux_loss
            metrics["aux/ce_loss"] = aux_loss.item()

        return total_loss, metrics

    def _filter_batch(self, batch: Dict[str, Any], mask: torch.Tensor) -> Dict[str, Any]:
        indices = mask.nonzero(as_tuple=True)[0].tolist()
        filtered: Dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                filtered[key] = value[mask.to(value.device)]
            elif isinstance(value, list):
                filtered[key] = [value[i] for i in indices]
            else:
                filtered[key] = value
        return filtered

    def _compute_aux_ce_loss(self, model, batch: Dict[str, Any], sr_mask: torch.Tensor) -> torch.Tensor:
        if "aux_input_ids" in batch and "aux_labels" in batch:
            input_ids = batch["aux_input_ids"][sr_mask.to(batch["aux_input_ids"].device)]
            attention_mask = batch.get("aux_attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask[sr_mask.to(attention_mask.device)]
            labels = batch["aux_labels"][sr_mask.to(batch["aux_labels"].device)]
        elif "completion_input_ids" in batch and "completion_labels" in batch:
            input_ids = batch["completion_input_ids"][sr_mask.to(batch["completion_input_ids"].device)]
            attention_mask = batch.get("completion_attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask[sr_mask.to(attention_mask.device)]
            labels = batch["completion_labels"][sr_mask.to(batch["completion_labels"].device)]
        elif "input_ids" in batch and "labels" in batch:
            input_ids = batch["input_ids"][sr_mask.to(batch["input_ids"].device)]
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask[sr_mask.to(attention_mask.device)]
            labels = batch["labels"][sr_mask.to(batch["labels"].device)]
        else:
            available = ", ".join(sorted(batch.keys()))
            raise KeyError(f"Cannot compute aux CE loss from batch keys: {available}")

        input_ids = input_ids.to(self.accelerator.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.accelerator.device)
        labels = labels.to(self.accelerator.device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--ref_model_path", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, required=True, help="KTO HF Dataset directory.")
    parser.add_argument("--aux_dataset_path", type=str, default=None, help="Optional auxiliary SFT(CE) HF Dataset directory.")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--max_prompt_length", type=int, default=None)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--desirable_weight", type=str, default="1.0")
    parser.add_argument("--undesirable_weight", type=str, default="1.0")
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--run_name", type=str, default=None)
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
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--target_true_ratio", type=float, default=0.5)
    parser.add_argument("--group_cycle_length", type=int, default=16)
    parser.add_argument(
        "--aux_loss_weight",
        type=str,
        default="auto",
        help="Aux CE loss weight, or 'auto' to use len(aux_dataset) / len(kto_dataset).",
    )
    parser.add_argument("--debug_batch_ratio", action="store_true")
    parser.add_argument("--debug_batch_ratio_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "bfloat16", "float16", "float32"],
    )
    return parser.parse_args()


def resolve_torch_dtype(dtype_name: str):
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(dtype_name, "auto")


def is_bf16_gpu_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    if hasattr(torch.cuda, "is_bf16_supported"):
        try:
            return bool(torch.cuda.is_bf16_supported())
        except Exception:
            pass
    try:
        major, _ = torch.cuda.get_device_capability(0)
        return major >= 8
    except Exception:
        return False


def resolve_precision_mode(dtype_name: str) -> tuple[bool, bool, str]:
    if dtype_name == "auto":
        if is_bf16_gpu_supported():
            return True, False, "bfloat16"
        if torch.cuda.is_available():
            return False, True, "float16"
        return False, False, "float32"

    if dtype_name == "bfloat16" and not is_bf16_gpu_supported():
        if torch.cuda.is_available():
            print("[WARN] bf16 unsupported on this GPU. Falling back to float16.", flush=True)
            return False, True, "float16"
        print("[WARN] bf16 requested but no CUDA GPU. Falling back to float32.", flush=True)
        return False, False, "float32"

    if dtype_name == "float16" and not torch.cuda.is_available():
        print("[WARN] float16 requested but no CUDA GPU. Falling back to float32.", flush=True)
        return False, False, "float32"

    if dtype_name == "bfloat16":
        return True, False, "bfloat16"
    if dtype_name == "float16":
        return False, True, "float16"
    return False, False, "float32"


def _completion_text(value: Any) -> str:
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value or "")


def _common_prefix(left: str, right: str) -> str:
    return "".join(x for x, _ in takewhile(lambda pair: pair[0] == pair[1], zip(left, right, strict=False)))


def render_prompt_completion_fields(tokenizer, prompt: Any, completion: Any) -> tuple[str, str]:
    if (
        isinstance(prompt, list)
        and isinstance(completion, list)
        and all(isinstance(item, dict) for item in prompt)
        and all(isinstance(item, dict) for item in completion)
    ):
        rendered_prompt = tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
        )
        rendered_full = tokenizer.apply_chat_template(
            prompt + completion,
            tokenize=False,
            add_generation_prompt=False,
        )
        prefix = _common_prefix(rendered_prompt, rendered_full)
        return prefix, rendered_full[len(prefix) :]

    if isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt):
        rendered_prompt = tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
        )
    elif isinstance(prompt, dict):
        rendered_prompt = json.dumps(prompt, ensure_ascii=False)
    else:
        rendered_prompt = str(prompt or "")

    return rendered_prompt, _completion_text(completion)


def apply_chat_template_if_needed(tokenizer):
    def preprocess(example: Dict[str, Any]) -> Dict[str, Any]:
        prompt, completion = render_prompt_completion_fields(
            tokenizer,
            example.get("prompt"),
            example.get("completion"),
        )
        example["prompt"] = prompt
        example["completion"] = completion
        return example

    return preprocess


def _to_rows(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        return [row for row in obj if isinstance(row, dict)]
    if isinstance(obj, dict):
        if any(key in obj for key in ("prompt", "completion", "messages")):
            return [obj]
        keys = list(obj.keys())
        n_rows = max((len(obj[key]) for key in keys if isinstance(obj[key], list)), default=0)
        return [
            {
                key: obj[key][idx] if isinstance(obj[key], list) and idx < len(obj[key]) else obj[key]
                for key in keys
            }
            for idx in range(n_rows)
        ]
    raise ValueError(f"Unsupported dataset JSON type: {type(obj).__name__}")


def _normalise_message_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalised: List[Dict[str, Any]] = []
    for row in rows:
        row = dict(row)
        if "messages" in row and ("prompt" not in row or "completion" not in row):
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                continue
            row["prompt"] = messages[:-1]
            row["completion"] = [messages[-1]]
            row.setdefault("label", True)
        normalised.append(row)
    return normalised


def _load_dataset(path: str, split: str) -> Dataset:
    if os.path.isdir(path):
        dataset = load_from_disk(path)
    elif os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            rows = _normalise_message_rows(_to_rows(json.load(f)))
        if not rows:
            raise ValueError(f"No usable rows found in JSON dataset: {path}")
        dataset = Dataset.from_list(rows)
    else:
        raise FileNotFoundError(path)

    if isinstance(dataset, DatasetDict):
        if split not in dataset:
            raise KeyError(f"Split '{split}' not found in {path}; available={list(dataset.keys())}")
        dataset = dataset[split]
    return dataset


def add_phase_group(dataset: Dataset, source: str) -> Dataset:
    def mapper(example: Dict[str, Any]) -> Dict[str, Any]:
        group = infer_training_group(example)
        example["phase_group"] = group
        example["sample_source"] = source
        return example

    return dataset.map(mapper)


def load_kto_dataset(path: str, split: str) -> Dataset:
    dataset = _load_dataset(path, split)
    required = {"prompt", "completion", "label"}
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(f"KTO dataset missing columns {sorted(missing)}: {path}")
    if "phase" not in dataset.column_names:
        dataset = dataset.map(lambda example: {**example, "phase": ""})
    dataset = add_phase_group(dataset, "kto")
    label_counts = Counter(_label_is_true(label) for label in dataset["label"])
    group_counts = Counter(dataset["phase_group"])
    print(
        f"[data] KTO rows={len(dataset)} labels={{True: {label_counts[True]}, False: {label_counts[False]}}}",
        flush=True,
    )
    print(f"[data] KTO groups={dict(group_counts)}", flush=True)
    return dataset


def load_state_recon_dataset(path: str, split: str) -> Dataset:
    dataset = _load_dataset(path, split)
    required = {"prompt", "completion"}
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(f"state_recon dataset missing columns {sorted(missing)}: {path}")

    if "label" in dataset.column_names:
        before = len(dataset)
        dataset = dataset.filter(lambda example: _label_is_true(example.get("label", True)))
        print(
            f"[data] state_recon label=True filter rows={len(dataset)} dropped={before - len(dataset)}",
            flush=True,
        )

    def mapper(example: Dict[str, Any]) -> Dict[str, Any]:
        original_phase = str(example.get("phase", "") or "")
        example["label"] = True
        example["phase"] = "state_reconstruction"
        example["phase_group"] = "state_recon"
        example["sample_source"] = "state_recon"
        example["source_phase"] = original_phase
        return example

    dataset = dataset.map(mapper)
    state_points = Counter(dataset["state_point"]) if "state_point" in dataset.column_names else Counter()
    state_timings = Counter(dataset["state_timing"]) if "state_timing" in dataset.column_names else Counter()
    print(f"[data] state_recon rows={len(dataset)}", flush=True)
    if state_points:
        print(f"[data] state_recon state_points={dict(state_points)}", flush=True)
    if state_timings:
        print(f"[data] state_recon state_timings={dict(state_timings)}", flush=True)
    return dataset


def align_and_concat(datasets: Iterable[Dataset]) -> Dataset:
    dataset_list = list(datasets)
    all_columns = sorted({column for dataset in dataset_list for column in dataset.column_names})

    aligned = []
    for dataset in dataset_list:
        missing = [column for column in all_columns if column not in dataset.column_names]
        if missing:
            dataset = dataset.map(lambda example, cols=missing: {**example, **{col: "" for col in cols}})
        aligned.append(dataset.select_columns(all_columns))
    return concatenate_datasets(aligned)


def build_training_args(args, output_dir: str, use_bf16: bool, use_fp16: bool) -> KTOConfig:
    kwargs = {
        "output_dir": output_dir,
        "beta": args.beta,
        "desirable_weight": args.desirable_weight,
        "undesirable_weight": args.undesirable_weight,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": args.lr_scheduler_type,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "warmup_steps": args.warmup_steps,
        "logging_steps": args.logging_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "save_safetensors": args.save_safetensors,
        "save_only_model": args.save_only_model,
        "bf16": use_bf16,
        "fp16": use_fp16,
        "optim": args.optim,
        "report_to": args.report_to,
        "run_name": args.run_name or os.path.basename(output_dir.rstrip(os.sep)),
        "disable_tqdm": False,
        "logging_first_step": True,
        "gradient_checkpointing": args.gradient_checkpointing,
        "seed": args.seed,
        "remove_unused_columns": False,
    }
    if args.gradient_checkpointing:
        kwargs["gradient_checkpointing_kwargs"] = {
            "use_reentrant": bool(args.gradient_checkpointing_use_reentrant)
        }
    if args.warmup_ratio is not None and args.warmup_steps <= 0:
        kwargs["warmup_ratio"] = args.warmup_ratio
    if args.max_prompt_length is not None:
        kwargs["max_prompt_length"] = args.max_prompt_length

    signature = inspect.signature(KTOConfig)
    has_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if not has_var_kwargs:
        unsupported = sorted(set(kwargs) - set(signature.parameters))
        if unsupported:
            quiet_unsupported = {"save_safetensors"}
            warn_unsupported = [key for key in unsupported if key not in quiet_unsupported]
            if _rank() == 0:
                if warn_unsupported:
                    print(
                        f"[WARN] KTOConfig does not support these arguments in this TRL version; "
                        f"skipping: {warn_unsupported}",
                        flush=True,
                    )
            kwargs = {key: value for key, value in kwargs.items() if key not in unsupported}

    return KTOConfig(**kwargs)


def resolve_aux_loss_weight(value: str, kto_rows: int, aux_rows: int) -> float:
    raw = str(value).strip().lower()
    if raw == "auto":
        if kto_rows <= 0:
            return 0.0
        return aux_rows / kto_rows
    return float(value)


def resolve_model_ref(label: str, value: str) -> tuple[str, bool]:
    expanded = os.path.expanduser(value)
    if os.path.isdir(expanded):
        return expanded, True
    if value.startswith(("/", "./", "../", "~")):
        raise FileNotFoundError(f"[{label}] path does not exist: {value}")
    return value, False


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if not 0.0 <= args.target_true_ratio <= 1.0:
        raise ValueError("--target_true_ratio must be between 0.0 and 1.0.")

    use_bf16, use_fp16, effective_dtype = resolve_precision_mode(args.torch_dtype)
    if effective_dtype != args.torch_dtype:
        print(f"[INFO] torch_dtype adjusted: {args.torch_dtype} -> {effective_dtype}", flush=True)

    model_path, model_is_local = resolve_model_ref("model_path", args.model_path)
    ref_model_path, ref_is_local = resolve_model_ref("ref_model_path", args.ref_model_path or args.model_path)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=model_is_local,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": resolve_torch_dtype(effective_dtype),
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": model_is_local,
    }
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    ref_kwargs = {**model_kwargs, "local_files_only": ref_is_local}
    ref_model = AutoModelForCausalLM.from_pretrained(ref_model_path, **ref_kwargs)

    kto_dataset = load_kto_dataset(args.dataset_path, args.dataset_split)
    datasets_to_merge = [kto_dataset]
    aux_dataset = None
    if args.aux_dataset_path:
        aux_dataset = load_state_recon_dataset(args.aux_dataset_path, args.dataset_split)
        datasets_to_merge.append(aux_dataset)
    aux_loss_weight = resolve_aux_loss_weight(
        args.aux_loss_weight,
        kto_rows=len(kto_dataset),
        aux_rows=len(aux_dataset) if aux_dataset is not None else 0,
    )
    print(f"[data] aux_loss_weight={aux_loss_weight:.6g} (setting={args.aux_loss_weight})", flush=True)

    raw_dataset = align_and_concat(datasets_to_merge)
    print(f"[data] merged training rows={len(raw_dataset)} groups={dict(Counter(raw_dataset['phase_group']))}", flush=True)
    args.desirable_weight, args.undesirable_weight = resolve_kto_loss_weights(
        args.desirable_weight,
        args.undesirable_weight,
        kto_dataset["label"],
        rank=_rank(),
    )

    train_dataset = raw_dataset.map(apply_chat_template_if_needed(tokenizer))

    training_args = build_training_args(args, args.output_dir, use_bf16, use_fp16)
    trainer = KTOWithStateReconTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        target_true_ratio=args.target_true_ratio,
        group_cycle_length=args.group_cycle_length,
        debug_batch_ratio=args.debug_batch_ratio,
        debug_batch_ratio_steps=args.debug_batch_ratio_steps,
        aux_loss_weight=aux_loss_weight,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
