"""
KTO Training — Phase-Balanced Batch Sampler (Among Us)

The dataset is prebuilt by rewards_with_belief.build_dataset. It already carries
the with_belief-compatible metadata columns (`kto_bucket`, `source_model`,
`actor_role`). This trainer uses `kto_bucket` to create homogeneous batches for
task actions, meeting votes, and meeting speeches while keeping the requested
True/False ratio within each bucket.

Dataset labels are built before training with rewards_with_belief.build_dataset.
This training script only loads a prebuilt HuggingFace Dataset from disk.

Usage:
    accelerate launch --config_file configs/accelerate_zero3_bf16.yaml \\
        kto_batch.py \\
        --model_path /path/to/sft_model \\
        --dataset_path /path/to/kto_dataset \\
        --output_dir /path/to/output
"""
import argparse
import inspect
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.distributed as dist
from datasets import load_from_disk
from torch.utils.data import BatchSampler, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import KTOConfig, KTOTrainer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from kto_weighting import resolve_kto_loss_weights


# ── KTO bucket groups ─────────────────────────────────────────────────────────

GROUP_SEQUENCE_ORDER = [
    "meeting_speech",
    "meeting_vote",
    "task_action",
]


def infer_training_group(sample: dict) -> str:
    bucket = str(sample.get("kto_bucket", "") or sample.get("_kto_bucket", "")).strip()
    if bucket in GROUP_SEQUENCE_ORDER:
        return bucket

    phase = str(sample.get("phase", "") or "")
    if "task" in phase.lower():
        return "task_action"
    if "meeting" in phase.lower():
        return "meeting_speech"
    return "other"


def _rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


class PhaseBalancedBatchSampler(BatchSampler):
    """
    Each local batch is homogeneous in KTO bucket:
    task_action / meeting_vote / meeting_speech.

    In distributed training, every process builds the same global batch, and
    each rank yields only its own per-device slice.
    """

    def __init__(
        self,
        dataset,
        per_device_batch_size: int,
        seed: int = 42,
        target_true_ratio: float = 0.5,
        group_cycle_length: int = 16,
        debug_batch_ratio: bool = False,
        debug_batch_ratio_steps: int = 20,
    ):
        self.per_device_bs = per_device_batch_size
        self.seed = seed
        self.epoch = 0
        self.target_true_ratio = min(max(target_true_ratio, 0.0), 1.0)
        self.group_cycle_length = max(1, group_cycle_length)
        self.debug_batch_ratio = debug_batch_ratio
        self.debug_batch_ratio_steps = max(0, debug_batch_ratio_steps)

        self.groups: dict[str, dict[bool, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for idx in range(len(dataset)):
            sample = dataset[idx]
            group = infer_training_group(sample)
            if group == "other":
                continue
            self.groups[group][bool(sample["label"])].append(idx)

        self.group_total = {
            g: len(d[True]) + len(d[False])
            for g, d in self.groups.items()
            if len(d[True]) + len(d[False]) > 0
        }
        self.total = sum(self.group_total.values())
        if self.total == 0:
            raise ValueError("No valid samples for PhaseBalancedBatchSampler.")

        self.group_sequence = [
            group for group in GROUP_SEQUENCE_ORDER if self.group_total.get(group, 0) > 0
        ]
        if not self.group_sequence:
            self.group_sequence = sorted(self.group_total)
        self.group_cycle = self._build_group_cycle()

    def _build_group_cycle(self) -> list[str]:
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
        sequence: list[str] = []
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

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        rank = _rank()
        world_size = _world_size()
        global_bs = self.per_device_bs * world_size
        rng = np.random.default_rng(self.seed + self.epoch)

        shuffled: dict[str, dict[bool, list[int]]] = {}
        ptrs: dict[str, dict[bool, int]] = {}
        for g, label_dict in self.groups.items():
            shuffled[g] = {}
            ptrs[g] = {}
            for lbl, idxs in label_dict.items():
                shuffled[g][lbl] = list(rng.permutation(idxs))
                ptrs[g][lbl] = 0

        def next_idx(g: str, lbl: bool) -> int:
            pool = shuffled[g].get(lbl, [])
            if not pool:
                raise ValueError(f"No samples for group={g} label={lbl}")
            ptr = ptrs[g][lbl]
            if ptr >= len(pool):
                shuffled[g][lbl] = list(rng.permutation(self.groups[g][lbl]))
                ptrs[g][lbl] = 0
                ptr = 0
            idx = shuffled[g][lbl][ptr]
            ptrs[g][lbl] += 1
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

                global_batch: list[int] = []
                global_batch_labels: list[bool] = []
                global_batch_groups: list[str] = []

                for _rank_slot in range(world_size):
                    local_batch: list[int] = []
                    local_labels: list[bool] = []
                    local_groups: list[str] = []

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
                        f"target_group={target_group} "
                        f"global(T/F)={global_true}/{global_false} "
                        f"local_rank{rank}(T/F)={local_true}/{local_false} "
                        f"local_groups={dict(Counter(local_groups))}",
                        flush=True,
                    )

                yield global_batch[start:end]
        finally:
            self.epoch += 1

    def __len__(self) -> int:
        world_size = _world_size()
        return self.total // (self.per_device_bs * world_size)


class PhaseBalancedKTOTrainer(KTOTrainer):
    """Use PhaseBalancedBatchSampler for the KTOTrainer DataLoader."""

    def __init__(
        self,
        *args,
        target_true_ratio: float = 0.5,
        group_cycle_length: int = 16,
        debug_batch_ratio: bool = False,
        debug_batch_ratio_steps: int = 20,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.target_true_ratio = target_true_ratio
        self.group_cycle_length = group_cycle_length
        self.debug_batch_ratio = debug_batch_ratio
        self.debug_batch_ratio_steps = debug_batch_ratio_steps

    def get_train_dataloader(self) -> DataLoader:
        dataset = self.train_dataset
        sampler = PhaseBalancedBatchSampler(
            dataset=dataset,
            per_device_batch_size=self.args.per_device_train_batch_size,
            seed=getattr(self.args, "seed", 42),
            target_true_ratio=self.target_true_ratio,
            group_cycle_length=self.group_cycle_length,
            debug_batch_ratio=self.debug_batch_ratio,
            debug_batch_ratio_steps=self.debug_batch_ratio_steps,
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


# ── Args ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--ref_model_path", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default=None, help="Prebuilt HF Dataset directory.")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Deprecated alias for --dataset_path.",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--max_prompt_length", type=int, default=3584)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--desirable_weight", type=str, default="0.7")
    parser.add_argument("--undesirable_weight", type=str, default="1.2")
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--save_safetensors", action="store_true")
    parser.add_argument("--save_only_model", action="store_true")
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--target_true_ratio", type=float, default=0.5)
    parser.add_argument("--group_cycle_length", type=int, default=16)
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


def preprocess_builder(tokenizer):
    def preprocess(example):
        if isinstance(example["prompt"], list):
            example["prompt"] = tokenizer.apply_chat_template(
                example["prompt"], tokenize=False, add_generation_prompt=True
            )
        if isinstance(example["completion"], list):
            example["completion"] = tokenizer.apply_chat_template(
                example["completion"], tokenize=False, add_generation_prompt=False
            )
        return example
    return preprocess


def load_prebuilt_dataset(args):
    """Load a prebuilt HF dataset from disk."""
    dataset_path = args.dataset_path or args.cache_dir
    if not dataset_path:
        raise ValueError("--dataset_path is required. Build it with rewards_with_belief.build_dataset first.")
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Prebuilt dataset not found: {dataset_path}")

    print(f"[data] loading prebuilt dataset from {dataset_path}", flush=True)
    dataset = load_from_disk(dataset_path)
    pos = sum(dataset["label"])
    neg = len(dataset) - pos
    print(f"[data] dataset size={len(dataset)} positive={pos} negative={neg}", flush=True)
    if "kto_bucket" in dataset.column_names:
        print(f"[data] kto_buckets={dict(Counter(dataset['kto_bucket']))}", flush=True)
    if "actor_role" in dataset.column_names:
        print(f"[data] actor_roles={dict(Counter(dataset['actor_role']))}", flush=True)
    return dataset


def build_training_args(args, output_dir: str, use_bf16: bool, use_fp16: bool) -> KTOConfig:
    kwargs = {
        "output_dir": output_dir,
        "beta": args.beta,
        "desirable_weight": args.desirable_weight,
        "undesirable_weight": args.undesirable_weight,
        "max_length": args.max_length,
        "max_prompt_length": args.max_prompt_length,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": args.lr_scheduler_type,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "warmup_ratio": args.warmup_ratio,
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
        "gradient_checkpointing": args.gradient_checkpointing,
        "seed": args.seed,
    }

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
            if _rank() == 0 and warn_unsupported:
                print(
                    f"[WARN] KTOConfig does not support these arguments in this TRL version; "
                    f"skipping: {warn_unsupported}",
                    flush=True,
                )
            kwargs = {key: value for key, value in kwargs.items() if key not in unsupported}

    return KTOConfig(**kwargs)


def main():
    args = parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if not 0.0 <= args.target_true_ratio <= 1.0:
        raise ValueError("--target_true_ratio must be between 0.0 and 1.0.")

    use_bf16, use_fp16, effective_dtype = resolve_precision_mode(args.torch_dtype)
    if effective_dtype != args.torch_dtype:
        print(f"[INFO] torch_dtype adjusted: {args.torch_dtype} -> {effective_dtype}", flush=True)

    for label, path in [
        ("model_path", args.model_path),
        ("ref_model_path", args.ref_model_path or args.model_path),
    ]:
        if not os.path.isdir(path):
            raise FileNotFoundError(f"[{label}] path does not exist: '{path}'")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=os.path.isdir(args.model_path),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": resolve_torch_dtype(effective_dtype),
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": os.path.isdir(args.model_path),
    }
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)

    ref_model_path = args.ref_model_path or args.model_path
    ref_kwargs = {**model_kwargs, "local_files_only": os.path.isdir(ref_model_path)}
    ref_model = AutoModelForCausalLM.from_pretrained(ref_model_path, **ref_kwargs)

    raw_dataset = load_prebuilt_dataset(args)
    args.desirable_weight, args.undesirable_weight = resolve_kto_loss_weights(
        args.desirable_weight,
        args.undesirable_weight,
        raw_dataset["label"],
        rank=_rank(),
    )
    train_dataset = raw_dataset.map(preprocess_builder(tokenizer))

    training_args = build_training_args(args, args.output_dir, use_bf16, use_fp16)

    trainer = PhaseBalancedKTOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        target_true_ratio=args.target_true_ratio,
        group_cycle_length=args.group_cycle_length,
        debug_batch_ratio=args.debug_batch_ratio,
        debug_batch_ratio_steps=args.debug_batch_ratio_steps,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
