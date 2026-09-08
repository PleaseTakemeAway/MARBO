
import argparse
import inspect
import json
import os
from collections import Counter, defaultdict
import numpy as np
import torch
import torch.distributed as dist
from datasets import Dataset, concatenate_datasets
from torch.utils.data import BatchSampler, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import KTOConfig, KTOTrainer


# ── Phase Definition ────────────────────────────────────────────────────────────

PHASE_PATTERNS = {
    "state_recon": ["state_reconstruction", "state_recon"],
    "night_skill": ["night_skill"],
    "vote":        ["vote"],
    "speech":      ["speech"],
}
PHASE_SEQUENCE_ORDER = ["speech", "vote", "night_skill", "state_recon"]


def get_phase_group(phase: str) -> str:
    for group, patterns in PHASE_PATTERNS.items():
        if any(p in phase for p in patterns):
            return group
    return "other"


# ── DDP Helper Functions ───────────────────────────────────────────────────────

def _rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


# ── Phase-Balanced Batch Sampler ───────────────────────────────────────────────
class PhaseBalancedBatchSampler(BatchSampler):
    """
    Batch-sequence phase sampler.

    """

    def __init__(
        self,
        dataset,
        per_device_batch_size: int,
        seed: int = 42,
        target_true_ratio: float = 0.5,
        phase_cycle_length: int = 16,
        debug_batch_ratio: bool = False,
        debug_batch_ratio_steps: int = 20,
    ):
        self.per_device_bs = per_device_batch_size
        self.seed = seed
        self.epoch = 0
        self.target_true_ratio = min(max(target_true_ratio, 0.0), 1.0)
        self.phase_cycle_length = max(1, phase_cycle_length)
        self.debug_batch_ratio = debug_batch_ratio
        self.debug_batch_ratio_steps = max(0, debug_batch_ratio_steps)

     
        self.groups: dict[str, dict[bool, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for idx in range(len(dataset)):
            sample = dataset[idx]
            group = get_phase_group(sample["phase"])
            if group == "other":
                continue
            self.groups[group][sample["label"]].append(idx)


        self.group_total = {
            g: len(d[True]) + len(d[False])
            for g, d in self.groups.items()
            if (len(d[True]) + len(d[False])) > 0
        }
        self.total = sum(self.group_total.values())
        if self.total == 0:
            raise ValueError("No valid samples for sampler (phase groups are empty).")

        self.phase_groups = [
            g for g in PHASE_SEQUENCE_ORDER if self.group_total.get(g, 0) > 0
        ]
        if not self.phase_groups:
            self.phase_groups = sorted(self.group_total.keys())

        self.phase_cycle = self._build_phase_cycle()

    def _build_phase_cycle(self) -> list[str]:
        groups = self.phase_groups
        cycle_len = self.phase_cycle_length

        raw = {
            g: cycle_len * self.group_total[g] / self.total
            for g in groups
        }
        counts = {g: int(np.floor(raw[g])) for g in groups}
        remaining = cycle_len - sum(counts.values())
        if remaining > 0:
            by_remainder = sorted(
                groups,
                key=lambda g: (raw[g] - counts[g], self.group_total[g]),
                reverse=True,
            )
            for g in by_remainder[:remaining]:
                counts[g] += 1

        used = {g: 0 for g in groups}
        sequence: list[str] = []
        for step in range(cycle_len):
            candidates = [g for g in groups if used[g] < counts[g]]
            if not candidates:
                break
            chosen = max(
                candidates,
                key=lambda g: (
                    ((step + 1) * counts[g] / cycle_len) - used[g],
                    -used[g],
                    counts[g],
                    -groups.index(g),
                ),
            )
            sequence.append(chosen)
            used[chosen] += 1

        if not sequence:
            sequence = [groups[0]]
        return sequence

    def _display_group(self, group: str) -> str:
        return "role_pred" if group == "state_recon" else group

    def _compute_label_alloc_for_group(self, group: str, batch_size: int) -> tuple[int, int]:
        true_cnt = len(self.groups[group].get(True, []))
        false_cnt = len(self.groups[group].get(False, []))
        if batch_size <= 0:
            return 0, 0
        if true_cnt == 0:
            return 0, batch_size
        if false_cnt == 0:
            return batch_size, 0

        n_true = int(round(batch_size * self.target_true_ratio))
        n_true = min(max(n_true, 0), batch_size)
        if self.target_true_ratio > 0.0 and n_true == 0:
            n_true = 1
        if self.target_true_ratio < 1.0 and n_true == batch_size:
            n_true = batch_size - 1
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
            
            ptr = ptrs[g][lbl]
            if ptr >= len(shuffled[g][lbl]):
                shuffled[g][lbl] = list(rng.permutation(self.groups[g][lbl]))
                ptrs[g][lbl] = 0
                ptr = 0
            idx = shuffled[g][lbl][ptr]
            ptrs[g][lbl] += 1
            return idx

        n_batches = self.total // global_bs

        for batch_idx in range(n_batches):
            cycle_idx = (batch_idx + self.epoch) % len(self.phase_cycle)
            target_group = self.phase_cycle[cycle_idx]
            n_true, n_false = self._compute_label_alloc_for_group(
                target_group,
                self.per_device_bs,
            )

            global_batch: list[int] = []
            global_batch_labels: list[bool] = []
            global_batch_groups: list[str] = []

            for rank_slot in range(world_size):
                local_batch: list[int] = []
                local_batch_labels: list[bool] = []
                local_batch_groups: list[str] = []

                for _ in range(n_true):
                    local_batch.append(next_idx(target_group, True))
                    local_batch_labels.append(True)
                    local_batch_groups.append(target_group)
                for _ in range(n_false):
                    local_batch.append(next_idx(target_group, False))
                    local_batch_labels.append(False)
                    local_batch_groups.append(target_group)

                if local_batch:
                    perm = rng.permutation(len(local_batch))
                    local_batch = [local_batch[i] for i in perm]
                    local_batch_labels = [local_batch_labels[i] for i in perm]
                    local_batch_groups = [local_batch_groups[i] for i in perm]

                global_batch.extend(local_batch)
                global_batch_labels.extend(local_batch_labels)
                global_batch_groups.extend(local_batch_groups)

            start = rank * self.per_device_bs
            end   = start + self.per_device_bs

            if self.debug_batch_ratio and batch_idx < self.debug_batch_ratio_steps:
                global_size = len(global_batch_labels)
                global_true = sum(global_batch_labels)
                global_false = global_size - global_true
                local_labels = global_batch_labels[start:end]
                local_groups = global_batch_groups[start:end]
                local_size = len(local_labels)
                local_true = sum(local_labels)
                local_false = local_size - local_true
                global_true_ratio = global_true / global_size if global_size else 0.0
                global_false_ratio = global_false / global_size if global_size else 0.0
                local_true_ratio = local_true / local_size if local_size else 0.0
                local_false_ratio = local_false / local_size if local_size else 0.0
                local_group_counts = Counter(local_groups)

                print(
                    "[DEBUG][batch_ratio] "
                    f"epoch={self.epoch} batch={batch_idx + 1}/{n_batches} "
                    f"target_group={self._display_group(target_group)} "
                    f"global(T/F)={global_true}/{global_false} "
                    f"global_ratio={global_true_ratio:.3f}/{global_false_ratio:.3f} "
                    f"local_rank{rank}(T/F)={local_true}/{local_false} "
                    f"local_ratio={local_true_ratio:.3f}/{local_false_ratio:.3f} "
                    f"local_groups={ {self._display_group(k): v for k, v in local_group_counts.items()} }",
                    flush=True,
                )

            yield global_batch[start:end]

    def __len__(self) -> int:
        world_size = _world_size()
        return self.total // (self.per_device_bs * world_size)


# ── Phase-Balanced KTO Trainer ─────────────────────────────────────────────────

class PhaseBalancedKTOTrainer(KTOTrainer):


    def __init__(
        self,
        *args,
        target_true_ratio: float = 0.5,
        phase_cycle_length: int = 16,
        debug_batch_ratio: bool = False,
        debug_batch_ratio_steps: int = 20,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.target_true_ratio = target_true_ratio
        self.phase_cycle_length = max(1, phase_cycle_length)
        self.debug_batch_ratio = debug_batch_ratio
        self.debug_batch_ratio_steps = debug_batch_ratio_steps

    def get_train_dataloader(self) -> DataLoader:
        dataset = self.train_dataset
        sampler = PhaseBalancedBatchSampler(
            dataset=dataset,
            per_device_batch_size=self.args.per_device_train_batch_size,
            seed=getattr(self.args, "seed", 42),
            target_true_ratio=self.target_true_ratio,
            phase_cycle_length=self.phase_cycle_length,
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

import torch.nn.functional as F


# ── KTO + Auxiliary Loss Trainer ───────────────────────────────────────────────

class KTOWithAuxTrainer(PhaseBalancedKTOTrainer):
    """
    total_loss = kto_loss + aux_loss_weight * aux_loss
    """

    def __init__(self, *args, aux_loss_weight: float = 0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.aux_loss_weight = aux_loss_weight

    def get_batch_loss_metrics(self, model, batch):
        phases = batch.get("phase", None)

        if phases is None:
            return super().get_batch_loss_metrics(model, batch)

        sr_mask = torch.tensor(
            [get_phase_group(p) == "state_recon" for p in phases],
            dtype=torch.bool,
        )
        kto_mask = ~sr_mask

        device = self.accelerator.device
        total_loss = torch.tensor(0.0, device=device)
        metrics: dict = {}

        # ── KTO loss ──────────────────────────────────
        if kto_mask.any():
            kto_batch = self._filter_batch(batch, kto_mask)
            kto_loss, kto_metrics = super().get_batch_loss_metrics(model, kto_batch)
            total_loss = total_loss + kto_loss
            metrics.update(kto_metrics)

        # ── Auxiliary CE loss ─────────────────────────
        if sr_mask.any():
            aux_loss = self._compute_aux_ce_loss(model, batch, sr_mask)
            total_loss = total_loss + self.aux_loss_weight * aux_loss
            metrics["aux/ce_loss"] = aux_loss.item()

        return total_loss, metrics

    def _filter_batch(self, batch: dict, mask: torch.Tensor) -> dict:

        indices = mask.nonzero(as_tuple=True)[0].tolist()
        filtered: dict = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                filtered[k] = v[mask]
            elif isinstance(v, list):
                filtered[k] = [v[i] for i in indices]
            else:
                filtered[k] = v
        return filtered

    def _compute_aux_ce_loss(
        self, model, batch: dict, sr_mask: torch.Tensor
    ) -> torch.Tensor:

        input_ids = batch["completion_input_ids"][sr_mask]
        attention_mask = batch["completion_attention_mask"][sr_mask]
        labels = batch["completion_labels"][sr_mask]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )


# ── Args ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--ref_model_path", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--desirable_weight", type=float, default=1.0)
    parser.add_argument("--undesirable_weight", type=float, default=0.7)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
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
    parser.add_argument(
        "--phase_cycle_length",
        type=int,
        default=16,
        help="Batch-phase ratio",
    )
    parser.add_argument("--debug_batch_ratio", action="store_true")
    parser.add_argument("--debug_batch_ratio_steps", type=int, default=20)
    parser.add_argument("--aux_dataset_path", type=str, default=None,
                        help="state_recon aux CE loss")
    parser.add_argument("--aux_loss_weight", type=float, default=0.2)
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "bfloat16", "float16", "float32"],
    )
    return parser.parse_args()


def resolve_torch_dtype(dtype_name: str):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(
        dtype_name, "auto"
    )


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
    """
    Returns:
      (use_bf16, use_fp16, effective_dtype_name)
    """
    if dtype_name == "auto":
        if is_bf16_gpu_supported():
            return True, False, "bfloat16"
        if torch.cuda.is_available():
            return False, True, "float16"
        return False, False, "float32"

    if dtype_name == "bfloat16" and not is_bf16_gpu_supported():
        if torch.cuda.is_available():
            print(
                "[WARN] bf16 is not supported on this GPU. Falling back to float16.",
                flush=True,
            )
            return False, True, "float16"
        print(
            "[WARN] bf16 requested but CUDA GPU is unavailable. Falling back to float32.",
            flush=True,
        )
        return False, False, "float32"

    if dtype_name == "float16" and not torch.cuda.is_available():
        print(
            "[WARN] float16 requested but CUDA GPU is unavailable. Falling back to float32.",
            flush=True,
        )
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


def _looks_like_message_list(value) -> bool:
    return (
        isinstance(value, list)
        and all(
            isinstance(item, dict)
            and "role" in item
            and "content" in item
            for item in value
        )
    )


def _rows_from_json_obj(obj) -> list[dict]:
    if isinstance(obj, list):
        return [row for row in obj if isinstance(row, dict)]

    if not isinstance(obj, dict):
        raise ValueError(f"Unsupported dataset type: {type(obj).__name__}")

    list_like_values = [
        value for value in obj.values()
        if isinstance(value, list) and not _looks_like_message_list(value)
    ]
    if not list_like_values:
        return [obj]

    row_count = max(len(value) for value in list_like_values)
    rows = []
    for idx in range(row_count):
        row = {}
        for key, value in obj.items():
            if isinstance(value, list) and not _looks_like_message_list(value):
                row[key] = value[idx] if idx < len(value) else None
            else:
                row[key] = value
        rows.append(row)
    return rows


def _normalize_chat_field(value, default_role: str) -> list[dict]:
    if _looks_like_message_list(value):
        messages = []
        for item in value:
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            role = str(item.get("role", "")).strip() or default_role
            messages.append({"role": role, "content": content})
        return messages

    if isinstance(value, list):
        messages = []
        for item in value:
            text = str(item).strip()
            if not text:
                continue
            messages.append({"role": default_role, "content": text})
        return messages

    if isinstance(value, dict):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value or "").strip()

    if not text:
        return []
    return [{"role": default_role, "content": text}]


def _coerce_bool_label(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes", "y", "desirable", "positive"}:
        return True
    if text in {"false", "0", "no", "n", "undesirable", "negative"}:
        return False
    return None


def _infer_phase(example: dict, force_phase: str | None = None) -> str:
    if force_phase:
        return force_phase

    phase = str(example.get("phase", "") or "").strip()
    if phase:
        return phase

    candidates = [
        example.get("sample_type"),
        example.get("type"),
        example.get("pair_source"),
        example.get("state_recon"),
    ]
    joined = " ".join(str(value or "") for value in candidates).lower()
    if "state_reconstruction" in joined or "state_recon" in joined:
        return "state_reconstruction"
    if "night_skill" in joined or "action" in joined:
        return "night_skill"
    if "vote" in joined:
        return "vote"
    if "speech" in joined:
        return "speech"
    return "other"


def _normalize_kto_record(example: dict, force_phase: str | None = None) -> dict | None:
    prompt = _normalize_chat_field(example.get("prompt"), default_role="user")
    completion = _normalize_chat_field(example.get("completion"), default_role="assistant")
    label = _coerce_bool_label(example.get("label"))

    if not prompt or not completion or label is None:
        return None

    return {
        "prompt": prompt,
        "completion": completion,
        "label": label,
        "phase": _infer_phase(example, force_phase=force_phase),
    }


def _load_kto_dataset(path: str, tokenizer, dataset_name: str, force_phase: str | None = None) -> Dataset:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{dataset_name} dataset file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    raw_rows = _rows_from_json_obj(obj)
    normalized_rows = []
    skipped_rows = 0
    for row in raw_rows:
        normalized = _normalize_kto_record(row, force_phase=force_phase)
        if normalized is None:
            skipped_rows += 1
            continue
        normalized_rows.append(normalized)

    if not normalized_rows:
        raise ValueError(f"No valid rows found in {dataset_name} dataset: {path}")

    phase_counts = Counter(row["phase"] for row in normalized_rows)
    label_counts = Counter(row["label"] for row in normalized_rows)
    print(
        f"[INFO] Loaded {dataset_name} dataset: total={len(raw_rows)} "
        f"valid={len(normalized_rows)} skipped={skipped_rows}",
        flush=True,
    )
    print(f"[INFO] {dataset_name} phase counts: {dict(phase_counts)}", flush=True)
    print(
        f"[INFO] {dataset_name} label counts: "
        f"{{True: {label_counts.get(True, 0)}, False: {label_counts.get(False, 0)}}}",
        flush=True,
    )

    return Dataset.from_list(normalized_rows).map(preprocess_builder(tokenizer))


def build_training_args(
    args,
    output_dir: str,
    use_bf16: bool,
    use_fp16: bool,
    num_train_epochs: int,
) -> KTOConfig:
    config_kwargs = {
        "output_dir": output_dir,
        "beta": args.beta,
        "desirable_weight": args.desirable_weight,
        "undesirable_weight": args.undesirable_weight,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": args.lr_scheduler_type,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": num_train_epochs,
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
    }
    supported_args = set(inspect.signature(KTOConfig).parameters)
    unsupported_args = sorted(set(config_kwargs) - supported_args)
    if unsupported_args:
        print(
            f"[WARN] Current trl.KTOConfig does not support: {unsupported_args}. "
            "Ignoring these arguments.",
            flush=True,
        )
    return KTOConfig(
        **{
            key: value
            for key, value in config_kwargs.items()
            if key in supported_args
        }
    )

# ── Aux dataset loader ────────────────────────────────────────────────────────

def _prepare_aux_dataset(path: str, tokenizer, dataset_split: str):
    del dataset_split
    return _load_kto_dataset(
        path=path,
        tokenizer=tokenizer,
        dataset_name="aux",
        force_phase="state_reconstruction",
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if not 0.0 <= args.target_true_ratio <= 1.0:
        raise ValueError("target_true_ratio")
    use_bf16, use_fp16, effective_dtype = resolve_precision_mode(args.torch_dtype)
    if effective_dtype != args.torch_dtype:
        print(
            f"[INFO] torch_dtype adjusted: {args.torch_dtype} -> {effective_dtype}",
            flush=True,
        )

    for label, path in [
        ("model_path", args.model_path),
        ("ref_model_path", args.ref_model_path or args.model_path),
    ]:
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"No such directory for {label}: {path}"
            )

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

    train_dataset = _load_kto_dataset(
        path=args.dataset_path,
        tokenizer=tokenizer,
        dataset_name="train",
    )

    if args.aux_dataset_path and os.path.isfile(args.aux_dataset_path):
        print(f"[INFO] Loading aux (state_recon) dataset: {args.aux_dataset_path}", flush=True)
        aux_dataset = _prepare_aux_dataset(args.aux_dataset_path, tokenizer, args.dataset_split)
        common_cols = [col for col in train_dataset.column_names if col in aux_dataset.column_names]
        train_dataset = concatenate_datasets([
            train_dataset.select_columns(common_cols),
            aux_dataset.select_columns(common_cols),
        ])
        print(
            f"[INFO] Dataset merged — KTO: {len(train_dataset) - len(aux_dataset)}, "
            f"Aux: {len(aux_dataset)}, Total: {len(train_dataset)}",
            flush=True,
        )
    elif args.aux_dataset_path:
        print(f"[WARN] aux_dataset_path does not exist; ignoring: {args.aux_dataset_path}", flush=True)

    print(f"[INFO] KTO training dataset size: {len(train_dataset)}", flush=True)
    training_args = build_training_args(
        args=args,
        output_dir=args.output_dir,
        use_bf16=use_bf16,
        use_fp16=use_fp16,
        num_train_epochs=args.num_train_epochs,
    )
    trainer = KTOWithAuxTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        target_true_ratio=args.target_true_ratio,
        phase_cycle_length=args.phase_cycle_length,
        debug_batch_ratio=args.debug_batch_ratio,
        debug_batch_ratio_steps=args.debug_batch_ratio_steps,
        aux_loss_weight=args.aux_loss_weight,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    print(
        "[INFO] Single-pass KTO training completed. "
        "SFT iteration/rollout is handled only by the outer orchestration script.",
        flush=True,
    )


if __name__ == "__main__":
    main()
