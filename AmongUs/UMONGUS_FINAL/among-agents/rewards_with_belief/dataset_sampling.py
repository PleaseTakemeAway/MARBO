import math
import random
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar

from datasets import Dataset


T = TypeVar("T")
Bucket = Tuple[Any, ...]


def normalize_bucket_key(key: Any) -> Bucket:
    if isinstance(key, tuple):
        return key
    if isinstance(key, list):
        return tuple(key)
    return (key,)


def normalize_model_filter_values(include_models: Optional[Iterable[str]]) -> Tuple[str, ...]:
    if not include_models:
        return ()
    filters: List[str] = []
    for raw_model in include_models:
        filters.extend(model.strip().lower() for model in str(raw_model).split(",") if model.strip())
    return tuple(filters)


def resolve_model_ratios(model_ratios: Any, include_models: Optional[Iterable[str]]) -> Dict[str, float]:
    if not model_ratios:
        return {}

    filters = normalize_model_filter_values(include_models)
    if isinstance(model_ratios, str):
        if model_ratios.lower() not in {"equal", "equal_include_models", "include_models"}:
            raise ValueError(f"Unsupported model ratio mode: {model_ratios}")
        return {model_filter: 1.0 for model_filter in filters}

    ratios = {}
    for raw_model, raw_weight in dict(model_ratios).items():
        weight = float(raw_weight)
        if weight > 0:
            ratios[str(raw_model).strip().lower()] = weight
    if filters:
        matched = {
            model_key: weight
            for model_key, weight in ratios.items()
            if any(model_key in model_filter or model_filter in model_key for model_filter in filters)
        }
        if matched:
            return matched
        return {model_filter: 1.0 for model_filter in filters}
    return ratios


def model_ratio_key(model: str, model_ratios: Dict[str, float]) -> Optional[str]:
    if not model_ratios:
        return None
    normalized_model = str(model or "").strip().lower()
    matches = [key for key in model_ratios if key in normalized_model]
    if not matches:
        return None
    return max(matches, key=len)


def add_model_ratios_to_weights(weights: Dict[Any, float], model_ratios: Dict[str, float]) -> Dict[Bucket, float]:
    normalized_weights = {
        normalize_bucket_key(key): float(weight)
        for key, weight in weights.items()
        if float(weight) > 0
    }
    if not model_ratios:
        return normalized_weights

    combined: Dict[Bucket, float] = {}
    model_total = sum(float(weight) for weight in model_ratios.values() if float(weight) > 0)
    if model_total <= 0:
        return normalized_weights

    for bucket, bucket_weight in normalized_weights.items():
        for model_key, model_weight in model_ratios.items():
            if float(model_weight) <= 0:
                continue
            combined[(*bucket, model_key)] = bucket_weight * float(model_weight) / model_total
    return combined


def append_model_ratio_key(base_key: Any, model: str, model_ratios: Dict[str, float]) -> Bucket:
    key = normalize_bucket_key(base_key)
    if not model_ratios:
        return key
    matched_model = model_ratio_key(model, model_ratios)
    if matched_model is None:
        return (*key, "__unmatched_model__")
    return (*key, matched_model)


def allocate_counts(total: int, weights: Dict[Bucket, float]) -> Dict[Bucket, int]:
    clean = {key: float(weight) for key, weight in weights.items() if float(weight) > 0}
    if total <= 0 or not clean:
        return {}

    weight_sum = sum(clean.values())
    raw = {key: total * weight / weight_sum for key, weight in clean.items()}
    counts = {key: math.floor(value) for key, value in raw.items()}
    remainder = total - sum(counts.values())
    fractional_order = sorted(
        raw,
        key=lambda key: (raw[key] - counts[key], clean[key]),
        reverse=True,
    )
    for key in fractional_order[:remainder]:
        counts[key] += 1
    return counts


def infer_feasible_total(available: Counter, weights: Dict[Bucket, float]) -> int:
    feasible = {
        key: weight
        for key, weight in weights.items()
        if weight > 0 and available.get(key, 0) > 0
    }
    if not feasible:
        return 0

    weight_sum = sum(feasible.values())
    limits = [
        math.floor(available[key] / (weight / weight_sum))
        for key, weight in feasible.items()
    ]
    return max(0, min(limits))


def stratified_sample_records(
    records: Sequence[T],
    key_fn: Callable[[T], Any],
    weights: Dict[Any, float],
    total_size: Optional[int],
    seed: int,
    allow_oversample: bool,
    fill_shortage: bool,
) -> Tuple[List[T], Dict[str, Any]]:
    normalized_weights = {
        normalize_bucket_key(key): float(weight)
        for key, weight in weights.items()
        if float(weight) > 0
    }

    buckets: Dict[Bucket, List[T]] = defaultdict(list)
    for record in records:
        key = normalize_bucket_key(key_fn(record))
        if key in normalized_weights:
            buckets[key].append(record)

    rng = random.Random(seed)
    for bucket_records in buckets.values():
        rng.shuffle(bucket_records)

    available = Counter({key: len(value) for key, value in buckets.items()})
    if total_size is None:
        total_size = len(records) if allow_oversample else infer_feasible_total(available, normalized_weights)

    requested = allocate_counts(total_size, normalized_weights)
    selected_by_bucket: Dict[Bucket, List[T]] = defaultdict(list)
    selected: List[T] = []
    shortage = 0

    for key, requested_count in requested.items():
        pool = buckets.get(key, [])
        if allow_oversample:
            if not pool:
                chosen: List[T] = []
            elif requested_count <= len(pool):
                chosen = pool[:requested_count]
            else:
                chosen = list(pool) + [rng.choice(pool) for _ in range(requested_count - len(pool))]
        else:
            chosen = pool[: min(requested_count, len(pool))]

        selected_by_bucket[key].extend(chosen)
        selected.extend(chosen)
        shortage += max(0, requested_count - len(chosen))

    if fill_shortage and shortage > 0 and not allow_oversample:
        fill_keys = sorted(requested, key=lambda key: requested[key], reverse=True)
        while shortage > 0:
            added = False
            for key in fill_keys:
                pool = buckets.get(key, [])
                used = len(selected_by_bucket[key])
                if used >= len(pool):
                    continue
                item = pool[used]
                selected_by_bucket[key].append(item)
                selected.append(item)
                shortage -= 1
                added = True
                if shortage == 0:
                    break
            if not added:
                break

    selected_counts = Counter({key: len(value) for key, value in selected_by_bucket.items()})
    report = {
        "requested_total": sum(requested.values()),
        "selected_total": len(selected),
        "available_by_bucket": dict(available),
        "requested_by_bucket": dict(requested),
        "selected_by_bucket": dict(selected_counts),
        "missing_by_bucket": {
            key: max(0, requested.get(key, 0) - selected_counts.get(key, 0))
            for key in requested
        },
    }
    return selected, report


def sample_dataset(
    dataset: Dataset,
    key_fn: Callable[[Dict[str, Any]], Any],
    weights: Dict[Any, float],
    total_size: Optional[int],
    seed: int,
    allow_oversample: bool,
    fill_shortage: bool,
) -> Tuple[Dataset, Dict[str, Any]]:
    indices = list(range(len(dataset)))
    selected_indices, report = stratified_sample_records(
        indices,
        key_fn=lambda idx: key_fn(dataset[int(idx)]),
        weights=weights,
        total_size=total_size,
        seed=seed,
        allow_oversample=allow_oversample,
        fill_shortage=fill_shortage,
    )

    if len(set(selected_indices)) == len(selected_indices):
        return dataset.select(selected_indices), report
    return Dataset.from_list([dataset[int(idx)] for idx in selected_indices]), report


def _normalise_role_ratios(role_ratios: Any, label: bool) -> Dict[str, float]:
    if not role_ratios:
        return {}

    raw = dict(role_ratios)
    if label in raw or str(label) in raw:
        raw = dict(raw.get(label, raw.get(str(label), {})))

    ratios: Dict[str, float] = {}
    for role, weight in raw.items():
        role_name = str(role).strip().lower()
        if role_name in {"crewmate", "crew"}:
            key = "Crewmate"
        elif role_name in {"impostor", "imposter", "imp"}:
            key = "Impostor"
        else:
            key = str(role).strip()
        value = float(weight)
        if key and value > 0:
            ratios[key] = value
    return ratios


def flatten_kto_weights(
    label_ratios: Dict[bool, float],
    detail_ratios: Dict[bool, Dict[str, float]],
    role_ratios: Any = None,
) -> Dict[Bucket, float]:
    weights: Dict[Bucket, float] = {}
    for label, label_weight in label_ratios.items():
        details = detail_ratios.get(label, {})
        detail_total = sum(float(value) for value in details.values() if float(value) > 0)
        if float(label_weight) <= 0 or detail_total <= 0:
            continue
        roles = _normalise_role_ratios(role_ratios, bool(label))
        role_total = sum(float(value) for value in roles.values() if float(value) > 0)
        for bucket_name, detail_weight in details.items():
            if float(detail_weight) <= 0:
                continue
            base_weight = float(label_weight) * float(detail_weight) / detail_total
            if role_total > 0:
                for role, role_weight in roles.items():
                    weights[(bucket_name, bool(label), role)] = base_weight * float(role_weight) / role_total
            else:
                weights[(bucket_name, bool(label))] = base_weight
    return weights


def stringify_report(report: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: {
            "|".join(str(part) for part in bucket): value
            for bucket, value in bucket_counts.items()
        }
        if isinstance(bucket_counts, dict)
        else bucket_counts
        for key, bucket_counts in report.items()
    }
