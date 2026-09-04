from typing import Any, Iterable, Tuple


def _label_is_true(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _parse_weight(value: Any) -> float | None:
    if isinstance(value, str) and value.strip().lower() == "auto":
        return None
    return float(value)


def resolve_kto_loss_weights(
    desirable_weight: Any,
    undesirable_weight: Any,
    labels: Iterable[Any],
    *,
    rank: int = 0,
) -> Tuple[float, float]:
    """Resolve KTO loss weights, accepting "auto" for one or both values.

    TRL's warning is based on Eq. (8) from the KTO paper. When both values are
    auto, keep the minority class at 1.0 and downweight the majority class.
    """

    desired = _parse_weight(desirable_weight)
    undesired = _parse_weight(undesirable_weight)
    label_list = list(labels)
    num_desirable = max(sum(_label_is_true(label) for label in label_list), 1)
    num_undesirable = max(len(label_list) - num_desirable, 1)

    if desired is None and undesired is None:
        if num_desirable >= num_undesirable:
            desired = round(num_undesirable / num_desirable, 2)
            undesired = 1.0
        else:
            desired = 1.0
            undesired = round(num_desirable / num_undesirable, 2)
    elif desired is None:
        desired = round(num_undesirable * float(undesired) / num_desirable, 2)
    elif undesired is None:
        undesired = round(num_desirable * float(desired) / num_undesirable, 2)

    desired = float(desired)
    undesired = float(undesired)

    if rank == 0:
        des_lower = num_undesirable * undesired / num_desirable
        des_upper = des_lower * 1.33
        und_upper = num_desirable * desired / num_undesirable
        und_lower = und_upper / 1.33
        print(
            "[data] KTO loss weights "
            f"desirable={desired:.6g} undesirable={undesired:.6g} "
            f"(labels True={num_desirable} False={num_undesirable}; "
            f"recommended desirable=[{des_lower:.3g}, {des_upper:.3g}] "
            f"or undesirable=[{und_lower:.3g}, {und_upper:.3g}])",
            flush=True,
        )

    return desired, undesired
