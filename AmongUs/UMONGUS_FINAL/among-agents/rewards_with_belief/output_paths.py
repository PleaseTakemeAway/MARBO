import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


_SLUG_SPLIT_RE = re.compile(r"[^a-z0-9]+")


def normalize_include_model_filters(include_models: Optional[Iterable[str]]) -> Tuple[str, ...]:
    if not include_models:
        return ()

    filters: List[str] = []
    for raw_model in include_models:
        filters.extend(model.strip().lower() for model in str(raw_model).split(",") if model.strip())
    return tuple(filters)


def _model_family(model_filter: str) -> Optional[str]:
    candidate = model_filter.strip().lower().rsplit("/", 1)[-1]
    match = re.match(r"[a-z]+", candidate)
    if match:
        return match.group(0)

    tokens = [token for token in _SLUG_SPLIT_RE.split(candidate) if token]
    return tokens[0] if tokens else None


def include_model_family_slug(include_models: Optional[Iterable[str]]) -> Optional[str]:
    families: List[str] = []
    for model_filter in normalize_include_model_filters(include_models):
        family = _model_family(model_filter)
        if family and family not in families:
            families.append(family)

    if not families:
        return None
    return families[0] if len(families) == 1 else "_".join(sorted(families))


def _path_mentions_slug(path: Path, slug: str) -> bool:
    for part in path.parts:
        part_lower = part.lower()
        if part_lower == slug:
            return True
        tokens = [token for token in _SLUG_SPLIT_RE.split(part_lower) if token]
        if slug in tokens:
            return True
    return False


def add_include_model_scope_to_output_root(output_root: Path, include_models: Optional[Iterable[str]]) -> Path:
    slug = include_model_family_slug(include_models)
    if not slug or _path_mentions_slug(output_root, slug):
        return output_root
    return output_root / slug


def add_include_model_scope_to_dataset_dir(output_dir: Path, include_models: Optional[Iterable[str]]) -> Path:
    slug = include_model_family_slug(include_models)
    if not slug or _path_mentions_slug(output_dir, slug):
        return output_dir
    return output_dir.parent / slug / output_dir.name
