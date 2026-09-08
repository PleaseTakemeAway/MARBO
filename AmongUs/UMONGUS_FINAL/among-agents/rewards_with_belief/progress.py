from typing import Any, Iterable, Optional


class _NullProgress:
    def __init__(self, iterable: Optional[Iterable[Any]] = None, **_: Any) -> None:
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable or ())

    def __enter__(self):
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def update(self, _: int = 1) -> None:
        return None


try:
    from tqdm.auto import tqdm
except ImportError:

    def iter_progress(iterable: Iterable[Any], **kwargs: Any):
        return _NullProgress(iterable, **kwargs)

    def progress_bar(**kwargs: Any):
        return _NullProgress(**kwargs)

else:

    def iter_progress(iterable: Iterable[Any], **kwargs: Any):
        return tqdm(iterable, **kwargs)

    def progress_bar(**kwargs: Any):
        return tqdm(**kwargs)
