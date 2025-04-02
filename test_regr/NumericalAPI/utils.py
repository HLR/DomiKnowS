from typing import TypeVar, Iterable, Optional, Callable

sample_type = TypeVar("sample_type")
def batch_iterator(
        data: Iterable[sample_type],
        batch_size: int,
        filter_f: Optional[Callable[[sample_type], bool]] = None
    ) -> Iterable[list[sample_type]]:

    batch = []
    for item in data:
        if filter_f is not None and not filter_f(item):
            continue

        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []

    if len(batch) > 0:
        yield batch
