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


def call_once(init_f: Callable) -> Callable:
    """
    Decorator, when applied to a function, will result in that function
    only being able to be called once.
    Any calls after the first one uses the value cached from the first call.
    """
    obj = None

    def _try_call(*args, **kwargs):
        nonlocal obj

        if obj is not None:
            print('using cached')
            return obj

        print('calling init')
        obj = init_f(*args, **kwargs)

        return obj

    return _try_call
