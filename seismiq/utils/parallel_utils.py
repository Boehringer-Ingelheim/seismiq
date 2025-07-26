from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable, Iterable, Iterator, Optional, TypeVar

from tqdm import tqdm

T = TypeVar("T")
U = TypeVar("U")


def _make_bar(it: Iterable[T], enable: bool, description: Optional[str] = None, **kwargs: Any) -> Iterable[T]:
    if not enable:
        return it
    bar = tqdm(it, **kwargs)
    if description:
        bar.set_description(description)
    return bar


def parallel_threads(
    items: Iterable[T],
    worker_function: Callable[[list[T]], U],
    job_size: int,
    n_jobs: int,
    progress_bar: bool = True,
) -> Iterator[U]:
    """Processes all items in parallel threads using the given worker function.

    Args:
        items (Iterable[T]): The items to be processed.
        worker_function (Callable[[list[T]], U]): The worker function.
        job_size (int): Size of the batches that are passed to the workers.
        n_jobs (int): Maximum number of parallel threads running concurrently.

    Yields:
        Iterator[U]: The result of processing each batch of items.
    """

    if n_jobs > 1:
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            job_batches = batches(_make_bar(items, progress_bar, ncols=0, description="Queued batches"), job_size)
            yield from executor.map(worker_function, job_batches)
    else:
        for batch in _make_bar(batches(items, job_size), progress_bar, ncols=0, description="Queued batches"):
            res = worker_function(batch)
            yield res


def parallel_processes(
    items: Iterable[T],
    worker_function: Callable[[list[T]], U],
    job_size: int,
    n_jobs: int,
    progress_bar: bool = True,
) -> Iterator[U]:
    """Processes all items in parallel processes using the given worker function.

    Args:
        items (Iterable[T]): The items to be processed.
        worker_function (Callable[[list[T]], U]): The worker function.
        job_size (int): Size of the batches that are passed to the workers.
        n_jobs (int): Maximum number of parallel threads running concurrently.

    Yields:
        Iterator[U]: The result of processing each batch of items.
    """

    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs, max_tasks_per_child=10000) as executor:
            job_batches = batches(_make_bar(items, progress_bar, ncols=0, description="Queued batches"), job_size)
            yield from executor.map(worker_function, job_batches)
    else:
        for batch in _make_bar(batches(items, job_size), progress_bar, ncols=0, description="Queued batches"):
            res = worker_function(batch)
            yield res


def batches(iter: Iterable[T], batch_size: int) -> Iterator[list[T]]:
    batch = []
    for x in iter:
        batch.append(x)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
