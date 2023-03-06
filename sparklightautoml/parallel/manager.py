from abc import ABC, abstractmethod
from enum import Enum
from typing import Sequence, TypeVar, List

from mypy.typeshed.stdlib.multiprocessing.pool import ThreadPool
from mypy.typeshed.stdlib.typing import Callable, Any, Optional
from pyspark import inheritable_thread_target

__parallel_computations_manager__: Optional['ParallelComputationsManager'] = None

T = TypeVar("T")


class PoolType(Enum):
    ML_PIPELINES = "ML_PIPELINES"
    ML_ALGOS = "ML_ALGOS"
    DEFAULT = "DEFAULT"


class ParallelComputationsManager(ABC):
    @abstractmethod
    def parallel(self, tasks: list[Callable[T]], pool_type: PoolType) -> List[T]:
        ...


class DefaultParallelComputationsManager(ParallelComputationsManager):
    def get_pool(self, pool_type: str) -> ThreadPool:
        pool = ThreadPool(processes=min(1, 2))
        return pool

    def parallel(self, tasks: List[Callable[[], T]], pool_type: str) -> List[T]:
        pool = self.get_pool(pool_type)

        ptasks = map(inheritable_thread_target, tasks)
        results = sorted(
            (result for result in pool.imap_unordered(lambda f: f(), ptasks) if result),
            key=lambda x: x[0]
        )

        return results


def parallel_computations_manager() -> ParallelComputationsManager:
    global __parallel_computations_manager__
    if not __parallel_computations_manager__:
        __parallel_computations_manager__ = DefaultParallelComputationsManager()

    return __parallel_computations_manager__


def compute_parallel(tasks: List[Callable[[], T]], pool_type: PoolType = PoolType.DEFAULT) -> List[T]:
    return parallel_computations_manager().parallel(tasks, pool_type)
