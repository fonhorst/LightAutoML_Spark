import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import TypeVar, List

from mypy.typeshed.stdlib.multiprocessing.pool import ThreadPool
from mypy.typeshed.stdlib.typing import Callable, Optional
from pyspark import inheritable_thread_target

ENV_VAR_SLAMA_COMPUTATIONS_MANAGER = "SLAMA_COMPUTATIONS_MANAGER"

__computations_manager__: Optional['ComputationsManager'] = None

T = TypeVar("T")


class PoolType(Enum):
    ML_PIPELINES = "ML_PIPELINES"
    ML_ALGOS = "ML_ALGOS"
    DEFAULT = "DEFAULT"


class ComputationsManager(ABC):
    @abstractmethod
    def compute(self, tasks: list[Callable[T]], pool_type: PoolType) -> List[T]:
        ...


class ParallelComputationsManager(ComputationsManager):
    def get_pool(self, pool_type: str) -> ThreadPool:
        pool = ThreadPool(processes=min(1, 2))
        return pool

    def compute(self, tasks: List[Callable[[], T]], pool_type: str) -> List[T]:
        pool = self.get_pool(pool_type)

        ptasks = map(inheritable_thread_target, tasks)
        results = sorted(
            (result for result in pool.imap_unordered(lambda f: f(), ptasks) if result),
            key=lambda x: x[0]
        )

        return results


class SequentialComputationsManager(ComputationsManager):
    def compute(self, tasks: list[Callable[T]], pool_type: PoolType) -> List[T]:
        return [task() for task in tasks]


def computations_manager() -> ComputationsManager:
    global __computations_manager__
    if not __computations_manager__:
        comp_manager_type = os.environ.get(ENV_VAR_SLAMA_COMPUTATIONS_MANAGER, "sequential")
        if comp_manager_type == "parallel":
            comp_manager = ParallelComputationsManager()
        elif comp_manager_type == "sequential":
            comp_manager = SequentialComputationsManager()
        else:
            raise Exception(f"Incorrect computations manager type: {comp_manager_type}. "
                            f"Supported values: [parallel, sequential]")
        __computations_manager__ = comp_manager

    return __computations_manager__


def compute_tasks(tasks: List[Callable[[], T]], pool_type: PoolType = PoolType.DEFAULT) -> List[T]:
    return computations_manager().compute(tasks, pool_type)
