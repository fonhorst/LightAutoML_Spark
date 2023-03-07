import logging
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import TypeVar, List

from mypy.typeshed.stdlib.multiprocessing.pool import ThreadPool
from mypy.typeshed.stdlib.typing import Callable, Optional
from pyspark import inheritable_thread_target

logger = logging.getLogger(__name__)

ENV_VAR_SLAMA_COMPUTATIONS_MANAGER = "SLAMA_COMPUTATIONS_MANAGER"

__computations_manager__: Optional['ComputationsManager'] = None

T = TypeVar("T")


def _compute_sequential(tasks: List[Callable[[], T]]) -> List[T]:
    return [task() for task in tasks]


class PoolType(Enum):
    ML_PIPELINES = "ML_PIPELINES"
    ML_ALGOS = "ML_ALGOS"
    DEFAULT = "DEFAULT"


class ComputationsManager(ABC):
    @abstractmethod
    def compute(self, tasks: List[Callable[T]], pool_type: PoolType) -> List[T]:
        ...

    @abstractmethod
    def compute_exclusively(self, tasks: List[Callable[T]]) -> List[T]:
        ...


class ParallelComputationsManager(ComputationsManager):
    def __init__(self, ml_pipes_pool_size: int, ml_algos_pool_size: int, default_pool_size: int):
        self._ml_pipes_pool: Optional[ThreadPool] = \
            ThreadPool(processes=ml_pipes_pool_size) if ml_pipes_pool_size > 0 else None
        self._ml_algos_pool: Optional[ThreadPool] = \
            ThreadPool(processes=ml_algos_pool_size) if ml_algos_pool_size > 0 else None
        self._default_pool: Optional[ThreadPool] = \
            ThreadPool(processes=default_pool_size) if default_pool_size > 0 else None

    def get_pool(self, pool_type: PoolType) -> Optional[ThreadPool]:
        if pool_type == pool_type.ML_PIPELINES:
            return self._ml_pipes_pool
        elif pool_type == pool_type.ML_ALGOS:
            return self._ml_algos_pool
        else:
            return self._default_pool

    def compute(self, tasks: List[Callable[[], T]], pool_type: PoolType) -> List[T]:
        pool = self.get_pool(pool_type)

        if not pool:
            return _compute_sequential(tasks)

        ptasks = map(inheritable_thread_target, tasks)
        results = sorted(
            (result for result in pool.imap_unordered(lambda f: f(), ptasks) if result),
            key=lambda x: x[0]
        )

        return results


class SequentialComputationsManager(ComputationsManager):
    def compute(self, tasks: list[Callable[T]], pool_type: PoolType) -> List[T]:
        return _compute_sequential(tasks)


def computations_manager() -> ComputationsManager:
    global __computations_manager__
    if not __computations_manager__:
        comp_manager_type = os.environ.get(ENV_VAR_SLAMA_COMPUTATIONS_MANAGER, "sequential")
        logger.info(f"Initializing computations manager with type and params: {comp_manager_type}")
        if comp_manager_type.startswith("parallel"):
            comp_manager_args = comp_manager_type[comp_manager_type.index('[') + 1: comp_manager_type.index(']')]
            comp_manager_args = [int(arg.strip()) for arg in comp_manager_args.split(',')]
            comp_manager = ParallelComputationsManager(*comp_manager_args)
        elif comp_manager_type.startswith("sequential"):
            comp_manager = SequentialComputationsManager()
        else:
            raise Exception(f"Incorrect computations manager type: {comp_manager_type}. "
                            f"Supported values: [parallel, sequential]")
        __computations_manager__ = comp_manager

    return __computations_manager__


def compute_tasks(tasks: List[Callable[[], T]], pool_type: PoolType = PoolType.DEFAULT) -> List[T]:
    return computations_manager().compute(tasks, pool_type)


def compute_tasks_exclusively(tasks: List[Callable[[], T]]) -> List[T]:
    raise NotImplementedError()
