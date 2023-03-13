import logging
import os
import threading
from abc import ABC, abstractmethod
from concurrent.futures.thread import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar, List, Iterator

from mypy.typeshed.stdlib.multiprocessing.pool import ThreadPool
from mypy.typeshed.stdlib.typing import Callable, Optional
from pyspark import inheritable_thread_target

from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.utils import SparkDataFrame

logger = logging.getLogger(__name__)

ENV_VAR_SLAMA_COMPUTATIONS_MANAGER = "SLAMA_COMPUTATIONS_MANAGER"

__computations_manager__: Optional['ComputationsManager'] = None

T = TypeVar("T")
S = TypeVar("S", bound='Slot')


def _compute_sequential(tasks: List[Callable[[], T]]) -> List[T]:
    return [task() for task in tasks]


class Slot(ABC):
    ...


@dataclass
class LGBMDatasetSlot(Slot):
    dataset: SparkDataset
    num_tasks: int
    num_threads: int
    use_barrier_execution_mode: bool
    free: bool



class PoolType(Enum):
    ML_PIPELINES = "ML_PIPELINES"
    ML_ALGOS = "ML_ALGOS"
    DEFAULT = "DEFAULT"


class ComputationsManager(ABC):
    @abstractmethod
    def compute(self, tasks: List[Callable[T]], pool_type: PoolType) -> List[T]:
        ...

    @abstractmethod
    def compute_with_slots(self, name: str, slots: List[S],
                           tasks: List[Callable[[S], T]], pool_type: PoolType) -> List[T]:
        ...


class ParallelComputationsManager(ComputationsManager):
    def __init__(self, ml_pipes_pool_size: int = 10, ml_algos_pool_size: int = 20, default_pool_size: int = 10):
        self._pools ={
            PoolType.ML_PIPELINES: ThreadPool(processes=ml_pipes_pool_size) if ml_pipes_pool_size > 0 else None,
            PoolType.ML_ALGOS: ThreadPool(processes=ml_algos_pool_size) if ml_algos_pool_size > 0 else None,
            PoolType.DEFAULT: ThreadPool(processes=default_pool_size) if default_pool_size > 0 else None
        }

        self._pools_lock = threading.Lock()

    def _get_pool(self, pool_type: PoolType) -> Optional[ThreadPool]:
        return self._pools.get(pool_type, None)

    def compute(self, tasks: List[Callable[[], T]], pool_type: PoolType) -> List[T]:
        # TODO: wouldn't be it a problem
        pool = self._get_pool(pool_type)

        if not pool:
            return _compute_sequential(tasks)

        with self._pools_lock:
            ptasks = map(inheritable_thread_target, tasks)
            results = sorted(
                (result for result in pool.imap_unordered(lambda f: f(), ptasks) if result),
                key=lambda x: x[0]
            )

        return results

    def compute_with_slots(self, name: str, slots: List[S],
                           tasks: List[Callable[[S], T]], pool_type: PoolType) -> List[T]:
        with self._block_all_pools(pool_type):
            # check pools
            assert all((pool is None) for _pool_type, pool in self._pools.items() if _pool_type != pool_type), \
                f"All thread pools except {pool_type} should be None"
            pool = self._get_pool(pool_type)
            # TODO: check the pool is empty or check threads by name?
            slots_lock = threading.Lock()

            @contextmanager
            def _use_train() -> Iterator[S]:
                with slots_lock:
                    free_slot = next((slot for slot in slots if slot.free))
                    free_slot.free = False

                yield free_slot

                with slots_lock:
                    free_slot.free = True

            def _func(task):
                with _use_train() as slot:
                    return task(slot)

            f_tasks = [_func(task) for task in tasks]

            return self.compute(f_tasks, pool_type.DEFAULT)

    @contextmanager
    def _block_all_pools(self, pool_type: PoolType):
        # need to block pool
        with self._pools_lock:
            yield self._get_pool(pool_type)


class SequentialComputationsManager(ComputationsManager):
    def compute_with_slots(self, name: str, prepare_slots: Callable[List[S]], tasks: List[Callable[[S], T]],
                           pool_type: PoolType) -> List[T]:
        raise NotImplementedError()

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
