import logging
import os
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from multiprocessing.pool import ThreadPool
from typing import Callable, Optional, Iterator, Iterable, ContextManager
from typing import TypeVar, List, Iterator

from pyspark import inheritable_thread_target

from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.validation.base import SparkBaseTrainValidIterator, TrainVal

logger = logging.getLogger(__name__)

ENV_VAR_SLAMA_COMPUTATIONS_MANAGER = "SLAMA_COMPUTATIONS_MANAGER"

__computations_manager__: Optional['ComputationsManager'] = None

T = TypeVar("T")
S = TypeVar("S", bound='Slot')


def _compute_sequential(tasks: List[Callable[[], T]]) -> List[T]:
    return [task() for task in tasks]


def build_named_parallelism_settings(config_name: str, parallelism: int):
    parallelism_config = {
        "no_parallelism": {
            "feature_selector_parallelism": 1,
            "mlpipes_parallelism": 1,
            "mlalgos_parallelism": 1,
            "linear_l2": {},
            "lgb": {},
            "lgb_tuned": {}
        },
        "intra_mlpipe_parallelism": {
            "feature_selector_parallelism": parallelism,
            "mlpipes_parallelism": 1,
            "mlalgos_parallelism": 1,
            "linear_l2": {
                "coeff_opt_parallelism": 1
            },
            "lgb": {
                "folds_parallelism": parallelism,
                "use_barrier_execution_mode": True,
                "use_slot_based_parallelism": False
            },
            "lgb_tuned": {
                "optimization_parallelism": parallelism,
                "folds_parallelism": parallelism,
                "use_barrier_execution_mode": True,
                "use_slot_based_parallelism": False
            }
        },
        "intra_mlpipe_parallelism_with_experimental_features": {
            "feature_selector_parallelism": parallelism,
            "mlpipes_parallelism": 1,
            "mlalgos_parallelism": 1,
            "linear_l2": {
                "coeff_opt_parallelism": 1
            },
            "lgb": {
                "folds_parallelism": parallelism,
                "use_barrier_execution_mode": True,
                "use_slot_based_parallelism": True
            },
            "lgb_tuned": {
                "optimization_parallelism": parallelism,
                "folds_parallelism": parallelism,
                "use_barrier_execution_mode": True,
                "use_slot_based_parallelism": True
            }
        },
        "mlpipe_level_parallelism": {
            "feature_selector_parallelism": 1,
            "mlpipes_parallelism": parallelism,
            "mlalgos_parallelism": 1,
            "linear_l2": {},
            "lgb": {},
            "lgb_tuned": {}

        },
        "all_levels_parallelism_with_experimental_features": {
            "feature_selector_parallelism": parallelism,
            "mlpipes_parallelism": parallelism,
            "mlalgos_parallelism": parallelism,
            "linear_l2": {
                "coeff_opt_parallelism": parallelism
            },
            "lgb": {
                "folds_parallelism": parallelism,
                "use_barrier_execution_mode": True,
                "use_slot_based_parallelism": False
            },
            "lgb_tuned": {
                "optimization_parallelism": parallelism,
                "folds_parallelism": parallelism,
                "use_barrier_execution_mode": True,
                "use_slot_based_parallelism": False
            }
        },
        "all_levels_parallelism": {
            "feature_selector_parallelism": parallelism,
            "mlpipes_parallelism": parallelism,
            "mlalgos_parallelism": parallelism,
            "linear_l2": {
                "coeff_opt_parallelism": parallelism
            },
            "lgb": {
                "folds_parallelism": parallelism,
                "use_barrier_execution_mode": True,
                "use_slot_based_parallelism": False
            },
            "lgb_tuned": {
                "optimization_parallelism": parallelism,
                "folds_parallelism": parallelism,
                "use_barrier_execution_mode": True,
                "use_slot_based_parallelism": False
            }
        }
    }

    return parallelism_config[config_name]



@dataclass
class DatasetSlot:
    dataset: SparkDataset
    free: bool


@dataclass
class SlotSize:
    num_tasks: int
    num_threads_per_executor: int


class PoolType(Enum):
    ML_PIPELINES = "ML_PIPELINES"
    ML_ALGOS = "ML_ALGOS"
    DEFAULT = "DEFAULT"


class SlotAllocator(ABC):
    @abstractmethod
    @contextmanager
    def allocate(self) -> DatasetSlot:
        ...

    @property
    @abstractmethod
    def slot_size(self) -> SlotSize:
        ...


class ParallelSlotAllocator(SlotAllocator):
    def __init__(self, slot_size: SlotSize, slots: List[DatasetSlot], pool: ThreadPool):
        self._slot_size = slot_size
        self._slots = slots
        self._pool = pool
        self._slots_lock = threading.Lock()

    @property
    def slot_size(self) -> SlotSize:
        return self._slot_size

    @contextmanager
    def allocate(self) -> DatasetSlot:
        with self._slots_lock:
            free_slot = next((slot for slot in self._slots if slot.free))
            free_slot.free = False

        yield free_slot

        with self._slots_lock:
            free_slot.free = True


class ComputationsManager(ABC):
    @abstractmethod
    def compute(self, tasks: List[Callable[[], T]], pool_type: PoolType) -> List[T]:
        ...

    @abstractmethod
    def compute_with_slots(self, name: str, slots: List[S],
                           tasks: List[Callable[[S], T]], pool_type: PoolType) -> List[T]:
        ...

    @contextmanager
    @abstractmethod
    def slots(self, dataset: SparkDataset, parallelism: int, pool_type: PoolType) -> SlotAllocator:
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
    def slots(self, dataset: SparkDataset, parallelism: int, pool_type: PoolType) -> SlotAllocator:
        with self._block_all_pools(pool_type):
            assert all((pool is None) for _pool_type, pool in self._pools.items() if _pool_type != pool_type), \
                f"All thread pools except {pool_type} should be None"

            pool = self._get_pool(pool_type)
            # TODO: parallel
            slot_size = SlotSize(num_tasks=, num_threads_per_executor=)
            slots = self._prepare_trains(paralellism_mode=, train_df=dataset.data, max_job_parallelism=)

            yield ParallelSlotAllocator(slot_size, slots, pool)

    @contextmanager
    def _block_all_pools(self, pool_type: PoolType):
        # need to block pool
        with self._pools_lock:
            yield self._get_pool(pool_type)

    def _prepare_trains(self, paralellism_mode, train_df, max_job_parallelism: int) -> List[DatasetSlot]:
        if self.parallelism_mode == ParallelismMode.pref_locs:
            execs_per_job = max(1, math.floor(len(self._executors) / max_job_parallelism))

            if len(self._executors) % max_job_parallelism != 0:
                warnings.warn(f"Uneven number of executors per job. Setting execs per job: {execs_per_job}.")

            slots_num = int(len(self._executors) / execs_per_job)

            _train_slots = []

            def _coalesce_df_to_locs(df: SparkDataFrame, pref_locs: List[str]):
                df = PrefferedLocsPartitionCoalescerTransformer(pref_locs=pref_locs).transform(df)
                df = df.cache()
                df.write.mode('overwrite').format('noop').save()
                return df

            for i in range(slots_num):
                pref_locs = self._executors[i * execs_per_job: (i + 1) * execs_per_job]

                # prepare train
                train_df = _coalesce_df_to_locs(train_df, pref_locs)

                # prepare test
                test_df = _coalesce_df_to_locs(test_df, pref_locs)

                _train_slots.append(DatasetSlot(
                    train_df=train_df,
                    test_df=test_df,
                    pref_locs=pref_locs,
                    num_tasks=len(pref_locs) * self._cores_per_exec,
                    num_threads=-1,
                    use_single_dataset_mode=True,
                    free=True
                ))

                print(f"Pref lcos for slot #{i}: {pref_locs}")
        elif self.parallelism_mode == ParallelismMode.no_single_dataset_mode:
            num_tasks_per_job = max(1, math.floor(len(self._executors) * self._cores_per_exec / max_job_parallelism))
            _train_slots = [DatasetSlot(
                train_df=self.train_dataset,
                test_df=self.test_dataset,
                pref_locs=None,
                num_tasks=num_tasks_per_job,
                num_threads=-1,
                use_single_dataset_mode=False,
                free=False
            )]
        elif self.parallelism_mode == ParallelismMode.single_dataset_mode:
            num_tasks_per_job = max(1, math.floor(len(self._executors) * self._cores_per_exec / max_job_parallelism))
            num_threads_per_exec = max(1, math.floor(num_tasks_per_job / len(self._executors)))

            if num_threads_per_exec != 1:
                warnings.warn(f"Num threads per exec {num_threads_per_exec} != 1. "
                              f"Overcommitting or undercommiting may happen due to "
                              f"uneven allocations of cores between executors for a job")

            _train_slots = [DatasetSlot(
                train_df=self.train_dataset,
                test_df=self.test_dataset,
                pref_locs=None,
                num_tasks=num_tasks_per_job,
                num_threads=num_threads_per_exec,
                use_single_dataset_mode=True,
                free=False
            )]
        else:
            _train_slots = [DatasetSlot(
                train_df=self.train_dataset,
                test_df=self.test_dataset,
                pref_locs=None,
                num_tasks=self.train_dataset.rdd.getNumPartitions(),
                num_threads=-1,
                use_single_dataset_mode=True,
                free=False
            )]

        return _train_slots


class SequentialComputationsManager(ComputationsManager):
    def compute_with_slots(self, name: str, prepare_slots: Callable[[], List[S]], tasks: List[Callable[[S], T]],
                           pool_type: PoolType) -> List[T]:
        raise NotImplementedError()

    def compute(self, tasks: list[Callable[[], T]], pool_type: PoolType) -> List[T]:
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


class _SlotBasedTVIter(SparkBaseTrainValidIterator):
    def __init__(self, slots: Callable[[], DatasetSlot], tviter: SparkBaseTrainValidIterator):
        super().__init__(None)
        self._slots = slots
        self._tviter = tviter
        self._curr_pos = 0

    def __iter__(self) -> Iterable:
        self._curr_pos = 0
        return self

    def __next__(self) -> TrainVal:
        with self._slots() as slot:
            tviter = deepcopy(self._tviter)
            tviter.train = slot.dataset

            self._curr_pos += 1

            try:
                curr_tv = None
                for i in range(self._curr_pos):
                    curr_tv = next(tviter)
            except StopIteration:
                self._curr_pos = 0
                raise StopIteration()

        return curr_tv

    def freeze(self) -> 'SparkBaseTrainValidIterator':
        raise NotImplementedError()

    def unpersist(self, skip_val: bool = False):
        raise NotImplementedError()

    @property
    def train_val_single_dataset(self) -> 'SparkDataset':
        return self._tviter.train_val_single_dataset

    def get_validation_data(self) -> SparkDataset:
        return self._tviter.get_validation_data()


class _SlotInitiatedTVIter(SparkBaseTrainValidIterator):
    def __init__(self, slot_allocator: SlotAllocator, tviter: SparkBaseTrainValidIterator):
        super().__init__(None)
        self._slot_allocator = slot_allocator
        self._tviter = deepcopy(tviter)

    def __iter__(self) -> Iterable:
        def _iter():
            with self._slot_allocator.allocate() as slot:
                tviter = deepcopy(self._tviter)
                tviter.train = slot.dataset
                for elt in tviter:
                    yield elt

        return _iter()

    def __next__(self):
        raise NotImplementedError("NotSupportedMethod")

    def freeze(self) -> 'SparkBaseTrainValidIterator':
        raise NotImplementedError("NotSupportedMethod")

    def unpersist(self, skip_val: bool = False):
        raise NotImplementedError("NotSupportedMethod")

    @property
    def train_val_single_dataset(self) -> 'SparkDataset':
        return self._tviter.train_val_single_dataset

    def get_validation_data(self) -> SparkDataset:
        return self._tviter.get_validation_data()