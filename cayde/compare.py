"""
    PerformanceComparator API:
    A tool for comparing the performance of various methods on a class.

    Example method:

    def getAccuracies(folds, max_depth, C):
        "Get the accuracies of various classifiers"
        accuracies = pd.DataFrame()
        comparator = PerformanceComparator([
            DecisionTreeClassifier(well, max_depth=max_depth),
            SVMClassifier(well, C=C),
            LogisticClassifier(well, C=C)
        ], n_threads=2)
        comparator.addTask('accuracy', args=(folds,))
        comparator.performTasks()

        if comparator.exceptions:
            st.write("Exceptions happened: ", comparator.exceptions)
            raise Exception(comparator.exceptions)

        classifier: Classifier
        output: Output
        for classifier, output in comparator.output['accuracy'].items():
            accuracies[classifier.__class__.__name__] = output.output.mean()

        return accuracies

"""

from typing import List, Dict, Any, TypeVar, Generic, TypedDict, Tuple, Optional
from threading import Thread
from dataclasses import dataclass
import time

T = TypeVar('T')
OutputT = Dict[str, Dict[T, 'Output']]

__all__ = ['PerformanceComparator']

@dataclass
class Task:
    method: str
    args: Tuple
    kwargs: Dict[str, Any]

@dataclass
class Output(Generic[T]):
    output: Any
    args: Tuple
    kwargs: Dict[str, Any]
    time: Optional[float]

class Worker(Generic[T]):
    _task_queue: List[Task]
    _performers: List[T]
    _output: OutputT
    _exceptions: List[Exception]
    _record_time: bool

    def __init__(self, performers: List[T], task_queue: List[Task], output: OutputT, exceptions: List[Exception], record_time: bool):
        self._task_queue = task_queue
        self._performers = performers
        self._output = output
        self._exceptions = exceptions
        self._record_time = record_time

    def _singleTask(self, task: Task, performer: T):
        self._output.setdefault(task.method, dict())
        total_time = -1.
        try:
            if self._record_time:
                start_time = time.time()
            result = getattr(performer, task.method)(*task.args, **task.kwargs)
            if self._record_time:
                end_time = time.time()
                total_time = end_time - total_time
            self._output[task.method][performer] = Output(
                result, task.args, task.kwargs, total_time
            )
        except Exception as e:
            self._exceptions.append(e)

    def run(self):
        thread_pool = []
        while self._task_queue:
            task = self._task_queue.pop(0)
            for performer in self._performers:
                thread_pool.append(Thread(
                    target=self._singleTask,
                    args=(task, performer),
                ))
                thread_pool[-1].start()
            for thread in thread_pool:
                thread.join()

class PerformanceComparator(Generic[T]):

    class NonEmptyTaskQueueException(Exception):
        "Thrown when an operation requires the task queue to be empty, but it currently isn't"

    _performers: List[T]
    _output: Dict[str, Dict[T, Any]]
    # {
    #   command: {
    #     performer: result
    #   }
    # }
    _total_threads: int
    _record_times: bool
    _task_queue: List[Task]
    _exceptions: List[Exception]

    def __init__(self, performers: List[T] = list(), record_times: bool = False, n_threads: int = 1):
        """
            Note that there will be len(self._performers) * total_threads total possible threads.
        """
        self._performers = performers[:]
        self._record_times = record_times
        self._total_threads = n_threads
        self._task_queue = []
        self._output = {}
        self._exceptions = []

    def addPerformer(self, performer: T, force: bool = False):
        if self._task_queue and not force:
            raise PerformanceComparator.NonEmptyTaskQueueException(
                "Cannot add performers when tasks are pending. pass in force=True to override this."
            )
        self._performers.append(performer)

    def addTasks(self, method_name: str, args: List[Tuple] = [], kwargs: List[Dict[str, Any]] = []):
        for arg in args:
            for kwarg in kwargs:
                self.addTask(method_name, arg, kwarg)

    def addTask(self, method_name: str, args: Tuple = tuple(), kwargs: Dict[str, Any] = {}):
        for performer in self._performers:
            if not hasattr(performer, method_name):
                raise AttributeError(f"not all performers have method name '{method_name}'")
        self._task_queue.append(Task(method_name, args, kwargs))

    def performTasks(self, block: bool = True):
        workers = [
            Worker(
                self._performers, self._task_queue,
                self._output, self._exceptions,
                self._record_times
            ) for _ in range(self._total_threads)
        ]
        threads = []

        for worker in workers:
            threads.append(Thread(target=worker.run))
            threads[-1].start()

        if block:
            for thread in threads:
                thread.join()

    @property
    def output(self) -> OutputT:
        return self._output.copy()

    @property
    def exceptions(self) -> List[Exception]:
        return self._exceptions[:]
