
from typing import TYPE_CHECKING, Any

import dask.dataframe as dd
from dask.tokenize import tokenize

from dask_cuda.collectives._core import CollectiveId

from dask.typing import Key
from distributed.core import ErrorMessage, OKMessage, error_message
from distributed.diagnostics.plugin import SchedulerPlugin

if TYPE_CHECKING:
    from distributed.scheduler import (
        Recs,
        Scheduler,
        TaskState,
        TaskStateState,
        WorkerState,
    )


class TaskPinnerPlugin(SchedulerPlugin):
    """Scheduler Plugin to pin tasks to specific workers"""

    scheduler: Scheduler
    _pinned_tasks: set

    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler
        self.scheduler.handlers.update({"pin_tasks": self.pin_tasks})
        self.scheduler.add_plugin(self, name="task_pinner")
        self._pinned_tasks = set()

    def pin_tasks(self, keys: list[Key], worker: str) -> OKMessage | ErrorMessage:
        try:
            for key in keys:
                ts = self.scheduler.tasks[key]
                self._set_restriction(ts, worker)
            return {"status": "OK"}
        except Exception as e:
            return error_message(e)

    def _set_restriction(self, ts: TaskState, worker: str) -> None:
        if ts.annotations and "task_pinner_restrictions" in ts.annotations:
            # This task is already pinned
            return
        if ts.annotations is None:
            ts.annotations = dict()
        ts.annotations["task_pinner_restrictions"] = (
            ts.worker_restrictions.copy()
            if ts.worker_restrictions is not None
            else None
        )
        self.scheduler.set_restrictions({ts.key: {worker}})


# def balance(df: dd.DataFrame):
#     # Extract graph from input
#     df = df.optimize()
#     name = df._name
#     dsk = df.dask

#     #
#     token = tokenize(df)
#     _barrier_key = barrier_key(CollectiveId(token))