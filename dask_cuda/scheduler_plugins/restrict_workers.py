from __future__ import annotations

from collections.abc import MutableMapping, Sequence

import dask.dataframe as dd
from dask.typing import Key
from distributed import Client
from distributed.diagnostics.plugin import SchedulerPlugin
from distributed.scheduler import Scheduler, TaskState


class WorkerRestrictorPlugin(SchedulerPlugin):
    """Scheduler Plugin to restrict tasks to specific workers"""

    scheduler: Scheduler
    _tasks_to_restrict: MutableMapping[tuple, str]
    _restricted_tasks: MutableMapping[tuple, str]

    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler
        self.scheduler.stream_handlers.update(
            {"add_worker_restrictions": self.add_worker_restrictions}
        )
        self.scheduler.add_plugin(self, name="worker_restrictor")
        self._tasks_to_restrict = {}
        self._restricted_tasks = {}

    def add_worker_restrictions(self, *args, **kwargs) -> None:
        key_worker_map = kwargs.pop("key_worker_map", {})
        for k, w in key_worker_map.items():
            self._tasks_to_restrict[k] = {w}

    def update_graph(self, *args, keys: set[Key], **kwargs) -> None:
        for key in keys:
            if key in self._tasks_to_restrict:
                worker = self._tasks_to_restrict.pop(key)
                ts: TaskState = self.scheduler.tasks[key]
                self.scheduler.set_restrictions({ts.key: worker})
                self._restricted_tasks[key] = worker


def pin_and_persist(
    df: dd.DataFrame,
    client: Client | None = None,
    ranks: Sequence[str] | None = None,
    partition_map: MutableMapping[int, int] | None = None,
    enforce: bool = True,
) -> dd.DataFrame:
    """Pin DataFrame partitions to workers and persist.

    Parameters
    ----------
    df :
        Dask DataFrame object to pin and persist.
    client : Optional
        Distributed client object. If none is provided,
        `get_client` will be used to infer the client.
    ranks : Optional
        List of worker addresses. The order of this list
        determines the 'rank' if each worker.
    partition_map : Optional
        Dictionary mapping of partition id's to ranks.
        For example, `{2: 0}` means that partition `2`
        should be mapped to rank `0`. By default,
        partitions will be mapped to ranks in a round-robin
        fashion.
    enforce : Optional
        Whether the location of all partitions should be
        validated after persisting. Specifying `True`,
        the default, requires an internal `wait` operation.
    """
    from distributed import get_client, wait

    df = df.optimize()
    name = df._name
    client = client or get_client()
    ranks = ranks or list(client.scheduler_info()["workers"].keys())
    partition_map = partition_map or {}

    n_workers = len(ranks)
    key_worker_map = {
        (name, i): ranks[partition_map.get(i, i % n_workers)]
        for i in range(df.npartitions)
    }

    client._send_to_scheduler(
        {
            "op": "add_worker_restrictions",
            "key_worker_map": key_worker_map,
        }
    )

    df = df.persist()
    if enforce:
        wait(df)
        for k, v in client.who_has(df).items():
            if (key_worker_map[k],) != v:
                raise ValueError(
                    f"Failed to restrict {k} to worker {key_worker_map[k]}. "
                    f"Got worker {v}."
                )

    return df


def dask_setup(scheduler):
    WorkerRestrictorPlugin(scheduler)
