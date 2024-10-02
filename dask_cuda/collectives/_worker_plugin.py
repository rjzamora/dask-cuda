from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, overload

from dask.context import thread_state
from dask.typing import Key

from distributed.core import ErrorMessage, OKMessage, clean_exception, error_message
from distributed.diagnostics.plugin import WorkerPlugin
from distributed.protocol.serialize import ToPickle
from distributed.utils import log_errors, sync

from dask_cuda.collectives._core import CollectiveId, CollectiveRun, CollectiveRunSpec, CollectiveSpec
from dask_cuda.collectives._exceptions import CollectiveConsistencyError, CollectiveClosedError

if TYPE_CHECKING:
    from distributed.worker import Worker


logger = logging.getLogger(__name__)


class _CollectiveRunManager:
    closed: bool
    _active_runs: dict[CollectiveId, CollectiveRun]
    _runs: set[CollectiveRun]
    #: Mapping of collective IDs to the largest stale run ID.
    #: This is used to prevent race conditions between fetching collective run data
    #: from the scheduler and failing a collective run.
    #: TODO: Remove once ordering between fetching and failing is guaranteed.
    _stale_run_ids: dict[CollectiveId, int]
    _runs_cleanup_condition: asyncio.Condition
    _plugin: CollectiveWorkerPlugin

    def __init__(self, plugin: CollectiveWorkerPlugin) -> None:
        self.closed = False
        self._active_runs = {}
        self._runs = set()
        self._stale_run_ids = {}
        self._runs_cleanup_condition = asyncio.Condition()
        self._plugin = plugin

    # def heartbeat(self) -> dict[CollectiveId, Any]:
    #     return {
    #         id: collective_run.heartbeat() for id, collective_run in self._active_runs.items()
    #     }

    def fail(self, collective_id: CollectiveId, run_id: int, message: str) -> None:
        stale_run_id = self._stale_run_ids.setdefault(collective_id, run_id)
        if stale_run_id < run_id:
            self._stale_run_ids[collective_id] = run_id

        collective_run = self._active_runs.get(collective_id, None)
        if collective_run is None or collective_run.run_id != run_id:
            return
        self._active_runs.pop(collective_id)
        exception = CollectiveConsistencyError(message)
        collective_run.fail(exception)

        self._plugin.worker._ongoing_background_tasks.call_soon(self.close, collective_run)

    @log_errors
    async def close(self, collective_run: CollectiveRun) -> None:
        try:
            await collective_run.close()
        finally:
            async with self._runs_cleanup_condition:
                self._runs.remove(collective_run)
                self._runs_cleanup_condition.notify_all()

    async def teardown(self) -> None:
        assert not self.closed
        self.closed = True

        while self._active_runs:
            _, collective_run = self._active_runs.popitem()
            self._plugin.worker._ongoing_background_tasks.call_soon(
                self.close, collective_run
            )

        async with self._runs_cleanup_condition:
            await self._runs_cleanup_condition.wait_for(lambda: not self._runs)

    async def get_with_run_id(self, collective_id: CollectiveId, run_id: int) -> CollectiveRun:
        """Get the collective matching the ID and run ID.

        If necessary, this method fetches the collective run from the scheduler plugin.

        Parameters
        ----------
        collective_id
            Unique identifier of the collective
        run_id
            Unique identifier of the collective run

        Raises
        ------
        KeyError
            If the collective does not exist
        CollectiveConsistencyError
            If the run_id is stale
        collectiveClosedError
            If the run manager has been closed
        """
        collective_run = self._active_runs.get(collective_id, None)
        if collective_run is None or collective_run.run_id < run_id:
            collective_run = await self._refresh(collective_id=collective_id)

        if collective_run.run_id > run_id:
            raise CollectiveConsistencyError(f"{run_id=} stale, got {collective_run}")
        elif collective_run.run_id < run_id:
            raise CollectiveConsistencyError(f"{run_id=} invalid, got {collective_run}")

        if self.closed:
            raise CollectiveClosedError(f"{self} has already been closed")
        if collective_run._exception:
            raise collective_run._exception
        return collective_run

    async def get_or_create(self, spec: CollectiveSpec, key: Key) -> CollectiveRun:
        """Get or create a collective matching the ID and data spec.

        Parameters
        ----------
        collective_id
            Unique identifier of the collective
        type:
            Type of the collective operation
        key:
            Task key triggering the function
        """
        collective_run = self._active_runs.get(spec.id, None)
        if collective_run is None:
            collective_run = await self._refresh(
                collective_id=spec.id,
                spec=spec,
                key=key,
            )

        if self.closed:
            raise CollectiveClosedError(f"{self} has already been closed")
        if collective_run._exception:
            raise collective_run._exception
        return collective_run

    async def get_most_recent(
        self, collective_id: CollectiveId, run_ids: Sequence[int]
    ) -> CollectiveRun:
        """Get the collective matching the ID and most recent run ID.

        If necessary, this method fetches the collective run from the scheduler plugin.

        Parameters
        ----------
        collective_id
            Unique identifier of the collective
        run_ids
            Sequence of possibly different run IDs

        Raises
        ------
        KeyError
            If the collective does not exist
        CollectiveConsistencyError
            If the most recent run_id is stale
        """
        return await self.get_with_run_id(collective_id=collective_id, run_id=max(run_ids))

    async def _fetch(
        self,
        collective_id: CollectiveId,
        spec: CollectiveSpec | None = None,
        key: Key | None = None,
    ) -> CollectiveRunSpec:
        if spec is None:
            response = await self._plugin.worker.scheduler.collective_get(
                id=collective_id,
                worker=self._plugin.worker.address,
            )
        else:
            response = await self._plugin.worker.scheduler.collective_get_or_create(
                spec=ToPickle(spec),
                key=key,
                worker=self._plugin.worker.address,
            )

        status = response["status"]
        if status == "error":
            _, exc, tb = clean_exception(**response)
            assert exc
            raise exc.with_traceback(tb)
        assert status == "OK"
        return response["run_spec"]

    @overload
    async def _refresh(
        self,
        collective_id: CollectiveId,
    ) -> CollectiveRun: ...

    @overload
    async def _refresh(
        self,
        collective_id: CollectiveId,
        spec: CollectiveSpec,
        key: Key,
    ) -> CollectiveRun: ...

    async def _refresh(
        self,
        collective_id: CollectiveId,
        spec: CollectiveSpec | None = None,
        key: Key | None = None,
    ) -> CollectiveRun:
        result = await self._fetch(collective_id=collective_id, spec=spec, key=key)
        if self.closed:
            raise CollectiveClosedError(f"{self} has already been closed")
        if existing := self._active_runs.get(collective_id, None):
            if existing.run_id >= result.run_id:
                return existing
            else:
                self.fail(
                    collective_id,
                    existing.run_id,
                    f"{existing!r} stale, expected run_id=={result.run_id}",
                )
        stale_run_id = self._stale_run_ids.get(collective_id, None)
        if stale_run_id is not None and stale_run_id >= result.run_id:
            raise CollectiveConsistencyError(
                f"Received stale collective run with run_id={result.run_id};"
                f" expected run_id > {stale_run_id}"
            )
        collective_run = result.spec.create_run_on_worker(
            run_id=result.run_id,
            worker_for=result.worker_for,
            plugin=self._plugin,
            span_id=result.span_id,
        )
        self._active_runs[collective_id] = collective_run
        self._runs.add(collective_run)
        return collective_run


class CollectiveWorkerPlugin(WorkerPlugin):
    """Interface between a Worker and a collective.

    This extension is responsible for

    - Lifecycle of collective instances
    - ensuring connectivity between remote collective instances
    - ensuring connectivity and integration with the scheduler
    - routing concurrent calls to the appropriate `Collective` based on its `CollectiveId`
    - collecting instrumentation of ongoing collectives and route to scheduler/worker
    """

    worker: Worker
    collective_runs: _CollectiveRunManager
    closed: bool

    def setup(self, worker: Worker) -> None:
        # Attach to worker
        worker.handlers["collective_receive"] = self.collective_receive
        worker.handlers["collective_inputs_done"] = self.collective_inputs_done
        worker.stream_handlers["collective-fail"] = self.collective_fail
        worker.extensions["collective"] = self

        # Initialize
        self.worker = worker
        self.collective_runs = _CollectiveRunManager(self)
        self.closed = False
        self._executor = ThreadPoolExecutor(self.worker.state.nthreads)

    def __str__(self) -> str:
        return f"CollectiveWorkerPlugin on {self.worker.address}"

    def __repr__(self) -> str:
        return f"<CollectiveWorkerPlugin, worker={self.worker.address_safe!r}, closed={self.closed}>"

    # Handlers
    ##########
    # NOTE: handlers are not threadsafe, but they're called from async comms, so that's okay

    # def heartbeat(self) -> dict[CollectiveId, Any]:
    #     return self.collective_runs.heartbeat()

    async def collective_receive(
        self,
        collective_id: CollectiveId,
        run_id: int,
        data: list[tuple[int, Any]] | bytes,
    ) -> OKMessage | ErrorMessage:
        """
        Handler: Receive an incoming shard of data from a peer worker.
        Using an unknown ``collective_id`` is an error.
        """
        try:
            collective_run = await self._get_collective_run(collective_id, run_id)
            return await collective_run.receive(data)
        except CollectiveConsistencyError as e:
            return error_message(e)

    async def collective_inputs_done(self, collective_id: CollectiveId, run_id: int) -> None:
        """
        Handler: Inform the extension that all input partitions have been handed off to extensions.
        Using an unknown ``collective_id`` is an error.
        """
        collective_run = await self._get_collective_run(collective_id, run_id)
        await collective_run.inputs_done()

    def collective_fail(self, collective_id: CollectiveId, run_id: int, message: str) -> None:
        """Fails the collective run with the message as exception and triggers cleanup.

        .. warning::
            To guarantee the correct order of operations, collective_fail must be
            synchronous. See
            https://github.com/dask/distributed/pull/7486#discussion_r1088857185
            for more details.
        """
        self.collective_runs.fail(collective_id=collective_id, run_id=run_id, message=message)

    def add_partition(
        self,
        data: Any,
        partition_id: int,
        spec: CollectiveSpec,
        **kwargs: Any,
    ) -> int:
        spec.validate_data(data)
        collective_run = self.get_or_create_collective(spec)
        return collective_run.add_partition(
            data=data,
            partition_id=partition_id,
            **kwargs,
        )

    async def _barrier(self, collective_id: CollectiveId, run_ids: Sequence[int]) -> int:
        """
        Task: Note that the barrier task has been reached (`add_partition` called for all input partitions)

        Using an unknown ``collective_id`` is an error. Calling this before all partitions have been
        added is undefined.
        """
        collective_run = await self.collective_runs.get_most_recent(collective_id, run_ids)
        # Tell all peers that we've reached the barrier
        # Note that this will call `collective_inputs_done` on our own worker as well
        return await collective_run.barrier(run_ids)

    async def _get_collective_run(
        self,
        collective_id: CollectiveId,
        run_id: int,
    ) -> CollectiveRun:
        return await self.collective_runs.get_with_run_id(
            collective_id=collective_id, run_id=run_id
        )

    async def _get_or_create_collective(
        self,
        spec: CollectiveSpec,
        key: Key,
    ) -> CollectiveRun:
        return await self.collective_runs.get_or_create(spec=spec, key=key)

    async def teardown(self, worker: Worker) -> None:
        assert not self.closed

        self.closed = True
        await self.collective_runs.teardown()
        try:
            self._executor.shutdown(cancel_futures=True)
        except Exception:  # pragma: no cover
            self._executor.shutdown()

    #############################
    # Methods for worker thread #
    #############################

    def barrier(self, collective_id: CollectiveId, run_ids: Sequence[int]) -> int:
        result = sync(self.worker.loop, self._barrier, collective_id, run_ids)
        return result

    def get_collective_run(
        self,
        collective_id: CollectiveId,
        run_id: int,
    ) -> CollectiveRun:
        return sync(
            self.worker.loop,
            self.collective_runs.get_with_run_id,
            collective_id,
            run_id,
        )

    def get_or_create_collective(
        self,
        spec: CollectiveSpec,
    ) -> CollectiveRun:
        key = thread_state.key
        return sync(
            self.worker.loop,
            self.collective_runs.get_or_create,
            spec,
            key,
        )

    def get_output_partition(
        self,
        collective_id: CollectiveId,
        run_id: int,
        partition_id: int,
        meta: Any | None = None,
    ) -> Any:
        """
        Task: Retrieve a final output partition from the CollectiveWorkerPlugin.

        Calling this for a ``collective_id`` which is unknown or incomplete is an error.
        """
        collective_run = self.get_collective_run(collective_id, run_id)
        key = thread_state.key
        return collective_run.get_output_partition(
            partition_id=partition_id,
            key=key,
            meta=meta,
        )
