from __future__ import annotations

import contextlib
import itertools
import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from dask.typing import Key

from distributed.core import ErrorMessage, OKMessage, error_message
from distributed.diagnostics.plugin import SchedulerPlugin
from distributed.metrics import time
from distributed.protocol.pickle import dumps
from distributed.protocol.serialize import ToPickle
from distributed.utils import log_errors

from dask_cuda.collectives._core import (
    CollectiveId,
    CollectiveRunSpec,
    CollectiveSpec,
    SchedulerCollectiveState,
    RunSpecMessage,
    barrier_key,
    id_from_key,
)
from dask_cuda.collectives._exceptions import (
    CollectiveConsistencyError,
    CollectiveIllegalStateError,
)
from dask_cuda.collectives._worker_plugin import  CollectiveWorkerPlugin


if TYPE_CHECKING:
    from distributed.scheduler import (
        Recs,
        Scheduler,
        TaskState,
        TaskStateState,
        WorkerState,
    )

logger = logging.getLogger(__name__)


class CollectiveSchedulerPlugin(SchedulerPlugin):
    """
    Collective plugin for the scheduler
    This coordinates the individual worker plugins to ensure correctness.
    See Also
    --------
    CollectiveWorkerPlugin
    """

    scheduler: Scheduler
    active_collectives: dict[CollectiveId, SchedulerCollectiveState]
    #heartbeats: defaultdict[CollectiveId, dict]
    _collectives: defaultdict[CollectiveId, set[SchedulerCollectiveState]]
    _archived_by_stimulus: defaultdict[str, set[SchedulerCollectiveState]]
    _shift_counter: itertools.count[int]

    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler
        self.scheduler.handlers.update(
            {
                "collective_barrier": self.barrier,
                "collective_get": self.get,
                "collective_get_or_create": self.get_or_create,
                "collective_restrict_task": self.restrict_task,
            }
        )
        #self.heartbeats = defaultdict(lambda: defaultdict(dict))
        self.active_collectives = {}
        self.scheduler.add_plugin(self, name="collective")
        self._collectives = defaultdict(set)
        self._archived_by_stimulus = defaultdict(set)
        self._shift_counter = itertools.count()

    async def start(self, scheduler: Scheduler) -> None:
        worker_plugin = CollectiveWorkerPlugin()
        await self.scheduler.register_worker_plugin(
            None, dumps(worker_plugin), name="collective", idempotent=False
        )

    def collective_ids(self) -> set[CollectiveId]:
        return set(self.active_collectives)

    async def barrier(self, id: CollectiveId, run_id: int, consistent: bool) -> None:
        collective = self.active_collectives[id]
        if collective.run_id != run_id:
            raise ValueError(f"{run_id=} does not match {collective}")
        if not consistent:
            logger.warning(
                "Collective %s restarted due to data inconsistency during barrier",
                collective.id,
            )
            return self._restart_collective(
                collective.id,
                self.scheduler,
                stimulus_id=f"collective-barrier-inconsistent-{time()}",
            )
        msg = {"op": "collective_inputs_done", "collective_id": id, "run_id": run_id}
        await self.scheduler.broadcast(
            msg=msg,
            workers=list(collective.participating_workers),
        )

    def restrict_task(
        self, id: CollectiveId, run_id: int, key: Key, worker: str
    ) -> OKMessage | ErrorMessage:
        try:
            collective = self.active_collectives[id]
            if collective.run_id > run_id:
                raise CollectiveConsistencyError(
                    f"Request stale, expected {run_id=} for {collective}"
                )
            elif collective.run_id < run_id:
                raise CollectiveConsistencyError(
                    f"Request invalid, expected {run_id=} for {collective}"
                )
            ts = self.scheduler.tasks[key]
            self._set_restriction(ts, worker)
            return {"status": "OK"}
        except CollectiveConsistencyError as e:
            return error_message(e)

    # def heartbeat(self, ws: WorkerState, data: dict) -> None:
    #     for collective_id, d in data.items():
    #         if collective_id in self.collective_ids():
    #             self.heartbeats[collective_id][ws.address].update(d)

    def get(self, id: CollectiveId, worker: str) -> RunSpecMessage | ErrorMessage:
        try:
            try:
                run_spec = self._get(id, worker)
                return {"status": "OK", "run_spec": ToPickle(run_spec)}
            except KeyError as e:
                raise CollectiveConsistencyError(
                    f"No active collective with {id=!r} found"
                ) from e
        except CollectiveConsistencyError as e:
            return error_message(e)

    def _get(self, id: CollectiveId, worker: str) -> CollectiveRunSpec:
        if worker not in self.scheduler.workers:
            # This should never happen
            raise CollectiveConsistencyError(
                f"Scheduler is unaware of this worker {worker!r}"
            )  # pragma: nocover
        state = self.active_collectives[id]
        state.participating_workers.add(worker)
        return state.run_spec

    def _create(self, spec: CollectiveSpec, key: Key, worker: str) -> CollectiveRunSpec:
        # FIXME: The current implementation relies on the barrier task to be
        # known by its name. If the name has been mangled, we cannot guarantee
        # that the collective works as intended and should fail instead.
        self._raise_if_barrier_unknown(spec.id)
        self._raise_if_task_not_processing(key)
        worker_for = self._calculate_worker_for(spec)
        self._ensure_output_tasks_are_non_rootish(spec)
        state = spec.create_new_run(
            worker_for=worker_for, span_id=self.scheduler.tasks[key].group.span_id
        )
        self.active_collectives[spec.id] = state
        self._collectives[spec.id].add(state)
        state.participating_workers.add(worker)
        logger.warning(
            "collective %s initialized by task %r executed on worker %s",
            spec.id,
            key,
            worker,
        )
        return state.run_spec

    def get_or_create(
        self,
        spec: CollectiveSpec,
        key: Key,
        worker: str,
    ) -> RunSpecMessage | ErrorMessage:
        try:
            run_spec = self._get(spec.id, worker)
        except CollectiveConsistencyError as e:
            return error_message(e)
        except KeyError:
            try:
                run_spec = self._create(spec, key, worker)
            except CollectiveConsistencyError as e:
                return error_message(e)
        return {"status": "OK", "run_spec": ToPickle(run_spec)}

    def _raise_if_barrier_unknown(self, id: CollectiveId) -> None:
        key = barrier_key(id)
        try:
            self.scheduler.tasks[key]
        except KeyError:
            raise CollectiveConsistencyError(
                f"Barrier task with key {key!r} does not exist. This may be caused by "
                "task fusion during graph generation. Please let us know that you ran "
                "into this by leaving a comment at distributed#7816."
            )

    def _raise_if_task_not_processing(self, key: Key) -> None:
        task = self.scheduler.tasks[key]
        if task.state != "processing":
            raise CollectiveConsistencyError(
                f"Expected {task} to be processing, is {task.state}."
            )

    def _calculate_worker_for(self, spec: CollectiveSpec) -> dict[Any, str]:
        """Pin the outputs of a collective to specific workers.

        The collective implementation of a hash join combines the loading of collective output
        partitions for the left and right side with the actual merge operation into a
        single output task. As a consequence, we need to make sure that collectives with
        shared output tasks align on the output mapping.

        Parameters
        ----------
        id: ID of the collective to pin
        output_partitions: Output partition IDs to pin
        pick: Function that picks a worker given a partition ID and sequence of worker

        .. note:
            This function assumes that the barrier task and the output tasks share
            the same worker restrictions.
        """
        existing: dict[Any, str] = {}
        collective_id = spec.id
        barrier = self.scheduler.tasks[barrier_key(collective_id)]

        if barrier.worker_restrictions:
            workers = list(barrier.worker_restrictions)
        else:
            workers = list(self.scheduler.workers)

        # Ensure homogeneous cluster utilization when there are multiple small,
        # independent collectives going on at the same time, e.g. due to partial rechunking
        shift_by = next(self._shift_counter) % len(workers)
        workers = workers[shift_by:] + workers[:shift_by]

        # Check if this collective shares output tasks with a different collective that has
        # already been initialized and needs to be taken into account when
        # mapping output partitions to workers.
        # Naively, you could delete this whole paragraph and just call
        # spec.pick_worker; it would return two identical sets of results on both calls
        # of this method... until the set of available workers changes between the two
        # calls, which would cause misaligned collective outputs and a deadlock.
        seen = {barrier}
        for dependent in barrier.dependents:
            for possible_barrier in dependent.dependencies:
                if possible_barrier in seen:
                    continue
                seen.add(possible_barrier)
                if not (other_barrier_key := id_from_key(possible_barrier.key)):
                    continue
                if not (collective := self.active_collectives.get(other_barrier_key)):
                    continue
                current_worker_for = collective.run_spec.worker_for
                # This is a fail-safe for future three-ways merges. At the moment there
                # should only ever be at most one other collective that shares output
                # tasks, so existing will always be empty.
                if existing:  # pragma: nocover
                    for shared_key in existing.keys() & current_worker_for.keys():
                        if existing[shared_key] != current_worker_for[shared_key]:
                            raise CollectiveIllegalStateError(
                                f"Failed to initialize collective {spec.id} because "
                                "it cannot align output partition mappings between "
                                f"existing collectives {seen}. "
                                f"Mismatch encountered for output partition {shared_key!r}: "
                                f"{existing[shared_key]} != {current_worker_for[shared_key]}."
                            )
                existing.update(current_worker_for)

        worker_for = {}
        for partition in spec.output_partitions:
            if (worker := existing.get(partition, None)) is None:
                worker = spec.pick_worker(partition, workers)
            worker_for[partition] = worker
        return worker_for

    def _ensure_output_tasks_are_non_rootish(self, spec: CollectiveSpec) -> None:
        """Output tasks are created without worker restrictions and run once with the
        only purpose of setting the worker restriction and then raising Reschedule, and
        then running again properly on the correct worker. It would be non-trivial to
        set the worker restriction before they're first run due to potential task
        fusion.

        Most times, this lack of initial restrictions would cause output tasks to be
        labelled as rootish on their first (very fast) run, which in turn would break
        the design assumption that the worker-side queue of rootish tasks will last long
        enough to cover the round-trip to the scheduler to receive more tasks, which in
        turn would cause a measurable slowdown on the overall runtime of the collective
        operation.

        This method ensures that, given M output tasks and N workers, each worker-side
        queue is pre-loaded with M/N output tasks which can be flushed very fast as
        they all raise Reschedule() in quick succession.

        See Also
        --------
        CollectiveRun._ensure_output_worker
        """
        barrier = self.scheduler.tasks[barrier_key(spec.id)]
        for dependent in barrier.dependents:
            dependent._queueable = False

    @log_errors()
    def _set_restriction(self, ts: TaskState, worker: str) -> None:
        if ts.annotations and "collective_original_restrictions" in ts.annotations:
            # This may occur if multiple barriers share the same output task,
            # e.g. in a hash join.
            return
        if ts.annotations is None:
            ts.annotations = dict()
        ts.annotations["collective_original_restrictions"] = (
            ts.worker_restrictions.copy()
            if ts.worker_restrictions is not None
            else None
        )
        self.scheduler.set_restrictions({ts.key: {worker}})

    @log_errors()
    def _unset_restriction(self, ts: TaskState) -> None:
        # collective_original_restrictions is only set if the task was first scheduled
        # on the wrong worker
        if (
            ts.annotations is None
            or "collective_original_restrictions" not in ts.annotations
        ):
            return
        original_restrictions = ts.annotations.pop("collective_original_restrictions")
        self.scheduler.set_restrictions({ts.key: original_restrictions})

    def _restart_recommendations(self, id: CollectiveId) -> Recs:
        barrier_task = self.scheduler.tasks[barrier_key(id)]
        recs: Recs = {}

        for dt in barrier_task.dependents:
            if dt.state == "erred":
                return {}
            recs.update({dt.key: "released"})

        if barrier_task.state == "erred":
            # This should never happen, a dependent of the barrier should already
            # be `erred`
            raise CollectiveIllegalStateError(
                f"Expected dependents of {barrier_task=} to be 'erred' if "
                "the barrier is."
            )  # pragma: no cover
        recs.update({barrier_task.key: "released"})

        for dt in barrier_task.dependencies:
            if dt.state == "erred":
                # This should never happen, a dependent of the barrier should already
                # be `erred`
                raise CollectiveIllegalStateError(
                    f"Expected barrier and its dependents to be "
                    f"'erred' if the barrier's dependency {dt} is."
                )  # pragma: no cover
            recs.update({dt.key: "released"})
        return recs

    def _restart_collective(
        self, id: CollectiveId, scheduler: Scheduler, *, stimulus_id: str
    ) -> None:
        recs = self._restart_recommendations(id)
        self.scheduler.transitions(recs, stimulus_id=stimulus_id)
        self.scheduler.stimulus_queue_slots_maybe_opened(stimulus_id=stimulus_id)
        logger.warning("Collective %s restarted due to stimulus '%s", id, stimulus_id)

    def remove_worker(
        self, scheduler: Scheduler, worker: str, *, stimulus_id: str, **kwargs: Any
    ) -> None:
        """Restart all active collectives when a participating worker leaves the cluster.

        .. note::
            Due to the order of operations in :meth:`~Scheduler.remove_worker`, the
            collective may have already been archived by
            :meth:`~CollectiveSchedulerPlugin.transition`. In this case, the
            ``stimulus_id`` is used as a transaction identifier and all archived collectives
            with a matching `stimulus_id` are restarted.
        """

        # If processing the transactions causes a task to get released, this
        # removes the collective from self.active_collectives. Therefore, we must iterate
        # over a copy.
        for collective_id, collective in self.active_collectives.copy().items():
            if worker not in collective.participating_workers:
                continue
            logger.debug(
                "Worker %s removed during active collective %s due to stimulus '%s'",
                worker,
                collective_id,
                stimulus_id,
            )
            exception = CollectiveConsistencyError(
                f"Worker {worker} left during active {collective}"
            )
            self._fail_on_workers(collective, str(exception))
            self._clean_on_scheduler(collective_id, stimulus_id)

        for collective in self._archived_by_stimulus.get(stimulus_id, set()):
            self._restart_collective(collective.id, scheduler, stimulus_id=stimulus_id)

    def transition(
        self,
        key: Key,
        start: TaskStateState,
        finish: TaskStateState,
        *args: Any,
        stimulus_id: str,
        **kwargs: Any,
    ) -> None:
        """Clean up scheduler and worker state once a collective becomes inactive."""
        if finish not in ("released", "erred", "forgotten"):
            return

        if finish == "erred":
            ts = self.scheduler.tasks[key]
            for active_collective in self.active_collectives.values():
                # Log once per active collective
                if active_collective._failed:
                    continue
                # Log IFF a collective task is the root cause
                if ts.exception_blame != ts:
                    continue
                barrier = self.scheduler.tasks[barrier_key(active_collective.id)]
                if (
                    ts == barrier
                    or ts in barrier.dependents
                    or ts in barrier.dependencies
                ):
                    active_collective._failed = True
                    self.scheduler.log_event(
                        "collective",
                        {
                            "action": "collective-failed",
                            "collective": active_collective.id,
                            "stimulus": stimulus_id,
                        },
                    )
                    return

        collective_id = id_from_key(key)
        if not collective_id:
            return

        if collective := self.active_collectives.get(collective_id):
            self._fail_on_workers(collective, message=f"{collective} forgotten")
            self._clean_on_scheduler(collective_id, stimulus_id=stimulus_id)
            logger.debug(
                "collective %s forgotten because task %r transitioned to %s due to "
                "stimulus '%s'",
                collective_id,
                key,
                finish,
                stimulus_id,
            )

        if finish == "forgotten":
            collectives = self._collectives.pop(collective_id, set())
            for collective in collectives:
                if collective._archived_by:
                    archived = self._archived_by_stimulus[collective._archived_by]
                    archived.remove(collective)
                    if not archived:
                        del self._archived_by_stimulus[collective._archived_by]

    def valid_workers_downscaling(
        self, scheduler: Scheduler, workers: list[WorkerState]
    ) -> list[WorkerState]:
        all_participating_workers = set()
        for collective in self.active_collectives.values():
            all_participating_workers.update(collective.participating_workers)
        return [w for w in workers if w.address not in all_participating_workers]

    def _fail_on_workers(self, collective: SchedulerCollectiveState, message: str) -> None:
        worker_msgs = {
            worker: [
                {
                    "op": "collective-fail",
                    "collective_id": collective.id,
                    "run_id": collective.run_id,
                    "message": message,
                }
            ]
            for worker in collective.participating_workers
        }
        self.scheduler.send_all({}, worker_msgs)

    def _clean_on_scheduler(self, id: CollectiveId, stimulus_id: str) -> None:
        collective = self.active_collectives.pop(id)
        logger.warning("collective %s deactivated due to stimulus '%s'", id, stimulus_id)
        if not collective._archived_by:
            collective._archived_by = stimulus_id
            self._archived_by_stimulus[stimulus_id].add(collective)

        #with contextlib.suppress(KeyError):
        #    del self.heartbeats[id]

        barrier_task = self.scheduler.tasks[barrier_key(id)]
        for dt in barrier_task.dependents:
            self._unset_restriction(dt)

    def restart(self, scheduler: Scheduler) -> None:
        self.active_collectives.clear()
        #self.heartbeats.clear()
        self._collectives.clear()
        self._archived_by_stimulus.clear()
