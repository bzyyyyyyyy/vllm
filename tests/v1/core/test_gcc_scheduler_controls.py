# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from collections import deque
from types import SimpleNamespace
from typing import Any, cast

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.inflight_prefix import InFlightPrefixTracker
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

pytestmark = pytest.mark.gcc_extension


class _Queue:
    def __init__(self) -> None:
        self.requests: list[Request] = []

    def add_request(self, request: Request) -> None:
        self.requests.append(request)

    def prepend_request(self, request: Request) -> None:
        self.requests.insert(0, request)

    def remove_requests(self, requests: list[Request]) -> None:
        removed = set(requests)
        self.requests = [request for request in self.requests if request not in removed]

    def __len__(self) -> int:
        return len(self.requests)


class _KVCacheManager:
    def __init__(self) -> None:
        self.cached: list[tuple[str, int]] = []
        self.freed: list[str] = []
        self.pinned_prefixes = False
        self.reset_allowed = True

    def has_pinned_prefix(self, pin_id: str) -> bool:
        return False

    def has_pinned_prefixes(self) -> bool:
        return self.pinned_prefixes

    def can_reset_prefix_cache(self) -> bool:
        return not self.pinned_prefixes and self.reset_allowed

    def cache_blocks(self, request: Request, num_tokens: int) -> None:
        self.cached.append((request.request_id, num_tokens))

    def free(self, request: Request) -> None:
        self.freed.append(request.request_id)

    def get_blocks(self, request_id: str):
        return SimpleNamespace(blocks=())

    def release_retained_blocks(self, blocks) -> None:
        raise AssertionError("test has no retained in-flight blocks")


class _CPUConnector:
    def __init__(
        self,
        *,
        pin_error: Exception | None = None,
        unpin_error: Exception | None = None,
    ) -> None:
        self.ready = False
        self.unpinned: list[str] = []
        self.pin_calls: list[tuple[str, str, int]] = []
        self.pin_error = pin_error
        self.unpin_error = unpin_error
        self.pinned_prefixes = False
        self.pin_ids: set[str] = set()

    def get_prefix_pin_alignment(self) -> int:
        return 16

    def pin_request_kv(
        self, pin_id: str, request: Request, num_computed_tokens: int
    ) -> bool:
        self.pin_calls.append((pin_id, request.request_id, num_computed_tokens))
        if self.pin_error is not None:
            raise self.pin_error
        self.pin_ids.add(pin_id)
        return True

    def pin_request_kv_with_snapshot(
        self, pin_id: str, request: Request, num_computed_tokens: int, blocks
    ) -> bool:
        return self.pin_request_kv(pin_id, request, num_computed_tokens)

    def get_prefix_pin_error(self, pin_id: str) -> str | None:
        return None

    def is_prefix_pin_ready(self, pin_id: str) -> bool:
        return self.ready

    def unpin_prefix(self, pin_id: str) -> bool:
        self.unpinned.append(pin_id)
        if self.unpin_error is not None:
            raise self.unpin_error
        self.pin_ids.discard(pin_id)
        return True

    def has_pinned_prefixes(self) -> bool:
        return self.pinned_prefixes or bool(self.pin_ids)

    def has_pinned_prefix(self, pin_id: str) -> bool:
        return pin_id in self.pin_ids


def _request(request_id: str = "request") -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=list(range(32)),
        sampling_params=SamplingParams(max_tokens=4),
        pooling_params=None,
    )


def _request_with_prompt(prompt_tokens: int) -> Request:
    return Request(
        request_id=f"prefix-{prompt_tokens}",
        prompt_token_ids=list(range(prompt_tokens)),
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
    )


def _scheduler(connector=None) -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.log_stats = False
    scheduler.cache_config = SimpleNamespace(enable_prefix_caching=True)
    scheduler.hash_block_size = 16
    scheduler.block_size = 64
    scheduler.connector = connector
    scheduler.kv_cache_manager = _KVCacheManager()
    scheduler.requests = {}
    scheduler.running = []
    scheduler.waiting = _Queue()
    scheduler.skipped_waiting = _Queue()
    scheduler.sched_step_seq = 0
    scheduler.processed_step_seq = 0
    scheduler.deferred_frees = deque()
    scheduler.reset_preempted_req_ids = set()
    scheduler._inflight_prefills = set()
    scheduler.encoder_cache_manager = SimpleNamespace(free=lambda _request: None)
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = None
    scheduler.num_waiting_for_streaming_input = 0
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler.use_pp = False
    scheduler.num_lookahead_tokens = 4
    scheduler._prefix_pin_sampler_bypass_supported = True
    scheduler._prefix_pins = {}
    scheduler._request_to_prefix_pin = {}
    scheduler._completed_prefix_pin_tiers = {}
    scheduler._completed_prefix_pin_request_ids = {}
    scheduler._inflight_prefixes = InFlightPrefixTracker(16, 64)
    scheduler._prefix_reservations = {}
    scheduler._paused_requests = {}
    scheduler._pause_resume_status = {}
    scheduler._pending_pause_req_ids = set()
    scheduler._pending_resume_req_ids = set()
    scheduler._pause_cpu_pin_ids = {}
    scheduler._pause_cpu_waiting = set()
    scheduler._pause_cpu_backed_tokens = {}
    scheduler._pause_original_computed_tokens = {}
    scheduler._resuming_cpu_pauses = set()
    scheduler._pause_ack_ready_ids = set()
    scheduler._pause_waiters = []
    scheduler._resume_waiters = []
    return scheduler


def test_gpu_pause_keeps_request_owned_kv_table() -> None:
    scheduler = _scheduler()
    scheduler.max_num_running_reqs = 1
    request = _request()
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = 16
    scheduler.requests[request.request_id] = request
    scheduler.running.append(request)

    paused = scheduler.pause_requests([request.request_id])

    assert paused.done()
    assert request.status == RequestStatus.PAUSED
    assert scheduler.kv_cache_manager.freed == []
    assert scheduler.kv_cache_manager.cached == [(request.request_id, 16)]
    assert not scheduler._has_free_request_slot()

    resumed = scheduler.resume_requests([request.request_id])
    assert resumed.done()
    assert request.status == RequestStatus.RUNNING
    assert scheduler.running == [request]


def test_prefix_pin_uses_hash_unit_and_keeps_metadata_scheduler_local() -> None:
    scheduler = _scheduler()
    request = _request_with_prompt(31)

    future = scheduler.pin_prefix("pin", request)

    assert not future.done()
    assert request.num_prompt_tokens == 16
    assert request.prompt_token_ids == list(range(16))
    assert request.request_id in scheduler._request_to_prefix_pin
    assert not hasattr(request, "pin_prefix_id")


def test_prefix_pin_rejects_prompt_shorter_than_hash_unit() -> None:
    scheduler = _scheduler()

    with pytest.raises(ValueError, match="prefix-match unit \\(16 tokens\\)"):
        scheduler.pin_prefix("pin", _request_with_prompt(15))


def test_unpin_reports_success_when_cancelling_pending_pin() -> None:
    scheduler = _scheduler()
    future = scheduler.pin_prefix("pin", _request_with_prompt(32))

    assert scheduler.unpin_prefix("pin")
    with pytest.raises(RuntimeError, match="prefix pin was cancelled"):
        future.result()


def test_cpu_pause_frees_gpu_only_after_exact_snapshot_is_ready() -> None:
    connector = _CPUConnector()
    scheduler = _scheduler(connector)
    request = _request()
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = 16
    scheduler.requests[request.request_id] = request
    scheduler.running.append(request)

    paused = scheduler.pause_requests([request.request_id])
    assert not paused.done()
    assert scheduler.kv_cache_manager.freed == []

    connector.ready = True
    scheduler._poll_pause_cpu_backing()
    scheduler._flush_request_operation_waiters()

    assert paused.done()
    assert scheduler.kv_cache_manager.freed == [request.request_id]
    assert request.num_computed_tokens == 0

    resumed = scheduler.resume_requests([request.request_id])
    assert not resumed.done()
    assert request.status == RequestStatus.PREEMPTED
    assert request in scheduler.waiting.requests

    scheduler._complete_cpu_pause_restore(request.request_id)
    scheduler._flush_request_operation_waiters()

    assert resumed.done()
    assert connector.unpinned == [
        scheduler._pause_cpu_pin_id(request.request_id)
    ]


def test_cpu_pause_full_external_hit_completes_restore_after_last_token_recompute(
) -> None:
    connector = _CPUConnector()
    scheduler = _scheduler(connector)
    request = _request()
    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    request.num_computed_tokens = request.num_tokens
    scheduler.requests[request.request_id] = request

    pin_id = scheduler._pause_cpu_pin_id(request.request_id)
    scheduler._pause_cpu_pin_ids[request.request_id] = pin_id
    scheduler._pause_cpu_backed_tokens[request.request_id] = request.num_tokens
    scheduler._pause_original_computed_tokens[request.request_id] = (
        request.num_tokens
    )
    scheduler._pause_resume_status[request.request_id] = RequestStatus.RUNNING
    scheduler._resuming_cpu_pauses.add(request.request_id)
    scheduler.finished_recving_kv_req_ids = {request.request_id}
    scheduler.failed_recving_kv_req_ids = set()

    assert scheduler._try_promote_blocked_waiting_request(request)
    assert request.num_computed_tokens == request.num_tokens - 1
    assert request.status == RequestStatus.PREEMPTED
    assert scheduler.kv_cache_manager.freed == []
    assert request.request_id not in scheduler._resuming_cpu_pauses
    assert request.request_id not in scheduler._pause_cpu_backed_tokens
    assert connector.unpinned == [pin_id]


def test_cpu_pause_partial_external_hit_releases_snapshot_and_recomputes_suffix(
) -> None:
    connector = _CPUConnector()
    scheduler = _scheduler(connector)
    request = _request()
    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    request.num_computed_tokens = 16
    scheduler.requests[request.request_id] = request

    pin_id = scheduler._pause_cpu_pin_id(request.request_id)
    scheduler._pause_cpu_pin_ids[request.request_id] = pin_id
    scheduler._pause_cpu_backed_tokens[request.request_id] = request.num_tokens
    scheduler._resuming_cpu_pauses.add(request.request_id)
    scheduler.finished_recving_kv_req_ids = {request.request_id}
    scheduler.failed_recving_kv_req_ids = set()

    assert scheduler._try_promote_blocked_waiting_request(request)
    assert request.num_computed_tokens == 16
    assert request.status == RequestStatus.PREEMPTED
    assert scheduler.kv_cache_manager.freed == []
    assert request.request_id not in scheduler._resuming_cpu_pauses
    assert connector.unpinned == [pin_id]


def test_cpu_pause_failed_external_load_preserves_snapshot_for_retry() -> None:
    connector = _CPUConnector()
    scheduler = _scheduler(connector)
    request = _request()
    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    request.num_computed_tokens = 0
    scheduler.requests[request.request_id] = request

    pin_id = scheduler._pause_cpu_pin_id(request.request_id)
    scheduler._pause_cpu_pin_ids[request.request_id] = pin_id
    scheduler._pause_cpu_backed_tokens[request.request_id] = request.num_tokens
    scheduler._resuming_cpu_pauses.add(request.request_id)
    scheduler.finished_recving_kv_req_ids = {request.request_id}
    scheduler.failed_recving_kv_req_ids = {request.request_id}

    assert scheduler._try_promote_blocked_waiting_request(request)

    assert request.num_computed_tokens == 0
    assert request.status == RequestStatus.PREEMPTED
    assert scheduler.kv_cache_manager.freed == [request.request_id]
    assert request.request_id in scheduler._resuming_cpu_pauses
    assert scheduler._pause_cpu_backed_tokens[request.request_id] == request.num_tokens
    assert connector.unpinned == []


def test_cpu_pause_local_fallback_releases_snapshot_after_admission() -> None:
    connector = _CPUConnector()
    scheduler = _scheduler(connector)
    request = _request()
    pin_id = scheduler._pause_cpu_pin_id(request.request_id)
    scheduler._pause_cpu_pin_ids[request.request_id] = pin_id
    scheduler._pause_cpu_backed_tokens[request.request_id] = request.num_tokens
    scheduler._resuming_cpu_pauses.add(request.request_id)

    scheduler._complete_cpu_pause_restore_after_local_admission(
        request.request_id
    )

    assert request.request_id not in scheduler._resuming_cpu_pauses
    assert request.request_id not in scheduler._pause_cpu_backed_tokens
    assert connector.unpinned == [pin_id]


def test_duplicate_pause_waits_for_pending_cpu_snapshot() -> None:
    connector = _CPUConnector()
    scheduler = _scheduler(connector)
    request = _request()
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = 16
    scheduler.requests[request.request_id] = request
    scheduler.running.append(request)

    first_pause = scheduler.pause_requests([request.request_id])
    duplicate_pause = scheduler.pause_requests([request.request_id])

    assert not first_pause.done()
    assert not duplicate_pause.done()
    assert request.request_id not in scheduler._pause_ack_ready_ids

    connector.ready = True
    scheduler._poll_pause_cpu_backing()
    scheduler._flush_request_operation_waiters()

    assert first_pause.done()
    assert duplicate_pause.done()


def test_duplicate_resume_joins_pending_cpu_restore() -> None:
    connector = _CPUConnector()
    scheduler = _scheduler(connector)
    request = _request()
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = 16
    scheduler.requests[request.request_id] = request
    scheduler.running.append(request)

    paused = scheduler.pause_requests([request.request_id])
    connector.ready = True
    scheduler._poll_pause_cpu_backing()
    scheduler._flush_request_operation_waiters()
    assert paused.done()

    first_resume = scheduler.resume_requests([request.request_id])
    duplicate_resume = scheduler.resume_requests([request.request_id])

    assert not first_resume.done()
    assert not duplicate_resume.done()
    assert request.request_id in scheduler._resuming_cpu_pauses

    scheduler._complete_cpu_pause_restore(request.request_id)
    scheduler._flush_request_operation_waiters()

    assert first_resume.done()
    assert duplicate_resume.done()


def test_repause_during_cpu_restore_fails_without_corrupting_resume() -> None:
    connector = _CPUConnector()
    scheduler = _scheduler(connector)
    request = _request()
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = 16
    scheduler.requests[request.request_id] = request
    scheduler.running.append(request)

    paused = scheduler.pause_requests([request.request_id])
    connector.ready = True
    scheduler._poll_pause_cpu_backing()
    scheduler._flush_request_operation_waiters()
    assert paused.done()

    resumed = scheduler.resume_requests([request.request_id])
    repaused = scheduler.pause_requests([request.request_id])

    assert not resumed.done()
    assert repaused.done()
    with pytest.raises(RuntimeError, match="CPU KV restore is in progress"):
        repaused.result()
    assert request.status == RequestStatus.PREEMPTED
    assert request in scheduler.waiting.requests
    assert request.request_id in scheduler._resuming_cpu_pauses
    assert request.request_id not in scheduler._paused_requests

    scheduler._complete_cpu_pause_restore(request.request_id)
    scheduler._flush_request_operation_waiters()
    assert resumed.done()


def test_cpu_pause_releases_request_slot_only_after_worker_reset() -> None:
    connector = _CPUConnector()
    scheduler = _scheduler(connector)
    scheduler.max_num_running_reqs = 1
    request = _request("paused")
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = 16
    scheduler.requests[request.request_id] = request
    scheduler.running.append(request)
    scheduler.waiting.add_request(_request("waiting"))

    paused = scheduler.pause_requests([request.request_id])

    assert not paused.done()
    assert not scheduler._has_free_request_slot()
    assert [request.request_id for request in scheduler.waiting.requests] == ["waiting"]

    connector.ready = True
    scheduler._poll_pause_cpu_backing()
    scheduler._flush_request_operation_waiters()

    assert paused.done()
    assert scheduler.reset_preempted_req_ids == {"paused"}
    assert scheduler._has_free_request_slot()


def test_immediate_cpu_pause_pin_failure_keeps_gpu_state(
    caplog: pytest.LogCaptureFixture,
) -> None:
    connector = _CPUConnector(
        pin_error=RuntimeError("CPU pin capacity exhausted"),
        unpin_error=RuntimeError("partial reservation cleanup failed"),
    )
    scheduler = _scheduler(connector)
    request = _request()
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = 16
    scheduler.requests[request.request_id] = request
    scheduler.running.append(request)

    with caplog.at_level(logging.WARNING):
        paused = scheduler.pause_requests([request.request_id])

    pin_id = scheduler._pause_cpu_pin_id(request.request_id)
    assert paused.done()
    assert request.status == RequestStatus.PAUSED
    assert scheduler.kv_cache_manager.freed == []
    assert connector.unpinned == [pin_id]
    assert request.request_id not in scheduler._pause_cpu_pin_ids
    assert request.request_id not in scheduler._pause_cpu_waiting
    assert "retaining its GPU KV state" in caplog.text
    assert "Failed to clean up partial CPU pause snapshot" in caplog.text


def test_deferred_cpu_pause_pin_failure_keeps_gpu_state() -> None:
    connector = _CPUConnector(pin_error=RuntimeError("connector unavailable"))
    scheduler = _scheduler(connector)
    request = _request()
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = 16
    request.last_sched_seq = 1
    scheduler.requests[request.request_id] = request
    scheduler.running.append(request)

    paused = scheduler.pause_requests([request.request_id])
    assert not paused.done()

    scheduler.processed_step_seq = 1
    scheduler._finalize_pending_pauses()
    scheduler._flush_request_operation_waiters()

    assert paused.done()
    assert request.status == RequestStatus.PAUSED
    assert scheduler.kv_cache_manager.freed == []
    assert connector.unpinned == [
        scheduler._pause_cpu_pin_id(request.request_id)
    ]
    assert request.request_id not in scheduler._pause_cpu_pin_ids
    assert request.request_id not in scheduler._pause_cpu_waiting


def test_unaligned_pause_uses_gpu_fallback() -> None:
    connector = _CPUConnector()
    scheduler = _scheduler(connector)
    request = _request()
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = 17
    scheduler.requests[request.request_id] = request
    scheduler.running.append(request)

    paused = scheduler.pause_requests([request.request_id])

    assert paused.done()
    assert scheduler.kv_cache_manager.freed == []
    assert request.num_computed_tokens == 17


def test_cpu_pause_pin_id_collision_keeps_foreign_pin_and_gpu_state() -> None:
    connector = _CPUConnector()
    scheduler = _scheduler(connector)
    request = _request("collision")
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = 16
    scheduler.requests[request.request_id] = request
    scheduler.running.append(request)
    foreign_pin_id = scheduler._pause_cpu_pin_id(request.request_id)
    connector.pin_ids.add(foreign_pin_id)

    paused = scheduler.pause_requests([request.request_id])

    assert paused.done()
    assert request.status == RequestStatus.PAUSED
    assert connector.pin_calls == []
    assert connector.unpinned == []
    assert foreign_pin_id in connector.pin_ids
    assert scheduler.kv_cache_manager.freed == []
    assert request.request_id not in scheduler._pause_cpu_pin_ids


def test_can_reset_prefix_cache_is_read_only_and_mirrors_blockers() -> None:
    connector = _CPUConnector()
    scheduler = _scheduler(connector)

    assert scheduler.can_reset_prefix_cache()
    assert scheduler.can_reset_prefix_cache(reset_connector=True)

    pin_future = scheduler.pin_prefix("gpu-pin", _request_with_prompt(16))
    assert not scheduler.can_reset_prefix_cache(reset_running_requests=True)
    assert scheduler.unpin_prefix("gpu-pin")
    with pytest.raises(RuntimeError, match="prefix pin was cancelled"):
        pin_future.result()

    scheduler.kv_cache_manager.pinned_prefixes = True
    assert not scheduler.can_reset_prefix_cache(reset_running_requests=True)
    scheduler.kv_cache_manager.pinned_prefixes = False

    connector.pinned_prefixes = True
    assert not scheduler.can_reset_prefix_cache(reset_connector=True)
    connector.pinned_prefixes = False

    scheduler._pending_pause_req_ids.add("pending")
    assert not scheduler.can_reset_prefix_cache(reset_running_requests=True)
    scheduler._pending_pause_req_ids.clear()

    scheduler._paused_requests["paused"] = _request("paused")
    assert not scheduler.can_reset_prefix_cache(reset_running_requests=True)
    scheduler._paused_requests.clear()

    scheduler._inflight_prefixes = SimpleNamespace(has_state=lambda: True)
    assert not scheduler.can_reset_prefix_cache()
    assert scheduler.can_reset_prefix_cache(reset_running_requests=True)

    scheduler._inflight_prefixes = SimpleNamespace(has_state=lambda: False)
    scheduler.kv_cache_manager.reset_allowed = False
    assert not scheduler.can_reset_prefix_cache()
    assert scheduler.can_reset_prefix_cache(reset_running_requests=True)

    scheduler.num_waiting_for_streaming_input = 1
    assert not scheduler.can_reset_prefix_cache(reset_running_requests=True)


def test_has_requests_polls_pause_handshakes_but_not_retained_pauses() -> None:
    scheduler = _scheduler()
    scheduler.ec_connector = None
    assert not scheduler.has_requests()

    scheduler._pending_pause_req_ids.add("remote-kv")
    assert scheduler.has_requests()
    scheduler._pending_pause_req_ids.clear()

    scheduler._pause_cpu_waiting.add("cpu-backing")
    assert scheduler.has_requests()
    scheduler._pause_cpu_waiting.clear()

    paused = _request("retained")
    paused.status = RequestStatus.PAUSED
    scheduler.requests[paused.request_id] = paused
    scheduler._paused_requests[paused.request_id] = paused
    assert not scheduler.has_requests()


def test_reset_preflight_preserves_waiting_remote_request_state() -> None:
    scheduler = _scheduler()
    request = _request("remote")
    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    scheduler.requests[request.request_id] = request
    scheduler.skipped_waiting.add_request(request)
    queue_before = tuple(scheduler.skipped_waiting.requests)

    assert not scheduler.can_reset_prefix_cache(reset_running_requests=True)

    assert scheduler.requests == {request.request_id: request}
    assert tuple(scheduler.skipped_waiting.requests) == queue_before
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS

    scheduler.skipped_waiting.remove_requests([request])
    request.status = RequestStatus.FINISHED_STOPPED
    assert not scheduler.can_reset_prefix_cache(reset_running_requests=True)
    assert scheduler.requests == {request.request_id: request}
    assert request.status == RequestStatus.FINISHED_STOPPED


def test_reset_preflight_waits_for_step_and_deferred_free_fences() -> None:
    scheduler = _scheduler()
    scheduler.sched_step_seq = 1

    assert not scheduler.can_reset_prefix_cache(reset_running_requests=True)

    scheduler.processed_step_seq = 1
    scheduler.deferred_frees.append((2, []))
    assert not scheduler.can_reset_prefix_cache(reset_running_requests=True)

    scheduler.processed_step_seq = 2
    assert scheduler.can_reset_prefix_cache(reset_running_requests=True)


def test_pending_resume_does_not_consume_pause_acknowledgement() -> None:
    connector = _CPUConnector()
    scheduler = _scheduler(connector)
    request = _request()
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = 16
    scheduler.requests[request.request_id] = request
    scheduler.running.append(request)

    paused = scheduler.pause_requests([request.request_id])
    resumed = scheduler.resume_requests([request.request_id])
    assert not paused.done()
    assert not resumed.done()

    connector.ready = True
    scheduler._poll_pause_cpu_backing()
    scheduler._flush_request_operation_waiters()

    assert paused.done()
    assert not resumed.done()
    scheduler._complete_cpu_pause_restore(request.request_id)
    scheduler._flush_request_operation_waiters()
    assert resumed.done()


def test_abort_resolves_pending_pause_and_resume_operations() -> None:
    connector = _CPUConnector()
    scheduler = _scheduler(connector)
    request = _request()
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = 16
    scheduler.requests[request.request_id] = request
    scheduler.running.append(request)

    paused = scheduler.pause_requests([request.request_id])
    resumed = scheduler.resume_requests([request.request_id])
    assert not paused.done()
    assert not resumed.done()
    setattr(scheduler, "_free_request", lambda *_args, **_kwargs: (None, None))

    scheduler.finish_requests(
        request.request_id,
        RequestStatus.FINISHED_ABORTED,
    )

    assert paused.done()
    assert resumed.done()


def test_streaming_wait_count_is_symmetric_across_gpu_pause() -> None:
    scheduler = _scheduler()
    request = Request(
        request_id="streaming",
        prompt_token_ids=list(range(16)),
        sampling_params=SamplingParams(max_tokens=4),
        pooling_params=None,
        resumable=True,
    )
    request.streaming_queue = deque()
    request.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    scheduler.requests[request.request_id] = request
    scheduler.skipped_waiting.add_request(request)
    scheduler.num_waiting_for_streaming_input = 1

    assert scheduler.pause_requests([request.request_id]).done()
    assert scheduler.num_waiting_for_streaming_input == 0
    assert scheduler.get_num_unfinished_requests() == 0

    assert scheduler.resume_requests([request.request_id]).done()
    assert request.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    assert scheduler.num_waiting_for_streaming_input == 1
    assert scheduler.get_num_unfinished_requests() == 0


def test_streaming_update_queued_while_paused_is_consumed_on_resume() -> None:
    scheduler = _scheduler()
    request = Request(
        request_id="paused-stream-update",
        prompt_token_ids=list(range(16)),
        sampling_params=SamplingParams(max_tokens=4),
        pooling_params=None,
        resumable=True,
    )
    request.streaming_queue = deque()
    request.num_computed_tokens = request.num_tokens
    request.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    scheduler.requests[request.request_id] = request
    scheduler.skipped_waiting.add_request(request)
    scheduler.num_waiting_for_streaming_input = 1

    assert scheduler.pause_requests([request.request_id]).done()
    continuation = Request(
        request_id=request.request_id,
        prompt_token_ids=[101, 102],
        sampling_params=SamplingParams(max_tokens=4),
        pooling_params=None,
        resumable=True,
    )
    scheduler.add_request(continuation)

    assert scheduler.resume_requests([request.request_id]).done()
    assert request.status == RequestStatus.WAITING
    assert request.streaming_queue == deque()
    assert request.prompt_token_ids[-2:] == [101, 102]
    assert scheduler.num_waiting_for_streaming_input == 0
    assert request in scheduler.waiting.requests
    assert request not in scheduler.skipped_waiting.requests


def test_inflight_stream_end_without_update_resumes_as_streaming_wait() -> None:
    scheduler = _scheduler()
    request = Request(
        request_id="inflight-stream-wait",
        prompt_token_ids=list(range(16)),
        sampling_params=SamplingParams(max_tokens=4),
        pooling_params=None,
        resumable=True,
    )
    request.streaming_queue = deque()
    request.status = RequestStatus.RUNNING
    request.last_sched_seq = 1
    scheduler.requests[request.request_id] = request
    scheduler.running.append(request)

    paused = scheduler.pause_requests([request.request_id])
    assert not paused.done()
    request.status = RequestStatus.FINISHED_STOPPED
    assert not scheduler._handle_stopped_request(request)
    assert request.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    assert scheduler.num_waiting_for_streaming_input == 1

    scheduler.processed_step_seq = 1
    scheduler._finalize_pending_pauses()
    scheduler._flush_request_operation_waiters()

    assert paused.done()
    assert request.status == RequestStatus.PAUSED
    assert scheduler._pause_resume_status[request.request_id] == (
        RequestStatus.WAITING_FOR_STREAMING_REQ
    )
    assert scheduler.num_waiting_for_streaming_input == 0
    assert request not in scheduler.skipped_waiting.requests

    assert scheduler.resume_requests([request.request_id]).done()
    assert request.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    assert scheduler.num_waiting_for_streaming_input == 1
    assert request in scheduler.skipped_waiting.requests


def test_inflight_stream_end_with_update_resumes_as_waiting() -> None:
    scheduler = _scheduler()
    request = Request(
        request_id="inflight-stream-update",
        prompt_token_ids=list(range(16)),
        sampling_params=SamplingParams(max_tokens=4),
        pooling_params=None,
        resumable=True,
    )
    request.streaming_queue = deque()
    request.num_computed_tokens = request.num_tokens
    request.status = RequestStatus.RUNNING
    request.last_sched_seq = 1
    scheduler.requests[request.request_id] = request
    scheduler.running.append(request)

    paused = scheduler.pause_requests([request.request_id])
    assert not paused.done()
    continuation = Request(
        request_id=request.request_id,
        prompt_token_ids=[201, 202],
        sampling_params=SamplingParams(max_tokens=4),
        pooling_params=None,
        resumable=True,
    )
    scheduler.add_request(continuation)
    request.status = RequestStatus.FINISHED_STOPPED
    assert not scheduler._handle_stopped_request(request)
    assert request.status == RequestStatus.WAITING
    assert request.prompt_token_ids[-2:] == [201, 202]

    scheduler.processed_step_seq = 1
    scheduler._finalize_pending_pauses()
    scheduler._flush_request_operation_waiters()

    assert paused.done()
    assert request.status == RequestStatus.PAUSED
    assert scheduler._pause_resume_status[request.request_id] == RequestStatus.WAITING
    assert request not in scheduler.waiting.requests

    assert scheduler.resume_requests([request.request_id]).done()
    assert request.status == RequestStatus.WAITING
    assert scheduler.num_waiting_for_streaming_input == 0
    assert request in scheduler.waiting.requests
    assert request not in scheduler.running


def test_streaming_wait_uses_gpu_pause_even_when_cpu_connector_is_ready() -> None:
    connector = _CPUConnector()
    connector.ready = True
    scheduler = _scheduler(connector)
    request = Request(
        request_id="streaming-cpu-guard",
        prompt_token_ids=list(range(16)),
        sampling_params=SamplingParams(max_tokens=4),
        pooling_params=None,
        resumable=True,
    )
    request.streaming_queue = deque()
    request.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    request.num_computed_tokens = 16
    scheduler.requests[request.request_id] = request
    scheduler.skipped_waiting.add_request(request)
    scheduler.num_waiting_for_streaming_input = 1

    paused = scheduler.pause_requests([request.request_id])

    assert paused.done()
    assert request.status == RequestStatus.PAUSED
    assert scheduler.num_waiting_for_streaming_input == 0
    assert connector.pin_calls == []
    assert scheduler.kv_cache_manager.freed == []
    assert request.request_id not in scheduler._pause_cpu_pin_ids

    resumed = scheduler.resume_requests([request.request_id])

    assert resumed.done()
    assert request.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    assert scheduler.num_waiting_for_streaming_input == 1
    assert request in scheduler.skipped_waiting.requests


def test_any_speculative_runner_disables_prefix_sampler_bypass() -> None:
    scheduler = _scheduler()
    scheduled = {"prefix": 16}
    prefix_only = {"prefix"}

    assert Scheduler._supports_prefix_sampler_bypass(
        cast(Any, SimpleNamespace(speculative_config=None))
    )
    assert not Scheduler._supports_prefix_sampler_bypass(
        cast(Any, SimpleNamespace(speculative_config=object()))
    )
    assert scheduler._should_bypass_sampler_for_prefix_batch(
        scheduled, prefix_only
    )
    scheduler._prefix_pin_sampler_bypass_supported = False
    assert not scheduler._should_bypass_sampler_for_prefix_batch(
        scheduled, prefix_only
    )


def test_prefix_pin_never_allocates_speculative_lookahead() -> None:
    scheduler = _scheduler()
    scheduler._request_to_prefix_pin["prefix"] = "pin"

    assert scheduler._request_lookahead_tokens("prefix") == 0
    assert scheduler._request_lookahead_tokens("user") == 4
    assert scheduler._request_lookahead_tokens("user", load_kv_async=True) == 0


def test_snapshot_pin_hook_delegates_to_legacy_three_argument_override() -> None:
    request = _request("legacy-connector")
    calls: list[tuple[str, str, int]] = []

    def legacy_pin(pin_id: str, req: Request, num_tokens: int) -> bool:
        calls.append((pin_id, req.request_id, num_tokens))
        return True

    connector = cast(
        KVConnectorBase_V1,
        cast(object, SimpleNamespace(pin_request_kv=legacy_pin)),
    )
    blocks = cast(KVCacheBlocks, cast(object, SimpleNamespace()))

    assert KVConnectorBase_V1.pin_request_kv_with_snapshot(
        connector, "pin", request, 16, blocks
    )
    assert calls == [("pin", request.request_id, 16)]


def test_conditional_unpin_rejects_stale_pending_owner() -> None:
    scheduler = _scheduler()
    state = SimpleNamespace(request_id="replacement-owner")
    scheduler._prefix_pins["shared"] = state
    aborted: list[object] = []
    scheduler._abort_prefix_pin = lambda pin, _message: aborted.append(pin)

    assert not scheduler.unpin_prefix(
        "shared", expected_request_id="cancelled-owner"
    )
    assert scheduler._prefix_pins["shared"] is state
    assert aborted == []

    assert scheduler.unpin_prefix(
        "shared", expected_request_id="replacement-owner"
    )
    assert aborted == [state]


def test_conditional_unpin_rejects_stale_completed_owner() -> None:
    scheduler = _scheduler()
    scheduler._completed_prefix_pin_tiers["shared"] = "gpu"
    scheduler._completed_prefix_pin_request_ids["shared"] = "replacement-owner"
    unpinned: list[str] = []
    scheduler.kv_cache_manager.unpin_prefix = lambda pin_id: (
        unpinned.append(pin_id) or True
    )

    assert not scheduler.unpin_prefix(
        "shared", expected_request_id="cancelled-owner"
    )
    assert unpinned == []
    assert scheduler._completed_prefix_pin_tiers["shared"] == "gpu"

    assert scheduler.unpin_prefix(
        "shared", expected_request_id="replacement-owner"
    )
    assert unpinned == ["shared"]
    assert "shared" not in scheduler._completed_prefix_pin_request_ids
