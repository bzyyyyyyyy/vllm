# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
    RequestOffloadState,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_offload.base import (
    LookupResult,
    ReqContext,
    make_offload_key,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.request import Request

pytestmark = [pytest.mark.cpu_test, pytest.mark.gcc_extension]


def _request_state(**attributes: object) -> RequestOffloadState:
    attributes.setdefault("max_offload_tokens", None)
    req = attributes.get("req")
    req_context = attributes.get("req_context")
    if (
        req is not None
        and req_context is not None
        and not hasattr(req, "request_id")
    ):
        setattr(req, "request_id", getattr(req_context, "req_id"))
    return cast(
        RequestOffloadState,
        cast(object, SimpleNamespace(**attributes)),
    )


def _scheduler(*, is_eagle_group: bool = False) -> OffloadingConnectorScheduler:
    scheduler = OffloadingConnectorScheduler.__new__(OffloadingConnectorScheduler)
    scheduler.config = SimpleNamespace(
        tokens_per_hash=16,
        kv_group_configs=(
            SimpleNamespace(
                group_idx=0,
                tokens_per_block=16,
                tokens_per_chunk=1056,
                is_eagle_group=is_eagle_group,
                alignment_chunk_count=None,
                sliding_window_size_in_chunks=None,
            ),
        ),
    )
    scheduler._sliding_window_groups = ()
    scheduler._current_batch_allocated_block_ids = set()
    scheduler._current_batch_jobs_to_flush = set()
    scheduler._block_id_to_pending_jobs = {}
    scheduler._stable_source_pin_ids = {}
    scheduler.manager = MagicMock()
    return scheduler


def test_partial_pin_uses_exact_hash_boundary_below_physical_chunk() -> None:
    scheduler = _scheduler()
    key_16 = make_offload_key(b"hash-16", 0)
    key_32 = make_offload_key(b"hash-32", 0)
    group_state = SimpleNamespace(
        offload_keys=[],
        hash_offload_keys=[key_16, key_32],
    )
    req_status = _request_state(
        req=SimpleNamespace(num_prompt_tokens=32),
        group_states=(group_state,),
        update_offload_keys=lambda: None,
    )

    keys = scheduler._get_prefix_pin_keys(req_status, num_tokens=32)

    assert scheduler.get_prefix_pin_alignment() == 16
    assert keys == [key_32]


def test_partial_lookup_returns_sub_chunk_token_count() -> None:
    scheduler = _scheduler()
    manager = cast(MagicMock, scheduler.manager)
    key_16 = make_offload_key(b"hash-16", 0)
    key_32 = make_offload_key(b"hash-32", 0)
    manager.lookup.side_effect = lambda key, _: (
        LookupResult.HIT if key == key_32 else LookupResult.MISS
    )
    manager.is_prefix_key_pinned.return_value = True
    req_status = _request_state(
        req=SimpleNamespace(num_tokens=32),
        req_context=ReqContext(req_id="partial-lookup"),
        group_states=(
            SimpleNamespace(hash_offload_keys=[key_16, key_32]),
        ),
    )

    hit_tokens = scheduler._lookup_partial_prefix(
        req_status,
        num_computed_tokens=0,
        num_chunk_hit_tokens=0,
    )

    assert hit_tokens == 32


def test_partial_lookup_requires_hard_pinned_key() -> None:
    scheduler = _scheduler()
    manager = cast(MagicMock, scheduler.manager)
    key_16 = make_offload_key(b"hash-16", 0)
    key_32 = make_offload_key(b"hash-32", 0)
    req_context = ReqContext(req_id="hard-pin-only")
    req_status = _request_state(
        req=SimpleNamespace(num_tokens=32),
        req_context=req_context,
        group_states=(
            SimpleNamespace(hash_offload_keys=[key_16, key_32]),
        ),
    )
    manager.lookup.return_value = LookupResult.HIT
    manager.is_prefix_key_pinned.return_value = False

    unpinned_hit_tokens = scheduler._lookup_partial_prefix(
        req_status,
        num_computed_tokens=0,
        num_chunk_hit_tokens=0,
    )

    assert unpinned_hit_tokens == 0
    manager.lookup.assert_not_called()

    manager.is_prefix_key_pinned.side_effect = lambda key: key == key_32
    pinned_hit_tokens = scheduler._lookup_partial_prefix(
        req_status,
        num_computed_tokens=0,
        num_chunk_hit_tokens=0,
    )

    assert pinned_hit_tokens == 32
    manager.lookup.assert_called_once_with(key_32, req_context)


def test_eagle_group_does_not_extend_partial_lookup() -> None:
    scheduler = _scheduler(is_eagle_group=True)
    manager = cast(MagicMock, scheduler.manager)
    req_status = _request_state(
        req=SimpleNamespace(num_tokens=32),
        req_context=ReqContext(req_id="eagle-partial"),
        group_states=(
            SimpleNamespace(
                hash_offload_keys=[
                    make_offload_key(b"hash-16", 0),
                    make_offload_key(b"hash-32", 0),
                ]
            ),
        ),
    )

    hit_tokens = scheduler._lookup_partial_prefix(
        req_status,
        num_computed_tokens=0,
        num_chunk_hit_tokens=16,
    )

    assert hit_tokens == 16
    manager.is_prefix_key_pinned.assert_not_called()
    manager.lookup.assert_not_called()


@pytest.mark.parametrize("num_prompt_tokens", [32, 1056])
def test_eagle_cpu_prefix_pin_fails_before_reserving_any_keys(
    num_prompt_tokens: int,
) -> None:
    scheduler = _scheduler(is_eagle_group=True)
    scheduler.on_new_request = MagicMock()
    scheduler._prefix_pin_req_ids = {}
    scheduler._forced_store_req_ids = {}
    scheduler._partial_pin_boundaries = {}
    scheduler._pending_partial_pin_req_ids = set()
    scheduler._failed_prefix_pins = {}
    request = cast(
        Request,
        cast(
            object,
            SimpleNamespace(
                request_id="eagle-pin",
                num_prompt_tokens=num_prompt_tokens,
            ),
        ),
    )

    with pytest.raises(RuntimeError, match="CPU prefix pin does not support EAGLE"):
        scheduler.pin_prefix("pin", request)

    scheduler.on_new_request.assert_not_called()
    cast(MagicMock, scheduler.manager).pin_prefix.assert_not_called()
    assert scheduler._prefix_pin_req_ids == {}
    assert scheduler._forced_store_req_ids == {}
    assert scheduler._partial_pin_boundaries == {}
    assert scheduler._pending_partial_pin_req_ids == set()
    assert scheduler._failed_prefix_pins == {}


def test_zero_step_cpu_pin_records_authoritative_source_block_table() -> None:
    scheduler = _scheduler()
    manager = cast(MagicMock, scheduler.manager)
    manager.pin_prefix.return_value = [7, 8]
    scheduler.on_new_request = MagicMock()
    scheduler._prefix_pin_req_ids = {}
    scheduler._forced_store_req_ids = {}
    scheduler._stable_source_pin_ids = {}
    scheduler._partial_pin_boundaries = {}
    scheduler._pending_partial_pin_req_ids = set()
    scheduler._block_id_to_pending_jobs = {41: {9}}
    scheduler._sliding_window_groups = (0,)
    scheduler.config.blocks_per_chunk = 1
    scheduler.config.kv_group_configs[0].sliding_window_size_in_chunks = 1

    key_16 = make_offload_key(b"hash-16", 0)
    key_32 = make_offload_key(b"hash-32", 0)
    group_state = SimpleNamespace(
        offload_keys=[],
        hash_offload_keys=[key_16, key_32],
        block_ids=[],
        next_stored_chunk_idx=3,
    )
    request = cast(
        Request,
        cast(
            object,
            SimpleNamespace(
                request_id="zero-step-pin",
                num_prompt_tokens=32,
            ),
        ),
    )
    req_status = _request_state(
        req=request,
        req_context=ReqContext(req_id=request.request_id),
        offloading_context=SimpleNamespace(policy=None),
        group_states=(group_state,),
        update_offload_keys=lambda: None,
        max_offload_tokens=16,
    )
    scheduler._req_status = {request.request_id: req_status}
    source_block_ids = [41, 0, 43]
    blocks = MagicMock(spec=KVCacheBlocks)
    blocks.get_block_ids.return_value = (source_block_ids,)

    pinned_ids = scheduler.pin_request_kv("pin", request, 32, blocks)
    source_block_ids.append(44)
    empty_output = cast(
        SchedulerOutput,
        cast(
            object,
            SimpleNamespace(
                scheduled_new_reqs=[],
                scheduled_cached_reqs=SimpleNamespace(
                    req_ids=[], new_block_ids=[], resumed_req_ids=set()
                ),
            ),
        ),
    )
    scheduler._update_req_states(empty_output)

    assert pinned_ids == [7, 8]
    assert group_state.block_ids == [41, 0, 43]
    assert group_state.next_stored_chunk_idx == 0
    assert scheduler._current_batch_allocated_block_ids == set()
    assert scheduler._current_batch_jobs_to_flush == {9}
    assert scheduler._forced_store_req_ids == {request.request_id: 32}
    assert scheduler._stable_source_pin_ids == {request.request_id: "pin"}


def test_truncated_snapshot_is_rejected_before_reservation_or_mutation() -> None:
    scheduler = _scheduler()
    manager = cast(MagicMock, scheduler.manager)
    scheduler.on_new_request = MagicMock()
    scheduler._prefix_pin_req_ids = {}
    scheduler._forced_store_req_ids = {}
    scheduler._stable_source_pin_ids = {}
    scheduler._partial_pin_boundaries = {}
    scheduler._pending_partial_pin_req_ids = set()
    existing_group_state = SimpleNamespace(block_ids=[91, 92], next_stored_chunk_idx=2)
    request = cast(
        Request,
        cast(
            object,
            SimpleNamespace(
                request_id="truncated-source",
                num_prompt_tokens=32,
            ),
        ),
    )
    scheduler._req_status = {
        request.request_id: _request_state(group_states=(existing_group_state,))
    }
    blocks = MagicMock(spec=KVCacheBlocks)
    blocks.get_block_ids.return_value = ([41],)

    assert scheduler.pin_request_kv("pin", request, 32, blocks) == []

    scheduler.on_new_request.assert_not_called()
    manager.pin_prefix.assert_not_called()
    assert existing_group_state.block_ids == [91, 92]
    assert existing_group_state.next_stored_chunk_idx == 2
    assert scheduler._prefix_pin_req_ids == {}
    assert scheduler._forced_store_req_ids == {}
    assert scheduler._stable_source_pin_ids == {}
    assert scheduler._partial_pin_boundaries == {}
    assert scheduler._pending_partial_pin_req_ids == set()


def test_pin_owner_collision_does_not_mutate_request_source_table() -> None:
    scheduler = _scheduler()
    scheduler.on_new_request = MagicMock()
    scheduler._prefix_pin_req_ids = {"collision": "foreign-owner"}
    scheduler._forced_store_req_ids = {}
    scheduler._stable_source_pin_ids = {}
    scheduler._partial_pin_boundaries = {}
    scheduler._pending_partial_pin_req_ids = set()
    existing_group_state = SimpleNamespace(block_ids=[91], next_stored_chunk_idx=2)
    request = cast(
        Request,
        cast(
            object,
            SimpleNamespace(
                request_id="new-owner",
                num_prompt_tokens=32,
            ),
        ),
    )
    source_block_ids = [41, 42]
    blocks = MagicMock(spec=KVCacheBlocks)
    blocks.get_block_ids.return_value = (source_block_ids,)
    scheduler._req_status = {
        request.request_id: _request_state(group_states=(existing_group_state,))
    }

    with pytest.raises(ValueError, match="foreign-owner"):
        scheduler.pin_request_kv("collision", request, 32, blocks)

    scheduler.on_new_request.assert_not_called()
    blocks.get_block_ids.assert_not_called()
    assert existing_group_state.block_ids == [91]
    assert existing_group_state.next_stored_chunk_idx == 2
    assert scheduler._current_batch_allocated_block_ids == set()


def test_stable_mamba_snapshot_restores_subchunk_boundary() -> None:
    scheduler = _scheduler()
    scheduler._stable_source_pin_ids = {"mamba-small": "pin"}
    scheduler._lookup_groups = (0,)
    scheduler._sliding_window_groups = (0,)
    scheduler._mamba_align_size = 1056
    scheduler._chunks_being_loaded = None
    scheduler.config.kv_group_configs[0].sliding_window_size_in_chunks = 1
    key_16 = make_offload_key(b"hash-16", 0)
    key_32 = make_offload_key(b"hash-32", 0)
    req_status = _request_state(
        req=SimpleNamespace(request_id="mamba-small", num_tokens=32),
        req_context=ReqContext(req_id="mamba-small"),
        num_locally_computed_tokens=0,
        group_states=(
            SimpleNamespace(
                offload_keys=[],
                hash_offload_keys=[key_16, key_32],
            ),
        ),
    )
    manager = cast(MagicMock, scheduler.manager)
    manager.is_prefix_key_pinned.return_value = True
    manager.lookup.return_value = LookupResult.HIT

    chunk_hit = scheduler._lookup(req_status)
    exact_hit = scheduler._lookup_partial_prefix(req_status, 0, chunk_hit or 0)

    assert chunk_hit == 0
    assert exact_hit == 32


def test_stable_mamba_snapshot_bypasses_ordinary_physical_chunk_cap() -> None:
    scheduler = _scheduler()
    scheduler._stable_source_pin_ids = {"mamba-full": "pin"}
    scheduler._lookup_groups = (0,)
    scheduler._sliding_window_groups = (0,)
    scheduler._mamba_align_size = 1056
    scheduler._chunks_being_loaded = None
    scheduler.config.kv_group_configs[0].sliding_window_size_in_chunks = 1
    key_1056 = make_offload_key(b"hash-1056", 0)
    key_2112 = make_offload_key(b"hash-2112", 0)
    req_status = _request_state(
        req=SimpleNamespace(request_id="mamba-full", num_tokens=2112),
        req_context=ReqContext(req_id="mamba-full"),
        num_locally_computed_tokens=0,
        group_states=(
            SimpleNamespace(
                offload_keys=[key_1056, key_2112],
                hash_offload_keys=[],
            ),
        ),
    )
    cast(MagicMock, scheduler.manager).lookup.return_value = LookupResult.HIT

    assert scheduler._lookup(req_status) == 2112


def test_forced_store_missing_source_fails_pin_and_releases_reservation() -> None:
    scheduler = OffloadingConnectorScheduler.__new__(OffloadingConnectorScheduler)
    scheduler.config = SimpleNamespace(
        blocks_per_chunk=1,
        num_workers=1,
        kv_group_configs=(
            SimpleNamespace(
                alignment_chunk_count=None,
                is_eagle_group=False,
                sliding_window_size_in_chunks=None,
                tokens_per_chunk=16,
            ),
        ),
    )
    manager = CPUOffloadingManager(num_blocks=1)
    scheduler.manager = manager
    scheduler._connector_stats = MagicMock()
    scheduler._forced_store_req_ids = {"forced-source": 16}
    scheduler._prefix_pin_req_ids = {"pin": "forced-source"}
    scheduler._failed_prefix_pins = {}
    scheduler._pending_partial_pin_req_ids = {"forced-source"}
    key = make_offload_key(b"forced-key", 0)
    replacement_key = make_offload_key(b"replacement-key", 0)
    req_context = ReqContext(req_id="forced-source")
    manager.pin_prefix("pin", [key], req_context)
    group_state = SimpleNamespace(
        offload_keys=[key],
        block_ids=[0],
        next_stored_chunk_idx=0,
    )

    def advance_stored_idx(_num_tokens: int) -> None:
        group_state.next_stored_chunk_idx = 1

    req_status = _request_state(
        req=SimpleNamespace(num_tokens=16),
        req_context=req_context,
        group_states=(group_state,),
        storable_chunks=lambda *_args: 1,
        advance_stored_idx=advance_stored_idx,
        transfer_jobs=set(),
    )
    scheduler._req_status = {"forced-source": req_status}
    scheduler_output = cast(
        SchedulerOutput,
        cast(
            object,
            SimpleNamespace(num_scheduled_tokens={}, finished_req_ids=None),
        ),
    )

    assert scheduler._build_store_jobs(scheduler_output) == {}
    assert scheduler.get_prefix_pin_error("pin") == (
        "forced CPU pin source block is unavailable"
    )
    assert scheduler._pending_partial_pin_req_ids == set()
    assert not manager.is_prefix_pin_ready("pin")
    assert group_state.next_stored_chunk_idx == 1

    assert manager.unpin_prefix("pin")
    assert manager.pin_prefix("replacement", [replacement_key], req_context) == [0]


def test_forced_store_allocation_failure_disables_later_ordinary_store() -> None:
    scheduler = OffloadingConnectorScheduler.__new__(OffloadingConnectorScheduler)
    scheduler.config = SimpleNamespace(
        blocks_per_chunk=1,
        num_workers=1,
        offload_prompt_only=False,
        kv_group_configs=(
            SimpleNamespace(
                alignment_chunk_count=None,
                is_eagle_group=False,
                sliding_window_size_in_chunks=None,
                tokens_per_chunk=16,
            ),
        ),
    )
    manager = MagicMock()
    manager.prepare_store.return_value = None
    manager.get_prefix_pin_ids_for_keys.return_value = {"pin"}
    scheduler.manager = manager
    scheduler._connector_stats = MagicMock()
    scheduler._forced_store_req_ids = {"forced-allocation": 16}
    scheduler._prefix_pin_req_ids = {"pin": "forced-allocation"}
    scheduler._failed_prefix_pins = {}
    scheduler._pending_partial_pin_req_ids = {"forced-allocation"}
    key = make_offload_key(b"forced-allocation-key", 0)
    group_state = SimpleNamespace(
        offload_keys=[key],
        block_ids=[41],
        next_stored_chunk_idx=0,
    )

    def advance_stored_idx(_num_tokens: int) -> None:
        group_state.next_stored_chunk_idx = 1

    req_status = _request_state(
        req=SimpleNamespace(
            status=None,
            num_tokens=16,
            is_finished=lambda: True,
        ),
        req_context=ReqContext(req_id="forced-allocation"),
        group_states=(group_state,),
        storable_chunks=lambda *_args: 1,
        advance_stored_idx=advance_stored_idx,
        transfer_jobs=set(),
    )
    scheduler._req_status = {"forced-allocation": req_status}
    empty_output = cast(
        SchedulerOutput,
        cast(
            object,
            SimpleNamespace(num_scheduled_tokens={}, finished_req_ids=None),
        ),
    )

    assert scheduler._build_store_jobs(empty_output) == {}
    assert group_state.next_stored_chunk_idx == 1
    assert scheduler.get_prefix_pin_error("pin") == (
        "forced CPU pin store could not allocate storage"
    )

    manager.prepare_store.reset_mock()
    finished_output = cast(
        SchedulerOutput,
        cast(
            object,
            SimpleNamespace(
                num_scheduled_tokens={},
                finished_req_ids={"forced-allocation"},
            ),
        ),
    )
    assert scheduler._build_store_jobs(finished_output) == {}
    manager.prepare_store.assert_not_called()


def test_forced_full_store_internal_source_hole_fails_closed() -> None:
    scheduler = OffloadingConnectorScheduler.__new__(OffloadingConnectorScheduler)
    scheduler.config = SimpleNamespace(
        blocks_per_chunk=3,
        num_workers=1,
        kv_group_configs=(
            SimpleNamespace(
                alignment_chunk_count=None,
                is_eagle_group=False,
                sliding_window_size_in_chunks=None,
                tokens_per_chunk=48,
            ),
        ),
    )
    manager = MagicMock()
    key = make_offload_key(b"forced-internal-hole", 0)
    manager.prepare_store.return_value = SimpleNamespace(
        keys_to_store=[key],
        store_spec=MagicMock(),
    )
    manager.get_prefix_pin_ids_for_keys.return_value = {"pin"}
    scheduler.manager = manager
    scheduler._connector_stats = MagicMock()
    scheduler._forced_store_req_ids = {"forced-hole": 48}
    scheduler._prefix_pin_req_ids = {"pin": "forced-hole"}
    scheduler._failed_prefix_pins = {}
    scheduler._pending_partial_pin_req_ids = {"forced-hole"}
    group_state = SimpleNamespace(
        offload_keys=[key],
        block_ids=[41, 0, 42],
        next_stored_chunk_idx=0,
    )

    def advance_stored_idx(_num_tokens: int) -> None:
        group_state.next_stored_chunk_idx = 1

    req_status = _request_state(
        req=SimpleNamespace(num_tokens=48),
        req_context=ReqContext(req_id="forced-hole"),
        group_states=(group_state,),
        storable_chunks=lambda *_args: 1,
        advance_stored_idx=advance_stored_idx,
        transfer_jobs=set(),
    )
    scheduler._req_status = {"forced-hole": req_status}
    scheduler_output = cast(
        SchedulerOutput,
        cast(
            object,
            SimpleNamespace(num_scheduled_tokens={}, finished_req_ids=None),
        ),
    )

    assert scheduler._build_store_jobs(scheduler_output) == {}
    assert scheduler.get_prefix_pin_error("pin") == (
        "forced CPU pin source block layout is non-contiguous"
    )
    assert scheduler._pending_partial_pin_req_ids == set()
    assert group_state.next_stored_chunk_idx == 1
    manager.complete_store.assert_called_once_with(
        {key}, req_status.req_context, success=False
    )


def test_partial_store_flushes_source_reallocated_in_same_step() -> None:
    scheduler = _scheduler()
    manager = cast(MagicMock, scheduler.manager)
    scheduler.config.blocks_per_chunk = 66
    scheduler.config.num_workers = 1
    scheduler.config.kv_group_configs[0].tokens_per_block = 16
    scheduler._pending_partial_pin_req_ids = {"partial-store"}
    scheduler._partial_pin_boundaries = {"partial-store": 32}
    scheduler._stable_source_pin_ids = {}
    scheduler._mamba_align_group_ids = set()
    scheduler._prefix_pin_req_ids = {"pin": "partial-store"}
    scheduler._failed_prefix_pins = {}
    scheduler._job_counter = 0
    scheduler._jobs = {}
    scheduler._block_id_to_pending_jobs = {}
    scheduler._current_batch_allocated_block_ids = {42}
    scheduler._current_batch_jobs_to_flush = set()
    key_16 = make_offload_key(b"hash-16", 0)
    key_32 = make_offload_key(b"hash-32", 0)
    req_status = _request_state(
        req_context=ReqContext(req_id="partial-store"),
        group_states=(
            SimpleNamespace(
                hash_offload_keys=[key_16, key_32],
                block_ids=[41, 42],
            ),
        ),
        transfer_jobs=set(),
    )
    scheduler._req_status = {"partial-store": req_status}
    manager.prepare_store.return_value = SimpleNamespace(
        keys_to_store=[key_32],
        store_spec=MagicMock(),
    )
    scheduler_output = cast(
        SchedulerOutput,
        cast(object, SimpleNamespace(partial_tail_offloads=None)),
    )

    jobs = scheduler._build_partial_pin_store_jobs(scheduler_output)

    assert set(jobs) == {0}
    assert scheduler._current_batch_jobs_to_flush == {0}
    assert scheduler._block_id_to_pending_jobs == {41: {0}, 42: {0}}
