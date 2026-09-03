# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, call

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
    TransferJob,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
    RequestOffloadState,
    TransferJobStatus,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    ReqContext,
    make_offload_key,
)
from vllm.v1.outputs import KVConnectorOutput

pytestmark = [pytest.mark.cpu_test, pytest.mark.gcc_extension]


def _worker_without_device_state() -> OffloadingConnectorWorker:
    worker = OffloadingConnectorWorker.__new__(OffloadingConnectorWorker)
    worker.worker = MagicMock()
    worker._load_jobs = {}
    worker._load_job_block_ids = {}
    worker._failed_load_req_ids = set()
    worker._invalid_load_block_ids = set()
    worker._unsubmitted_store_jobs = []
    worker._connector_worker_meta = OffloadingWorkerMetadata()
    return worker


def test_failed_load_submission_reports_request_and_invalid_blocks() -> None:
    worker = _worker_without_device_state()
    offloading_worker = cast(MagicMock, worker.worker)
    offloading_worker.submit_load.return_value = False
    offloading_worker.get_finished.return_value = []
    metadata = OffloadingConnectorMetadata(
        load_jobs={
            7: TransferJob(
                req_id="failed-load",
                src_spec=MagicMock(),
                dst_spec=GPULoadStoreSpec(
                    [3, 0, 4],
                    group_sizes=[3],
                    block_indices=[0],
                ),
            )
        },
        store_jobs={},
    )

    worker.start_kv_transfers(metadata)
    _, finished_recving = worker.get_finished(set())

    assert finished_recving == {"failed-load"}
    assert worker.get_block_ids_with_load_errors() == {3, 4}
    assert worker.get_block_ids_with_load_errors() == set()
    result = worker.build_connector_worker_meta()
    assert result is not None
    assert result.failed_jobs == {7: 1}


def test_worker_metadata_aggregates_failed_jobs() -> None:
    first = OffloadingWorkerMetadata(failed_jobs={42: 1})
    second = OffloadingWorkerMetadata(failed_jobs={42: 1, 7: 1})

    result = first.aggregate(second)

    assert isinstance(result, OffloadingWorkerMetadata)
    assert result.failed_jobs == {42: 2, 7: 1}


def test_shared_failed_key_marks_all_overlapping_prefix_pins() -> None:
    scheduler = OffloadingConnectorScheduler.__new__(OffloadingConnectorScheduler)
    shared_key = make_offload_key(b"shared-key", 0)
    req_context = ReqContext(req_id="shared-source")
    manager = MagicMock()
    scheduler.manager = manager
    manager.get_prefix_pin_ids_for_keys.return_value = {
        "pin-a",
        "pin-b",
    }
    scheduler._connector_stats = MagicMock()
    scheduler._stale_job_threshold = 0
    scheduler._jobs = {
        9: TransferJobStatus(
            req_id="shared-source",
            pending_count=1,
            keys={shared_key},
            is_store=True,
        )
    }
    scheduler._req_status = {
        "shared-source": cast(
            RequestOffloadState,
            cast(
                object,
                SimpleNamespace(
                    req=SimpleNamespace(is_finished=lambda: False),
                    req_context=req_context,
                    transfer_jobs={9},
                    finished_signaled=False,
                ),
            ),
        )
    }
    scheduler._chunks_being_loaded = None
    scheduler._block_id_to_pending_jobs = {}
    scheduler._deferred_prefix_unpins = set()
    scheduler._prefix_pin_req_ids = {}
    scheduler._stable_source_pin_ids = {}
    scheduler._partial_pin_boundaries = {}
    scheduler._pending_partial_pin_req_ids = set()
    scheduler._failed_prefix_pins = {}
    connector_output = KVConnectorOutput(
        kv_connector_worker_meta=OffloadingWorkerMetadata(failed_jobs={9: 1})
    )

    scheduler.update_connector_output(connector_output)

    assert scheduler._failed_prefix_pins == {
        "pin-a": "CPU KV transfer failed",
        "pin-b": "CPU KV transfer failed",
    }
    assert manager.method_calls == [
        call.get_prefix_pin_ids_for_keys({shared_key}),
        call.complete_store({shared_key}, req_context, success=False),
    ]


def test_completed_hybrid_store_removes_overlapping_block_fence_once() -> None:
    scheduler = OffloadingConnectorScheduler.__new__(OffloadingConnectorScheduler)
    req_context = ReqContext(req_id="hybrid-source")
    stored_key = make_offload_key(b"hybrid-key", 0)
    manager = MagicMock()
    scheduler.manager = manager
    scheduler._connector_stats = MagicMock()
    scheduler._stale_job_threshold = 0
    scheduler._jobs = {
        9: TransferJobStatus(
            req_id="hybrid-source",
            pending_count=1,
            keys={stored_key},
            is_store=True,
            non_sliding_window_block_ids=[8, 9],
            sliding_window_block_ids=[8, 10],
        )
    }
    req_status = cast(
        RequestOffloadState,
        cast(
            object,
            SimpleNamespace(
                req=SimpleNamespace(is_finished=lambda: True),
                req_context=req_context,
                transfer_jobs={9},
                finished_signaled=False,
            ),
        ),
    )
    scheduler._req_status = {"hybrid-source": req_status}
    scheduler._chunks_being_loaded = None
    scheduler._block_id_to_pending_jobs = {
        8: {9},
        9: {9},
        10: {9},
    }
    scheduler._deferred_prefix_unpins = set()
    scheduler._prefix_pin_req_ids = {}
    scheduler._stable_source_pin_ids = {}
    scheduler._partial_pin_boundaries = {}
    scheduler._pending_partial_pin_req_ids = set()
    scheduler._failed_prefix_pins = {}
    connector_output = KVConnectorOutput(
        kv_connector_worker_meta=OffloadingWorkerMetadata(completed_jobs={9: 1})
    )

    scheduler.update_connector_output(connector_output)

    assert scheduler._block_id_to_pending_jobs == {}
    assert scheduler._jobs == {}
    assert req_status.transfer_jobs == set()
    manager.complete_store.assert_called_once_with(
        {stored_key}, req_context, success=True
    )


def test_partial_source_internal_hole_marks_all_overlapping_prefix_pins() -> None:
    scheduler = OffloadingConnectorScheduler.__new__(OffloadingConnectorScheduler)
    manager = MagicMock()
    scheduler.manager = manager
    scheduler.config = SimpleNamespace(
        tokens_per_hash=16,
        blocks_per_chunk=66,
        num_workers=1,
        kv_group_configs=(
            SimpleNamespace(
                group_idx=0,
                tokens_per_block=16,
                tokens_per_chunk=1056,
            ),
        ),
    )
    scheduler._pending_partial_pin_req_ids = {"partial-source"}
    scheduler._partial_pin_boundaries = {"partial-source": 64}
    scheduler._stable_source_pin_ids = {}
    scheduler._mamba_align_group_ids = set()
    scheduler._prefix_pin_req_ids = {"pin-a": "partial-source"}
    scheduler._failed_prefix_pins = {}
    scheduler._job_counter = 0
    scheduler._jobs = {}
    scheduler._block_id_to_pending_jobs = {}
    scheduler._current_batch_allocated_block_ids = set()
    scheduler._current_batch_jobs_to_flush = set()
    key_16 = make_offload_key(b"hash-16", 0)
    key_32 = make_offload_key(b"hash-32", 0)
    key_48 = make_offload_key(b"hash-48", 0)
    shared_key = make_offload_key(b"shared-key", 0)
    req_context = ReqContext(req_id="partial-source")
    scheduler._req_status = {
        "partial-source": cast(
            RequestOffloadState,
            cast(
                object,
                SimpleNamespace(
                    req_context=req_context,
                    group_states=(
                        SimpleNamespace(
                            hash_offload_keys=[key_16, key_32, key_48, shared_key],
                            # Leading SWA nulls are legal, but a hole after the
                            # first source cannot be represented by the
                            # contiguous GPULoadStoreSpec layout.
                            block_ids=[0, 51, 0, 52],
                        ),
                    ),
                    transfer_jobs=set(),
                ),
            ),
        )
    }
    store_spec = MagicMock()
    manager.prepare_store.return_value = SimpleNamespace(
        keys_to_store=[shared_key],
        store_spec=store_spec,
    )
    manager.get_prefix_pin_ids_for_keys.return_value = {
        "pin-a",
        "pin-b",
    }
    scheduler_output = cast(
        SchedulerOutput,
        cast(object, SimpleNamespace(partial_tail_offloads=None)),
    )

    jobs = scheduler._build_partial_pin_store_jobs(scheduler_output)

    assert jobs == {}
    assert scheduler._pending_partial_pin_req_ids == set()
    assert scheduler._failed_prefix_pins == {
        "pin-a": "partial CPU pin source block is unavailable",
        "pin-b": "partial CPU pin source block is unavailable",
    }
    assert manager.method_calls == [
        call.prepare_store([shared_key], req_context),
        call.get_prefix_pin_ids_for_keys({shared_key}),
        call.complete_store({shared_key}, req_context, success=False),
    ]
