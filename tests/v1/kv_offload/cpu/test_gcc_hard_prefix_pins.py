# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadKey,
    ReqContext,
    make_offload_key,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

pytestmark = [pytest.mark.cpu_test, pytest.mark.gcc_extension]

_REQ_CONTEXT = ReqContext(req_id="gcc-hard-pin")


def _keys(*values: int) -> list[OffloadKey]:
    return [make_offload_key(str(value).encode(), 0) for value in values]


@pytest.mark.parametrize("cache_policy", ["lru", "arc"])
def test_hard_pin_reserves_pending_store_and_bypasses_threshold(
    cache_policy: str,
) -> None:
    manager = CPUOffloadingManager(
        num_blocks=2,
        cache_policy=cache_policy,
        store_threshold=2,
    )
    keys = _keys(1, 2)

    assert manager.pin_prefix("pin", keys, _REQ_CONTEXT) == [0, 1]
    assert not manager.is_prefix_pin_ready("pin")

    store = manager.prepare_store(keys, _REQ_CONTEXT)
    assert store is not None
    assert store.keys_to_store == keys
    manager.complete_store(keys, _REQ_CONTEXT)

    assert manager.is_prefix_pin_ready("pin")
    assert manager.get_prefix_pin_block_ids("pin") == [0, 1]
    assert manager.lookup(keys[0], _REQ_CONTEXT) is LookupResult.HIT
    with pytest.raises(RuntimeError, match="hard prefix pins"):
        manager.reset_cache()


def test_overlapping_pins_release_only_the_last_reference() -> None:
    manager = CPUOffloadingManager(num_blocks=1)
    key = _keys(1)
    store = manager.prepare_store(key, _REQ_CONTEXT)
    assert store is not None
    manager.complete_store(key, _REQ_CONTEXT)

    manager.pin_prefix("first", key, _REQ_CONTEXT)
    manager.pin_prefix("second", key, _REQ_CONTEXT)
    assert manager.unpin_prefix("first")
    assert manager.prepare_store(_keys(2), _REQ_CONTEXT) is None

    assert manager.unpin_prefix("second")
    assert manager.prepare_store(_keys(2), _REQ_CONTEXT) is not None


def test_pin_capacity_failure_is_atomic() -> None:
    manager = CPUOffloadingManager(num_blocks=2)
    manager.pin_prefix("first", _keys(1, 2), _REQ_CONTEXT)

    with pytest.raises(RuntimeError, match="insufficient CPU KV cache capacity"):
        manager.pin_prefix("second", _keys(3), _REQ_CONTEXT)

    assert not manager.has_pinned_prefix("second")
    assert manager.unpin_prefix("first")
    assert manager.pin_prefix("second", _keys(3), _REQ_CONTEXT) == [1]


def test_failed_reserved_store_reports_pin_not_ready_until_release() -> None:
    manager = CPUOffloadingManager(num_blocks=1)
    key = _keys(1)
    manager.pin_prefix("pin", key, _REQ_CONTEXT)
    store = manager.prepare_store(key, _REQ_CONTEXT)
    assert store is not None

    manager.complete_store(key, _REQ_CONTEXT, success=False)

    assert not manager.is_prefix_pin_ready("pin")
    with pytest.raises(RuntimeError, match="prefix pin is unavailable"):
        manager.get_prefix_pin_block_ids("pin")
    assert manager.unpin_prefix("pin")
    assert not manager.has_pinned_prefixes()
