# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.sched.inflight_prefix import InFlightPrefixTracker

pytestmark = pytest.mark.gcc_extension


def _hashes(count: int) -> list[BlockHash]:
    return [BlockHash(index.to_bytes(4, "big")) for index in range(count)]


def test_dependency_is_published_only_at_durable_block_boundary() -> None:
    tracker = InFlightPrefixTracker(hash_block_size=16, durable_block_size=64)
    hashes = _hashes(4)

    assert not tracker.register(
        "producer",
        hashes,
        ready_tokens=0,
        prompt_tokens=64,
        max_wait_tokens=64,
        wait_for_pending=True,
    )
    assert tracker.limit_cache_hit_length("consumer", hashes, 64) == 0
    assert tracker.register(
        "consumer",
        hashes,
        ready_tokens=0,
        prompt_tokens=64,
        max_wait_tokens=64,
        wait_for_pending=True,
    )

    assert not tracker.update_from_output("producer", 16)
    assert not tracker.update_from_output("producer", 16)
    assert not tracker.update_from_output("producer", 16)
    assert tracker.has_unready_dependency("consumer")
    assert tracker.update_from_output("producer", 16)
    assert tracker.dependency_is_ready("consumer")
    assert tracker.consume_ready_dependency("consumer") == "producer"


def test_partial_physical_block_never_becomes_inflight_dependency() -> None:
    tracker = InFlightPrefixTracker(hash_block_size=16, durable_block_size=64)
    hashes = _hashes(3)

    assert not tracker.register(
        "producer",
        hashes,
        ready_tokens=0,
        prompt_tokens=48,
        max_wait_tokens=48,
        wait_for_pending=True,
    )
    assert tracker.limit_cache_hit_length("consumer", hashes, 48) == 48
    assert not tracker.register(
        "consumer",
        hashes,
        ready_tokens=0,
        prompt_tokens=48,
        max_wait_tokens=48,
        wait_for_pending=True,
    )


def test_removing_producer_releases_unready_waiter() -> None:
    tracker = InFlightPrefixTracker(hash_block_size=16, durable_block_size=64)
    hashes = _hashes(4)
    tracker.register(
        "producer",
        hashes,
        ready_tokens=0,
        prompt_tokens=64,
        max_wait_tokens=64,
        wait_for_pending=True,
    )
    assert tracker.register(
        "consumer",
        hashes,
        ready_tokens=0,
        prompt_tokens=64,
        max_wait_tokens=64,
        wait_for_pending=True,
    )

    tracker.remove_request("producer")

    assert not tracker.has_unready_dependency("consumer")
