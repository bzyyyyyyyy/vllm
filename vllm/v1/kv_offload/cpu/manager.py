# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import Counter, OrderedDict
from collections.abc import Collection, Iterable

from typing_extensions import override

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.v1.kv_offload.base import (
    LoadStoreSpec,
    LookupResult,
    Medium,
    OffloadingEvent,
    OffloadingManager,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
)
from vllm.v1.kv_offload.cpu.common import (
    CPULoadStoreSpec,
    CPUOffloadingMetrics,
)
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy
from vllm.v1.kv_offload.cpu.policies.factory import CachePolicyFactory


class CPUOffloadingManager(OffloadingManager):
    """
    An OffloadingManager with a pluggable CachePolicy, resolved by name via
    CachePolicyFactory (built in: "lru", "arc"; external policies can either
    register their own or be loaded out-of-tree via cache_policy_module_path).

    The manager owns all shared logic: ref-counting, event emission,
    block pool management, and the prepare_store/complete_store skeletons.
    Policy-specific block organization and eviction decisions are delegated
    to the CachePolicy implementation.
    """

    def __init__(
        self,
        num_blocks: int,
        cache_policy: str = "lru",
        cache_policy_module_path: str | None = None,
        enable_events: bool = False,
        store_threshold: int = 1,
        max_tracker_size: int = 64_000,
    ):
        self.medium: Medium = Medium.CPU
        self._num_blocks: int = num_blocks
        self._num_allocated_blocks: int = 0
        self._free_list: list[int] = []
        self.events: list[OffloadingEvent] | None = [] if enable_events else None
        policy_cls = CachePolicyFactory.get_cache_policy_cls(
            cache_policy, cache_policy_module_path
        )
        self._policy: CachePolicy = policy_cls(cache_capacity=num_blocks)
        # Track the number of blocks in the cache that are evictable. i.e. ref_cnt 0.
        self._num_evictable_cache_blocks: int = 0
        # Track blocks with an in-flight store (ref_cnt -1, not yet completed).
        self._num_write_pending_blocks: int = 0

        self.store_threshold: int = store_threshold
        self.max_tracker_size: int = max_tracker_size
        self.stores_skipped_in_current_batch: int = 0
        self.allocation_sizes_in_current_batch: list[int] = []

        # Number of block references. It is ordered so can evict the LRU entry in O(1).
        self.counts: OrderedDict[OffloadKey, int] | None = (
            OrderedDict() if store_threshold >= 2 else None
        )

        # A hard pin owns a separate reference from transient load jobs.
        # Write-pending blocks use ref_cnt == -1, so references which become
        # active after the store finishes are tracked separately.
        self._pin_id_to_keys: dict[str, tuple[OffloadKey, ...]] = {}
        self._pin_ref_counts: Counter[OffloadKey] = Counter()
        self._reserved_store_keys: set[OffloadKey] = set()

    # --- block pool ---

    def _get_num_free_blocks(self) -> int:
        return len(self._free_list) + self._num_blocks - self._num_allocated_blocks

    def _allocate_blocks(self, keys: list[OffloadKey]) -> list[BlockStatus]:
        num_fresh = min(len(keys), self._num_blocks - self._num_allocated_blocks)
        num_reused = len(keys) - num_fresh
        assert len(self._free_list) >= num_reused

        # allocate fresh blocks
        blocks: list[BlockStatus] = []
        for _ in range(num_fresh):
            blocks.append(BlockStatus(self._num_allocated_blocks))
            self._num_allocated_blocks += 1

        # allocate reused blocks
        for _ in range(num_reused):
            blocks.append(BlockStatus(self._free_list.pop()))
        return blocks

    def _free_block(self, block: BlockStatus) -> None:
        self._free_list.append(block.block_id)

    def _get_load_store_spec(
        self,
        keys: Iterable[OffloadKey],
        blocks: Iterable[BlockStatus],
    ) -> CPULoadStoreSpec:
        return CPULoadStoreSpec([block.block_id for block in blocks])

    # --- OffloadingManager interface ---

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        if self.counts is not None:
            if key in self.counts:
                self.counts.move_to_end(key)
                self.counts[key] += 1
            else:
                if len(self.counts) >= self.max_tracker_size:
                    self.counts.popitem(last=False)
                self.counts[key] = 1
        block = self._policy.get(key)
        if block is None:
            return LookupResult.MISS
        if not block.is_ready:
            return LookupResult.HIT_PENDING
        return LookupResult.HIT

    @override
    def prepare_load(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> LoadStoreSpec:
        blocks = []
        for key in keys:
            block = self._policy.get(key)
            assert block is not None, f"Block {key!r} not found in cache"
            assert block.is_ready, f"Block {key!r} is not ready for reading"
            if block.ref_cnt == 0:
                self._policy.mark_non_evictable(key)
                self._num_evictable_cache_blocks -= 1  # ref_cnt 0 -> 1
                assert self._num_evictable_cache_blocks >= 0
            block.ref_cnt += 1
            blocks.append(block)
        return self._get_load_store_spec(keys, blocks)

    @override
    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        self._policy.touch(keys, req_context)

    @override
    def complete_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> None:
        for key in keys:
            block = self._policy.get(key)
            assert block is not None, f"Block {key!r} not found"
            assert block.ref_cnt > 0, f"Block {key!r} ref_cnt is already 0"
            block.ref_cnt -= 1
            if block.ref_cnt == 0:
                self._num_evictable_cache_blocks += 1  # ref_cnt 1 -> 0
                self._policy.mark_evictable(key)

    @override
    def prepare_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> PrepareStoreOutput | None:
        unique_keys = list(dict.fromkeys(keys))
        reserved_keys = self._reserved_store_keys.intersection(unique_keys)
        regular_keys = [key for key in unique_keys if key not in reserved_keys]
        if self.counts is not None:
            num_keys = len(regular_keys)
            regular_keys = [
                key
                for key in regular_keys
                if self.counts.get(key, 0) >= self.store_threshold
            ]
            self.stores_skipped_in_current_batch += num_keys - len(regular_keys)

        # Reserved keys already own write-pending slots and deliberately
        # bypass the ordinary store threshold. Regular ready/pending keys do
        # not need another store.
        regular_keys = [
            key for key in regular_keys if self._policy.get(key) is None
        ]
        keys_to_store = [
            key
            for key in unique_keys
            if key in reserved_keys or key in regular_keys
        ]

        if not keys_to_store:
            return PrepareStoreOutput(
                keys_to_store=[],
                store_spec=self._get_load_store_spec([], []),
                evicted_keys=[],
            )

        if regular_keys:
            self.allocation_sizes_in_current_batch.append(len(regular_keys))
        num_blocks_to_evict = len(regular_keys) - self._get_num_free_blocks()

        to_evict: list[OffloadKey] = []
        if num_blocks_to_evict > 0:
            if num_blocks_to_evict > self._num_evictable_cache_blocks:
                # Eviction will fail.
                return None
            # There is a still a chance for eviction failure as some of the
            # idle blocks might be in the protected list.

            # Blocks from the original input are excluded from eviction candidates:
            # a block that was already stored must remain in the cache after this call.
            protected = set(unique_keys)
            evicted = self._policy.evict(num_blocks_to_evict, protected)
            if evicted is None:
                return None

            # cache-policy removes only idle blocks.
            self._num_evictable_cache_blocks -= len(evicted)
            assert self._num_evictable_cache_blocks >= 0

            for key, block in evicted:
                self._free_block(block)
                to_evict.append(key)

        if to_evict and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    keys=to_evict,
                    medium=self.medium,
                    removed=True,
                )
            )

        new_blocks = self._allocate_blocks(regular_keys)
        assert len(new_blocks) == len(regular_keys), (
            "Block pool did not allocate the expected number of blocks"
        )

        for key, block in zip(regular_keys, new_blocks):
            self._policy.insert(key, block)
        self._num_write_pending_blocks += len(regular_keys)

        blocks = [self._policy.get(key) for key in keys_to_store]
        assert all(block is not None for block in blocks)
        pending_blocks = [block for block in blocks if block is not None]
        assert all(not block.is_ready for block in pending_blocks)
        self._reserved_store_keys.difference_update(keys_to_store)

        # build store specs for allocated blocks
        store_spec = self._get_load_store_spec(keys_to_store, pending_blocks)

        return PrepareStoreOutput(
            keys_to_store=keys_to_store,
            store_spec=store_spec,
            evicted_keys=to_evict,
        )

    @override
    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ) -> None:
        stored_keys: list[OffloadKey] = []

        if success:
            for key in keys:
                block = self._policy.get(key)
                if block is not None and not block.is_ready:
                    block.ref_cnt = self._pin_ref_counts.get(key, 0)
                    self._num_write_pending_blocks -= 1
                    if block.ref_cnt == 0:
                        self._num_evictable_cache_blocks += 1
                        self._policy.mark_evictable(key)
                    stored_keys.append(key)
        else:
            for key in keys:
                block = self._policy.get(key)
                if block is not None and not block.is_ready:
                    self._num_write_pending_blocks -= 1
                    self._policy.remove(key)
                    self._free_block(block)

        if stored_keys and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    keys=stored_keys,
                    medium=self.medium,
                    removed=False,
                )
            )

    @override
    def pin_prefix(
        self,
        pin_id: str,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> list[int]:
        if pin_id in self._pin_id_to_keys:
            raise ValueError(f"prefix pin already exists: {pin_id!r}")

        unique_keys = tuple(dict.fromkeys(keys))
        if not unique_keys:
            raise ValueError("CPU prefix pin contains no offloadable chunks")

        missing_keys = [key for key in unique_keys if self._policy.get(key) is None]
        if missing_keys:
            self.allocation_sizes_in_current_batch.append(len(missing_keys))
        num_blocks_to_evict = len(missing_keys) - self._get_num_free_blocks()
        to_evict: list[OffloadKey] = []
        if num_blocks_to_evict > 0:
            if num_blocks_to_evict > self._num_evictable_cache_blocks:
                raise RuntimeError(
                    "insufficient CPU KV cache capacity for hard prefix pin"
                )
            evicted = self._policy.evict(num_blocks_to_evict, set(unique_keys))
            if evicted is None:
                raise RuntimeError(
                    "insufficient CPU KV cache capacity for hard prefix pin"
                )
            self._num_evictable_cache_blocks -= len(evicted)
            assert self._num_evictable_cache_blocks >= 0
            for key, block in evicted:
                self._free_block(block)
                to_evict.append(key)

        new_blocks = self._allocate_blocks(missing_keys)
        for key, block in zip(missing_keys, new_blocks):
            self._policy.insert(key, block)
        self._num_write_pending_blocks += len(missing_keys)
        self._reserved_store_keys.update(missing_keys)

        self._pin_id_to_keys[pin_id] = unique_keys
        for key in unique_keys:
            block = self._policy.get(key)
            assert block is not None
            self._pin_ref_counts[key] += 1
            if block.is_ready:
                if block.ref_cnt == 0:
                    self._policy.mark_non_evictable(key)
                    self._num_evictable_cache_blocks -= 1
                    assert self._num_evictable_cache_blocks >= 0
                block.ref_cnt += 1

        if to_evict and self.events is not None:
            self.events.append(
                OffloadingEvent(keys=to_evict, medium=self.medium, removed=True)
            )

        return self.get_prefix_pin_block_ids(pin_id)

    @override
    def is_prefix_pin_ready(self, pin_id: str) -> bool:
        keys = self._pin_id_to_keys.get(pin_id)
        return keys is not None and all(
            (block := self._policy.get(key)) is not None and block.is_ready
            for key in keys
        )

    @override
    def get_prefix_pin_block_ids(self, pin_id: str) -> list[int]:
        keys = self._pin_id_to_keys.get(pin_id)
        if keys is None:
            raise KeyError(pin_id)
        blocks = [self._policy.get(key) for key in keys]
        if any(block is None for block in blocks):
            raise RuntimeError(f"prefix pin is unavailable: {pin_id!r}")
        return [block.block_id for block in blocks if block is not None]

    @override
    def unpin_prefix(self, pin_id: str) -> bool:
        keys = self._pin_id_to_keys.pop(pin_id, None)
        if keys is None:
            return False

        for key in keys:
            self._pin_ref_counts[key] -= 1
            if self._pin_ref_counts[key] == 0:
                del self._pin_ref_counts[key]

            block = self._policy.get(key)
            if block is None:
                continue
            if block.is_ready:
                assert block.ref_cnt > 0
                block.ref_cnt -= 1
                if block.ref_cnt == 0:
                    self._num_evictable_cache_blocks += 1
                    self._policy.mark_evictable(key)
            elif key in self._reserved_store_keys and key not in self._pin_ref_counts:
                self._reserved_store_keys.remove(key)
                self._num_write_pending_blocks -= 1
                self._policy.remove(key)
                self._free_block(block)
        return True

    @override
    def has_pinned_prefix(self, pin_id: str) -> bool:
        return pin_id in self._pin_id_to_keys

    @override
    def has_pinned_prefixes(self) -> bool:
        return bool(self._pin_id_to_keys)

    @override
    def is_prefix_key_pinned(self, key: OffloadKey) -> bool:
        return key in self._pin_ref_counts

    @override
    def get_prefix_pin_ids_for_keys(
        self, keys: Collection[OffloadKey]
    ) -> set[str]:
        key_set = set(keys)
        return {
            pin_id
            for pin_id, pin_keys in self._pin_id_to_keys.items()
            if not key_set.isdisjoint(pin_keys)
        }

    @override
    def reset_cache(self) -> None:
        if self.has_pinned_prefixes():
            raise RuntimeError(
                "cannot reset CPU KV cache while hard prefix pins are present"
            )
        # Clear ALL blocks unconditionally. The scheduler's _stale_job_threshold
        # guarantees that complete_load / complete_store are never called for
        # pre-reset jobs, so no lazy cleanup is needed. The scheduler also
        # flushes in-flight load job IDs to the workers before any new stores
        # can begin, preventing a cross-direction data race on reused offload block IDs.
        self._policy.clear()
        self._num_evictable_cache_blocks = 0
        self._num_write_pending_blocks = 0

        self._free_list.clear()
        self._num_allocated_blocks = 0
        self._pin_id_to_keys.clear()
        self._pin_ref_counts.clear()
        self._reserved_store_keys.clear()

    @override
    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()

    def get_stats(self) -> OffloadingConnectorStats | None:
        stats = OffloadingConnectorStats()

        # Compute cache usage.
        num_used = (
            self._num_allocated_blocks
            - len(self._free_list)
            - self._num_evictable_cache_blocks
        )
        usage = num_used / self._num_blocks if self._num_blocks > 0 else 0.0
        stats.set_gauge(CPUOffloadingMetrics.CPU_CACHE_USAGE_PERC, usage)

        for allocation_size in self.allocation_sizes_in_current_batch:
            stats.observe_histogram(
                CPUOffloadingMetrics.CPU_ALLOCATION_SIZE, allocation_size
            )
        self.allocation_sizes_in_current_batch.clear()

        write_usage = (
            self._num_write_pending_blocks / self._num_blocks
            if self._num_blocks > 0
            else 0.0
        )
        read_usage = max(usage - write_usage, 0.0)
        stats.set_gauge(CPUOffloadingMetrics.CPU_CACHE_WRITE_USAGE_PERC, write_usage)
        stats.set_gauge(CPUOffloadingMetrics.CPU_CACHE_READ_USAGE_PERC, read_usage)

        if self.store_threshold >= 2:
            stats.increase_counter(
                CPUOffloadingMetrics.STORES_SKIPPED,
                self.stores_skipped_in_current_batch,
            )
            self.stores_skipped_in_current_batch = 0

        return stats
