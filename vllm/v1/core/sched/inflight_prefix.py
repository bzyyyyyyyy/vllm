# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashList,
    resolve_block_hashes,
)


@dataclass(slots=True)
class PrefixDependency:
    producer_id: str
    target_hash: BlockHash
    target_block_idx: int
    ready: bool = False


class InFlightPrefixTracker:
    """Coordinates durable prefix blocks that are not computed yet.

    Prefix hashes may be finer grained than scheduler blocks. Dependencies are
    intentionally tracked only through the physical block view so a waiter is
    never released before hybrid/Mamba state is durable.
    """

    def __init__(self, hash_block_size: int, durable_block_size: int) -> None:
        if durable_block_size % hash_block_size:
            raise ValueError(
                "durable block size must be a multiple of the hash block size"
            )
        self.hash_block_size = hash_block_size
        self.durable_block_size = durable_block_size
        self._owners: dict[BlockHash, str] = {}
        self._claims: dict[str, list[tuple[int, BlockHash]]] = {}
        self._dependencies: dict[str, PrefixDependency] = {}
        self._completed_tokens: dict[str, int] = {}
        self._prompt_tokens: dict[str, int] = {}

    def _resolve_durable_hashes(
        self, block_hashes: BlockHashList
    ) -> BlockHashList:
        return resolve_block_hashes(
            block_hashes,
            self.hash_block_size,
            self.durable_block_size,
        )

    def limit_cache_hit_length(
        self,
        request_id: str,
        block_hashes: BlockHashList,
        max_cache_hit_length: int,
    ) -> int:
        """Stop a cache lookup before a unit still owned by another request."""
        durable_hashes = self._resolve_durable_hashes(block_hashes)
        max_blocks = min(
            len(durable_hashes),
            (max_cache_hit_length + self.durable_block_size - 1)
            // self.durable_block_size,
        )
        for block_idx, block_hash in enumerate(durable_hashes[:max_blocks]):
            owner = self._owners.get(block_hash)
            if owner is not None and owner != request_id:
                return min(
                    max_cache_hit_length,
                    block_idx * self.durable_block_size,
                )
        return max_cache_hit_length

    def register(
        self,
        request_id: str,
        block_hashes: BlockHashList,
        ready_tokens: int,
        prompt_tokens: int,
        max_wait_tokens: int,
        *,
        wait_for_pending: bool,
    ) -> bool:
        """Claim missing prompt units and return whether the request must wait."""
        durable_hashes = self._resolve_durable_hashes(block_hashes)
        num_prompt_blocks = min(
            len(durable_hashes),
            prompt_tokens // self.durable_block_size,
        )
        first_missing_block = min(
            ready_tokens // self.durable_block_size,
            num_prompt_blocks,
        )
        dependency: PrefixDependency | None = None
        claims = self._claims.setdefault(request_id, [])
        claimed_hashes = {block_hash for _, block_hash in claims}

        for block_idx in range(first_missing_block, num_prompt_blocks):
            block_hash = durable_hashes[block_idx]
            owner = self._owners.get(block_hash)
            if owner is None:
                self._owners[block_hash] = request_id
                if block_hash not in claimed_hashes:
                    claims.append((block_idx, block_hash))
                    claimed_hashes.add(block_hash)
            else:
                if (
                    owner != request_id
                    and wait_for_pending
                    and (block_idx + 1) * self.durable_block_size
                    <= max_wait_tokens
                ):
                    dependency = PrefixDependency(
                        producer_id=owner,
                        target_hash=block_hash,
                        target_block_idx=block_idx,
                    )

        self._completed_tokens.setdefault(request_id, ready_tokens)
        self._prompt_tokens[request_id] = prompt_tokens
        current_dependency = self._dependencies.get(request_id)
        if dependency is not None:
            self._dependencies[request_id] = dependency
            return True
        return current_dependency is not None and not current_dependency.ready

    def update_from_output(self, request_id: str, num_scheduled_tokens: int) -> bool:
        """Publish newly completed units and report newly-ready waiters."""
        prompt_tokens = self._prompt_tokens.get(request_id)
        if prompt_tokens is None:
            return False

        completed_tokens = min(
            prompt_tokens,
            self._completed_tokens.get(request_id, 0) + num_scheduled_tokens,
        )
        self._completed_tokens[request_id] = completed_tokens
        completed_blocks = completed_tokens // self.durable_block_size

        claims = self._claims.get(request_id, [])
        remaining_claims: list[tuple[int, BlockHash]] = []
        for block_idx, block_hash in claims:
            if block_idx < completed_blocks:
                if self._owners.get(block_hash) == request_id:
                    del self._owners[block_hash]
            else:
                remaining_claims.append((block_idx, block_hash))
        if remaining_claims:
            self._claims[request_id] = remaining_claims
        else:
            self._claims.pop(request_id, None)

        waiter_became_ready = False
        for dependency in self._dependencies.values():
            if (
                not dependency.ready
                and dependency.producer_id == request_id
                and dependency.target_block_idx < completed_blocks
            ):
                dependency.ready = True
                waiter_became_ready = True
        return waiter_became_ready

    def has_unready_dependency(self, request_id: str) -> bool:
        dependency = self._dependencies.get(request_id)
        return dependency is not None and not dependency.ready

    def dependency_is_ready(self, request_id: str) -> bool:
        dependency = self._dependencies.get(request_id)
        return dependency is not None and dependency.ready

    def consume_ready_dependency(self, request_id: str) -> str | None:
        dependency = self._dependencies.get(request_id)
        if dependency is None or not dependency.ready:
            return None
        del self._dependencies[request_id]
        return dependency.producer_id

    def producer_needs_reservation(self, producer_id: str) -> bool:
        return any(
            dependency.ready and dependency.producer_id == producer_id
            for dependency in self._dependencies.values()
        )

    def completed_tokens(self, request_id: str) -> int | None:
        return self._completed_tokens.get(request_id)

    def remove_request(self, request_id: str) -> set[str]:
        """Remove a request and return producers whose reservation may release."""
        maybe_releasable: set[str] = set()
        dependency = self._dependencies.pop(request_id, None)
        if dependency is not None and dependency.ready:
            maybe_releasable.add(dependency.producer_id)

        for _, block_hash in self._claims.pop(request_id, []):
            if self._owners.get(block_hash) == request_id:
                del self._owners[block_hash]

        for waiter_id, waiter_dependency in list(self._dependencies.items()):
            if (
                waiter_dependency.producer_id == request_id
                and not waiter_dependency.ready
            ):
                del self._dependencies[waiter_id]

        self._completed_tokens.pop(request_id, None)
        self._prompt_tokens.pop(request_id, None)
        return maybe_releasable

    def has_state(self) -> bool:
        return bool(self._owners or self._claims or self._dependencies)

    def clear(self) -> None:
        self._owners.clear()
        self._claims.clear()
        self._dependencies.clear()
        self._completed_tokens.clear()
        self._prompt_tokens.clear()
