# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import asyncio
import time
import uuid
from typing import Any, Literal

from vllm import AsyncEngineArgs, AsyncLLMEngine


PinLevel = Literal["gpu", "cpu"]


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke-test GPU and CPU prefix pin residency."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--level",
        choices=("gpu", "cpu", "both"),
        default="both",
    )
    parser.add_argument(
        "--prompt",
        default="This is a fixed prefix used to test vLLM prefix pinning. ",
    )
    parser.add_argument("--prompt-repetitions", type=int, default=128)
    parser.add_argument("--kv-offloading-size", type=float, default=32.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def _levels(value: str) -> tuple[PinLevel, ...]:
    if value == "both":
        return "gpu", "cpu"
    if value == "gpu":
        return ("gpu",)
    if value == "cpu":
        return ("cpu",)
    raise ValueError(f"unsupported pin level: {value!r}")


async def _pin_once(
    engine: AsyncLLMEngine,
    prompt: str,
    level: PinLevel,
) -> None:
    pin_id = f"prefix-pin-smoke-{level}-{uuid.uuid4().hex}"
    started = time.perf_counter()
    result: dict[str, Any] = await engine.pin_prefix(
        prompt,
        pin_id,
        level=level,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000

    try:
        if result.get("level") != level:
            raise RuntimeError(f"unexpected pin level result: {result!r}")
        if not isinstance(result.get("pinned_tokens"), int):
            raise RuntimeError(f"missing pinned token count: {result!r}")
        if result["pinned_tokens"] <= 0:
            raise RuntimeError(f"no tokens were pinned: {result!r}")
        if not result.get("block_ids"):
            raise RuntimeError(f"no KV blocks were pinned: {result!r}")

        print(
            f"PASS level={level} elapsed_ms={elapsed_ms:.1f} "
            f"pinned_tokens={result['pinned_tokens']} "
            f"pinned_blocks={len(result['block_ids'])}"
        )
    finally:
        if not await engine.unpin_prefix(pin_id):
            raise RuntimeError(f"failed to unpin prefix {pin_id!r}")


async def main(args: argparse.Namespace) -> None:
    if args.prompt_repetitions <= 0:
        raise ValueError("--prompt-repetitions must be positive")
    if args.kv_offloading_size <= 0:
        raise ValueError("--kv-offloading-size must be positive")

    engine = AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(
            model=args.model,
            trust_remote_code=args.trust_remote_code,
            enable_prefix_caching=True,
            kv_offloading_size=args.kv_offloading_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    )
    prompt = args.prompt * args.prompt_repetitions

    try:
        for level in _levels(args.level):
            await _pin_once(engine, prompt, level)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main(create_parser().parse_args()))
