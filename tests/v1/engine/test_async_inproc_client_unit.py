# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Model-free tests for the async in-process EngineCore bridge."""

import asyncio
import inspect
import queue
import sys
import threading
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import Future
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

import vllm.distributed.elastic_ep.elastic_execute as elastic_execute_module
import vllm.distributed.elastic_ep.elastic_state as elastic_state_module
import vllm.v1.engine.async_llm as async_llm_module
import vllm.v1.engine.coordinator as coordinator_module
import vllm.v1.engine.core as core_module
import vllm.v1.engine.core_client as core_client_module
import vllm.v1.engine.utils as engine_utils_module
import vllm.v1.worker.gpu_model_runner as gpu_model_runner_module
import vllm.v1.worker.gpu_worker as gpu_worker_module
from vllm.config import ParallelConfig, VllmConfig
from vllm.engine.protocol import EngineClient
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequest,
    EngineCoreRequestType,
    FinishReason,
    PrefixPinResult,
    UtilityOutput,
    UtilityResult,
)
from vllm.v1.core.sched.interface import PauseState, SchedulerInterface
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import DPEngineCoreProc, EngineCore, EngineCoreProc
from vllm.v1.engine.core_client import (
    AsyncInprocClient,
    DPLBAsyncMPClient,
    EngineCoreClient,
    InprocBackgroundResources,
)
# noinspection PyProtectedMember
from vllm.v1.engine.core_client import (
    _copy_exception_without_traceback as copy_exception_without_traceback,
)
from vllm.v1.engine.exceptions import EngineDeadError
from vllm.v1.executor.abstract import Executor
from vllm.v1.executor.uniproc_executor import UniProcExecutor
from vllm.v1.request import RequestStatus
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from vllm.v1.worker.sentinel.gpu_worker_sentinel import WorkerSentinel
from vllm.v1.worker.worker_base import WorkerWrapperBase

pytestmark = pytest.mark.gcc_extension


@dataclass
class _CustomUtilityValue:
    values: list[int]


def test_worker_sentinel_reads_elastic_dp_coordinates_dynamically() -> None:
    parallel_config = SimpleNamespace(
        data_parallel_rank=1,
        data_parallel_size=2,
        data_parallel_master_ip="10.0.0.1",
    )
    sentinel = WorkerSentinel.__new__(WorkerSentinel)
    sentinel.worker = cast(Any, SimpleNamespace(parallel_config=parallel_config))

    assert (sentinel.dp_rank, sentinel.dp_size) == (1, 2)
    assert sentinel.data_parallel_master_ip == "10.0.0.1"

    parallel_config.data_parallel_rank = 3
    parallel_config.data_parallel_size = 4
    parallel_config.data_parallel_master_ip = "10.0.0.2"

    assert (sentinel.dp_rank, sentinel.dp_size) == (3, 4)
    assert sentinel.data_parallel_master_ip == "10.0.0.2"


def test_elastic_ep_refreshes_async_fault_probe_for_new_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_runner = SimpleNamespace(
        model_config=SimpleNamespace(is_moe=True),
        check_ep_fault=False,
    )
    executor = SimpleNamespace(
        worker=SimpleNamespace(model_runner=model_runner)
    )
    monkeypatch.setattr(
        elastic_execute_module,
        "get_ep_all2all_manager",
        lambda: SimpleNamespace(support_fault_tolerance=True),
    )

    elastic_execute_module.ElasticEPScalingExecutor._refresh_ep_fault_detection(
        cast(Any, executor), 4
    )
    assert model_runner.check_ep_fault

    elastic_execute_module.ElasticEPScalingExecutor._refresh_ep_fault_detection(
        cast(Any, executor), 1
    )
    assert not model_runner.check_ep_fault


def test_prefix_controls_do_not_expand_engine_client_abstract_contract() -> None:
    extension_methods = {
        "pin_prefix",
        "unpin_prefix",
        "pause_prefix",
        "resume_prefix",
        "pause",
        "resume",
    }

    assert extension_methods.isdisjoint(EngineClient.__abstractmethods__)


def test_prefix_controls_do_not_expand_scheduler_abstract_contract() -> None:
    extension_methods = {
        "pin_prefix",
        "unpin_prefix",
        "pause_prefix",
        "resume_prefix",
        "pause_requests",
        "resume_requests",
        "get_paused_request_ids",
    }

    assert extension_methods.isdisjoint(SchedulerInterface.__abstractmethods__)
    assert SchedulerInterface.get_paused_request_ids(cast(Any, object())) == ()


def test_gcc_request_states_preserve_upstream_serialized_values() -> None:
    assert [
        RequestStatus.WAITING,
        RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR,
        RequestStatus.WAITING_FOR_REMOTE_KVS,
        RequestStatus.WAITING_FOR_STREAMING_REQ,
        RequestStatus.RUNNING,
        RequestStatus.PREEMPTED,
        RequestStatus.FINISHED_STOPPED,
        RequestStatus.FINISHED_LENGTH_CAPPED,
        RequestStatus.FINISHED_ABORTED,
        RequestStatus.FINISHED_IGNORED,
        RequestStatus.FINISHED_ERROR,
        RequestStatus.FINISHED_REPETITION,
    ] == list(range(1, 13))
    assert not RequestStatus.is_finished(RequestStatus.WAITING_FOR_PREFIX)
    assert not RequestStatus.is_finished(RequestStatus.PAUSED)


class _FakeEngineCore:
    def __init__(self) -> None:
        self.input_queue: queue.Queue[tuple[EngineCoreRequestType, Any]] = queue.Queue()
        self.aborts_queue: queue.Queue[list[str]] = queue.Queue()
        self.shutdown_state: Any = None


class _RecordingEngineCoreClient(EngineCoreClient):
    def __init__(self, responses: dict[str, Any]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.client_index = 3

    def shutdown(self, timeout: float | None = None) -> None:
        pass

    async def call_utility_async(self, method: str, *args: Any) -> Any:
        self.calls.append((method, args))
        return self.responses.get(method)


def _config(**parallel_overrides: Any) -> VllmConfig:
    parallel = {
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "data_parallel_size": 1,
        "data_parallel_size_local": 1,
        "data_parallel_rank": 0,
        "data_parallel_index": 0,
        "nnodes": 1,
        "_api_process_count": 1,
        "data_parallel_external_lb": False,
        "data_parallel_hybrid_lb": False,
        "enable_elastic_ep": False,
        "enable_fault_tolerance": False,
    }
    parallel.update(parallel_overrides)
    return cast(
        VllmConfig,
        cast(
            object,
            SimpleNamespace(parallel_config=SimpleNamespace(**parallel)),
        ),
    )


def _run_fake_owner(
    resources: InprocBackgroundResources,
    _vllm_config: Any,
    _executor_class: Any,
    _log_stats: bool,
) -> None:
    core = _FakeEngineCore()
    weight_version = "default"
    resources.attach_engine(core)  # type: ignore[arg-type]
    resources.startup_future.set_result(None)
    try:
        while True:
            request_type, request = core.input_queue.get()
            if request_type is EngineCoreRequestType.WAKEUP:
                break
            if request_type is EngineCoreRequestType.UTILITY:
                _client_index, call_id, method, args = request
                if method == "set_weight_version":
                    weight_version = args[0]
                    result = None
                elif method == "get_weight_version":
                    result = weight_version
                else:
                    raise AssertionError(f"unexpected utility method: {method}")
                resources.publish_output(
                    EngineCoreOutputs(
                        utility_output=UtilityOutput(
                            call_id=call_id,
                            result=UtilityResult(result),
                        )
                    )
                )
    finally:
        resources.mark_stopped()


def _output(request_id: str) -> EngineCoreOutputs:
    return EngineCoreOutputs(
        outputs=[EngineCoreOutput(request_id=request_id, new_token_ids=[1])]
    )


def _captured_error() -> RuntimeError:
    try:
        raise RuntimeError("detached failure")
    except RuntimeError as error:
        return error


def _request(request_id: str, data_parallel_rank: int) -> EngineCoreRequest:
    return cast(
        EngineCoreRequest,
        cast(
            object,
            SimpleNamespace(
                request_id=request_id,
                data_parallel_rank=data_parallel_rank,
                pooling_params=None,
            ),
        ),
    )


def _dplb_client() -> DPLBAsyncMPClient:
    client = DPLBAsyncMPClient.__new__(DPLBAsyncMPClient)
    client.resources = SimpleNamespace(engine_dead=False)
    client.client_count = 1
    client.client_index = 0
    client.core_engines = [b"engine-0", b"engine-1"]
    client.core_engine = client.core_engines[0]
    client.lb_engines = [[0, 0, 0.0], [0, 0, 0.0]]
    client.eng_start_index = 0
    client.reqs_in_flight = {}
    client.prefix_pins = {}
    client.engine_inflight = Counter()
    return client


def test_exception_copy_drops_traceback_context_and_cause() -> None:
    source = _captured_error()
    detached = copy_exception_without_traceback(source)

    assert detached is not source
    assert type(detached) is type(source)
    assert detached.args == source.args
    assert detached.__traceback__ is None
    assert detached.__context__ is None
    assert detached.__cause__ is None


@pytest.mark.asyncio
async def test_output_and_utility_bridge_preserves_order() -> None:
    resources = InprocBackgroundResources()
    resources.attach_engine(_FakeEngineCore())  # type: ignore[arg-type]

    pending = _output("pending")
    resources.publish_output(pending)
    resources.bind_loop(asyncio.get_running_loop())
    assert await asyncio.wait_for(resources.outputs_queue.get(), 1.0) is pending

    utility_future = asyncio.get_running_loop().create_future()
    resources.register_utility(7, utility_future)
    utility_output = EngineCoreOutputs(
        utility_output=UtilityOutput(call_id=7, result=UtilityResult("result"))
    )
    utility_thread = threading.Thread(
        target=resources.publish_output, args=(utility_output,)
    )
    utility_thread.start()
    utility_thread.join()
    assert await asyncio.wait_for(utility_future, 1.0) == "result"

    live = _output("live")
    output_thread = threading.Thread(target=resources.publish_output, args=(live,))
    output_thread.start()
    output_thread.join()
    assert await asyncio.wait_for(resources.outputs_queue.get(), 1.0) is live


@pytest.mark.asyncio
async def test_inproc_utility_output_matches_mp_serialization_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    resources = InprocBackgroundResources()
    resources.attach_engine(_FakeEngineCore())  # type: ignore[arg-type]
    resources.bind_loop(asyncio.get_running_loop())
    output_queue = core_client_module._InprocEngineOutputQueue(resources)

    owner_custom = _CustomUtilityValue(values=[3, 4])
    owner_result = {
        "plain": [1, 2],
        "custom": [owner_custom],
    }
    utility_output = EngineCoreOutputs(
        utility_output=UtilityOutput(
            call_id=13,
            result=UtilityResult(owner_result),
        )
    )
    mp_output = MsgpackDecoder(EngineCoreOutputs, share_mem=False).decode(
        MsgpackEncoder().encode(utility_output)
    )
    assert mp_output.utility_output is not None
    assert mp_output.utility_output.result is not None
    mp_result = mp_output.utility_output.result.result

    utility_future = asyncio.get_running_loop().create_future()
    resources.register_utility(13, utility_future)
    output_thread = threading.Thread(
        target=output_queue.put_nowait, args=((0, utility_output),)
    )
    output_thread.start()
    output_thread.join()

    # Mutating owner-side containers before loop delivery cannot leak across
    # the same serialization boundary used by AsyncMPClient.
    owner_result["plain"].append(99)
    owner_custom.values.append(99)
    frontend_result = await asyncio.wait_for(utility_future, 1.0)
    assert frontend_result == mp_result
    assert frontend_result is not owner_result
    assert frontend_result["plain"] is not owner_result["plain"]
    frontend_custom = frontend_result["custom"][0]
    assert isinstance(frontend_custom, _CustomUtilityValue)
    assert frontend_custom is not owner_custom
    assert frontend_custom.values is not owner_custom.values
    assert frontend_result["plain"] == [1, 2]
    assert frontend_custom.values == [3, 4]

    # Ordinary token outputs stay on the allocation-free direct path.
    ordinary = _output("ordinary")
    output_queue.put_nowait((0, ordinary))
    assert await asyncio.wait_for(resources.outputs_queue.get(), 1.0) is ordinary


def test_abort_is_visible_to_eager_and_fifo_queues() -> None:
    resources = InprocBackgroundResources()
    core = _FakeEngineCore()
    resources.attach_engine(core)  # type: ignore[arg-type]

    resources.enqueue_abort(["request-1"])

    assert core.aborts_queue.get_nowait() == ["request-1"]
    assert core.input_queue.get_nowait() == (
        EngineCoreRequestType.ABORT,
        ["request-1"],
    )


@pytest.mark.asyncio
async def test_abort_request_ids_restore_mutable_ownership_boundary() -> None:
    resources = InprocBackgroundResources()
    core = _FakeEngineCore()
    resources.attach_engine(core)  # type: ignore[arg-type]
    client = AsyncInprocClient.__new__(AsyncInprocClient)
    client.resources = resources
    frontend_request_ids = ["request-1"]

    await client.abort_requests_async(frontend_request_ids)
    frontend_request_ids.append("request-2")

    eager_request_ids = core.aborts_queue.get_nowait()
    request_type, fifo_request_ids = core.input_queue.get_nowait()
    assert request_type is EngineCoreRequestType.ABORT
    assert eager_request_ids is fifo_request_ids
    assert eager_request_ids is not frontend_request_ids
    assert eager_request_ids == ["request-1"]


@pytest.mark.asyncio
async def test_add_request_restores_mutable_ownership_boundary() -> None:
    resources = InprocBackgroundResources()
    core = _FakeEngineCore()
    resources.attach_engine(core)  # type: ignore[arg-type]
    client = AsyncInprocClient.__new__(AsyncInprocClient)
    client.resources = resources
    client.client_index = 0
    client._request_encoder = MsgpackEncoder()
    client._request_decoder = MsgpackDecoder(EngineCoreRequest, share_mem=False)
    frontend_tokens = [1, 2]
    frontend_embeds = torch.tensor([[1.0], [2.0]])
    frontend_reasoning = {"state": ["frontend"]}
    request = EngineCoreRequest(
        request_id="streaming-request",
        prompt_token_ids=frontend_tokens,
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        prompt_embeds=frontend_embeds,
        resumable=True,
        reasoning_parser_kwargs=frontend_reasoning,
    )

    await client.add_request_async(request)
    request_type, owner_request = core.input_queue.get_nowait()
    assert request_type is EngineCoreRequestType.ADD
    assert owner_request is not request
    assert owner_request.prompt_token_ids is not frontend_tokens
    assert owner_request.prompt_embeds is not frontend_embeds
    assert owner_request.prompt_embeds.data_ptr() != frontend_embeds.data_ptr()
    assert owner_request.reasoning_parser_kwargs is not frontend_reasoning
    assert owner_request.reasoning_parser_kwargs["state"] is not (
        frontend_reasoning["state"]
    )

    # Frontend and owner each apply the same streaming continuation once.
    frontend_tokens.append(3)
    owner_request.prompt_token_ids.append(3)
    frontend_embeds[0, 0] = 99.0
    assert frontend_tokens == [1, 2, 3]
    assert owner_request.prompt_token_ids == [1, 2, 3]
    assert owner_request.prompt_embeds[0, 0].item() == 1.0


@pytest.mark.asyncio
async def test_utility_request_restores_mutable_ownership_boundary() -> None:
    resources = InprocBackgroundResources()
    core = _FakeEngineCore()
    resources.attach_engine(core)  # type: ignore[arg-type]
    client = AsyncInprocClient.__new__(AsyncInprocClient)
    client.resources = resources
    client.client_index = 0
    client._request_encoder = MsgpackEncoder()
    client._utility_decoder = MsgpackDecoder(share_mem=False)
    frontend_tokens = list(range(16))
    request = EngineCoreRequest(
        request_id="prefix-request",
        prompt_token_ids=frontend_tokens,
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )

    utility_task = asyncio.create_task(
        client.call_utility_async("pin_prefix", "pin", request, "gpu")
    )
    await asyncio.sleep(0)
    request_type, payload = core.input_queue.get_nowait()
    assert request_type is EngineCoreRequestType.UTILITY
    _client_index, call_id, method_name, owner_args = payload
    assert method_name == "pin_prefix"
    assert owner_args[1] is not request
    assert owner_args[1][1] is not frontend_tokens

    engine = EngineCore.__new__(EngineCore)
    converted_args = EngineCoreProc._convert_msgspec_args(
        engine.pin_prefix, owner_args
    )
    owner_request = converted_args[1]
    assert isinstance(owner_request, EngineCoreRequest)
    assert owner_request.prompt_token_ids is not frontend_tokens
    owner_request.prompt_token_ids.append(16)
    assert frontend_tokens == list(range(16))

    resources.publish_output(
        EngineCoreOutputs(
            utility_output=UtilityOutput(
                call_id=call_id,
                result=UtilityResult(None),
            )
        )
    )
    assert await utility_task is None


@pytest.mark.asyncio
async def test_fatal_error_wakes_output_and_utility_waiters_once() -> None:
    resources = InprocBackgroundResources()
    resources.attach_engine(_FakeEngineCore())  # type: ignore[arg-type]
    resources.bind_loop(asyncio.get_running_loop())

    utility_future = asyncio.get_running_loop().create_future()
    resources.register_utility(11, utility_future)
    output_waiter = asyncio.create_task(resources.outputs_queue.get())
    await asyncio.sleep(0)

    fatal_thread = threading.Thread(
        target=resources.mark_fatal, args=(_captured_error(),)
    )
    fatal_thread.start()
    fatal_thread.join()

    terminal = await asyncio.wait_for(output_waiter, 1.0)
    assert isinstance(terminal, EngineDeadError)
    with pytest.raises(EngineDeadError):
        await asyncio.wait_for(utility_future, 1.0)

    assert resources.fatal_error is not None
    assert resources.fatal_error.__traceback__ is None
    assert resources.fatal_error.__context__ is None
    assert resources.fatal_error.__cause__ is None
    assert resources.terminal_output_published
    assert resources.terminal_output_delivered

    resources.mark_stopped()
    await asyncio.sleep(0)
    assert resources.outputs_queue.empty()


def test_deferred_utility_future_does_not_block_dispatch() -> None:
    pending: Future[str] = Future()
    output = UtilityOutput(call_id=17)
    delivered: list[UtilityOutput] = []

    EngineCoreProc._invoke_utility_method(
        "deferred", lambda: pending, output, delivered.append
    )
    assert delivered == []

    pending.set_result("complete")
    assert delivered == [output]
    assert output.result is not None
    assert output.result.result == "complete"


@pytest.mark.asyncio
async def test_typed_prefix_and_request_utility_proxies() -> None:
    pin_result: PrefixPinResult = {
        "pin_id": "shared-prefix",
        "level": "cpu",
        "pinned_tokens": 32,
        "block_ids": [1, 2],
    }
    client = _RecordingEngineCoreClient(
        {"pin_prefix": pin_result, "unpin_prefix": True}
    )
    request = _request("pin-request", data_parallel_rank=0)

    assert (
        await client.pin_prefix_async("shared-prefix", request, tier="cpu")
        is pin_result
    )
    assert request.client_index == 3
    assert await client.unpin_prefix_async("shared-prefix") is True
    await client.pause_prefix_async("shared-prefix")
    await client.resume_prefix_async("shared-prefix")
    await client.pause_requests_async(["request-1", "request-2"])
    await client.resume_requests_async(["request-1", "request-2"])

    assert client.calls == [
        ("pin_prefix", ("shared-prefix", request, "cpu")),
        ("unpin_prefix", ("shared-prefix", None)),
        ("pause_prefix", ("shared-prefix",)),
        ("resume_prefix", ("shared-prefix",)),
        ("pause_requests", (["request-1", "request-2"],)),
        ("resume_requests", (["request-1", "request-2"],)),
    ]


@pytest.mark.asyncio
async def test_async_llm_prefix_uses_legacy_level_result_contract() -> None:
    pin_result: PrefixPinResult = {
        "pin_id": "cpu-prefix",
        "level": "cpu",
        "pinned_tokens": 32,
        "block_ids": [1, 2],
    }

    class _Core:
        def __init__(self) -> None:
            self.resources = SimpleNamespace(engine_dead=False)
            self.calls: list[tuple[str, str]] = []

        async def pin_prefix_async(
            self, pin_id: str, _request: Any, level: str
        ) -> PrefixPinResult:
            self.calls.append((pin_id, level))
            return pin_result

    class _InputProcessor:
        async def process_inputs_async(self, *_args: Any, **_kwargs: Any) -> Any:
            return SimpleNamespace(
                request_id="pin-request",
                sampling_params=None,
                pooling_params=None,
                abort_immediately=False,
            )

        @staticmethod
        def assign_request_id(_request: Any) -> None:
            pass

    core = _Core()
    llm = AsyncLLM.__new__(AsyncLLM)
    llm.engine_core = cast(Any, core)
    llm.input_processor = cast(Any, _InputProcessor())
    llm.output_handler = None

    async def get_supported_tasks() -> tuple[str, ...]:
        return ("generate",)

    llm.get_supported_tasks = cast(Any, get_supported_tasks)

    assert await llm.pin_prefix("prompt", "cpu-prefix", level="cpu") is pin_result
    assert core.calls == [("cpu-prefix", "cpu")]
    with pytest.raises(ValueError, match="unsupported prefix pin level"):
        await llm.pin_prefix(
            "prompt", "invalid-prefix", level=cast(Any, "invalid")
        )


@pytest.mark.asyncio
async def test_dplb_prefix_utilities_stay_on_owning_engine() -> None:
    client = _dplb_client()
    request = _request("pin-request", data_parallel_rank=1)
    duplicate_request = _request("duplicate-request", data_parallel_rank=0)
    pin_result: PrefixPinResult = {
        "pin_id": "shared-prefix",
        "level": "gpu",
        "pinned_tokens": 16,
        "block_ids": [[7]],
    }
    calls: list[tuple[str, tuple[Any, ...], bytes]] = []

    async def call_utility(
        method: str, *args: Any, engine: bytes
    ) -> Any:
        calls.append((method, args, engine))
        return True if method == "unpin_prefix" else pin_result

    cast(Any, client)._call_utility_async = call_utility

    assert (
        await client.pin_prefix_async("shared-prefix", request, tier="gpu")
        is pin_result
    )
    with pytest.raises(ValueError, match="already exists"):
        await client.pin_prefix_async("shared-prefix", duplicate_request)

    await client.pause_prefix_async("shared-prefix")
    await client.resume_prefix_async("shared-prefix")
    assert await client.unpin_prefix_async("shared-prefix") is True

    owning_engine = client.core_engines[1]
    assert [call[0] for call in calls] == [
        "pin_prefix",
        "pause_prefix",
        "resume_prefix",
        "unpin_prefix",
    ]
    assert all(call[2] == owning_engine for call in calls)
    assert duplicate_request.request_id not in client.reqs_in_flight
    assert "shared-prefix" not in client.prefix_pins


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["pause_requests_async", "resume_requests_async"])
async def test_dplb_request_controls_route_only_to_owning_engines(
    method: str,
) -> None:
    client = _dplb_client()
    engine_0, engine_1 = client.core_engines
    client.reqs_in_flight.update(
        {
            "request-0-a": engine_0,
            "request-0-b": engine_0,
            "request-1": engine_1,
        }
    )
    calls: list[tuple[str, list[str], bytes]] = []

    async def call_utility(
        utility_method: str, request_ids: list[str], *, engine: bytes
    ) -> None:
        calls.append((utility_method, request_ids, engine))

    cast(Any, client)._call_utility_async = call_utility

    await getattr(client, method)(
        ["request-0-a", "unknown", "request-1", "request-0-b"]
    )

    expected_utility = method.removesuffix("_async")
    assert calls == [
        (expected_utility, ["request-0-a", "request-0-b"], engine_0),
        (expected_utility, ["request-1"], engine_1),
    ]


@pytest.mark.asyncio
async def test_dplb_failed_pin_releases_routing_reservation() -> None:
    client = _dplb_client()
    request = _request("failed-pin-request", data_parallel_rank=0)
    owning_engine = client.core_engines[0]

    async def fail_utility(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("pin failed")

    cast(Any, client)._call_utility_async = fail_utility

    with pytest.raises(RuntimeError, match="pin failed"):
        await client.pin_prefix_async("failed-prefix", request)

    assert "failed-prefix" not in client.prefix_pins
    assert request.request_id not in client.reqs_in_flight
    assert client.engine_inflight[owning_engine] == 0


@pytest.mark.asyncio
async def test_dplb_cancelled_pin_preserves_owner_route() -> None:
    client = _dplb_client()
    request = _request("cancelled-pin-request", data_parallel_rank=0)
    owning_engine = client.core_engines[0]

    async def cancel_utility(*_args: Any, **_kwargs: Any) -> Any:
        raise asyncio.CancelledError

    cast(Any, client)._call_utility_async = cancel_utility

    with pytest.raises(asyncio.CancelledError):
        await client.pin_prefix_async("cancelled-prefix", request)

    assert client.prefix_pins["cancelled-prefix"] == (
        owning_engine,
        request.request_id,
    )
    assert client.reqs_in_flight[request.request_id] == owning_engine
    assert client.engine_inflight[owning_engine] == 1


@pytest.mark.asyncio
async def test_async_llm_cancelled_pin_releases_after_admission_race() -> None:
    pin_result: PrefixPinResult = {
        "pin_id": "cancelled-prefix",
        "level": "gpu",
        "pinned_tokens": 16,
        "block_ids": [[7]],
    }

    class _RacingCore:
        def __init__(self) -> None:
            self.resources = SimpleNamespace(engine_dead=False)
            self.entered = asyncio.Event()
            self.admit = asyncio.Event()
            self.released = asyncio.Event()
            self.registered = False
            self.unpin_observations: list[bool] = []

        async def pin_prefix_async(self, *_args: Any) -> PrefixPinResult:
            self.entered.set()
            await self.admit.wait()
            self.registered = True
            return pin_result

        async def unpin_prefix_async(
            self,
            _pin_id: str,
            expected_request_id: str | None = None,
        ) -> bool:
            assert expected_request_id == "prefix-request"
            was_registered = self.registered
            self.unpin_observations.append(was_registered)
            if was_registered:
                self.registered = False
                self.released.set()
            return was_registered

        def shutdown(self, timeout: float | None = None) -> None:
            pass

    class _InputProcessor:
        async def process_inputs_async(self, *_args: Any, **_kwargs: Any) -> Any:
            return SimpleNamespace(
                request_id="prefix-request",
                sampling_params=None,
                pooling_params=None,
                abort_immediately=False,
            )

        @staticmethod
        def assign_request_id(_request: Any) -> None:
            pass

    core = _RacingCore()
    llm = AsyncLLM.__new__(AsyncLLM)
    llm.engine_core = cast(Any, core)
    llm.input_processor = cast(Any, _InputProcessor())
    llm.output_handler = None

    async def get_supported_tasks() -> tuple[str, ...]:
        return ("generate",)

    llm.get_supported_tasks = cast(Any, get_supported_tasks)
    pin_task = asyncio.create_task(llm.pin_prefix("prompt", "cancelled-prefix"))
    await core.entered.wait()
    pin_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await pin_task

    assert core.unpin_observations == []
    core.admit.set()
    await asyncio.wait_for(core.released.wait(), 1.0)
    assert core.unpin_observations == [True]
    assert not core.registered


@pytest.mark.asyncio
async def test_async_llm_cancelled_duplicate_does_not_release_existing_pin() -> None:
    class _DuplicateCore:
        def __init__(self) -> None:
            self.resources = SimpleNamespace(engine_dead=False)
            self.entered = asyncio.Event()
            self.reject = asyncio.Event()
            self.existing_pin = True
            self.unpin_calls = 0

        async def pin_prefix_async(self, *_args: Any) -> PrefixPinResult:
            self.entered.set()
            await self.reject.wait()
            raise ValueError("prefix pin already exists")

        async def unpin_prefix_async(
            self,
            _pin_id: str,
            expected_request_id: str | None = None,
        ) -> bool:
            self.unpin_calls += 1
            self.existing_pin = False
            return True

        def shutdown(self, timeout: float | None = None) -> None:
            pass

    class _InputProcessor:
        async def process_inputs_async(self, *_args: Any, **_kwargs: Any) -> Any:
            return SimpleNamespace(
                request_id="duplicate-prefix-request",
                sampling_params=None,
                pooling_params=None,
                abort_immediately=False,
            )

        @staticmethod
        def assign_request_id(_request: Any) -> None:
            pass

    core = _DuplicateCore()
    llm = AsyncLLM.__new__(AsyncLLM)
    llm.engine_core = cast(Any, core)
    llm.input_processor = cast(Any, _InputProcessor())
    llm.output_handler = None

    async def get_supported_tasks() -> tuple[str, ...]:
        return ("generate",)

    llm.get_supported_tasks = cast(Any, get_supported_tasks)
    pin_task = asyncio.create_task(llm.pin_prefix("prompt", "existing-prefix"))
    await core.entered.wait()
    pin_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await pin_task

    cleanup_tasks = list(llm._prefix_pin_cleanup_tasks)
    assert len(cleanup_tasks) == 1
    core.reject.set()
    await asyncio.gather(*cleanup_tasks)
    assert core.existing_pin
    assert core.unpin_calls == 0


@pytest.mark.asyncio
async def test_cancelled_pin_cleanup_does_not_release_replacement_owner() -> None:
    pin_result: PrefixPinResult = {
        "pin_id": "shared-prefix",
        "level": "gpu",
        "pinned_tokens": 16,
        "block_ids": [[7]],
    }

    class _ReplacementCore:
        def __init__(self) -> None:
            self.resources = SimpleNamespace(engine_dead=False)
            self.entered = asyncio.Event()
            self.admit = asyncio.Event()
            self.cleanup_entered = asyncio.Event()
            self.check_owner = asyncio.Event()
            self.owner_request_id: str | None = None
            self.cleanup_expected: str | None = None

        async def pin_prefix_async(
            self, _pin_id: str, request: Any, _tier: str
        ) -> PrefixPinResult:
            self.entered.set()
            await self.admit.wait()
            self.owner_request_id = request.request_id
            return pin_result

        async def unpin_prefix_async(
            self,
            _pin_id: str,
            expected_request_id: str | None = None,
        ) -> bool:
            self.cleanup_expected = expected_request_id
            self.cleanup_entered.set()
            await self.check_owner.wait()
            if self.owner_request_id != expected_request_id:
                return False
            self.owner_request_id = None
            return True

        def shutdown(self, timeout: float | None = None) -> None:
            pass

    class _InputProcessor:
        async def process_inputs_async(self, *_args: Any, **_kwargs: Any) -> Any:
            return SimpleNamespace(
                request_id="cancelled-owner",
                sampling_params=None,
                pooling_params=None,
                abort_immediately=False,
            )

        @staticmethod
        def assign_request_id(_request: Any) -> None:
            pass

    core = _ReplacementCore()
    llm = AsyncLLM.__new__(AsyncLLM)
    llm.engine_core = cast(Any, core)
    llm.input_processor = cast(Any, _InputProcessor())
    llm.output_handler = None

    async def get_supported_tasks() -> tuple[str, ...]:
        return ("generate",)

    llm.get_supported_tasks = cast(Any, get_supported_tasks)
    pin_task = asyncio.create_task(llm.pin_prefix("prompt", "shared-prefix"))
    await core.entered.wait()
    pin_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pin_task

    cleanup_tasks = list(llm._prefix_pin_cleanup_tasks)
    assert len(cleanup_tasks) == 1
    core.admit.set()
    await core.cleanup_entered.wait()
    # A caller released the old owner and installed a replacement before the
    # cancelled call's background cleanup reached EngineCore.
    core.owner_request_id = "replacement-owner"
    core.check_owner.set()
    await asyncio.gather(*cleanup_tasks)

    assert core.cleanup_expected == "cancelled-owner"
    assert core.owner_request_id == "replacement-owner"


@pytest.mark.asyncio
async def test_dplb_conditional_unpin_rejects_stale_route_owner() -> None:
    client = _dplb_client()
    engine = client.core_engines[0]
    client.prefix_pins["shared-prefix"] = (engine, "replacement-owner")
    calls: list[tuple[Any, ...]] = []

    async def record_call(*args: Any, **_kwargs: Any) -> bool:
        calls.append(args)
        return True

    cast(Any, client)._call_utility_async = record_call

    assert not await client.unpin_prefix_async(
        "shared-prefix", expected_request_id="cancelled-owner"
    )
    assert calls == []
    assert client.prefix_pins["shared-prefix"] == (
        engine,
        "replacement-owner",
    )


@pytest.mark.asyncio
async def test_dplb_cancelled_unpin_still_retires_owner_route() -> None:
    client = _dplb_client()
    engine = client.core_engines[0]
    client.prefix_pins["shared-prefix"] = (engine, "pin-owner")
    entered = asyncio.Event()
    complete = asyncio.Event()

    async def delayed_unpin(*_args: Any, **_kwargs: Any) -> bool:
        entered.set()
        await complete.wait()
        return True

    cast(Any, client)._call_utility_async = delayed_unpin
    unpin_task = asyncio.create_task(client.unpin_prefix_async("shared-prefix"))
    await entered.wait()
    unpin_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await unpin_task

    assert "shared-prefix" in client.prefix_pins
    complete.set()
    for _ in range(3):
        await asyncio.sleep(0)

    assert "shared-prefix" not in client.prefix_pins


def test_async_client_factory_selects_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mp_client = object()
    inproc_client = object()
    monkeypatch.setattr(
        EngineCoreClient,
        "make_async_mp_client",
        lambda *_args, **_kwargs: mp_client,
    )
    monkeypatch.setattr(
        core_client_module,
        "AsyncInprocClient",
        lambda *_args, **_kwargs: inproc_client,
    )

    args = (_config(), UniProcExecutor, False)
    assert EngineCoreClient.make_async_client("mp", *args) is mp_client
    assert EngineCoreClient.make_async_client("inproc", *args) is inproc_client
    assert EngineCoreClient.make_client(True, True, *args) is mp_client
    assert EngineCoreClient.make_client(False, True, *args) is inproc_client
    with pytest.raises(ValueError, match="Unsupported async EngineCore mode"):
        EngineCoreClient.make_async_client("invalid", *args)  # type: ignore[arg-type]


def test_async_llm_preserves_enable_multiprocessing_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for method in (
        AsyncLLM.__init__,
        AsyncLLM.from_vllm_config,
        AsyncLLM.from_engine_args,
    ):
        parameter = inspect.signature(method).parameters["enable_multiprocessing"]
        assert parameter.default is True

    captured: list[dict[str, Any]] = []

    class _RecordingAsyncLLM(AsyncLLM):
        def __init__(self, **kwargs: Any) -> None:
            captured.append(kwargs)

        def __del__(self) -> None:
            pass

    monkeypatch.setattr(Executor, "get_class", lambda _config: UniProcExecutor)
    config = _config()
    _RecordingAsyncLLM.from_vllm_config(
        config, enable_multiprocessing=False
    )
    engine_args = SimpleNamespace(
        create_engine_config=lambda _usage_context: config,
        enable_log_requests=False,
        disable_log_stats=True,
    )
    _RecordingAsyncLLM.from_engine_args(
        cast(Any, engine_args), enable_multiprocessing=False
    )

    assert [call["enable_multiprocessing"] for call in captured] == [False, False]


@pytest.mark.asyncio
async def test_async_llm_prefix_and_request_controls_forward_public_ids() -> None:
    calls: list[tuple[str, Any]] = []

    class _RecordingCore:
        resources = SimpleNamespace(engine_dead=False)

        async def unpin_prefix_async(self, pin_id: str) -> bool:
            calls.append(("unpin_prefix", pin_id))
            return True

        async def pause_prefix_async(self, pin_id: str) -> None:
            calls.append(("pause_prefix", pin_id))

        async def resume_prefix_async(self, pin_id: str) -> None:
            calls.append(("resume_prefix", pin_id))

        async def pause_requests_async(self, request_ids: list[str]) -> None:
            calls.append(("pause_requests", request_ids))

        async def resume_requests_async(self, request_ids: list[str]) -> None:
            calls.append(("resume_requests", request_ids))

    class _NoFinalizerAsyncLLM(AsyncLLM):
        def __del__(self) -> None:
            pass

    llm = _NoFinalizerAsyncLLM.__new__(_NoFinalizerAsyncLLM)
    llm.engine_core = cast(Any, _RecordingCore())
    llm.output_handler = None

    def resolve_ids(request_ids: Iterable[str]) -> list[str]:
        public_ids = list(request_ids)
        calls.append(("resolve", public_ids))
        return [f"internal:{request_id}" for request_id in public_ids]

    llm.output_processor = cast(
        Any, SimpleNamespace(get_internal_request_ids=resolve_ids)
    )

    assert await llm.unpin_prefix("pin")
    await llm.pause_prefix("pin")
    await llm.resume_prefix("pin")
    await llm.pause("external-a")
    await llm.resume(["external-a", "external-b"])

    assert calls == [
        ("unpin_prefix", "pin"),
        ("pause_prefix", "pin"),
        ("resume_prefix", "pin"),
        ("resolve", ["external-a"]),
        ("pause_requests", ["internal:external-a"]),
        ("resolve", ["external-a", "external-b"]),
        (
            "resume_requests",
            ["internal:external-a", "internal:external-b"],
        ),
    ]

    for method in (llm.unpin_prefix, llm.pause_prefix, llm.resume_prefix):
        with pytest.raises(TypeError, match="must be a string"):
            await method(cast(Any, 1))


@pytest.mark.parametrize(
    ("enable_multiprocessing", "expected_mode"),
    [(True, "mp"), (False, "inproc")],
)
def test_async_llm_maps_legacy_multiprocessing_flag_to_backend(
    monkeypatch: pytest.MonkeyPatch,
    enable_multiprocessing: bool,
    expected_mode: str,
) -> None:
    modes: list[str] = []
    renderer = SimpleNamespace(tokenizer=object(), shutdown=lambda: None)
    engine_core = SimpleNamespace(
        engine_ranks_managed=[0], shutdown=lambda timeout=None: None
    )
    config = SimpleNamespace(
        model_config=object(),
        observability_config=SimpleNamespace(otlp_traces_endpoint=None),
        scheduler_config=SimpleNamespace(stream_interval=1),
        profiler_config=SimpleNamespace(profiler=None),
    )

    monkeypatch.setattr(
        async_llm_module, "maybe_register_config_serialize_by_value", lambda: None
    )
    monkeypatch.setattr(
        async_llm_module, "renderer_from_config", lambda _config: renderer
    )
    monkeypatch.setattr(async_llm_module, "InputProcessor", lambda *_args: object())
    monkeypatch.setattr(
        async_llm_module,
        "OutputProcessor",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        async_llm_module, "load_stat_logger_plugin_factories", lambda: []
    )
    monkeypatch.setattr(async_llm_module, "shutdown_prometheus", lambda: None)

    def make_async_client(mode: str, *_args: Any, **_kwargs: Any) -> Any:
        modes.append(mode)
        return engine_core

    monkeypatch.setattr(EngineCoreClient, "make_async_client", make_async_client)
    llm = AsyncLLM(
        cast(Any, config),
        UniProcExecutor,
        log_stats=False,
        enable_multiprocessing=enable_multiprocessing,
    )
    llm.shutdown()

    assert modes == [expected_mode]


def test_fake_owner_thread_starts_and_shutdown_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(core_client_module, "_run_async_inproc_engine", _run_fake_owner)
    client = AsyncInprocClient(_config(), UniProcExecutor, log_stats=False)
    owner_thread = client.resources.thread

    assert owner_thread is not None
    assert owner_thread.is_alive()
    assert owner_thread.ident != threading.get_ident()

    client.shutdown(timeout=2.0)
    assert not owner_thread.is_alive()
    client.shutdown(timeout=2.0)


def test_concurrent_async_inproc_clients_fail_fast_and_release_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: False)
    monkeypatch.setattr(core_client_module, "_run_async_inproc_engine", _run_fake_owner)
    first = AsyncInprocClient(_config(), UniProcExecutor, log_stats=False)

    try:
        with pytest.raises(RuntimeError, match="another AsyncInprocClient"):
            AsyncInprocClient(_config(), UniProcExecutor, log_stats=False)
    finally:
        first.shutdown(timeout=2.0)

    replacement = AsyncInprocClient(_config(), UniProcExecutor, log_stats=False)
    replacement.shutdown(timeout=2.0)


@pytest.mark.asyncio
async def test_fake_client_shutdown_wakes_output_waiter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(core_client_module, "_run_async_inproc_engine", _run_fake_owner)
    client = AsyncInprocClient(_config(), UniProcExecutor, log_stats=False)
    owner_thread = client.resources.thread
    assert owner_thread is not None

    output_waiter = asyncio.create_task(client.get_output_async())
    await asyncio.sleep(0)
    client.shutdown(timeout=2.0)

    with pytest.raises(EngineDeadError):
        await asyncio.wait_for(output_waiter, 1.0)
    assert not owner_thread.is_alive()


@pytest.mark.asyncio
async def test_stuck_owner_shutdown_fails_output_and_utility_waiters() -> None:
    resources = InprocBackgroundResources()
    resources.bind_loop(asyncio.get_running_loop())
    utility_future = asyncio.get_running_loop().create_future()
    resources.register_utility(17, utility_future)
    output_waiter = asyncio.create_task(resources.outputs_queue.get())
    stop_owner = threading.Event()
    owner_thread = threading.Thread(target=stop_owner.wait, daemon=True)
    resources.thread = owner_thread
    owner_thread.start()

    try:
        with pytest.raises(TimeoutError, match="owner thread did not stop"):
            resources.shutdown(timeout=0.001)

        terminal = await asyncio.wait_for(output_waiter, 1.0)
        assert isinstance(terminal, EngineDeadError)
        with pytest.raises(EngineDeadError):
            await asyncio.wait_for(utility_future, 1.0)
        assert resources.engine_dead
        assert isinstance(resources.fatal_error, TimeoutError)
    finally:
        stop_owner.set()
        owner_thread.join(timeout=1.0)


@pytest.mark.parametrize("failing_resource", ["renderer", "prometheus"])
def test_async_llm_shutdown_attempts_core_despite_frontend_failure(
    monkeypatch: pytest.MonkeyPatch,
    failing_resource: str,
) -> None:
    calls: list[str] = []

    class _NoFinalizerAsyncLLM(AsyncLLM):
        def __del__(self) -> None:
            pass

    def shutdown_resource(name: str) -> None:
        calls.append(name)
        if name == failing_resource:
            raise RuntimeError(f"{name} shutdown failed")

    monkeypatch.setattr(
        async_llm_module,
        "shutdown_prometheus",
        lambda: shutdown_resource("prometheus"),
    )
    llm = _NoFinalizerAsyncLLM.__new__(_NoFinalizerAsyncLLM)
    llm.engine_core = cast(
        Any,
        SimpleNamespace(shutdown=lambda timeout=None: shutdown_resource("core")),
    )
    llm.renderer = SimpleNamespace(shutdown=lambda: shutdown_resource("renderer"))
    llm.output_handler = None
    llm._prefix_pin_cleanup_tasks = set()

    with pytest.raises(RuntimeError, match=f"{failing_resource} shutdown failed"):
        llm.shutdown(timeout=0.001)

    assert calls == ["core", "renderer", "prometheus"]


def test_async_llm_shutdown_preserves_core_timeout_over_later_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class _NoFinalizerAsyncLLM(AsyncLLM):
        def __del__(self) -> None:
            pass

    def fail(name: str, error: Exception) -> None:
        calls.append(name)
        raise error

    monkeypatch.setattr(
        async_llm_module,
        "shutdown_prometheus",
        lambda: fail("prometheus", RuntimeError("prometheus failed")),
    )
    llm = _NoFinalizerAsyncLLM.__new__(_NoFinalizerAsyncLLM)
    llm.engine_core = cast(
        Any,
        SimpleNamespace(
            shutdown=lambda timeout=None: fail(
                "core", TimeoutError("owner thread did not stop")
            )
        ),
    )
    llm.renderer = SimpleNamespace(
        shutdown=lambda: fail("renderer", RuntimeError("renderer failed"))
    )
    llm.output_handler = None
    llm._prefix_pin_cleanup_tasks = set()

    with pytest.raises(TimeoutError, match="owner thread did not stop"):
        llm.shutdown(timeout=0.001)

    assert calls == ["core", "renderer", "prometheus"]


@pytest.mark.asyncio
async def test_async_llm_shutdown_cancels_tasks_when_core_shutdown_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vllm.v1.engine.async_llm.shutdown_prometheus", lambda: None
    )

    class _FailingCore:
        def shutdown(self, timeout: float | None = None) -> None:
            raise RuntimeError("core shutdown failed")

    llm = AsyncLLM.__new__(AsyncLLM)
    llm.renderer = None
    llm.engine_core = cast(Any, _FailingCore())
    handler = asyncio.create_task(asyncio.Event().wait())
    cleanup = asyncio.create_task(asyncio.Event().wait())
    llm.output_handler = handler
    llm._prefix_pin_cleanup_tasks = {cleanup}

    with pytest.raises(RuntimeError, match="core shutdown failed"):
        llm.shutdown(timeout=0.001)
    await asyncio.sleep(0)

    assert handler.cancelled()
    assert cleanup.cancelled()
    llm.engine_core = None
    llm.output_handler = None
    llm._prefix_pin_cleanup_tasks.clear()


@pytest.mark.asyncio
async def test_weight_version_and_status_interfaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(core_client_module, "_run_async_inproc_engine", _run_fake_owner)
    client = AsyncInprocClient(_config(), UniProcExecutor, log_stats=False)
    try:
        await client.set_weight_version_async("weights-v2")
        assert await client.get_weight_version_async() == "weights-v2"
        assert await client.get_status() == {
            "schema_version": 1,
            "total_engines": 1,
            "engines": [{"id": 0, "status": "healthy"}],
        }
    finally:
        client.shutdown(timeout=2.0)


def test_failed_fake_startup_does_not_leak_owner_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner_threads: list[threading.Thread] = []

    def fail_startup(resources: InprocBackgroundResources, *_args: Any) -> None:
        owner_threads.append(threading.current_thread())
        resources.startup_future.set_exception(RuntimeError("fake startup failure"))
        resources.mark_stopped()

    monkeypatch.setattr(core_client_module, "_run_async_inproc_engine", fail_startup)
    with pytest.raises(RuntimeError, match="fake startup failure"):
        AsyncInprocClient(_config(), UniProcExecutor, log_stats=False)

    assert len(owner_threads) == 1
    assert not owner_threads[0].is_alive()

    monkeypatch.setattr(core_client_module, "_run_async_inproc_engine", _run_fake_owner)
    replacement = AsyncInprocClient(_config(), UniProcExecutor, log_stats=False)
    replacement.shutdown(timeout=2.0)


@pytest.mark.parametrize(
    ("inproc_engine", "expected_calls"),
    [(False, []), (True, ["driver_shutdown"])],
)
def test_executor_init_failure_cleans_partial_driver_only_inproc(
    monkeypatch: pytest.MonkeyPatch,
    inproc_engine: bool,
    expected_calls: list[str],
) -> None:
    calls: list[str] = []

    class _PartialDriver(WorkerWrapperBase):
        def shutdown(self) -> None:
            calls.append("driver_shutdown")
            super().shutdown()

    def fail_init(executor: UniProcExecutor) -> None:
        executor.driver_worker = cast(Any, _PartialDriver())
        raise RuntimeError("fake executor init failure")

    config = _config()
    for name in (
        "model_config",
        "cache_config",
        "lora_config",
        "load_config",
        "scheduler_config",
        "device_config",
        "speculative_config",
        "observability_config",
    ):
        setattr(config, name, None)
    monkeypatch.setattr(UniProcExecutor, "_init_executor", fail_init)

    with pytest.raises(RuntimeError, match="fake executor init failure"):
        UniProcExecutor(config, inproc_engine=inproc_engine)

    assert calls == expected_calls


def test_teardown_error_is_propagated_after_thread_stops() -> None:
    resources = InprocBackgroundResources()

    def finish_with_teardown_error() -> None:
        resources.teardown_error = copy_exception_without_traceback(
            RuntimeError("fake teardown failure")
        )
        resources.mark_stopped()

    owner_thread = threading.Thread(target=finish_with_teardown_error)
    resources.thread = owner_thread
    owner_thread.start()

    with pytest.raises(RuntimeError, match="fake teardown failure"):
        resources.shutdown(timeout=2.0)
    assert not owner_thread.is_alive()


@pytest.mark.parametrize(
    ("parallel_field", "value"),
    [
        ("tensor_parallel_size", 2),
        ("pipeline_parallel_size", 2),
        ("prefill_context_parallel_size", 2),
        ("decode_context_parallel_size", 2),
        ("data_parallel_size", 2),
        ("data_parallel_size_local", 2),
        ("data_parallel_rank", 1),
        ("nnodes", 2),
        ("_api_process_count", 2),
        ("data_parallel_external_lb", True),
        ("data_parallel_hybrid_lb", True),
        ("enable_elastic_ep", True),
        ("enable_fault_tolerance", True),
    ],
)
def test_parallel_fault_tolerance_and_api_configs_fail_fast(
    parallel_field: str,
    value: int | bool,
) -> None:
    with pytest.raises(ValueError, match=parallel_field):
        AsyncInprocClient._validate_config(
            _config(**{parallel_field: value}),
            UniProcExecutor,
            client_addresses=None,
            client_count=1,
            client_index=0,
        )


@pytest.mark.parametrize(
    ("client_addresses", "client_count", "client_index", "message"),
    [
        ({"inputs": "inproc://test"}, 1, 0, "client_addresses"),
        (None, 2, 0, "client_count=2"),
        (None, 1, 1, "client_index=1"),
    ],
)
def test_external_and_multiple_clients_fail_fast(
    client_addresses: dict[str, Any] | None,
    client_count: int,
    client_index: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        AsyncInprocClient._validate_config(
            _config(),
            UniProcExecutor,
            client_addresses=client_addresses,
            client_count=client_count,
            client_index=client_index,
        )


def test_non_uniproc_executor_fails_fast() -> None:
    with pytest.raises(ValueError, match="executor_class"):
        AsyncInprocClient._validate_config(
            _config(),
            Executor,
            client_addresses=None,
            client_count=1,
            client_index=0,
        )


def test_uniproc_executor_subclass_fails_fast() -> None:
    custom_executor = type("CustomUniProcExecutor", (UniProcExecutor,), {})

    with pytest.raises(ValueError, match="CustomUniProcExecutor"):
        AsyncInprocClient._validate_config(
            _config(),
            custom_executor,
            client_addresses=None,
            client_count=1,
            client_index=0,
        )


def test_engine_core_shutdown_attempts_every_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    engine = EngineCore.__new__(EngineCore)

    def cleanup(name: str, *, fail: bool = False) -> None:
        calls.append(name)
        if fail:
            raise RuntimeError(f"{name} failed")

    engine.structured_output_manager = SimpleNamespace(
        clear_backend=lambda: cleanup("structured", fail=True)
    )
    engine.model_executor = SimpleNamespace(
        shutdown=lambda: cleanup("executor")
    )
    cast(Any, engine).scheduler = SimpleNamespace(
        shutdown=lambda: cleanup("scheduler")
    )
    monkeypatch.setattr(core_module.gc, "unfreeze", lambda: cleanup("gc"))
    monkeypatch.setattr(
        "vllm.v1.engine.core.cleanup_dist_env_and_memory",
        lambda **_kwargs: cleanup("distributed"),
    )

    with pytest.raises(RuntimeError, match="structured failed"):
        engine.shutdown()

    assert calls == ["structured", "executor", "scheduler", "gc", "distributed"]


def test_async_inproc_rejects_preinitialized_host_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner_started = False

    def record_owner_start(*_args: Any) -> None:
        nonlocal owner_started
        owner_started = True

    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
    monkeypatch.setattr(
        core_client_module, "_run_async_inproc_engine", record_owner_start
    )

    with pytest.raises(RuntimeError, match="exclusive ownership"):
        AsyncInprocClient(_config(), UniProcExecutor, log_stats=False)

    assert not owner_started


def test_owner_recheck_does_not_cleanup_host_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dist_initialized = False
    shutdown_calls: list[str] = []
    monkeypatch.setattr(
        "torch.distributed.is_initialized", lambda: dist_initialized
    )
    token = core_client_module._reserve_inproc_distributed_ownership()
    resources = InprocBackgroundResources(dist_ownership_token=token)
    dist_initialized = True
    monkeypatch.setattr(
        core_client_module._AsyncInprocEngineCore,
        "shutdown",
        lambda _self: shutdown_calls.append("shutdown"),
    )

    core_client_module._run_async_inproc_engine(
        resources, _config(), UniProcExecutor, False
    )

    with pytest.raises(RuntimeError, match="after ownership was reserved"):
        resources.startup_future.result()
    assert shutdown_calls == []
    assert resources.dist_ownership_token is None

    dist_initialized = False
    replacement_token = core_client_module._reserve_inproc_distributed_ownership()
    core_client_module._release_inproc_distributed_ownership(replacement_token)


@pytest.mark.parametrize("inproc_engine", [False, True])
def test_engine_core_gc_freeze_respects_process_ownership(
    monkeypatch: pytest.MonkeyPatch, inproc_engine: bool
) -> None:
    calls: list[str] = []
    engine = EngineCore.__new__(EngineCore)
    engine.inproc_engine = inproc_engine
    monkeypatch.setattr(core_module, "freeze_gc_heap", lambda: calls.append("freeze"))
    monkeypatch.setattr(
        core_module,
        "maybe_attach_gc_debug_callback",
        lambda: calls.append("gc_debug"),
    )

    engine._freeze_gc_heap_if_owned()
    engine._attach_gc_debug_callback_if_owned()

    assert calls == ([] if inproc_engine else ["freeze", "gc_debug"])


@pytest.mark.parametrize(
    ("inproc_engine", "transitioned", "owns_transition"),
    [(False, True, False), (True, False, False), (True, True, True)],
)
def test_engine_core_env_cache_tracks_only_inproc_transition(
    monkeypatch: pytest.MonkeyPatch,
    inproc_engine: bool,
    transitioned: bool,
    owns_transition: bool,
) -> None:
    engine = EngineCore.__new__(EngineCore)
    engine.inproc_engine = inproc_engine
    monkeypatch.setattr(
        core_module.envs,
        "_enable_envs_cache_with_ownership",
        lambda: transitioned,
    )

    engine._enable_envs_cache_with_ownership()

    assert engine._owns_envs_cache is owns_transition


@pytest.mark.parametrize("owns_envs_cache", [False, True])
def test_inproc_engine_shutdown_preserves_host_process_globals(
    monkeypatch: pytest.MonkeyPatch, owns_envs_cache: bool
) -> None:
    calls: list[tuple[str, bool | None, bool | None]] = []
    engine = EngineCore.__new__(EngineCore)
    engine.inproc_engine = True
    engine._owns_envs_cache = owns_envs_cache
    engine.structured_output_manager = None
    engine.model_executor = None
    engine.scheduler = None
    monkeypatch.setattr(
        core_module.gc,
        "unfreeze",
        lambda: calls.append(("direct_unfreeze", None, None)),
    )
    monkeypatch.setattr(
        core_module,
        "cleanup_dist_env_and_memory",
        lambda *, unfreeze_gc, reset_envs_cache: calls.append(
            ("distributed", unfreeze_gc, reset_envs_cache)
        ),
    )

    engine.shutdown()

    assert calls == [("distributed", False, owns_envs_cache)]
    assert not engine._owns_envs_cache


@pytest.mark.parametrize("inproc_engine", [False, True])
def test_gpu_worker_gc_lifecycle_respects_process_ownership(
    monkeypatch: pytest.MonkeyPatch, inproc_engine: bool
) -> None:
    calls: list[str] = []
    worker = gpu_worker_module.Worker.__new__(gpu_worker_module.Worker)
    worker.inproc_engine = inproc_engine
    worker.model_runner = SimpleNamespace(inproc_engine=None)
    monkeypatch.setattr(
        gpu_worker_module, "freeze_gc_heap", lambda: calls.append("freeze")
    )
    monkeypatch.setattr(
        gpu_worker_module,
        "maybe_attach_gc_debug_callback",
        lambda: calls.append("gc_debug"),
    )
    monkeypatch.setattr(
        gpu_worker_module.gc, "unfreeze", lambda: calls.append("unfreeze")
    )

    worker._propagate_process_ownership_to_model_runner()
    worker._freeze_gc_heap_if_owned()
    worker._attach_gc_debug_callback_if_owned()
    worker._unfreeze_gc_heap_if_owned()

    assert calls == (
        [] if inproc_engine else ["freeze", "gc_debug", "unfreeze"]
    )
    assert worker.model_runner.inproc_engine is inproc_engine


@pytest.mark.parametrize("inproc_engine", [False, True])
def test_cudagraph_gc_lifecycle_respects_process_ownership(
    monkeypatch: pytest.MonkeyPatch, inproc_engine: bool
) -> None:
    events: list[str] = []
    runner = gpu_model_runner_module.GPUModelRunner.__new__(
        gpu_model_runner_module.GPUModelRunner
    )
    runner.inproc_engine = inproc_engine
    monkeypatch.setattr(
        gpu_model_runner_module,
        "envs",
        SimpleNamespace(VLLM_ENABLE_CUDAGRAPH_GC=False),
    )
    monkeypatch.setattr(gpu_model_runner_module.gc, "collect", lambda: None)
    monkeypatch.setattr(
        gpu_model_runner_module.gc, "freeze", lambda: events.append("freeze")
    )
    monkeypatch.setattr(
        gpu_model_runner_module.gc, "unfreeze", lambda: events.append("unfreeze")
    )

    with runner._freeze_gc():
        events.append("body")

    assert events == (
        ["body"] if inproc_engine else ["freeze", "body", "unfreeze"]
    )


@pytest.mark.parametrize(
    ("operation", "args"),
    [
        ("pin_prefix", ("pin", object(), "gpu")),
        ("pause_prefix", ("pin",)),
        ("resume_prefix", ("pin",)),
        ("pause_requests", (["streaming-request"],)),
        ("resume_requests", (["request"],)),
    ],
)
def test_dp_utility_admission_wakes_coordinator_wave(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    args: tuple[Any, ...],
) -> None:
    result: Future[Any] = Future()
    monkeypatch.setattr(EngineCore, operation, lambda *_args: result)
    engine = DPEngineCoreProc.__new__(DPEngineCoreProc)
    engine.has_coordinator = True
    engine.engines_running = False
    engine.current_wave = 7
    engine.output_queue = queue.Queue()
    engine.scheduler = SimpleNamespace(
        pause_state=PauseState.UNPAUSED,
        has_requests=lambda: True,
    )

    returned = getattr(DPEngineCoreProc, operation)(engine, *args)

    assert returned is result
    assert engine.engines_running
    client_index, output = engine.output_queue.get_nowait()
    assert client_index == -1
    assert output.start_wave == 7


@pytest.mark.parametrize(
    ("pause_state", "has_work"),
    [
        (PauseState.PAUSED_ALL, True),
        (PauseState.UNPAUSED, False),
    ],
)
def test_dp_utility_wake_respects_pause_and_local_work(
    pause_state: PauseState,
    has_work: bool,
) -> None:
    engine = DPEngineCoreProc.__new__(DPEngineCoreProc)
    engine.has_coordinator = True
    engine.engines_running = False
    engine.current_wave = 9
    engine.output_queue = queue.Queue()
    engine.scheduler = SimpleNamespace(
        pause_state=pause_state,
        has_requests=lambda: has_work,
    )

    engine._wake_dp_wave_for_local_work()

    assert not engine.engines_running
    assert engine.output_queue.empty()


@pytest.mark.parametrize(
    ("pending_pause", "ignore_start", "pause_state"),
    [
        (True, False, PauseState.UNPAUSED),
        (False, True, PauseState.UNPAUSED),
        (False, False, PauseState.PAUSED_NEW),
        (False, False, PauseState.PAUSED_ALL),
    ],
)
def test_dp_local_work_wake_is_blocked_by_every_pause_lifecycle_guard(
    pending_pause: bool,
    ignore_start: bool,
    pause_state: PauseState,
) -> None:
    engine = DPEngineCoreProc.__new__(DPEngineCoreProc)
    engine.has_coordinator = True
    engine.engines_running = False
    engine.pending_pause = pending_pause
    engine.ignore_start_dp_wave = ignore_start
    engine.current_wave = 9
    engine.output_queue = queue.Queue()
    engine.scheduler = SimpleNamespace(
        pause_state=pause_state,
        has_requests=lambda: True,
    )

    engine._wake_dp_wave_for_local_work()

    assert not engine.engines_running
    assert engine.output_queue.empty()


@pytest.mark.parametrize(
    ("operation", "args"),
    [
        ("pause_requests", (["streaming-request"],)),
        ("pause_prefix", ("cpu-prefix",)),
    ],
)
@pytest.mark.parametrize("has_coordinator", [False, True])
def test_dp_pause_control_wakes_new_connector_backing_work(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    args: tuple[Any, ...],
    has_coordinator: bool,
) -> None:
    has_connector_work = False
    result: Future[None] = Future()

    def create_connector_work(*_args: Any) -> Future[None]:
        nonlocal has_connector_work
        has_connector_work = True
        return result

    monkeypatch.setattr(EngineCore, operation, create_connector_work)
    engine = DPEngineCoreProc.__new__(DPEngineCoreProc)
    engine.has_coordinator = has_coordinator
    engine.engines_running = False
    engine.current_wave = 10
    engine.output_queue = queue.Queue()
    engine.scheduler = SimpleNamespace(
        pause_state=PauseState.UNPAUSED,
        has_requests=lambda: has_connector_work,
    )

    assert getattr(engine, operation)(*args) is result
    assert engine.engines_running
    if has_coordinator:
        client_index, output = engine.output_queue.get_nowait()
        assert client_index == -1
        assert output.start_wave == 10
    else:
        assert engine.output_queue.empty()


def test_dp_idle_paused_prefix_unpin_wakes_cleanup_wave(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    has_work = False

    def unpin_prefix(
        _engine: EngineCore,
        pin_id: str,
        expected_request_id: str | None = None,
    ) -> bool:
        nonlocal has_work
        assert (pin_id, expected_request_id) == ("pin", "owner")
        has_work = True
        return True

    monkeypatch.setattr(EngineCore, "unpin_prefix", unpin_prefix)
    engine = DPEngineCoreProc.__new__(DPEngineCoreProc)
    engine.has_coordinator = True
    engine.engines_running = False
    engine.current_wave = 11
    engine.output_queue = queue.Queue()
    engine.scheduler = SimpleNamespace(
        pause_state=PauseState.UNPAUSED,
        has_requests=lambda: has_work,
    )

    assert engine.unpin_prefix("pin", "owner")

    assert engine.engines_running
    client_index, output = engine.output_queue.get_nowait()
    assert client_index == -1
    assert output.start_wave == 11


def test_dp_idle_paused_request_duplicate_abort_emits_single_wave(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    has_work = False
    abort_calls: list[list[str]] = []

    def abort_requests(_engine: EngineCore, request_ids: list[str]) -> None:
        nonlocal has_work
        abort_calls.append(request_ids)
        has_work = True

    monkeypatch.setattr(EngineCore, "abort_requests", abort_requests)
    engine = DPEngineCoreProc.__new__(DPEngineCoreProc)
    engine.has_coordinator = True
    engine.engines_running = False
    engine.current_wave = 13
    engine.output_queue = queue.Queue()
    engine.scheduler = SimpleNamespace(
        pause_state=PauseState.UNPAUSED,
        has_requests=lambda: has_work,
    )

    # An eager abort from admission and the already-queued frontend abort may
    # both arrive. The second call must not announce a duplicate DP wave.
    engine.abort_requests(["request"])
    engine.abort_requests(["request"])

    assert abort_calls == [["request"], ["request"]]
    client_index, output = engine.output_queue.get_nowait()
    assert client_index == -1
    assert output.start_wave == 13
    assert engine.output_queue.empty()


def test_dp_liveness_includes_connector_only_work() -> None:
    engine = DPEngineCoreProc.__new__(DPEngineCoreProc)
    engine.scheduler = SimpleNamespace(
        has_unfinished_requests=lambda: False,
        has_requests=lambda: True,
    )

    assert not engine.scheduler.has_unfinished_requests()
    assert engine._has_local_dp_work()


def test_sleep_cache_reset_preflight_runs_before_abort() -> None:
    calls: list[str] = []
    scheduler = SimpleNamespace(pause_state=PauseState.UNPAUSED)
    scheduler.can_reset_prefix_cache = lambda **_kwargs: False
    scheduler.finish_requests = lambda *_args: calls.append("abort")

    def set_pause_state(state: PauseState) -> None:
        calls.append("pause")
        scheduler.pause_state = state

    scheduler.set_pause_state = set_pause_state
    engine = EngineCore.__new__(EngineCore)
    engine.scheduler = scheduler
    engine.model_executor = SimpleNamespace(
        sleep=lambda _level: calls.append("executor_sleep")
    )

    with pytest.raises(RuntimeError, match="pinned prefixes"):
        engine.sleep(level=1, mode="abort")

    assert calls == []
    assert scheduler.pause_state == PauseState.UNPAUSED


def test_sleep_does_not_discard_kv_when_cache_reset_is_rejected() -> None:
    calls: list[str] = []
    scheduler = SimpleNamespace(pause_state=PauseState.UNPAUSED)
    scheduler.can_reset_prefix_cache = lambda **_kwargs: True

    def set_pause_state(state: PauseState) -> None:
        scheduler.pause_state = state

    scheduler.set_pause_state = set_pause_state
    engine = EngineCore.__new__(EngineCore)
    engine.scheduler = scheduler
    engine.model_executor = SimpleNamespace(
        sleep=lambda _level: calls.append("executor_sleep")
    )
    engine.reset_prefix_cache = lambda **_kwargs: False
    engine.reset_mm_cache = lambda: calls.append("mm_reset")
    engine.reset_encoder_cache = lambda: calls.append("encoder_reset")

    with pytest.raises(RuntimeError, match="Cannot reset the prefix cache"):
        engine.sleep(level=1, mode="keep")

    assert calls == []
    assert scheduler.pause_state == PauseState.UNPAUSED


def test_deferred_sleep_reset_failure_is_nonfatal_and_rolls_back() -> None:
    calls: list[str] = []
    scheduler = SimpleNamespace(pause_state=PauseState.UNPAUSED)
    scheduler.can_reset_prefix_cache = lambda **_kwargs: True

    def set_pause_state(state: PauseState) -> None:
        scheduler.pause_state = state

    scheduler.set_pause_state = set_pause_state
    engine = EngineCoreProc.__new__(EngineCoreProc)
    engine.scheduler = scheduler
    engine.model_executor = SimpleNamespace(
        sleep=lambda _level: calls.append("executor_sleep")
    )
    engine._idle_state_callbacks = []
    engine._pause_complete = lambda: False

    def reject_reset() -> None:
        raise RuntimeError("dynamic reset rejection")

    engine._reset_caches = reject_reset
    sleep_future = engine.sleep(level=1, mode="keep")
    assert isinstance(sleep_future, Future)
    assert scheduler.pause_state == PauseState.PAUSED_ALL

    callback = engine._idle_state_callbacks.pop()
    callback(engine)

    with pytest.raises(RuntimeError, match="dynamic reset rejection"):
        sleep_future.result()
    assert calls == []
    assert scheduler.pause_state == PauseState.UNPAUSED


def _make_dp_lifecycle_engine(
    *, local_allowed: bool = True
) -> tuple[DPEngineCoreProc, list[str], list[tuple[bool, bool]]]:
    lifecycle_calls: list[str] = []
    reset_calls: list[tuple[bool, bool]] = []
    scheduler = SimpleNamespace(
        pause_state=PauseState.UNPAUSED,
        requests={},
        can_reset_prefix_cache=lambda **_kwargs: local_allowed,
        has_requests=lambda: False,
        has_unfinished_requests=lambda: False,
    )

    def finish_requests(*_args: Any) -> list[Any]:
        lifecycle_calls.append("abort")
        return []

    def set_pause_state(state: PauseState) -> None:
        lifecycle_calls.append(f"pause:{state.name}")
        scheduler.pause_state = state

    def reset_prefix_cache(
        reset_running_requests: bool, reset_connector: bool
    ) -> bool:
        reset_calls.append((reset_running_requests, reset_connector))
        return True

    scheduler.finish_requests = finish_requests
    scheduler.set_pause_state = set_pause_state
    scheduler.reset_prefix_cache = reset_prefix_cache
    engine = DPEngineCoreProc.__new__(DPEngineCoreProc)
    engine.scheduler = scheduler
    engine.dp_group = object()
    engine.dp_size = 2
    engine.step_counter = 31
    engine.pending_pause = False
    engine.ignore_start_dp_wave = False
    engine._pending_dp_control_op = None
    engine._pending_dp_control_completion = None
    engine._dp_control_partial_checkpoints = 0
    engine._dp_control_partial_started_at = None
    engine._dp_control_partial_active = False
    engine.engines_running = False
    engine.has_coordinator = True
    engine.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_external_lb=False,
            data_parallel_hybrid_lb=False,
        )
    )
    engine.current_wave = 4
    engine.output_queue = queue.Queue()
    engine._idle_state_callbacks = []
    engine.reset_mm_cache = lambda: lifecycle_calls.append("reset:mm")
    engine.reset_encoder_cache = lambda: lifecycle_calls.append("reset:encoder")
    return engine, lifecycle_calls, reset_calls


def test_elastic_ep_group_commit_updates_control_consensus_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, lifecycle_calls, _ = _make_dp_lifecycle_engine()
    old_group = object()
    new_group = SimpleNamespace(rank=lambda: 1, size=lambda: 3)
    new_store = object()
    scaling_state = SimpleNamespace(
        old_dp_group=old_group,
        new_dp_group=new_group,
        new_dp_store=new_store,
        engine_core=engine,
    )
    ft_epoch_resets: list[None] = []
    engine.enable_fault_tolerance = True
    engine.ft_sentinel = SimpleNamespace(
        reset_dp_reinit_epoch=lambda: ft_epoch_resets.append(None)
    )
    destroyed: list[object] = []
    monkeypatch.setattr(
        elastic_state_module,
        "stateless_destroy_torch_distributed_process_group",
        destroyed.append,
    )
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *_args, **_kwargs: None)

    elastic_state_module.ElasticEPScalingState._commit_new_dp_group(
        cast(Any, scaling_state)
    )

    assert destroyed == [old_group]
    assert engine.dp_group is new_group
    assert engine.dp_rank == 1
    assert engine.dp_size == 3
    assert engine.dp_store is new_store
    assert ft_epoch_resets == [None]

    consensus_sizes: list[int] = []

    def sync_state(
        _group: object, **kwargs: Any
    ) -> tuple[bool, int, int, int, int, int]:
        signature = kwargs["control_op_signature"]
        consensus_sizes.append(engine.dp_size)
        pause_count = engine.dp_size if signature < 0 else 0
        return (
            False,
            pause_count,
            engine.dp_size,
            0,
            engine.dp_size * signature,
            engine.dp_size * signature * signature,
        )

    monkeypatch.setattr(core_module.ParallelConfig, "sync_dp_state", sync_state)

    pause_future = engine.pause_scheduler(mode="keep", clear_cache=False)
    assert isinstance(pause_future, Future)
    engine.engines_running = engine._has_global_unfinished_reqs(local_work=False)
    assert not pause_future.done()
    engine.step_counter = 63
    engine.engines_running = engine._has_global_unfinished_reqs(local_work=False)
    assert pause_future.result() is None
    assert engine.scheduler.pause_state == PauseState.PAUSED_ALL
    assert not engine.engines_running

    resume_future = engine.resume_scheduler()
    engine.step_counter = 95
    engine.engines_running = engine._has_global_unfinished_reqs(local_work=False)
    assert resume_future.result() is None
    assert engine.scheduler.pause_state == PauseState.UNPAUSED
    assert consensus_sizes == [3, 3, 3]
    assert lifecycle_calls == [
        "pause:PAUSED_ALL",
        "pause:UNPAUSED",
    ]


@pytest.mark.parametrize("method", ["reset_prefix_cache", "pause_scheduler"])
def test_dp_cache_utility_dispatch_only_publishes_intent(
    monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    engine, lifecycle_calls, reset_calls = _make_dp_lifecycle_engine()
    sync_calls: list[dict[str, Any]] = []
    engine.scheduler.can_reset_prefix_cache = lambda **_kwargs: (_ for _ in ()).throw(
        AssertionError("preflight ran during utility dispatch")
    )
    monkeypatch.setattr(
        core_module.ParallelConfig,
        "sync_dp_state",
        lambda _group, **kwargs: sync_calls.append(kwargs),
    )
    monkeypatch.setattr(
        core_module.ParallelConfig,
        "has_unfinished_dp",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("out-of-cadence collective")
        ),
    )

    if method == "reset_prefix_cache":
        result = engine.reset_prefix_cache(reset_connector=True)
    else:
        result = engine.pause_scheduler(mode="abort", clear_cache=True)

    assert isinstance(result, Future)
    assert not result.done()
    assert engine.engines_running
    assert sync_calls == []
    assert lifecycle_calls == []
    assert reset_calls == []


@pytest.mark.parametrize(
    "entrypoint",
    ["reset", "pause", "resume", "sleep", "wake"],
)
@pytest.mark.parametrize("topology", ["external", "hybrid"])
def test_non_internal_dp_global_controls_fail_before_collective_or_mutation(
    monkeypatch: pytest.MonkeyPatch, entrypoint: str, topology: str
) -> None:
    engine, lifecycle_calls, reset_calls = _make_dp_lifecycle_engine()
    # Both external and hybrid MoE topologies may have a wave coordinator;
    # neither frontend broadcasts lifecycle descriptors to every global rank.
    engine.has_coordinator = True
    engine.vllm_config.parallel_config.data_parallel_external_lb = (
        topology == "external"
    )
    engine.vllm_config.parallel_config.data_parallel_hybrid_lb = topology == "hybrid"
    executor_calls: list[str] = []
    engine.model_executor = SimpleNamespace(
        is_sleeping=False,
        sleep=lambda _level: executor_calls.append("sleep"),
        wake_up=lambda _tags: executor_calls.append("wake"),
    )
    monkeypatch.setattr(
        core_module.ParallelConfig,
        "sync_dp_state",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("external-LB control entered a collective")
        ),
    )

    if entrypoint == "reset":
        result = engine.reset_prefix_cache()
    elif entrypoint == "pause":
        result = engine.pause_scheduler()
    elif entrypoint == "resume":
        result = engine.resume_scheduler()
    elif entrypoint == "sleep":
        result = engine.sleep(level=1)
    else:
        result = engine.wake_up()

    assert isinstance(result, Future)
    with pytest.raises(RuntimeError, match="pure internal"):
        result.result()
    assert not engine.engines_running
    assert lifecycle_calls == []
    assert reset_calls == []
    assert executor_calls == []


@pytest.mark.parametrize("local_allowed", [False, True])
def test_dp_pause_blocker_rejects_before_lifecycle_mutation(
    monkeypatch: pytest.MonkeyPatch, local_allowed: bool
) -> None:
    engine, lifecycle_calls, reset_calls = _make_dp_lifecycle_engine(
        local_allowed=local_allowed
    )
    future = engine.pause_scheduler(mode="abort", clear_cache=True)
    assert isinstance(future, Future)
    operation = cast(Any, engine._pending_dp_control_op)
    sync_inputs: list[dict[str, Any]] = []

    def sync_state(_group: object, **kwargs: Any) -> tuple[bool, int, int, int, int, int]:
        sync_inputs.append(kwargs)
        signature = operation.signature
        return False, 0, 2, 1, 2 * signature, 2 * signature * signature

    monkeypatch.setattr(core_module.ParallelConfig, "sync_dp_state", sync_state)

    assert not engine._has_global_unfinished_reqs(local_work=False)
    with pytest.raises(RuntimeError, match="preflight was rejected"):
        future.result()
    assert sync_inputs[0]["control_op_blocked"] is (not local_allowed)
    assert lifecycle_calls == []
    assert reset_calls == []
    assert engine.scheduler.pause_state == PauseState.UNPAUSED


def test_dp_direct_reset_blocker_returns_false_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, lifecycle_calls, reset_calls = _make_dp_lifecycle_engine(
        local_allowed=False
    )
    future = engine.reset_prefix_cache(reset_connector=True)
    assert isinstance(future, Future)
    operation = cast(Any, engine._pending_dp_control_op)
    signature = operation.signature
    monkeypatch.setattr(
        core_module.ParallelConfig,
        "sync_dp_state",
        lambda *_args, **_kwargs: (
            False,
            0,
            2,
            1,
            2 * signature,
            2 * signature * signature,
        ),
    )

    assert not engine._has_global_unfinished_reqs(local_work=False)
    assert future.result() is False
    assert lifecycle_calls == []
    assert reset_calls == []


def test_dp_direct_reset_partial_intent_and_retry_converge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, lifecycle_calls, reset_calls = _make_dp_lifecycle_engine()
    future = engine.reset_prefix_cache(
        reset_running_requests=False, reset_connector=True
    )
    assert isinstance(future, Future)
    assert (
        engine.reset_prefix_cache(
            reset_running_requests=False, reset_connector=True
        )
        is future
    )
    operation = cast(Any, engine._pending_dp_control_op)
    signature = operation.signature
    fully_admitted = False

    def sync_state(
        *_args: Any, **_kwargs: Any
    ) -> tuple[bool, int, int, int, int, int]:
        count = 2 if fully_admitted else 1
        return (
            not fully_admitted,
            0,
            count,
            0,
            count * signature,
            count * signature * signature,
        )

    now = [0.0]
    monkeypatch.setattr(core_module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        core_module.ParallelConfig,
        "sync_dp_state",
        sync_state,
    )

    # Exceed the checkpoint floor without reaching the minimum wall-time.
    # A delayed peer descriptor must still be able to join this operation.
    for _ in range(core_module._DP_CONTROL_PARTIAL_CHECKPOINT_LIMIT + 2):
        engine.step_counter = 31
        assert engine._has_global_unfinished_reqs(local_work=False)
    assert not future.done()
    assert reset_calls == []

    fully_admitted = True
    now[0] = core_module._DP_CONTROL_PARTIAL_TIMEOUT_S - 0.001
    engine.step_counter = 31
    assert not engine._has_global_unfinished_reqs(local_work=False)
    assert future.result() is True
    assert reset_calls == [(False, True)]
    assert lifecycle_calls == []


def test_dp_uniform_partial_broadcast_times_out_without_mutation_and_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, lifecycle_calls, reset_calls = _make_dp_lifecycle_engine()
    late_future = engine.reset_prefix_cache(reset_connector=True)
    assert isinstance(late_future, Future)
    operation = cast(Any, engine._pending_dp_control_op)
    signature = operation.signature
    now = [0.0]
    monkeypatch.setattr(core_module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        core_module.ParallelConfig,
        "sync_dp_state",
        lambda *_args, **_kwargs: (
            True,
            0,
            1,
            0,
            signature,
            signature * signature,
        ),
    )

    for _ in range(core_module._DP_CONTROL_PARTIAL_CHECKPOINT_LIMIT):
        engine.step_counter = 31
        assert engine._has_global_unfinished_reqs(local_work=False)

    assert not late_future.done()
    assert (
        engine._dp_control_partial_checkpoints
        == core_module._DP_CONTROL_PARTIAL_CHECKPOINT_LIMIT
    )
    now[0] = core_module._DP_CONTROL_PARTIAL_TIMEOUT_S
    engine.step_counter = 31
    assert engine._has_global_unfinished_reqs(local_work=False)
    with pytest.raises(RuntimeError, match="did not reach every rank"):
        late_future.result()
    assert engine._pending_dp_control_op is None
    assert engine._dp_control_partial_checkpoints == 0
    assert lifecycle_calls == []
    assert reset_calls == []

    # A later complete broadcast, including one with the same public
    # parameters, is a fresh operation and can commit normally.
    fresh_future = engine.reset_prefix_cache(reset_connector=True)
    assert isinstance(fresh_future, Future)
    fresh_operation = cast(Any, engine._pending_dp_control_op)
    fresh_signature = fresh_operation.signature
    monkeypatch.setattr(
        core_module.ParallelConfig,
        "sync_dp_state",
        lambda *_args, **_kwargs: (
            False,
            0,
            2,
            0,
            2 * fresh_signature,
            2 * fresh_signature * fresh_signature,
        ),
    )
    engine.step_counter = 31
    assert not engine._has_global_unfinished_reqs(local_work=False)
    assert fresh_future.result() is True
    assert reset_calls == [(False, True)]


def test_dp_absent_rank_and_late_descriptor_each_leave_partial_wave(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, lifecycle_calls, reset_calls = _make_dp_lifecycle_engine()
    signature = 17
    now = [0.0]
    monkeypatch.setattr(core_module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        core_module.ParallelConfig,
        "sync_dp_state",
        lambda *_args, **_kwargs: (
            True,
            0,
            1,
            0,
            signature,
            signature * signature,
        ),
    )

    # This rank never received the original frontend descriptor, but observes
    # and exits the same bounded partial wave without local mutation.
    for _ in range(core_module._DP_CONTROL_PARTIAL_CHECKPOINT_LIMIT):
        engine.step_counter = 31
        assert engine._has_global_unfinished_reqs(local_work=False)
    assert (
        engine._dp_control_partial_checkpoints
        == core_module._DP_CONTROL_PARTIAL_CHECKPOINT_LIMIT
    )
    now[0] = core_module._DP_CONTROL_PARTIAL_TIMEOUT_S
    engine.step_counter = 31
    assert engine._has_global_unfinished_reqs(local_work=False)
    assert engine._dp_control_partial_checkpoints == 0
    assert engine._pending_dp_control_op is None

    # A descriptor delivered after that wave is independent and is also
    # detached when its peers never publish a matching intent.
    late = engine.reset_prefix_cache(reset_connector=True)
    assert isinstance(late, Future)
    signature = cast(Any, engine._pending_dp_control_op).signature
    now[0] = 10.0
    for _ in range(core_module._DP_CONTROL_PARTIAL_CHECKPOINT_LIMIT):
        engine.step_counter = 31
        assert engine._has_global_unfinished_reqs(local_work=False)
    assert not late.done()
    now[0] = 10.0 + core_module._DP_CONTROL_PARTIAL_TIMEOUT_S
    engine.step_counter = 31
    assert engine._has_global_unfinished_reqs(local_work=False)
    with pytest.raises(RuntimeError, match="did not reach every rank"):
        late.result()
    assert lifecycle_calls == []
    assert reset_calls == []


def test_dp_pending_control_yields_only_without_model_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, _, _ = _make_dp_lifecycle_engine()
    sleeps: list[float] = []
    monkeypatch.setattr(core_module.time, "sleep", sleeps.append)

    # A rank without local intent still yields after observing a partial wave.
    engine._dp_control_partial_active = True
    engine._yield_for_pending_dp_control(model_executed=False)
    engine._yield_for_pending_dp_control(model_executed=True)
    assert sleeps == [core_module._DP_CONTROL_IDLE_YIELD_S]

    # A locally pending operation yields before the first partial checkpoint.
    engine._dp_control_partial_active = False
    future = engine.reset_prefix_cache(reset_connector=True)
    assert isinstance(future, Future)
    engine._yield_for_pending_dp_control(model_executed=False)
    assert sleeps == [
        core_module._DP_CONTROL_IDLE_YIELD_S,
        core_module._DP_CONTROL_IDLE_YIELD_S,
    ]


def test_dp_state_incompatibility_is_collective_preflight_blocker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, lifecycle_calls, reset_calls = _make_dp_lifecycle_engine()
    # RESUME is locally incompatible, but still publishes the descriptor so
    # peers cannot be left in a permanent partial wave.
    future = engine.resume_scheduler()
    assert isinstance(future, Future)
    operation = cast(Any, engine._pending_dp_control_op)
    assert operation.state_blocked
    signature = operation.signature
    monkeypatch.setattr(
        core_module.ParallelConfig,
        "sync_dp_state",
        lambda *_args, **_kwargs: (
            False,
            0,
            2,
            1,
            2 * signature,
            2 * signature * signature,
        ),
    )

    assert not engine._has_global_unfinished_reqs(local_work=False)
    with pytest.raises(RuntimeError, match="preflight was rejected"):
        future.result()
    assert lifecycle_calls == []
    assert reset_calls == []


def test_dp_control_signature_mismatch_fails_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, lifecycle_calls, reset_calls = _make_dp_lifecycle_engine()
    future = engine.reset_prefix_cache(reset_connector=True)
    assert isinstance(future, Future)
    operation = cast(Any, engine._pending_dp_control_op)
    local_signature = operation.signature
    peer_signature = local_signature + 1
    monkeypatch.setattr(
        core_module.ParallelConfig,
        "sync_dp_state",
        lambda *_args, **_kwargs: (
            False,
            0,
            2,
            0,
            local_signature + peer_signature,
            local_signature * local_signature + peer_signature * peer_signature,
        ),
    )

    with pytest.raises(RuntimeError, match="different lifecycle"):
        engine._has_global_unfinished_reqs(local_work=False)
    # The engine loop owns fail-stop propagation; it deliberately leaves the
    # local Future pending for the MP output fatal path to terminate.
    assert not future.done()
    assert lifecycle_calls == []
    assert reset_calls == []


@pytest.mark.parametrize("control", ["pause_no_clear", "sleep_level"])
def test_dp_incompatible_concurrent_lifecycle_controls_fail_on_both_ranks(
    monkeypatch: pytest.MonkeyPatch, control: str
) -> None:
    rank0, lifecycle0, reset0 = _make_dp_lifecycle_engine()
    rank1, lifecycle1, reset1 = _make_dp_lifecycle_engine()
    executor_calls: list[tuple[int, int]] = []
    rank0.model_executor = SimpleNamespace(
        sleep=lambda level: executor_calls.append((0, level))
    )
    rank1.model_executor = SimpleNamespace(
        sleep=lambda level: executor_calls.append((1, level))
    )

    if control == "pause_no_clear":
        future0 = rank0.pause_scheduler(mode="keep", clear_cache=False)
        future1 = rank1.pause_scheduler(mode="abort", clear_cache=False)
        rejected0 = rank0.pause_scheduler(mode="abort", clear_cache=False)
        rejected1 = rank1.pause_scheduler(mode="keep", clear_cache=False)
    else:
        future0 = rank0.sleep(level=1, mode="abort")
        future1 = rank1.sleep(level=2, mode="abort")
        rejected0 = rank0.sleep(level=2, mode="abort")
        rejected1 = rank1.sleep(level=1, mode="abort")

    for rejected in (rejected0, rejected1):
        with pytest.raises(RuntimeError, match="different DP lifecycle"):
            rejected.result()

    operation0 = cast(Any, rank0._pending_dp_control_op)
    operation1 = cast(Any, rank1._pending_dp_control_op)
    signature_sum = operation0.signature + operation1.signature
    signature_square_sum = (
        operation0.signature * operation0.signature
        + operation1.signature * operation1.signature
    )
    monkeypatch.setattr(
        core_module.ParallelConfig,
        "sync_dp_state",
        lambda *_args, **_kwargs: (
            False,
            0,
            2,
            0,
            signature_sum,
            signature_square_sum,
        ),
    )

    for rank in (rank0, rank1):
        with pytest.raises(RuntimeError, match="different lifecycle"):
            rank._has_global_unfinished_reqs(local_work=False)
    for future in (future0, future1):
        assert not future.done()
    assert lifecycle0 == lifecycle1 == []
    assert reset0 == reset1 == []
    assert executor_calls == []


def test_dp_pause_applies_cache_reset_at_global_idle_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, lifecycle_calls, reset_calls = _make_dp_lifecycle_engine()
    future = engine.pause_scheduler(mode="abort", clear_cache=True)
    assert isinstance(future, Future)
    operation = cast(Any, engine._pending_dp_control_op)
    signature = operation.signature
    responses = [
        (False, 0, 2, 0, 2 * signature, 2 * signature * signature),
        (False, 2, 2, 0, -2 * signature, 2 * signature * signature),
    ]
    monkeypatch.setattr(
        core_module.ParallelConfig,
        "sync_dp_state",
        lambda *_args, **_kwargs: responses.pop(0),
    )

    assert engine._has_global_unfinished_reqs(local_work=False)
    assert lifecycle_calls == ["abort", "pause:PAUSED_NEW"]
    assert reset_calls == []
    assert engine.pending_pause
    assert not future.done()

    engine.step_counter = 63
    assert not engine._has_global_unfinished_reqs(local_work=False)
    assert future.result() is None
    assert reset_calls == [(True, True)]
    assert lifecycle_calls == [
        "abort",
        "pause:PAUSED_NEW",
        "reset:mm",
        "reset:encoder",
    ]
    assert engine.ignore_start_dp_wave
    assert not engine.pending_pause


def test_dp_pause_apply_failure_is_fail_stop_without_rank_local_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, lifecycle_calls, _reset_calls = _make_dp_lifecycle_engine()
    engine.scheduler.reset_prefix_cache = lambda *_args, **_kwargs: False
    future = engine.pause_scheduler(mode="abort", clear_cache=True)
    assert isinstance(future, Future)
    operation = cast(Any, engine._pending_dp_control_op)
    signature = operation.signature
    responses = [
        (False, 0, 2, 0, 2 * signature, 2 * signature * signature),
        (False, 2, 2, 0, -2 * signature, 2 * signature * signature),
    ]
    monkeypatch.setattr(
        core_module.ParallelConfig,
        "sync_dp_state",
        lambda *_args, **_kwargs: responses.pop(0),
    )

    assert engine._has_global_unfinished_reqs(local_work=False)
    engine.step_counter = 63
    with pytest.raises(RuntimeError, match="invariant violated"):
        engine._has_global_unfinished_reqs(local_work=False)

    assert not future.done()
    assert engine.scheduler.pause_state == PauseState.PAUSED_NEW
    assert lifecycle_calls == ["abort", "pause:PAUSED_NEW"]


def test_dp_pause_completion_can_immediately_submit_resume_without_lost_wake(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, _lifecycle_calls, _reset_calls = _make_dp_lifecycle_engine()

    def sync_state(
        _group: object, **kwargs: Any
    ) -> tuple[bool, int, int, int, int, int]:
        signature = kwargs["control_op_signature"]
        pause_count = engine.dp_size if signature < 0 else 0
        return (
            False,
            pause_count,
            engine.dp_size,
            0,
            engine.dp_size * signature,
            engine.dp_size * signature * signature,
        )

    monkeypatch.setattr(core_module.ParallelConfig, "sync_dp_state", sync_state)
    pause_future = engine.pause_scheduler(mode="keep", clear_cache=False)
    callback_state: dict[str, Any] = {}

    def resume_immediately(_completed: Future[Any]) -> None:
        callback_state.update(
            pending_pause=engine.pending_pause,
            ignore_start_dp_wave=engine.ignore_start_dp_wave,
            engines_running=engine.engines_running,
            pause_state=engine.scheduler.pause_state,
        )
        callback_state["resume_future"] = engine.resume_scheduler()
        callback_state["resume_operation"] = engine._pending_dp_control_op

    pause_future.add_done_callback(resume_immediately)
    assert engine._has_global_unfinished_reqs(local_work=False)
    assert not pause_future.done()

    engine.step_counter = 63
    # Publishing pause completion synchronously submits resume. The old pause
    # cadence must not overwrite the wake performed by that new descriptor.
    assert engine._has_global_unfinished_reqs(local_work=False)
    assert pause_future.result() is None
    assert callback_state["pending_pause"] is False
    assert callback_state["ignore_start_dp_wave"] is True
    assert callback_state["engines_running"] is False
    assert callback_state["pause_state"] == PauseState.PAUSED_ALL
    resume_operation = cast(Any, callback_state["resume_operation"])
    assert not resume_operation.state_blocked
    assert engine._pending_dp_control_op is resume_operation
    assert engine.engines_running

    engine.step_counter = 95
    assert engine._has_global_unfinished_reqs(local_work=False)
    assert cast(Future[Any], callback_state["resume_future"]).result() is None
    assert engine.scheduler.pause_state == PauseState.UNPAUSED


@pytest.mark.parametrize("operation_kind", ["resume", "wake"])
def test_dp_partial_timeout_callback_can_immediately_retry_resume_or_wake(
    monkeypatch: pytest.MonkeyPatch,
    operation_kind: str,
) -> None:
    engine, _lifecycle_calls, _reset_calls = _make_dp_lifecycle_engine()
    engine.scheduler.pause_state = PauseState.PAUSED_ALL
    engine.ignore_start_dp_wave = True
    engine.engines_running = False
    if operation_kind == "wake":
        executor = SimpleNamespace(is_sleeping=True)

        def wake_up(_tags: list[str] | None) -> None:
            executor.is_sleeping = False

        executor.wake_up = wake_up
        engine.model_executor = executor

    def submit() -> Future[Any]:
        if operation_kind == "resume":
            return engine.resume_scheduler()
        future = engine.wake_up(["weights"])
        assert isinstance(future, Future)
        return future

    first_future = submit()
    first_operation = cast(Any, engine._pending_dp_control_op)
    first_signature = first_operation.signature
    engine._dp_control_partial_checkpoints = (
        core_module._DP_CONTROL_PARTIAL_CHECKPOINT_LIMIT - 1
    )
    engine._dp_control_partial_started_at = 0.0
    monkeypatch.setattr(
        core_module.time,
        "monotonic",
        lambda: core_module._DP_CONTROL_PARTIAL_TIMEOUT_S,
    )
    monkeypatch.setattr(
        core_module.ParallelConfig,
        "sync_dp_state",
        lambda *_args, **_kwargs: (
            False,
            engine.dp_size,
            1,
            0,
            first_signature,
            first_signature * first_signature,
        ),
    )
    callback_state: dict[str, Any] = {}

    def retry_immediately(completed: Future[Any]) -> None:
        completed.exception()
        callback_state["future"] = submit()
        callback_state["operation"] = engine._pending_dp_control_op

    first_future.add_done_callback(retry_immediately)
    assert engine._has_global_unfinished_reqs(local_work=False)
    with pytest.raises(RuntimeError, match="did not reach every rank"):
        first_future.result()

    retry_operation = cast(Any, callback_state["operation"])
    assert retry_operation is engine._pending_dp_control_op
    assert not retry_operation.state_blocked
    assert engine.engines_running
    retry_signature = retry_operation.signature
    monkeypatch.setattr(
        core_module.ParallelConfig,
        "sync_dp_state",
        lambda *_args, **_kwargs: (
            False,
            0,
            engine.dp_size,
            0,
            engine.dp_size * retry_signature,
            engine.dp_size * retry_signature * retry_signature,
        ),
    )
    engine.step_counter = 63
    assert engine._has_global_unfinished_reqs(local_work=False)
    assert cast(Future[Any], callback_state["future"]).result() is None
    assert engine.scheduler.pause_state == PauseState.UNPAUSED


def test_dp_resume_uses_fixed_cadence_descriptor_and_forces_followup_wave(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, lifecycle_calls, reset_calls = _make_dp_lifecycle_engine()
    engine.scheduler.pause_state = PauseState.PAUSED_NEW
    engine.ignore_start_dp_wave = True
    monkeypatch.setattr(
        core_module.ParallelConfig,
        "has_unfinished_dp",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("legacy resume collective")
        ),
    )

    future = engine.resume_scheduler()
    operation = cast(Any, engine._pending_dp_control_op)
    signature = operation.signature
    assert engine.scheduler.pause_state == PauseState.PAUSED_NEW
    assert engine.ignore_start_dp_wave
    assert not future.done()
    responses = [
        (True, 0, 1, 0, signature, signature * signature),
        (False, 0, 2, 0, 2 * signature, 2 * signature * signature),
    ]
    monkeypatch.setattr(
        core_module.ParallelConfig,
        "sync_dp_state",
        lambda *_args, **_kwargs: responses.pop(0),
    )

    assert engine._has_global_unfinished_reqs(local_work=False)
    assert engine.scheduler.pause_state == PauseState.PAUSED_NEW
    engine.step_counter = 63
    assert engine._has_global_unfinished_reqs(local_work=False)
    assert future.result() is None
    assert engine.scheduler.pause_state == PauseState.UNPAUSED
    assert not engine.ignore_start_dp_wave
    assert reset_calls == []
    assert lifecycle_calls == ["pause:UNPAUSED"]


def test_dp_sleep_applies_executor_level_once_at_global_idle_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, lifecycle_calls, _reset_calls = _make_dp_lifecycle_engine()
    sleep_calls: list[int] = []
    engine.model_executor = SimpleNamespace(
        sleep=lambda level: sleep_calls.append(level)
    )
    future = engine.sleep(level=2, mode="abort")
    duplicate = engine.sleep(level=2, mode="abort")
    assert future is duplicate
    operation = cast(Any, engine._pending_dp_control_op)
    signature = operation.signature
    responses = [
        (False, 0, 2, 0, 2 * signature, 2 * signature * signature),
        (False, 2, 2, 0, -2 * signature, 2 * signature * signature),
    ]
    monkeypatch.setattr(
        core_module.ParallelConfig,
        "sync_dp_state",
        lambda *_args, **_kwargs: responses.pop(0),
    )

    assert engine._has_global_unfinished_reqs(local_work=False)
    assert sleep_calls == []
    engine.step_counter = 63
    assert not engine._has_global_unfinished_reqs(local_work=False)
    assert future.result() is None
    assert sleep_calls == [2]
    assert lifecycle_calls[:2] == ["abort", "pause:PAUSED_NEW"]


def test_dp_wake_mutates_executor_only_after_descriptor_consensus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, lifecycle_calls, _reset_calls = _make_dp_lifecycle_engine()
    engine.scheduler.pause_state = PauseState.PAUSED_ALL
    engine.ignore_start_dp_wave = True
    wake_calls: list[list[str] | None] = []
    executor = SimpleNamespace(is_sleeping=True)

    def wake(tags: list[str] | None) -> None:
        wake_calls.append(tags)
        executor.is_sleeping = False

    executor.wake_up = wake
    engine.model_executor = executor
    future = engine.wake_up(["weights", "scheduling", "weights"])
    assert isinstance(future, Future)
    assert wake_calls == []
    operation = cast(Any, engine._pending_dp_control_op)
    signature = operation.signature
    monkeypatch.setattr(
        core_module.ParallelConfig,
        "sync_dp_state",
        lambda *_args, **_kwargs: (
            False,
            0,
            2,
            0,
            2 * signature,
            2 * signature * signature,
        ),
    )

    assert engine._has_global_unfinished_reqs(local_work=False)
    assert future.result() is None
    assert wake_calls == [["weights"]]
    assert engine.scheduler.pause_state == PauseState.UNPAUSED
    assert not engine.ignore_start_dp_wave
    assert lifecycle_calls == ["pause:UNPAUSED"]


def test_dp_paused_all_still_reports_finished_or_connector_work() -> None:
    engine, _lifecycle_calls, _reset_calls = _make_dp_lifecycle_engine()
    engine.scheduler.pause_state = PauseState.PAUSED_ALL
    engine.scheduler.has_requests = lambda: True

    assert engine._has_local_dp_work()


def test_elastic_scale_up_new_rank_aligns_before_descriptor_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, lifecycle_calls, _reset_calls = _make_dp_lifecycle_engine()
    engine._align_new_elastic_ep_rank_pause_state()

    assert engine.scheduler.pause_state == PauseState.PAUSED_ALL
    assert engine.ignore_start_dp_wave
    assert not engine.pending_pause
    assert not engine.engines_running
    future = engine.resume_scheduler()
    operation = cast(Any, engine._pending_dp_control_op)
    assert not operation.state_blocked
    signature = operation.signature
    monkeypatch.setattr(
        core_module.ParallelConfig,
        "sync_dp_state",
        lambda *_args, **_kwargs: (
            False,
            0,
            2,
            0,
            2 * signature,
            2 * signature * signature,
        ),
    )

    assert engine._has_global_unfinished_reqs(local_work=False)
    assert future.result() is None
    assert engine.scheduler.pause_state == PauseState.UNPAUSED
    assert lifecycle_calls == ["pause:PAUSED_ALL", "pause:UNPAUSED"]


@pytest.mark.asyncio
async def test_elastic_scale_down_uses_uniform_pause_then_removed_rank_abort() -> None:
    client = core_client_module.DPLBAsyncMPClient.__new__(
        core_client_module.DPLBAsyncMPClient
    )
    old_engines = [b"rank-0", b"rank-1", b"rank-2"]
    client.core_engines = old_engines.copy()
    client.lb_engines = [[0, 0, 0.0] for _ in old_engines]
    client.client_count = 1
    client.resources = SimpleNamespace(engine_dead=False)
    client._elastic_ep_routing_limit = None
    client.prefix_pins = {}
    lifecycle_calls: list[tuple[str, tuple[Any, ...], tuple[bytes, ...]]] = []
    local_calls: list[tuple[str, tuple[Any, ...], bytes]] = []

    async def lifecycle(
        method: str,
        args: tuple[Any, ...],
        engines: tuple[bytes, ...],
    ) -> None:
        lifecycle_calls.append((method, args, engines))

    async def local(method: str, *args: Any, engine: bytes) -> None:
        local_calls.append((method, args, engine))

    client._broadcast_dp_lifecycle_utility = lifecycle
    client._call_utility_async = local

    returned_engines, removed = await client._quiesce_scale_down_elastic_ep(2)

    assert returned_engines == old_engines
    assert removed == 1
    assert lifecycle_calls == [
        ("pause_scheduler", ("keep", False), tuple(old_engines))
    ]
    assert local_calls == [
        ("abort_for_elastic_ep_scale_down", (2,), b"rank-2")
    ]
    assert client.core_engines == old_engines[:2]
    assert len(client.lb_engines) == 2
    assert client._elastic_ep_routing_limit == 2


def test_elastic_scale_down_stats_and_explicit_routing_obey_gate() -> None:
    client = core_client_module.DPLBAsyncMPClient.__new__(
        core_client_module.DPLBAsyncMPClient
    )
    client.core_engines = [b"rank-0", b"rank-1", b"rank-2"]
    client.engine_ranks_managed = [0, 1, 2]
    client._engine_status = {
        rank: {"id": rank, "status": "healthy"}
        for rank in client.engine_ranks_managed
    }
    client.engine_ranks_managed = [0, 1, 2]
    client.resources = SimpleNamespace(engine_dead=False)
    client._elastic_ep_routing_limit = 2

    counts, count_slice = client._slice_managed_engine_counts(
        [[1, 0, 0.1], [2, 0, 0.2], [99, 99, 1.0]]
    )

    assert count_slice == slice(0, 3)
    assert counts == [[1, 0, 0.1], [2, 0, 0.2]]
    client.lb_engines = counts
    request = SimpleNamespace(
        request_id="removed-rank",
        data_parallel_rank=2,
        pooling_params=None,
    )
    with pytest.raises(ValueError, match="not routable"):
        client.get_core_engine_for_request(cast(Any, request))


def test_elastic_topology_keeps_fault_status_ranks_in_sync() -> None:
    client = core_client_module.DPAsyncMPClient.__new__(
        core_client_module.DPAsyncMPClient
    )
    rank_zero = {"id": 0, "status": "unhealthy", "reason": "test"}
    client.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(enable_fault_tolerance=True)
    )
    client._engine_status = {
        0: rank_zero,
        1: {"id": 1, "status": "healthy"},
        2: {"id": 2, "status": "healthy"},
    }

    client._apply_engine_rank_topology([0, 1, 3])

    assert client.engine_ranks_managed == [0, 1, 3]
    assert client._engine_status == {
        0: rank_zero,
        1: {"id": 1, "status": "healthy"},
        3: {"id": 3, "status": "healthy"},
    }


@pytest.mark.asyncio
async def test_elastic_scale_down_pause_rejection_restores_routing_gate() -> None:
    client = core_client_module.DPLBAsyncMPClient.__new__(
        core_client_module.DPLBAsyncMPClient
    )
    old_engines = [b"rank-0", b"rank-1", b"rank-2"]
    old_counts = [[rank, 0, 0.0] for rank in range(3)]
    client.core_engines = old_engines.copy()
    client.lb_engines = [counts.copy() for counts in old_counts]
    client.client_count = 1
    client.resources = SimpleNamespace(engine_dead=False)
    client._elastic_ep_routing_limit = None
    client.prefix_pins = {}

    async def reject_pause(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("common pause preflight rejected")

    client._broadcast_dp_lifecycle_utility = reject_pause

    with pytest.raises(RuntimeError, match="preflight rejected"):
        await client._quiesce_scale_down_elastic_ep(2)
    assert client.core_engines == old_engines
    assert client.lb_engines == old_counts
    assert client._elastic_ep_routing_limit is None


def _make_elastic_ep_transaction_client() -> Any:
    client = core_client_module.DPLBAsyncMPClient.__new__(
        core_client_module.DPLBAsyncMPClient
    )
    client.core_engines = [b"rank-0", b"rank-1", b"rank-2"]
    client.engine_ranks_managed = [0, 1, 2]
    client._engine_status = {
        rank: {"id": rank, "status": "healthy"}
        for rank in client.engine_ranks_managed
    }
    client.lb_engines = [[0, 0, 0.0] for _ in client.core_engines]
    client.client_count = 1
    client.client_index = 0
    client.resources = SimpleNamespace(engine_dead=False)
    client.utility_results = {}
    client.outputs_queue = asyncio.Queue()
    client.prefix_pins = {}
    client._prepared_elastic_ep = (2, 7)
    client._elastic_ep_transaction_pending = False
    client._elastic_ep_transaction_active = True
    client._elastic_ep_transaction_failed = False
    client._elastic_ep_commit_in_progress = False
    client._elastic_ep_prepare_mutated = False
    client._elastic_ep_commit_mutated = False
    client._elastic_ep_prepare_tasks = set()
    client._elastic_ep_commit_tasks = set()
    client._elastic_ep_fail_stop_tasks = set()
    client._dp_lifecycle_broadcast_tasks = set()
    client._dp_fault_recovery_tasks = set()
    client._elastic_ep_notification_tasks = set()
    client._elastic_ep_shutdown_requested = False
    client._shutdown_lock = threading.RLock()
    client._shutdown_complete = False
    client._elastic_ep_routing_limit = None
    client.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_backend="ray",
            data_parallel_size=3,
            data_parallel_size_local=3,
            data_parallel_rank=0,
            data_parallel_hybrid_lb=False,
            local_engines_only=False,
            enable_fault_tolerance=False,
            # Three ranks with twelve physical slots can safely scale to two
            # ranks while preserving all eight logical experts.
            eplb_config=SimpleNamespace(num_redundant_experts=4),
        ),
        model_config=SimpleNamespace(get_num_experts=lambda: 8),
    )
    return client


@pytest.mark.asyncio
async def test_dplb_fault_recovery_is_blocked_by_elastic_ep_transaction() -> None:
    client = _make_elastic_ep_transaction_client()

    with pytest.raises(RuntimeError, match="elastic-EP reconfiguration"):
        await client.call_utility_async(
            core_client_module.FT_UTILITY_METHOD,
            object(),
        )


@pytest.mark.asyncio
async def test_dplb_fault_recovery_bypasses_stalled_lifecycle_lock() -> None:
    client = _make_elastic_ep_transaction_client()
    client._prepared_elastic_ep = None
    client._elastic_ep_transaction_active = False
    client._elastic_ep_transaction_pending = True
    lifecycle_lock = client._get_dp_lifecycle_broadcast_lock()
    await lifecycle_lock.acquire()
    calls: list[tuple[str, tuple[Any, ...], tuple[bytes, ...]]] = []
    request = core_client_module.FaultToleranceRequest(
        instruction="retry", params={}, request_id="recovered"
    )

    async def broadcast(
        method: str,
        args: tuple[Any, ...],
        engines: tuple[bytes, ...],
    ) -> list[dict[str, Any]]:
        calls.append((method, args, engines))
        return [
            {"request_id": "recovered", "success": True}
            for _engine in engines
        ]

    client._broadcast_dp_lifecycle_utility = broadcast
    try:
        result = await asyncio.wait_for(
            client.call_utility_async(
                core_client_module.FT_UTILITY_METHOD,
                request,
            ),
            1.0,
        )
    finally:
        lifecycle_lock.release()

    assert result.success
    assert result.request_id == "recovered"
    assert calls == [
        (
            core_client_module.FT_UTILITY_METHOD,
            (request,),
            tuple(client.core_engines),
        )
    ]
    assert client._dp_fault_recovery_tasks == set()


@pytest.mark.asyncio
async def test_dplb_fault_recovery_aggregates_every_rank() -> None:
    client = _make_elastic_ep_transaction_client()
    client.engine_ranks_managed = [0, 1, 2]
    client._engine_status = {
        rank: {"id": rank, "status": "unhealthy"}
        for rank in client.engine_ranks_managed
    }
    request = core_client_module.FaultToleranceRequest(
        instruction="retry",
        params={},
        request_id="ft-request",
    )

    async def broadcast(
        _method: str, _args: tuple[Any, ...], _engines: tuple[bytes, ...]
    ) -> list[dict[str, Any]]:
        return [
            {"request_id": "ft-request", "success": True},
            {
                "request_id": "ft-request",
                "success": False,
                "reason": "worker recovery failed",
            },
            {"request_id": "wrong-request", "success": True},
        ]

    client._prepared_elastic_ep = None
    client._elastic_ep_transaction_active = False
    client._elastic_ep_transaction_pending = True
    client._broadcast_dp_lifecycle_utility = broadcast
    result = await client.handle_fault(request)

    assert not result.success
    assert result.reason is not None
    assert "rank 1: worker recovery failed" in result.reason
    assert "rank 2: mismatched request id" in result.reason
    assert client._engine_status[1]["last_ft_request_id"] == "ft-request"
    assert client._engine_status[2]["last_ft_request_id"] == "wrong-request"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw_results", "message"),
    [
        (None, "did not return per-rank results"),
        ([], "result count does not match"),
        ([{"request_id": "ft-request"}], "invalid per-rank result"),
    ],
)
async def test_dplb_fault_recovery_protocol_failures_mark_every_rank_unhealthy(
    raw_results: Any,
    message: str,
) -> None:
    client = _make_elastic_ep_transaction_client()
    client._prepared_elastic_ep = None
    client._elastic_ep_transaction_active = False
    client._elastic_ep_transaction_pending = True
    request = core_client_module.FaultToleranceRequest(
        instruction="retry", params={}, request_id="ft-request"
    )

    async def broadcast(*_args: Any, **_kwargs: Any) -> Any:
        return raw_results

    client._broadcast_dp_lifecycle_utility = broadcast
    with pytest.raises(RuntimeError, match=message):
        await client.handle_fault(request)

    assert {
        rank: status["status"] for rank, status in client._engine_status.items()
    } == {0: "unhealthy", 1: "unhealthy", 2: "unhealthy"}
    assert all(
        status["last_ft_request_id"] == "ft-request"
        for status in client._engine_status.values()
    )


def test_elastic_scale_up_partial_reservation_rolls_back_all_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    placement_group = object()
    removed: list[object] = []
    fake_ray = SimpleNamespace(
        util=SimpleNamespace(remove_placement_group=removed.append)
    )
    monkeypatch.setitem(sys.modules, "ray", fake_ray)

    manager = engine_utils_module.CoreEngineActorManager.__new__(
        engine_utils_module.CoreEngineActorManager
    )
    manager.local_engine_actors = [object(), object()]
    manager.remote_engine_actors = []
    manager.created_placement_groups = []
    manager.placement_group_is_local = [True, True]
    manager._actor_mutation_lock = threading.RLock()
    manager.manager_stopped = threading.Event()

    def return_partial_topology(
        _config: Any,
        _new_size: int,
        on_created: Any,
    ) -> tuple[list[object], list[int]]:
        on_created(placement_group, True)
        return [placement_group], [2]

    manager.add_dp_placement_groups = return_partial_topology
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=2)
    )

    with pytest.raises(RuntimeError, match="complete target topology"):
        manager.reserve_scale_up_elastic_ep(cast(Any, config), 4)

    assert removed == [placement_group]
    assert manager.created_placement_groups == []
    assert manager.placement_group_is_local == [True, True]
    assert len(manager.local_engine_actors) == 2


def test_elastic_scale_up_reservation_cleanup_failure_remains_owned_for_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    placement_group = object()
    cleanup_fails = True
    remove_calls: list[object] = []

    def remove_placement_group(pg: object) -> None:
        remove_calls.append(pg)
        if cleanup_fails:
            raise RuntimeError("Ray placement-group cleanup failed")

    fake_ray = SimpleNamespace(
        util=SimpleNamespace(remove_placement_group=remove_placement_group)
    )
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    manager = engine_utils_module.CoreEngineActorManager.__new__(
        engine_utils_module.CoreEngineActorManager
    )
    manager.local_engine_actors = [object(), object()]
    manager.remote_engine_actors = []
    manager.created_placement_groups = []
    manager.placement_group_is_local = [True, True]
    manager._actor_mutation_lock = threading.RLock()
    manager.manager_stopped = threading.Event()

    def return_partial_topology(
        _config: Any,
        _new_size: int,
        on_created: Any,
    ) -> tuple[list[object], list[int]]:
        on_created(placement_group, True)
        return [placement_group], [2]

    manager.add_dp_placement_groups = return_partial_topology
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=2)
    )

    with pytest.raises(
        engine_utils_module.ElasticEPScaleUpReservationError,
        match="shutdown is required",
    ):
        manager.reserve_scale_up_elastic_ep(cast(Any, config), 4)

    assert manager.created_placement_groups == [placement_group]
    assert manager.placement_group_is_local == [True, True, True]
    cleanup_fails = False
    assert manager._rollback_elastic_ep_scale_up([], [placement_group])
    assert remove_calls == [placement_group, placement_group]
    assert manager.created_placement_groups == []
    assert manager.placement_group_is_local == [True, True]


def test_elastic_scale_up_reservation_waits_until_placement_groups_are_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Any] = []

    class PlacementGroup:
        def ready(self) -> str:
            calls.append("ready")
            return "ready-ref"

    placement_group = PlacementGroup()

    def wait(
        refs: list[str], *, num_returns: int, timeout: float
    ) -> tuple[list[str], list[str]]:
        calls.append(("wait", refs, num_returns, timeout))
        return refs, []

    fake_ray = SimpleNamespace(
        wait=wait,
        get=lambda refs: calls.append(("get", refs)),
        util=SimpleNamespace(
            remove_placement_group=lambda pg: calls.append(("remove", pg))
        ),
    )
    monkeypatch.setitem(sys.modules, "ray", fake_ray)

    manager = engine_utils_module.CoreEngineActorManager.__new__(
        engine_utils_module.CoreEngineActorManager
    )
    manager.local_engine_actors = [object(), object()]
    manager.remote_engine_actors = []
    manager.created_placement_groups = []
    manager.placement_group_is_local = [True, True]
    manager._actor_mutation_lock = threading.RLock()
    manager.manager_stopped = threading.Event()

    def return_complete_topology(
        _config: Any,
        _new_size: int,
        on_created: Any,
    ) -> tuple[list[PlacementGroup], list[int]]:
        on_created(placement_group, True)
        return [placement_group], [2]

    manager.add_dp_placement_groups = return_complete_topology
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=2)
    )

    reservation = manager.reserve_scale_up_elastic_ep(cast(Any, config), 3)

    assert calls[0] == "ready"
    assert calls[1][0:3] == ("wait", ["ready-ref"], 1)
    assert calls[2] == ("get", ["ready-ref"])
    assert reservation.state == "reserved"
    manager.release_scale_up_elastic_ep_reservation(reservation)
    assert calls[3] == ("remove", placement_group)
    assert reservation.state == "released"
    assert manager.created_placement_groups == []
    assert manager.placement_group_is_local == [True, True]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected_mutated"),
    [
        (RuntimeError("insufficient Ray resources"), False),
        (
            engine_utils_module.ElasticEPScaleUpReservationError(
                "reservation cleanup failed"
            ),
            True,
        ),
    ],
)
async def test_elastic_scale_up_reservation_failure_does_not_touch_old_ranks(
    error: RuntimeError,
    expected_mutated: bool,
) -> None:
    client = _make_elastic_ep_transaction_client()
    manager = engine_utils_module.CoreEngineActorManager.__new__(
        engine_utils_module.CoreEngineActorManager
    )
    calls: list[str] = []

    def reject_reservation(_config: Any, _new_size: int) -> None:
        calls.append("reserve")
        raise error

    manager.reserve_scale_up_elastic_ep = reject_reservation
    client.resources.engine_manager = manager
    client._setup_elastic_ep_reconfig_bootstrap = lambda: calls.append(
        "bootstrap"
    )

    async def reconfigure(*_args: Any, **_kwargs: Any) -> None:
        calls.append("reconfigure")

    client._call_utility_async = reconfigure

    with pytest.raises(type(error), match=str(error)):
        await client._prepare_scale_up_elastic_ep(4, 4)

    assert calls == ["reserve"]
    assert client._elastic_ep_prepare_mutated is expected_mutated


@pytest.mark.asyncio
async def test_elastic_ep_prepare_rechecks_fault_health_before_activation() -> None:
    client = _make_elastic_ep_transaction_client()
    client._prepared_elastic_ep = None
    client._elastic_ep_transaction_active = False
    client.vllm_config.parallel_config.enable_fault_tolerance = True
    client.engine_ranks_managed = [0, 1, 2]
    client._engine_status = {
        0: {"id": 0, "status": "healthy"},
        1: {"id": 1, "status": "unhealthy"},
        2: {"id": 2, "status": "healthy"},
    }
    prepare_calls: list[int] = []

    async def utility(_method: str, *_args: Any, engine: bytes) -> bool:
        return False

    async def prepare_scale_down(new_size: int) -> None:
        prepare_calls.append(new_size)

    client._call_utility_async = utility
    client._prepare_scale_down_elastic_ep = prepare_scale_down

    with pytest.raises(RuntimeError, match=r"unhealthy ranks: \[1\]"):
        await client.prepare_elastic_ep(2)

    assert prepare_calls == []
    assert not client._elastic_ep_transaction_pending
    assert not client._elastic_ep_transaction_active


@pytest.mark.asyncio
@pytest.mark.parametrize("new_size", [True, 0, -1, 65537, 1.5])
async def test_elastic_ep_prepare_rejects_invalid_target_size(new_size: Any) -> None:
    client = _make_elastic_ep_transaction_client()
    client._prepared_elastic_ep = None
    client._elastic_ep_transaction_active = False

    with pytest.raises(ValueError, match="new_data_parallel_size"):
        await client.prepare_elastic_ep(new_size)

    assert not client._elastic_ep_transaction_pending
    assert not client._elastic_ep_transaction_active


@pytest.mark.asyncio
async def test_elastic_ep_prepare_same_size_is_side_effect_free() -> None:
    client = _make_elastic_ep_transaction_client()
    client._prepared_elastic_ep = None
    client._elastic_ep_transaction_active = False
    client._call_utility_async = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("same-size prepare reached EngineCore")
    )

    await client.prepare_elastic_ep(len(client.core_engines))

    assert not client._elastic_ep_transaction_pending
    assert not client._elastic_ep_transaction_active


@pytest.mark.asyncio
async def test_elastic_ep_prepare_rejects_insufficient_expert_capacity() -> None:
    client = _make_elastic_ep_transaction_client()
    client._prepared_elastic_ep = None
    client._elastic_ep_transaction_active = False
    client.vllm_config.parallel_config.eplb_config.num_redundant_experts = 0

    with pytest.raises(ValueError, match="fewer physical expert slots"):
        await client.prepare_elastic_ep(2)

    assert not client._elastic_ep_transaction_pending
    assert not client._elastic_ep_transaction_active


def test_elastic_ep_rejects_single_core_initial_topology() -> None:
    with pytest.raises(ValueError, match="initial data_parallel_size"):
        ParallelConfig(
            data_parallel_size=1,
            data_parallel_size_local=0,
            enable_eplb=True,
            enable_elastic_ep=True,
        )


@pytest.mark.asyncio
async def test_elastic_ep_shutdown_cancels_all_owned_control_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_elastic_ep_transaction_client()
    client._elastic_ep_transaction_pending = True
    client._elastic_ep_commit_in_progress = True
    client._prepared_elastic_ep = (2, 7)
    never = asyncio.Event()

    async def blocked() -> None:
        await never.wait()

    task_sets = (
        client._elastic_ep_prepare_tasks,
        client._elastic_ep_commit_tasks,
        client._elastic_ep_fail_stop_tasks,
        client._dp_lifecycle_broadcast_tasks,
        client._dp_fault_recovery_tasks,
        client._elastic_ep_notification_tasks,
    )
    tasks = [asyncio.create_task(blocked()) for _ in task_sets]
    for task_set, task in zip(task_sets, tasks):
        task_set.add(task)

    utility_future = asyncio.get_running_loop().create_future()
    client.utility_results[77] = utility_future
    base_shutdown_calls: list[float | None] = []
    monkeypatch.setattr(
        core_client_module.MPClient,
        "shutdown",
        lambda _self, timeout=None: base_shutdown_calls.append(timeout),
    )

    client.shutdown(timeout=0.25)
    results = await asyncio.gather(*tasks, return_exceptions=True)

    assert client.resources.engine_dead
    assert client._elastic_ep_shutdown_requested
    assert not client._elastic_ep_transaction_pending
    assert not client._elastic_ep_transaction_active
    assert not client._elastic_ep_commit_in_progress
    assert client._prepared_elastic_ep is None
    assert all(not task_set for task_set in task_sets)
    assert all(isinstance(result, asyncio.CancelledError) for result in results)
    with pytest.raises(EngineDeadError):
        await utility_future
    assert client.utility_results == {}
    assert base_shutdown_calls == [0.25]


def test_elastic_ep_shutdown_survives_owner_loop_close_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_elastic_ep_transaction_client()
    base_shutdown_calls: list[float | None] = []

    class ClosingLoop:
        @staticmethod
        def is_closed() -> bool:
            return False

        @staticmethod
        def is_running() -> bool:
            return True

        @staticmethod
        def call_soon_threadsafe(*_args: Any) -> None:
            raise RuntimeError("event loop closed during shutdown")

    class LoopOwnedTask:
        @staticmethod
        def get_loop() -> Any:
            return ClosingLoop()

        @staticmethod
        def done() -> bool:
            return False

    tracked = cast(asyncio.Task[None], cast(Any, LoopOwnedTask()))
    client._elastic_ep_prepare_tasks.add(tracked)
    concurrent_future: Future[Any] = Future()
    client.utility_results[91] = concurrent_future
    monkeypatch.setattr(
        core_client_module.MPClient,
        "shutdown",
        lambda _self, timeout=None: base_shutdown_calls.append(timeout),
    )

    client.shutdown(timeout=0.75)

    assert base_shutdown_calls == [0.75]
    assert client._elastic_ep_prepare_tasks == set()
    with pytest.raises(EngineDeadError):
        concurrent_future.result()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["prepare", "commit"])
async def test_elastic_ep_internal_cancel_after_mutation_fails_stop(
    phase: str,
) -> None:
    client = _make_elastic_ep_transaction_client()
    shutdown_calls: list[None] = []
    client.shutdown = lambda: shutdown_calls.append(None)

    async def cancel_prepare(_new_size: int) -> None:
        client._elastic_ep_prepare_mutated = True
        raise asyncio.CancelledError

    async def cancel_commit(_new_size: int) -> None:
        client._elastic_ep_commit_mutated = True
        raise asyncio.CancelledError

    if phase == "prepare":
        client._prepared_elastic_ep = None
        client._elastic_ep_transaction_pending = True
        client._elastic_ep_transaction_active = False

        async def utility(
            _method: str, *_args: Any, engine: bytes
        ) -> bool:
            return False

        client._call_utility_async = utility
        client._prepare_scale_down_elastic_ep = cancel_prepare
        operation = client._run_elastic_ep_prepare(2)
    else:
        client._commit_scale_down_elastic_ep = cancel_commit
        operation = client._run_elastic_ep_commit(2, 7)

    with pytest.raises(RuntimeError, match="cannot safely retry"):
        await operation

    async def wait_for_fail_stop_cleanup() -> None:
        while client._elastic_ep_fail_stop_tasks:
            await asyncio.sleep(0)

    await asyncio.wait_for(wait_for_fail_stop_cleanup(), 1.0)
    assert client.resources.engine_dead
    assert client._elastic_ep_transaction_failed
    assert shutdown_calls == [None]


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_elastic_scale_down_notification_completes_cleanup_ack(
    cleanup_fails: bool,
) -> None:
    client = _make_elastic_ep_transaction_client()
    completion = asyncio.get_running_loop().create_future()
    calls: list[tuple[int, int]] = []

    class Manager:
        local_engine_actors = [object(), object()]

        def scale_down_elastic_ep(self, old_size: int, new_size: int) -> None:
            calls.append((old_size, new_size))
            if cleanup_fails:
                raise RuntimeError("placement-group cleanup failed")

    client.resources.engine_manager = Manager()
    client.vllm_config.parallel_config.data_parallel_size_local = 3
    client.eep_scaling_cache = core_client_module.ElasticScalingCache(
        existing_core_engines=[b"rank-0", b"rank-1", b"rank-2"],
        num_new_core_engines=-1,
        pending_notifications={},
        completion_future=completion,
    )

    await DPLBAsyncMPClient.eep_process_engine_core_notification(
        client,
        (core_client_module.EEPNotificationType.SHUTDOWN_COMPLETE.value, 2),
    )

    assert calls == [(3, 2)]
    assert client.eep_scaling_cache is None
    if cleanup_fails:
        with pytest.raises(RuntimeError, match="placement-group cleanup failed"):
            await completion
    else:
        await completion
        assert client.vllm_config.parallel_config.data_parallel_size_local == 2


@pytest.mark.asyncio
async def test_elastic_scale_down_commit_waits_for_resource_cleanup_ack() -> None:
    client = _make_elastic_ep_transaction_client()
    old_engines = client.core_engines.copy()
    switch_future = asyncio.get_running_loop().create_future()
    events: list[str] = []

    manager = core_client_module.CoreEngineActorManager.__new__(
        core_client_module.CoreEngineActorManager
    )
    manager.local_engine_actors = [object(), object(), object()]
    manager.remove_run_refs_for_scale_down = lambda _removed: None

    def scale_down(old_size: int, new_size: int) -> None:
        assert (old_size, new_size) == (3, 2)
        events.append("cleanup")
        manager.local_engine_actors = manager.local_engine_actors[:new_size]

    manager.scale_down_elastic_ep = scale_down
    client.resources.engine_manager = manager

    async def quiesce(new_size: int) -> tuple[list[bytes], int]:
        client.core_engines = old_engines[:new_size]
        client._elastic_ep_commit_mutated = True
        return old_engines, len(old_engines) - new_size

    async def utility(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def broadcast(
        method: str,
        _args: tuple[Any, ...],
        _engines: tuple[bytes, ...],
    ) -> None:
        events.append(method)

    class FirstRequestSocket:
        @staticmethod
        async def send(_payload: bytes) -> None:
            events.append("marker")

    def wait_for_switch() -> asyncio.Future[Any]:
        client.utility_results[core_client_module.EEP_NOTIFICATION_CALL_ID] = (
            switch_future
        )
        return switch_future

    client._quiesce_scale_down_elastic_ep = quiesce
    client._call_utility_async = utility
    client._broadcast_dp_lifecycle_utility = broadcast
    client._make_reconfig_request = lambda *_args, **_kwargs: object()
    client._eep_wait_for_setup_switch_complete = wait_for_switch
    client._ensure_stats_update_task = lambda: None
    client.first_req_send_socket = FirstRequestSocket()

    commit = asyncio.create_task(client._commit_scale_down_elastic_ep(2))

    async def wait_for_cleanup_future() -> None:
        while client.eep_scaling_cache is None:
            await asyncio.sleep(0)

    await asyncio.wait_for(wait_for_cleanup_future(), 1.0)
    switch_future.set_result(None)
    await asyncio.sleep(0)
    assert not commit.done()
    assert "resume_scheduler" not in events

    await DPLBAsyncMPClient.eep_process_engine_core_notification(
        client,
        (core_client_module.EEPNotificationType.SHUTDOWN_COMPLETE.value, 2),
    )
    await asyncio.wait_for(commit, 1.0)

    assert events.index("cleanup") < events.index("resume_scheduler")
    assert client.eep_scaling_cache is None
    assert client.vllm_config.parallel_config.data_parallel_size == 2
    assert client.vllm_config.parallel_config.data_parallel_size_local == 2


def test_actor_manager_shutdown_retries_only_failed_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary_error = RuntimeError("actor kill failed")
    fail_actor = True
    kill_calls: list[str] = []
    remove_calls: list[str] = []

    def kill(actor: str) -> None:
        nonlocal fail_actor
        kill_calls.append(actor)
        if actor == "bad-actor" and fail_actor:
            raise primary_error

    def remove_placement_group(pg: str) -> None:
        remove_calls.append(pg)

    fake_ray = SimpleNamespace(
        kill=kill,
        util=SimpleNamespace(remove_placement_group=remove_placement_group),
    )
    monkeypatch.setitem(sys.modules, "ray", fake_ray)

    class Finalizer:
        alive = True

        def detach(self) -> object:
            self.alive = False
            return object()

    manager = core_client_module.CoreEngineActorManager.__new__(
        core_client_module.CoreEngineActorManager
    )
    manager.manager_stopped = threading.Event()
    manager._actor_mutation_lock = threading.RLock()
    manager.local_engine_actors = ["bad-actor"]
    manager.remote_engine_actors = ["good-actor"]
    manager.created_placement_groups = ["new-pg"]
    manager.placement_group_is_local = [False]
    manager.run_refs = ["bad-ref", "good-ref"]
    manager.actor_run_ref_dict = {
        "bad-actor": "bad-ref",
        "good-actor": "good-ref",
    }
    manager._finalizer = Finalizer()

    with pytest.raises(RuntimeError, match="actor kill failed") as exc_info:
        manager.shutdown()

    assert exc_info.value is primary_error
    assert manager.manager_stopped.is_set()
    assert kill_calls == ["bad-actor", "good-actor"]
    assert remove_calls == ["new-pg"]
    assert manager.local_engine_actors == ["bad-actor"]
    assert manager.remote_engine_actors == []
    assert manager.created_placement_groups == []
    assert manager.run_refs == ["bad-ref"]
    assert manager._finalizer.alive

    fail_actor = False
    manager.shutdown()
    assert kill_calls == ["bad-actor", "good-actor", "bad-actor"]
    assert remove_calls == ["new-pg"]
    assert manager.local_engine_actors == []
    assert manager.run_refs == []
    assert not manager._finalizer.alive


def test_background_resources_continue_cleanup_after_manager_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary_error = RuntimeError("manager cleanup failed")
    calls: list[str] = []

    class Manager:
        def shutdown(self, timeout: float | None = None) -> None:
            calls.append(f"manager:{timeout}")
            raise primary_error

    class Coordinator:
        def shutdown(self) -> None:
            calls.append("coordinator")

    monkeypatch.setattr(
        core_client_module,
        "close_sockets",
        lambda _sockets: calls.append("sockets"),
    )
    resources = core_client_module.BackgroundResources(
        ctx=cast(Any, SimpleNamespace()),
        engine_manager=cast(Any, Manager()),
        coordinator=cast(Any, Coordinator()),
    )

    with pytest.raises(RuntimeError, match="manager cleanup failed") as exc_info:
        resources()

    assert exc_info.value is primary_error
    assert calls == [
        f"manager:{core_client_module.envs.VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS}",
        "coordinator",
        "sockets",
    ]
    assert resources.coordinator is None


def test_background_resources_close_async_sockets_after_loop_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Any, ...]] = []

    class AsyncSocket:
        pass

    class ClosedLoop:
        @staticmethod
        def is_closed() -> bool:
            return True

    class FinishedTask:
        _loop = ClosedLoop()

        @staticmethod
        def done() -> bool:
            return True

        @staticmethod
        def cancelled() -> bool:
            return True

    output_socket = AsyncSocket()
    monkeypatch.setattr(core_client_module.zmq.asyncio, "Socket", AsyncSocket)
    monkeypatch.setattr(
        core_client_module,
        "close_sockets",
        lambda sockets: calls.append(tuple(sockets)),
    )
    resources = core_client_module.BackgroundResources(
        ctx=cast(Any, SimpleNamespace()),
        output_socket=cast(Any, output_socket),
        output_queue_task=cast(Any, FinishedTask()),
    )

    resources()

    assert calls == [(output_socket, None, None, None, None)]
    assert resources.output_queue_task is None


def test_background_resources_close_async_sockets_after_loop_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Any, ...]] = []

    class AsyncSocket:
        pass

    class StoppedLoop:
        @staticmethod
        def is_closed() -> bool:
            return False

        @staticmethod
        def is_running() -> bool:
            return False

        @staticmethod
        def call_soon_threadsafe(*_args: Any) -> None:
            raise AssertionError("stopped loop must not receive cleanup work")

    class PendingTask:
        _loop = StoppedLoop()

        @staticmethod
        def done() -> bool:
            return False

        @staticmethod
        def cancelled() -> bool:
            return False

    output_socket = AsyncSocket()
    monkeypatch.setattr(core_client_module.zmq.asyncio, "Socket", AsyncSocket)
    monkeypatch.setattr(
        core_client_module,
        "close_sockets",
        lambda sockets: calls.append(tuple(sockets)),
    )
    resources = core_client_module.BackgroundResources(
        ctx=cast(Any, SimpleNamespace()),
        output_socket=cast(Any, output_socket),
        output_queue_task=cast(Any, PendingTask()),
    )

    resources()

    assert calls == [(output_socket, None, None, None, None)]
    assert resources.output_socket is None
    assert resources.output_queue_task is None


def test_background_resources_uses_stats_task_owner_loop_without_output_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AsyncSocket:
        pass

    loop = asyncio.new_event_loop()
    loop_started = threading.Event()
    task_started = threading.Event()
    task_finished = threading.Event()
    owner_thread_ids: list[int] = []
    close_thread_ids: list[int] = []
    task_holder: dict[str, asyncio.Task[None]] = {}

    async def blocked_stats_update() -> None:
        task_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            task_finished.set()

    def run_owner_loop() -> None:
        asyncio.set_event_loop(loop)
        owner_thread_ids.append(threading.get_ident())
        loop_started.set()
        loop.run_forever()

    owner_thread = threading.Thread(target=run_owner_loop)
    owner_thread.start()
    assert loop_started.wait(1.0)

    def create_stats_task() -> None:
        task_holder["task"] = loop.create_task(blocked_stats_update())

    loop.call_soon_threadsafe(create_stats_task)
    assert task_started.wait(1.0)
    output_socket = AsyncSocket()
    monkeypatch.setattr(core_client_module.zmq.asyncio, "Socket", AsyncSocket)
    monkeypatch.setattr(
        core_client_module,
        "close_sockets",
        lambda _sockets: close_thread_ids.append(threading.get_ident()),
    )
    resources = core_client_module.BackgroundResources(
        ctx=cast(Any, SimpleNamespace()),
        output_socket=cast(Any, output_socket),
        stats_update_task=task_holder["task"],
    )

    try:
        resources()
        assert task_finished.wait(1.0)

        async def wait_for_task_terminal() -> None:
            while not task_holder["task"].done():
                await asyncio.sleep(0)

        asyncio.run_coroutine_threadsafe(
            wait_for_task_terminal(), loop
        ).result(1.0)
        assert task_holder["task"].cancelled()
        assert close_thread_ids == owner_thread_ids
        assert resources.stats_update_task is None
        assert resources.output_socket is None
    finally:
        task = task_holder.get("task")
        if task is not None and not task.done() and loop.is_running():
            loop.call_soon_threadsafe(task.cancel)
        loop.call_soon_threadsafe(loop.stop)
        owner_thread.join(1.0)
        assert not owner_thread.is_alive()
        loop.close()


def test_background_resources_wait_for_sync_output_thread_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ready = threading.Event()
    stopped = threading.Event()
    ready.set()
    calls: list[tuple[Any, ...]] = []

    class Sender:
        def setsockopt(self, option: int, value: int) -> None:
            calls.append(("setsockopt", option, value))

        def connect(self, path: str) -> None:
            calls.append(("connect", path))

        def send(self, payload: bytes, flags: int = 0) -> None:
            calls.append(("send", payload, flags))
            stopped.set()

        def close(self, linger: int = -1) -> None:
            calls.append(("close", linger))

    context = SimpleNamespace(socket=lambda _kind: Sender())
    monkeypatch.setattr(core_client_module, "close_sockets", lambda _sockets: None)
    resources = core_client_module.BackgroundResources(
        ctx=cast(Any, context),
        shutdown_path="inproc://sync-output-shutdown",
        sync_output_ready=ready,
        sync_output_stopped=stopped,
    )

    resources()

    assert ("send", b"", core_client_module.zmq.NOBLOCK) in calls
    assert calls[-1] == ("close", 0)
    assert resources.shutdown_path is None
    assert resources.sync_output_ready is None
    assert resources.sync_output_stopped is None


def test_mp_client_shutdown_runs_background_cleanup_after_manager_failure() -> None:
    primary_error = RuntimeError("direct manager shutdown failed")
    calls: list[str] = []

    class Finalizer:
        alive = True

        def detach(self) -> object:
            self.alive = False
            return object()

    class Manager:
        attempts = 0

        def shutdown(self, timeout: float | None = None) -> None:
            self.attempts += 1
            calls.append(f"manager:{timeout}")
            if self.attempts <= 2:
                raise primary_error

    class Resources:
        engine_dead = False
        engine_manager = Manager()

        def __call__(self) -> None:
            calls.append("resources")
            if self.engine_manager is not None:
                self.engine_manager.shutdown()
                self.engine_manager = None

    client = core_client_module.MPClient.__new__(core_client_module.MPClient)
    client._finalizer = Finalizer()
    client._shutdown_lock = threading.RLock()
    client._shutdown_complete = False
    client.resources = Resources()

    with pytest.raises(
        RuntimeError, match="direct manager shutdown failed"
    ) as exc_info:
        client.shutdown(timeout=0.5)

    assert exc_info.value is primary_error
    assert calls == ["manager:0.5", "resources", "manager:None"]
    assert client._finalizer.alive
    assert not client._shutdown_complete

    client.shutdown(timeout=0.5)
    assert calls == [
        "manager:0.5",
        "resources",
        "manager:None",
        "manager:0.5",
        "resources",
    ]
    assert not client._finalizer.alive
    assert client._shutdown_complete
    assert client.resources.engine_manager is None


@pytest.mark.parametrize("owner_kind", ["engine_manager", "coordinator"])
def test_process_owner_shutdown_keeps_finalizer_armed_for_retry(
    monkeypatch: pytest.MonkeyPatch,
    owner_kind: str,
) -> None:
    primary_error = RuntimeError("process cleanup failed")
    calls: list[float | None] = []
    process_args: list[list[Any]] = []

    class Finalizer:
        alive = True

        def detach(self) -> object:
            self.alive = False
            return object()

    def process_shutdown(_processes: list[Any], timeout: float | None = None) -> None:
        process_args.append(list(_processes))
        calls.append(timeout)
        if len(calls) == 1:
            raise primary_error

    if owner_kind == "engine_manager":
        monkeypatch.setattr(engine_utils_module, "shutdown", process_shutdown)
        owner = engine_utils_module.CoreEngineProcManager.__new__(
            engine_utils_module.CoreEngineProcManager
        )
        owner.processes = ["started", "not-started"]
        owner._started_processes = ["started"]
        owner.manager_stopped = threading.Event()
    else:
        monkeypatch.setattr(coordinator_module, "shutdown", process_shutdown)
        owner = coordinator_module.DPCoordinator.__new__(
            coordinator_module.DPCoordinator
        )
        owner.proc = "started"
        owner._started_processes = ["started"]

    owner._shutdown_lock = threading.RLock()
    owner._shutdown_complete = False
    owner._finalizer = Finalizer()

    with pytest.raises(RuntimeError, match="process cleanup failed") as exc_info:
        owner.shutdown(timeout=0.5)

    assert exc_info.value is primary_error
    assert owner._finalizer.alive
    assert not owner._shutdown_complete

    owner.shutdown(timeout=0.75)
    assert calls == [0.5, 0.75]
    assert process_args == [["started"], ["started"]]
    assert not owner._finalizer.alive
    assert owner._shutdown_complete


def _ray_launch_test_config() -> Any:
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=2,
            data_parallel_size_local=2,
            data_parallel_rank_local=None,
            data_parallel_rank=0,
            data_parallel_master_ip="127.0.0.1",
            local_engines_only=False,
            data_parallel_backend="ray",
        ),
        model_config=SimpleNamespace(multimodal_config=None, is_moe=True),
        needs_dp_coordinator=True,
    )


def test_ray_launch_manager_constructor_failure_cleans_coordinator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    primary_error = RuntimeError("actor manager construction failed")

    class Coordinator:
        proc = SimpleNamespace(pid=123)

        @staticmethod
        def get_engine_socket_addresses() -> tuple[str, str]:
            return "coord-in", "coord-out"

        @staticmethod
        def get_stats_publish_address() -> str:
            return "coord-stats"

        @staticmethod
        def shutdown() -> None:
            calls.append("coordinator")
            raise RuntimeError("coordinator cleanup failed")

    coordinator = Coordinator()
    monkeypatch.setattr(
        engine_utils_module, "DPCoordinator", lambda *_args, **_kwargs: coordinator
    )

    def fail_actor_manager(**_kwargs: Any) -> None:
        raise primary_error

    monkeypatch.setattr(
        engine_utils_module, "CoreEngineActorManager", fail_actor_manager
    )
    addresses = SimpleNamespace()

    with pytest.raises(RuntimeError, match="actor manager construction") as exc_info:
        with engine_utils_module.launch_core_engines(
            _ray_launch_test_config(),
            cast(Any, object),
            False,
            cast(Any, addresses),
        ):
            raise AssertionError("constructor failure unexpectedly yielded")

    assert exc_info.value is primary_error
    assert calls == ["coordinator"]


def test_ray_launch_body_failure_cleans_manager_then_coordinator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    primary_error = RuntimeError("launch body failed")

    class Coordinator:
        proc = SimpleNamespace(pid=123)

        @staticmethod
        def get_engine_socket_addresses() -> tuple[str, str]:
            return "coord-in", "coord-out"

        @staticmethod
        def get_stats_publish_address() -> str:
            return "coord-stats"

        @staticmethod
        def shutdown() -> None:
            calls.append("coordinator")

    class Manager:
        @staticmethod
        def shutdown() -> None:
            calls.append("manager")

    coordinator = Coordinator()
    manager = Manager()
    monkeypatch.setattr(
        engine_utils_module, "DPCoordinator", lambda *_args, **_kwargs: coordinator
    )
    monkeypatch.setattr(
        engine_utils_module,
        "CoreEngineActorManager",
        lambda **_kwargs: manager,
    )

    with pytest.raises(RuntimeError, match="launch body failed") as exc_info:
        with engine_utils_module.launch_core_engines(
            _ray_launch_test_config(),
            cast(Any, object),
            False,
            cast(Any, SimpleNamespace()),
        ):
            raise primary_error

    assert exc_info.value is primary_error
    assert calls == ["manager", "coordinator"]


def test_engine_monitor_failure_still_marks_dead_and_attempts_client_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor_error = RuntimeError("manager cleanup failed")
    cleanup_error = RuntimeError("client cleanup failed")
    shutdown_calls: list[None] = []

    class Finalizer:
        alive = True

    resources = SimpleNamespace(engine_manager=None, engine_dead=False)

    class Manager:
        @staticmethod
        def monitor_engine_liveness() -> None:
            # Exercise the path where another observer publishes death before
            # the manager itself fails during teardown.
            resources.engine_dead = True
            raise monitor_error

    resources.engine_manager = Manager()
    client = core_client_module.MPClient.__new__(core_client_module.MPClient)
    client.resources = resources
    client._finalizer = Finalizer()

    def fail_cleanup() -> None:
        shutdown_calls.append(None)
        raise cleanup_error

    client.shutdown = fail_cleanup

    class InlineThread:
        def __init__(self, *, target: Any, **_kwargs: Any) -> None:
            self.target = target

        def start(self) -> None:
            self.target()

    monkeypatch.setattr(core_client_module, "Thread", InlineThread)

    client.start_engine_core_monitor()

    assert resources.engine_dead
    assert shutdown_calls == [None]
    assert client._finalizer.alive


@pytest.mark.asyncio
async def test_elastic_ep_commit_holds_lifecycle_transaction_after_cancel() -> None:
    client = _make_elastic_ep_transaction_client()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def commit_scale_down(_new_size: int) -> None:
        assert client._get_dp_lifecycle_broadcast_lock().locked()
        entered.set()
        await release.wait()

    client._commit_scale_down_elastic_ep = commit_scale_down
    commit_waiter = asyncio.create_task(client.commit_elastic_ep())
    await asyncio.wait_for(entered.wait(), 1.0)

    with pytest.raises(RuntimeError, match="reconfiguration is pending"):
        await client._call_dp_lifecycle_utility("resume_scheduler", ())
    with pytest.raises(RuntimeError, match="already in progress"):
        await client.commit_elastic_ep()

    commit_waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await commit_waiter
    assert client._elastic_ep_commit_in_progress
    assert client._get_dp_lifecycle_broadcast_lock().locked()

    release.set()

    async def wait_for_commit_cleanup() -> None:
        while client._elastic_ep_commit_tasks:
            await asyncio.sleep(0)

    await asyncio.wait_for(wait_for_commit_cleanup(), 1.0)
    assert client._prepared_elastic_ep is None
    assert not client._elastic_ep_transaction_active
    assert not client._elastic_ep_commit_in_progress
    assert client.vllm_config.parallel_config.eplb_config.num_redundant_experts == 7


@pytest.mark.asyncio
async def test_elastic_ep_prepare_survives_caller_cancel_and_keeps_guard() -> None:
    client = _make_elastic_ep_transaction_client()
    client.core_engines = [b"rank-0", b"rank-1"]
    client._prepared_elastic_ep = None
    client._elastic_ep_transaction_active = False
    started = asyncio.Event()
    release = asyncio.Event()

    async def utility(method: str, *_args: Any, engine: bytes) -> bool:
        assert method in {"is_scheduler_paused", "is_sleeping"}
        assert engine in client.core_engines
        return False

    async def prepare_scale_up(_new_size: int, _redundant: int) -> None:
        started.set()
        await release.wait()

    client._call_utility_async = utility
    client._prepare_scale_up_elastic_ep = prepare_scale_up
    prepare_waiter = asyncio.create_task(client.prepare_elastic_ep(3))
    await asyncio.wait_for(started.wait(), 1.0)

    prepare_waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await prepare_waiter
    assert client._elastic_ep_transaction_active
    with pytest.raises(RuntimeError, match="reconfiguration is pending"):
        await client._call_dp_lifecycle_utility("pause_scheduler", ("keep", False))

    release.set()

    async def wait_for_prepare_cleanup() -> None:
        while client._elastic_ep_prepare_tasks:
            await asyncio.sleep(0)

    await asyncio.wait_for(wait_for_prepare_cleanup(), 1.0)
    assert client._prepared_elastic_ep is not None
    assert client._prepared_elastic_ep[0] == 3
    assert client._elastic_ep_transaction_active


@pytest.mark.asyncio
async def test_elastic_ep_prepare_rechecks_guard_after_lifecycle_queueing() -> None:
    client = _make_elastic_ep_transaction_client()
    client.core_engines = [b"rank-0", b"rank-1"]
    client._prepared_elastic_ep = None
    client._elastic_ep_transaction_active = False
    lock = client._get_dp_lifecycle_broadcast_lock()
    await lock.acquire()
    admissions: list[str] = []

    async def admit(*_args: Any, **_kwargs: Any) -> tuple[int, asyncio.Future[Any]]:
        admissions.append("admitted")
        future = asyncio.get_running_loop().create_future()
        future.set_result(None)
        return 1, future

    async def utility(method: str, *_args: Any, engine: bytes) -> bool:
        assert method in {"is_scheduler_paused", "is_sleeping"}
        return False

    async def prepare_scale_up(_new_size: int, _redundant: int) -> None:
        return None

    client._admit_utility_async = admit
    client._call_utility_async = utility
    client._prepare_scale_up_elastic_ep = prepare_scale_up
    lifecycle = asyncio.create_task(
        client._call_dp_lifecycle_utility("pause_scheduler", ("keep", False))
    )
    await asyncio.sleep(0)
    prepare = asyncio.create_task(client.prepare_elastic_ep(3))
    await asyncio.sleep(0)
    lock.release()

    with pytest.raises(RuntimeError, match="reconfiguration is pending"):
        await lifecycle
    await prepare
    assert admissions == []
    assert client._prepared_elastic_ep is not None


@pytest.mark.asyncio
async def test_elastic_ep_prepare_rejects_already_paused_old_group() -> None:
    client = _make_elastic_ep_transaction_client()
    client._prepared_elastic_ep = None
    client._elastic_ep_transaction_active = False
    prepare_calls: list[int] = []

    async def utility(method: str, *_args: Any, engine: bytes) -> bool:
        return method == "is_scheduler_paused"

    async def prepare_scale_down(new_size: int) -> None:
        prepare_calls.append(new_size)

    client._call_utility_async = utility
    client._prepare_scale_down_elastic_ep = prepare_scale_down
    with pytest.raises(RuntimeError, match="unpaused and fully awake"):
        await client.prepare_elastic_ep(2)

    assert prepare_calls == []
    assert not client._elastic_ep_transaction_pending
    assert not client._elastic_ep_transaction_active


@pytest.mark.asyncio
async def test_elastic_ep_prepare_waits_for_inflight_pause_then_rejects() -> None:
    client = _make_elastic_ep_transaction_client()
    client._prepared_elastic_ep = None
    client._elastic_ep_transaction_active = False
    pause_admitted = asyncio.Event()
    paused = False
    pause_futures: list[asyncio.Future[Any]] = []

    async def admit(
        _method: str, *_args: Any, engine: bytes
    ) -> tuple[int, asyncio.Future[Any]]:
        future = asyncio.get_running_loop().create_future()
        pause_futures.append(future)
        if len(pause_futures) == len(client.core_engines):
            pause_admitted.set()
        return len(pause_futures), future

    async def utility(method: str, *_args: Any, engine: bytes) -> bool:
        return paused and method == "is_scheduler_paused"

    async def prepare_scale_down(_new_size: int) -> None:
        raise AssertionError("paused group reached distributed EEP prepare")

    client._admit_utility_async = admit
    client._call_utility_async = utility
    client._prepare_scale_down_elastic_ep = prepare_scale_down
    pause = asyncio.create_task(
        client._call_dp_lifecycle_utility("pause_scheduler", ("keep", False))
    )
    await asyncio.wait_for(pause_admitted.wait(), 1.0)
    prepare = asyncio.create_task(client.prepare_elastic_ep(2))
    await asyncio.sleep(0)
    assert client._elastic_ep_transaction_pending

    paused = True
    for future in pause_futures:
        future.set_result(None)
    await pause
    with pytest.raises(RuntimeError, match="unpaused and fully awake"):
        await prepare
    assert not client._elastic_ep_transaction_pending
    assert not client._elastic_ep_transaction_active


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutated", "resource_dead"), [(True, False), (False, True)]
)
async def test_elastic_ep_commit_fails_stop_and_rejects_retry(
    mutated: bool, resource_dead: bool
) -> None:
    client = _make_elastic_ep_transaction_client()
    shutdown_calls: list[None] = []
    client.shutdown = lambda: shutdown_calls.append(None)

    async def fail_commit(_new_size: int) -> None:
        client._elastic_ep_commit_mutated = mutated
        client.resources.engine_dead = resource_dead
        raise RuntimeError("distributed commit failed")

    client._commit_scale_down_elastic_ep = fail_commit
    with pytest.raises(RuntimeError, match="cannot safely retry"):
        await client.commit_elastic_ep()
    assert client.resources.engine_dead
    assert client._elastic_ep_transaction_failed
    assert client._prepared_elastic_ep is None
    terminal = client.outputs_queue.get_nowait()
    assert isinstance(terminal, RuntimeError)

    with pytest.raises(RuntimeError, match="cannot be retried"):
        await client.commit_elastic_ep()

    async def wait_for_fail_stop_cleanup() -> None:
        while client._elastic_ep_fail_stop_tasks:
            await asyncio.sleep(0)

    await asyncio.wait_for(wait_for_fail_stop_cleanup(), 1.0)
    assert shutdown_calls == [None]


@pytest.mark.asyncio
@pytest.mark.parametrize("pending_pin", [False, True])
async def test_elastic_scale_down_prepare_rejects_removed_rank_prefix_pin(
    pending_pin: bool,
) -> None:
    client = _make_elastic_ep_transaction_client()
    client._prepared_elastic_ep = None
    client._elastic_ep_transaction_active = False
    pin_id = "pending" if pending_pin else "completed"
    client.prefix_pins[pin_id] = (b"rank-2", "pin-request")
    prepare_calls: list[int] = []

    async def utility(method: str, *_args: Any, engine: bytes) -> bool:
        assert method in {"is_scheduler_paused", "is_sleeping"}
        return False

    async def prepare_scale_down(new_size: int) -> None:
        prepare_calls.append(new_size)

    client._call_utility_async = utility
    client._prepare_scale_down_elastic_ep = prepare_scale_down
    with pytest.raises(RuntimeError, match="unpin them first"):
        await client.prepare_elastic_ep(2)

    assert prepare_calls == []
    assert not client._elastic_ep_transaction_pending
    assert not client._elastic_ep_transaction_active
    assert not client.resources.engine_dead


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["pending", "active"])
async def test_elastic_ep_transaction_rejects_new_prefix_pin_before_reservation(
    state: str,
) -> None:
    client = _make_elastic_ep_transaction_client()
    client.prefix_pins = {}
    client._elastic_ep_transaction_pending = state == "pending"
    client._elastic_ep_transaction_active = state == "active"
    request = SimpleNamespace(request_id="pin-request")

    with pytest.raises(RuntimeError, match="Prefix pins cannot be created"):
        await client.pin_prefix_async("new-pin", cast(Any, request))
    assert client.prefix_pins == {}


@pytest.mark.asyncio
async def test_async_llm_elastic_ep_commit_failure_clears_scaling_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states: list[bool] = []
    calls: list[str] = []

    class Core:
        async def prepare_elastic_ep(self, _new_size: int) -> None:
            calls.append("prepare")

        async def commit_elastic_ep(self) -> None:
            calls.append("commit")
            raise RuntimeError("commit failed")

    engine = AsyncLLM.__new__(AsyncLLM)
    engine.shutdown = lambda *_args, **_kwargs: None
    engine._client_count = 1
    engine.engine_core = Core()
    engine.log_stats = False
    engine.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=2)
    )
    monkeypatch.setattr(async_llm_module, "set_scaling_elastic_ep", states.append)
    monkeypatch.setattr(
        async_llm_module.envs,
        "VLLM_ELASTIC_EP_DRAIN_REQUESTS",
        False,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="commit failed"):
        await engine._scale_elastic_ep(3, 1)
    assert calls == ["prepare", "commit"]
    assert states == [True, False]


@pytest.mark.asyncio
@pytest.mark.parametrize("drain_timeout", [True, 0, -1, 1.5])
async def test_async_llm_elastic_ep_rejects_invalid_drain_timeout_before_prepare(
    drain_timeout: Any,
) -> None:
    engine = AsyncLLM.__new__(AsyncLLM)
    calls: list[int] = []

    async def unexpected_scale(new_size: int, _timeout: int) -> None:
        calls.append(new_size)

    engine._run_serialized_elastic_ep_scale = unexpected_scale

    with pytest.raises(ValueError, match="drain_timeout"):
        await engine.scale_elastic_ep(3, drain_timeout)
    assert calls == []


@pytest.mark.asyncio
async def test_async_llm_elastic_ep_multi_frontend_fails_before_prepare() -> None:
    calls: list[int] = []

    class Core:
        async def prepare_elastic_ep(self, new_size: int) -> None:
            calls.append(new_size)

    engine = AsyncLLM.__new__(AsyncLLM)
    engine.shutdown = lambda *_args, **_kwargs: None
    engine._client_count = 2
    engine.engine_core = Core()
    engine.log_stats = False
    engine.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=2)
    )

    with pytest.raises(RuntimeError, match="single lifecycle-owner"):
        await engine._scale_elastic_ep(3, 1)
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["pause", "resume"])
async def test_async_llm_request_control_rejects_dead_engine(method: str) -> None:
    engine = AsyncLLM.__new__(AsyncLLM)
    engine.shutdown = lambda *_args, **_kwargs: None
    engine.output_handler = None
    engine.engine_core = SimpleNamespace(resources=SimpleNamespace(engine_dead=True))

    with pytest.raises(EngineDeadError):
        await getattr(engine, method)("request")


@pytest.mark.asyncio
async def test_dplb_request_control_rejects_dead_but_abort_stays_idempotent() -> None:
    client = core_client_module.DPLBAsyncMPClient.__new__(
        core_client_module.DPLBAsyncMPClient
    )
    client.resources = SimpleNamespace(engine_dead=True)

    with pytest.raises(EngineDeadError):
        await client.pause_requests_async(["request"])
    with pytest.raises(EngineDeadError):
        await client.resume_requests_async(["request"])
    assert await client.abort_requests_async(["request"]) is None


def test_elastic_scale_down_removed_rank_emits_abort_outputs() -> None:
    engine, _lifecycle_calls, _reset_calls = _make_dp_lifecycle_engine()
    engine.vllm_config.parallel_config.enable_elastic_ep = True
    engine.dp_rank = 2
    engine.scheduler.pause_state = PauseState.PAUSED_ALL
    engine.ignore_start_dp_wave = True
    aborted = SimpleNamespace(request_id="removed-request", client_index=3)
    engine.scheduler.finish_requests = lambda *_args, **_kwargs: [aborted]

    engine.abort_for_elastic_ep_scale_down(2)

    client_index, outputs = engine.output_queue.get_nowait()
    assert client_index == 3
    assert outputs.finished_requests == ["removed-request"]
    assert outputs.outputs[0].finish_reason is FinishReason.ABORT


def test_dp_sync_state_uses_one_fixed_shape_for_work_pause_and_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reduced_inputs: list[torch.Tensor] = []
    group = SimpleNamespace(size=lambda: 2)

    def all_reduce(tensor: torch.Tensor, **_kwargs: Any) -> None:
        reduced_inputs.append(tensor.clone())
        tensor[:] = torch.tensor([0, 1, 1, 0, 37, 37 * 37], dtype=torch.int64)

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)

    result = core_module.ParallelConfig.sync_dp_state(
        cast(Any, group),
        has_unfinished=False,
        pending_pause=True,
        control_op_pending=True,
        control_op_blocked=False,
        control_op_signature=37,
    )

    assert reduced_inputs[0].tolist() == [0, 1, 1, 0, 37, 37 * 37]
    assert reduced_inputs[0].dtype == torch.int64
    assert result == (True, 1, 1, 0, 37, 37 * 37)


def test_mp_fatal_failure_detaches_and_finishes_all_utility_waiters() -> None:
    loop = asyncio.new_event_loop()
    try:
        async_future = loop.create_future()
        sync_future: Future[Any] = Future()
        utility_results = {1: async_future, 2: sync_future}
        original_error = RuntimeError("engine output channel failed")

        core_client_module._fail_utility_results(utility_results, original_error)

        assert utility_results == {}
        for future in (async_future, sync_future):
            assert future.done()
            error = future.exception()
            assert isinstance(error, RuntimeError)
            assert error is not original_error
            assert error.__traceback__ is None
    finally:
        loop.close()


def test_async_mp_utility_send_failure_discards_registered_waiter() -> None:
    client = core_client_module.AsyncMPClient.__new__(
        core_client_module.AsyncMPClient
    )
    client.client_index = 0
    client.core_engine = b"engine"
    client.utility_results = {}
    client.encoder = SimpleNamespace(encode=lambda _value: (b"payload",))

    async def fail_send(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("send failed")

    client._send_input_message = fail_send

    with pytest.raises(RuntimeError, match="send failed"):
        asyncio.run(client._call_utility_async("method", engine=b"engine"))
    assert client.utility_results == {}


def test_async_mp_cancelled_utility_discards_waiter_and_ignores_late_output() -> None:
    async def scenario() -> None:
        captured_call_id: list[int] = []
        client = core_client_module.AsyncMPClient.__new__(
            core_client_module.AsyncMPClient
        )
        client.client_index = 0
        client.core_engine = b"engine"
        client.utility_results = {}

        def encode(value: tuple[Any, ...]) -> tuple[bytes]:
            captured_call_id.append(value[1])
            return (b"payload",)

        async def send(*_args: Any, **_kwargs: Any) -> None:
            return None

        client.encoder = SimpleNamespace(encode=encode)
        client._send_input_message = send
        client._ensure_output_queue_task = lambda: None
        task = asyncio.create_task(
            client._call_utility_async("method", engine=b"engine")
        )
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert client.utility_results == {}
        core_client_module._process_utility_output(
            UtilityOutput(
                call_id=captured_call_id[0],
                result=UtilityResult("late"),
            ),
            client.utility_results,
        )
        assert client.utility_results == {}

    asyncio.run(scenario())


@pytest.mark.asyncio
async def test_async_inproc_cancelled_utility_discards_waiter_and_late_output() -> None:
    resources = InprocBackgroundResources()
    core = _FakeEngineCore()
    resources.attach_engine(core)  # type: ignore[arg-type]
    client = AsyncInprocClient.__new__(AsyncInprocClient)
    client.resources = resources
    client.client_index = 0
    client._request_encoder = MsgpackEncoder()
    client._utility_decoder = MsgpackDecoder(share_mem=False)

    utility_task = asyncio.create_task(client.call_utility_async("deferred"))
    await asyncio.sleep(0)
    request_type, payload = core.input_queue.get_nowait()
    assert request_type is EngineCoreRequestType.UTILITY
    _client_index, call_id, _method, _args = payload

    utility_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await utility_task
    assert resources.utility_results == {}

    resources.publish_output(
        EngineCoreOutputs(
            utility_output=UtilityOutput(
                call_id=call_id,
                result=UtilityResult("late"),
            )
        )
    )
    await asyncio.sleep(0)
    assert resources.utility_results == {}


@pytest.mark.asyncio
async def test_dplb_lifecycle_broadcast_survives_caller_cancellation() -> None:
    client = core_client_module.DPLBAsyncMPClient.__new__(
        core_client_module.DPLBAsyncMPClient
    )
    client.core_engines = [b"rank-0", b"rank-1"]
    client.client_count = 1
    client.resources = SimpleNamespace(engine_dead=False)
    release_admission = asyncio.Event()
    all_completed = asyncio.Event()
    admitted: list[bytes] = []
    completed: list[bytes] = []

    async def admit(
        _method: str, *_args: Any, engine: bytes
    ) -> tuple[int, asyncio.Future[Any]]:
        admitted.append(engine)
        await release_admission.wait()
        completed.append(engine)
        if len(completed) == len(client.core_engines):
            all_completed.set()
        future = asyncio.get_running_loop().create_future()
        future.set_result(None)
        return len(completed), future

    client._admit_utility_async = admit
    call = asyncio.create_task(client.call_utility_async("pause_scheduler"))
    await asyncio.sleep(0)
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call

    release_admission.set()
    await asyncio.wait_for(all_completed.wait(), 1.0)

    async def wait_for_broadcast_cleanup() -> None:
        while client._dp_lifecycle_broadcast_tasks:
            await asyncio.sleep(0)

    await asyncio.wait_for(wait_for_broadcast_cleanup(), 1.0)
    assert set(admitted) == {b"rank-0", b"rank-1"}
    assert set(completed) == set(admitted)
    assert client._dp_lifecycle_broadcast_tasks == set()


@pytest.mark.asyncio
async def test_dplb_lifecycle_lock_remains_held_after_caller_cancellation() -> None:
    client = core_client_module.DPLBAsyncMPClient.__new__(
        core_client_module.DPLBAsyncMPClient
    )
    client.core_engines = [b"rank-0", b"rank-1"]
    client.client_count = 1
    client.resources = SimpleNamespace(engine_dead=False)
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    first_count = 0
    second_admissions: list[bytes] = []

    async def admit(
        method: str, *_args: Any, engine: bytes
    ) -> tuple[int, asyncio.Future[Any]]:
        nonlocal first_count
        if method == "pause_scheduler":
            first_count += 1
            if first_count == len(client.core_engines):
                first_started.set()
            await release_first.wait()
        else:
            second_admissions.append(engine)
        future = asyncio.get_running_loop().create_future()
        future.set_result(None)
        return first_count + len(second_admissions), future

    client._admit_utility_async = admit
    first = asyncio.create_task(client.call_utility_async("pause_scheduler"))
    await asyncio.wait_for(first_started.wait(), 1.0)
    second = asyncio.create_task(client.call_utility_async("sleep", 1, "abort"))
    await asyncio.sleep(0)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    await asyncio.sleep(0)
    assert second_admissions == []

    release_first.set()
    assert await asyncio.wait_for(second, 1.0) is None
    assert set(second_admissions) == {b"rank-0", b"rank-1"}


@pytest.mark.asyncio
async def test_dplb_multi_api_lifecycle_control_fails_before_any_send() -> None:
    client = core_client_module.DPLBAsyncMPClient.__new__(
        core_client_module.DPLBAsyncMPClient
    )
    client.core_engines = [b"rank-0", b"rank-1"]
    client.client_count = 2
    client.resources = SimpleNamespace(engine_dead=False)
    admitted: list[bytes] = []

    async def admit(
        _method: str, *_args: Any, engine: bytes
    ) -> tuple[int, asyncio.Future[Any]]:
        admitted.append(engine)
        raise AssertionError("multi-API lifecycle control attempted a send")

    client._admit_utility_async = admit

    with pytest.raises(RuntimeError, match="single lifecycle-owner"):
        await client.call_utility_async("reset_prefix_cache")
    assert admitted == []


@pytest.mark.asyncio
async def test_dplb_partial_lifecycle_admission_fails_all_waiters_and_tears_down() -> None:
    client = core_client_module.DPLBAsyncMPClient.__new__(
        core_client_module.DPLBAsyncMPClient
    )
    client.core_engines = [b"rank-0", b"rank-1"]
    client.client_count = 1
    client.resources = SimpleNamespace(engine_dead=False)
    client.utility_results = {}
    shutdown_calls: list[None] = []
    client.shutdown = lambda: shutdown_calls.append(None)
    admitted_future = asyncio.get_running_loop().create_future()

    async def admit(
        _method: str, *_args: Any, engine: bytes
    ) -> tuple[int, asyncio.Future[Any]]:
        if engine == b"rank-1":
            raise RuntimeError("send failed")
        client.utility_results[7] = admitted_future
        return 7, admitted_future

    client._admit_utility_async = admit

    with pytest.raises(RuntimeError, match="every rank"):
        await client._broadcast_dp_lifecycle_utility(
            "pause_scheduler", (), tuple(client.core_engines)
        )
    assert client.resources.engine_dead
    assert client.utility_results == {}
    assert isinstance(admitted_future.exception(), RuntimeError)
    assert shutdown_calls == [None]


@pytest.mark.asyncio
async def test_async_mp_dead_resource_rejects_new_utility_without_waiter_leak() -> None:
    client = core_client_module.AsyncMPClient.__new__(
        core_client_module.AsyncMPClient
    )
    client.client_index = 0
    client.core_engine = b"engine"
    client.utility_results = {}
    client.resources = SimpleNamespace(engine_dead=True)
    client.encoder = SimpleNamespace(encode=lambda _value: (b"payload",))

    with pytest.raises(EngineDeadError):
        await client._call_utility_async("method", engine=b"engine")
    assert client.utility_results == {}


def test_sync_mp_utility_send_failure_discards_registered_waiter() -> None:
    client = core_client_module.SyncMPClient.__new__(
        core_client_module.SyncMPClient
    )
    client.utility_results = {}
    client._send_input = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("send failed")
    )

    with pytest.raises(RuntimeError, match="send failed"):
        client.call_utility("method")
    assert client.utility_results == {}


def test_wait_clear_rejects_active_resumable_before_pause_mutation() -> None:
    lifecycle_calls: list[str] = []
    scheduler = SimpleNamespace(
        pause_state=PauseState.UNPAUSED,
        requests={
            "stream": SimpleNamespace(
                resumable=True,
                is_finished=lambda: False,
            )
        },
        can_reset_prefix_cache=lambda **_kwargs: True,
        set_pause_state=lambda _state: lifecycle_calls.append("pause"),
    )
    engine = EngineCoreProc.__new__(EngineCoreProc)
    engine.scheduler = scheduler

    with pytest.raises(RuntimeError, match="resumable streaming"):
        engine.pause_scheduler(mode="wait", clear_cache=True)

    assert lifecycle_calls == []
    assert scheduler.pause_state == PauseState.UNPAUSED


def test_dp_wait_clear_rechecks_resumable_at_cadence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, lifecycle_calls, reset_calls = _make_dp_lifecycle_engine()
    engine.scheduler.requests = {
        "stream": SimpleNamespace(resumable=True, is_finished=lambda: False)
    }
    future = engine.pause_scheduler(mode="wait", clear_cache=True)
    assert isinstance(future, Future)
    operation = cast(Any, engine._pending_dp_control_op)
    signature = operation.signature
    sync_inputs: list[dict[str, Any]] = []

    def sync_state(_group: object, **kwargs: Any) -> tuple[bool, int, int, int, int, int]:
        sync_inputs.append(kwargs)
        return False, 0, 2, 1, 2 * signature, 2 * signature * signature

    monkeypatch.setattr(core_module.ParallelConfig, "sync_dp_state", sync_state)

    assert not engine._has_global_unfinished_reqs(local_work=False)
    with pytest.raises(RuntimeError, match="preflight was rejected"):
        future.result()
    assert sync_inputs[0]["control_op_blocked"] is True
    assert lifecycle_calls == []
    assert reset_calls == []


@pytest.mark.parametrize("destroy_fails", [False, True])
def test_dp_shutdown_always_attempts_stateless_group_cleanup_and_keeps_primary_error(
    monkeypatch: pytest.MonkeyPatch,
    destroy_fails: bool,
) -> None:
    engine = DPEngineCoreProc.__new__(DPEngineCoreProc)
    dp_group = object()
    engine.dp_group = dp_group
    primary_error = RuntimeError("parent shutdown failed")
    destroy_calls: list[object] = []

    def fail_parent_shutdown(_self: EngineCoreProc) -> None:
        raise primary_error

    def destroy_group(group: object) -> None:
        destroy_calls.append(group)
        if destroy_fails:
            raise RuntimeError("DP group cleanup failed")

    monkeypatch.setattr(EngineCoreProc, "shutdown", fail_parent_shutdown)
    monkeypatch.setattr(
        core_module,
        "stateless_destroy_torch_distributed_process_group",
        destroy_group,
    )

    with pytest.raises(RuntimeError, match="parent shutdown failed") as exc_info:
        engine.shutdown()

    assert exc_info.value is primary_error
    assert destroy_calls == [dp_group]
    assert engine.dp_group is (dp_group if destroy_fails else None)


@pytest.mark.parametrize("block_pool_free", [False, True])
def test_nonpreempting_cache_preflight_mirrors_block_pool_ownership(
    block_pool_free: bool,
) -> None:
    scheduler = SimpleNamespace(
        _prefix_pins={},
        kv_cache_manager=SimpleNamespace(
            has_pinned_prefixes=lambda: False,
            can_reset_prefix_cache=lambda: block_pool_free,
        ),
        connector=None,
        _paused_requests={},
        _pending_pause_req_ids=set(),
        _has_unsynchronized_kv_ownership=lambda: False,
        _inflight_prefixes=SimpleNamespace(has_state=lambda: False),
    )

    assert (
        Scheduler.can_reset_prefix_cache(
            cast(Any, scheduler), reset_running_requests=False
        )
        is block_pool_free
    )
