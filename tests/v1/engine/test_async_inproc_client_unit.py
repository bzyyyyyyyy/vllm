# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Model-free tests for the AsyncInprocClient thread/asyncio bridge."""

import asyncio
import queue
import threading
from types import SimpleNamespace
from typing import Any, cast

import pytest

import vllm.v1.engine.core_client as core_client_module
from vllm.config import VllmConfig
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequestType,
    UtilityOutput,
)
from vllm.v1.engine.core_client import (
    AsyncInprocClient,
    InprocBackgroundResources,
)
# noinspection PyProtectedMember
from vllm.v1.engine.core_client import _copy_exception_without_traceback
from vllm.v1.engine.exceptions import EngineDeadError
from vllm.v1.executor.abstract import Executor
from vllm.v1.executor.uniproc_executor import UniProcExecutor
from vllm.v1.serial_utils import UtilityResult


class _FakeEngineCore:
    def __init__(self) -> None:
        self.input_queue: queue.Queue[tuple[EngineCoreRequestType, Any]] = queue.Queue()
        self.aborts_queue: queue.Queue[list[str]] = queue.Queue()
        self.shutdown_state: Any = None


def _config(**parallel_overrides: Any) -> VllmConfig:
    parallel = {
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
        "data_parallel_size_local": 1,
        "enable_elastic_ep": False,
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
    resources.attach_engine(core)  # type: ignore[arg-type]
    resources.startup_future.set_result(None)
    try:
        while True:
            request_type, _request = core.input_queue.get()
            if request_type is EngineCoreRequestType.WAKEUP:
                break
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


def test_exception_copy_drops_traceback_context_and_cause() -> None:
    source = _captured_error()
    detached = _copy_exception_without_traceback(source)

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
    assert await asyncio.wait_for(resources.outputs_queue.get(), timeout=1.0) is pending

    utility_future = asyncio.get_running_loop().create_future()
    resources.register_utility(7, utility_future)
    utility_output = EngineCoreOutputs(
        utility_output=UtilityOutput(call_id=7, result=UtilityResult("utility-result"))
    )
    utility_thread = threading.Thread(
        target=resources.publish_output, args=(utility_output,)
    )
    utility_thread.start()
    utility_thread.join()
    assert await asyncio.wait_for(utility_future, timeout=1.0) == "utility-result"

    live = _output("live")
    output_thread = threading.Thread(target=resources.publish_output, args=(live,))
    output_thread.start()
    output_thread.join()
    assert await asyncio.wait_for(resources.outputs_queue.get(), timeout=1.0) is live


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

    terminal = await asyncio.wait_for(output_waiter, timeout=1.0)
    assert isinstance(terminal, EngineDeadError)
    with pytest.raises(EngineDeadError):
        await asyncio.wait_for(utility_future, timeout=1.0)

    assert resources.fatal_error is not None
    assert resources.fatal_error.__traceback__ is None
    assert resources.fatal_error.__context__ is None
    assert resources.fatal_error.__cause__ is None
    assert resources.terminal_output_published
    assert resources.terminal_output_delivered

    resources.mark_stopped()
    await asyncio.sleep(0)
    assert resources.outputs_queue.empty()


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
        await asyncio.wait_for(output_waiter, timeout=1.0)
    assert not owner_thread.is_alive()


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


def test_teardown_error_is_propagated_after_thread_stops() -> None:
    resources = InprocBackgroundResources()

    def finish_with_teardown_error() -> None:
        resources.teardown_error = _copy_exception_without_traceback(
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
        ("data_parallel_size", 2),
        ("data_parallel_size_local", 2),
        ("enable_elastic_ep", True),
    ],
)
def test_parallel_and_elastic_configs_fail_fast(
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
