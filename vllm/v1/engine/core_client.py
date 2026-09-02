# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import contextlib
import queue
import sys
import threading
import time
import uuid
import weakref
from abc import ABC, abstractmethod
from collections import Counter, defaultdict, deque
from collections.abc import Awaitable, Callable, Sequence
from concurrent.futures import Future, InvalidStateError as FutureInvalidStateError
from dataclasses import dataclass, field
from datetime import timedelta
from multiprocessing.connection import Connection
from multiprocessing.queues import Queue
from threading import Thread
from typing import Any, Literal, TypeAlias, TypeVar

import msgspec
import msgspec.msgpack
import torch
import zmq
import zmq.asyncio

from vllm import envs
from vllm.config import VllmConfig
from vllm.envs import VLLM_ENGINE_READY_TIMEOUT_S
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.tasks import SupportedTask
from vllm.tracing import instrument
from vllm.utils.async_utils import in_loop
from vllm.utils.network_utils import (
    close_sockets,
    get_open_zmq_inproc_path,
    make_zmq_socket,
)
from vllm.v1.engine import (
    EEP_NOTIFICATION_CALL_ID,
    FT_STATUS_CALL_ID,
    EEPNotificationType,
    EngineCoreOutputs,
    EngineCoreReadyResponse,
    EngineCoreRequest,
    EngineCoreRequestType,
    PauseMode,
    PrefixPinResult,
    PrefixPinTier,
    ReconfigureDistributedRequest,
    ReconfigureRankType,
    UtilityOutput,
)
from vllm.v1.engine.coordinator import DPCoordinator
from vllm.v1.engine.core import EngineCore, EngineCoreProc, EngineShutdownState
from vllm.v1.engine.exceptions import EngineDeadError
from vllm.v1.engine.tensor_ipc import TensorIpcSender
from vllm.v1.engine.utils import (
    CoreEngineActorManager,
    CoreEngineProcManager,
    ElasticEPScaleUpReservationError,
    get_engine_zmq_addresses,
    launch_core_engines,
)
from vllm.v1.executor import Executor, UniProcExecutor
from vllm.v1.fault_tolerance.engine_core_sentinel import FT_UTILITY_METHOD
from vllm.v1.fault_tolerance.utils import (
    FaultToleranceRequest,
    FaultToleranceResult,
)
from vllm.v1.pool.late_interaction import get_late_interaction_engine_index
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder, bytestr

logger = init_logger(__name__)

AnyFuture: TypeAlias = asyncio.Future[Any] | Future[Any]

_R = TypeVar("_R")  # Return type for collective_rpc

EngineIdentity = bytes
AsyncEngineCoreMode: TypeAlias = Literal["mp", "inproc"]
_MAX_DP_ENGINE_COUNT = 1 << 16  # Engine identities are encoded in two bytes.

_INPROC_DIST_OWNERSHIP_LOCK = threading.Lock()
_INPROC_DIST_OWNERSHIP_TOKEN: object | None = None
_ELASTIC_EP_STORE_WAIT_SLICE_S = 1.0


def _reserve_inproc_distributed_ownership() -> object:
    """Atomically reserve process-global distributed state for one client."""
    global _INPROC_DIST_OWNERSHIP_TOKEN
    with _INPROC_DIST_OWNERSHIP_LOCK:
        if _INPROC_DIST_OWNERSHIP_TOKEN is not None:
            raise RuntimeError(
                "AsyncInprocClient requires exclusive ownership of "
                "torch.distributed state; another AsyncInprocClient already "
                "owns the host process distributed state"
            )
        if torch.distributed.is_initialized():
            raise RuntimeError(
                "AsyncInprocClient requires exclusive ownership of "
                "torch.distributed state; the host process already has an "
                "initialized default process group"
            )
        token = object()
        _INPROC_DIST_OWNERSHIP_TOKEN = token
        return token


def _validate_inproc_distributed_ownership(token: object) -> None:
    """Validate the reservation immediately before owner-thread startup."""
    with _INPROC_DIST_OWNERSHIP_LOCK:
        if _INPROC_DIST_OWNERSHIP_TOKEN is not token:
            raise RuntimeError(
                "AsyncInprocClient lost its exclusive torch.distributed "
                "ownership reservation before EngineCore initialization"
            )
        if torch.distributed.is_initialized():
            raise RuntimeError(
                "AsyncInprocClient requires exclusive ownership of "
                "torch.distributed state; the host process initialized a "
                "default process group after ownership was reserved"
            )


def _release_inproc_distributed_ownership(token: object) -> None:
    """Release only the reservation identified by ``token``."""
    global _INPROC_DIST_OWNERSHIP_TOKEN
    with _INPROC_DIST_OWNERSHIP_LOCK:
        if _INPROC_DIST_OWNERSHIP_TOKEN is token:
            _INPROC_DIST_OWNERSHIP_TOKEN = None


def _copy_exception_without_traceback(error: Exception) -> Exception:
    """Copy an exception without retaining owner-thread frames or GPU objects."""
    builtins_module = sys.modules["builtins"]
    base_exception_group_type = getattr(
        builtins_module, "BaseExceptionGroup", None
    )
    if base_exception_group_type is not None and isinstance(
        error, base_exception_group_type
    ):
        nested = [
            _copy_exception_without_traceback(exc)
            if isinstance(exc, Exception)
            else RuntimeError(f"{type(exc).__name__}: {exc}")
            for exc in error.exceptions
        ]
        exception_group_type = getattr(builtins_module, "ExceptionGroup")
        detached = exception_group_type(
            getattr(error, "message", str(error)), nested
        )
        assert isinstance(detached, Exception)
    else:
        safe_arg_types = (str, bytes, int, float, bool, type(None))
        safe_args = all(isinstance(arg, safe_arg_types) for arg in error.args)
        try:
            # Common Python, PyTorch, and vLLM exceptions carry only scalar
            # args. Do not copy __dict__: custom attributes may own tensors.
            detached = (
                type(error)(*error.args)
                if safe_args
                else type(error)(str(error))
            )
        except Exception:
            detached = RuntimeError(f"{type(error).__name__}: {error}")

    add_note = getattr(detached, "add_note", None)
    if add_note is not None and not getattr(detached, "__notes__", None):
        for note in getattr(error, "__notes__", ()):
            add_note(note)
    detached.__traceback__ = None
    detached.__cause__ = None
    detached.__context__ = None
    return detached


class EngineCoreClient(ABC):
    """
    EngineCoreClient: subclasses handle different methods for pushing
        and pulling from the EngineCore for asyncio / multiprocessing.

    Subclasses:
    * InprocClient: In process EngineCore (for V0-style LLMEngine use)
    * SyncMPClient: ZMQ + background proc EngineCore (for LLM)
    * AsyncMPClient: ZMQ + background proc EngineCore w/ asyncio (for AsyncLLM)
    * AsyncInprocClient: owner-thread EngineCore w/ asyncio (for AsyncLLM)
    """

    @staticmethod
    def make_client(
        multiprocess_mode: bool,
        asyncio_mode: bool,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        client_addresses: dict[str, Any] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ) -> "EngineCoreClient":
        if asyncio_mode:
            return EngineCoreClient.make_async_client(
                "mp" if multiprocess_mode else "inproc",
                vllm_config,
                executor_class,
                log_stats,
                client_addresses,
                client_count,
                client_index,
            )

        if multiprocess_mode:
            return SyncMPClient(vllm_config, executor_class, log_stats)

        return InprocClient(vllm_config, executor_class, log_stats)

    @staticmethod
    def make_async_client(
        mode: AsyncEngineCoreMode,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        client_addresses: dict[str, Any] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ) -> "AsyncMPClient | AsyncInprocClient":
        """Create an asyncio client using the requested EngineCore backend."""
        client_args = (
            vllm_config,
            executor_class,
            log_stats,
            client_addresses,
            client_count,
            client_index,
        )
        if mode == "mp":
            return EngineCoreClient.make_async_mp_client(*client_args)
        if mode == "inproc":
            return AsyncInprocClient(*client_args)
        raise ValueError(f"Unsupported async EngineCore mode: {mode!r}")

    @staticmethod
    @instrument(span_name="Overall Loading")
    def make_async_mp_client(
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        client_addresses: dict[str, Any] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ) -> "AsyncMPClient":
        parallel_config = vllm_config.parallel_config
        client_args = (
            vllm_config,
            executor_class,
            log_stats,
            client_addresses,
            client_count,
            client_index,
        )
        if parallel_config.data_parallel_size > 1:
            if parallel_config.data_parallel_external_lb:
                # External load balancer - client per DP rank.
                return DPAsyncMPClient(*client_args)
            # Internal load balancer - client balances to all DP ranks.
            return DPLBAsyncMPClient(*client_args)
        return AsyncMPClient(*client_args)

    @abstractmethod
    def shutdown(self, timeout: float | None = None) -> None: ...

    def get_output(self) -> EngineCoreOutputs:
        raise NotImplementedError

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        raise NotImplementedError

    def add_request(self, request: EngineCoreRequest) -> None:
        raise NotImplementedError

    def profile(self, is_start: bool = True, profile_prefix: str | None = None) -> None:
        raise NotImplementedError

    def reset_mm_cache(self) -> None:
        raise NotImplementedError

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        raise NotImplementedError

    def reset_encoder_cache(self) -> None:
        raise NotImplementedError

    def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None:
        raise NotImplementedError

    def wake_up(self, tags: list[str] | None = None) -> None:
        raise NotImplementedError

    def is_sleeping(self) -> bool:
        raise NotImplementedError

    def execute_dummy_batch(self) -> None:
        raise NotImplementedError

    def set_weight_version(self, weight_version: str) -> None:
        raise NotImplementedError

    def get_weight_version(self) -> str:
        raise NotImplementedError

    async def execute_dummy_batch_async(self) -> None:
        raise NotImplementedError

    async def set_weight_version_async(self, weight_version: str) -> None:
        raise NotImplementedError

    async def get_weight_version_async(self) -> str:
        raise NotImplementedError

    def abort_requests(self, request_ids: list[str]) -> None:
        raise NotImplementedError

    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError

    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    def list_loras(self) -> set[int]:
        raise NotImplementedError

    def pin_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    def save_sharded_state(
        self, path: str, pattern: str | None = None, max_size: int | None = None
    ) -> None:
        raise NotImplementedError

    def collective_rpc(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        raise NotImplementedError

    def dp_engines_running(self) -> bool:
        """Returns True if data parallel engines are collectively in a
        running state."""
        raise NotImplementedError

    async def commit_elastic_ep(self) -> None:
        raise NotImplementedError

    async def prepare_elastic_ep(self, new_data_parallel_size: int) -> None:
        raise NotImplementedError

    async def get_output_async(self) -> EngineCoreOutputs:
        raise NotImplementedError

    async def get_supported_tasks_async(self) -> tuple[SupportedTask, ...]:
        raise NotImplementedError

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        raise NotImplementedError

    async def call_utility_async(self, method: str, *args) -> Any:
        raise NotImplementedError

    async def pin_prefix_async(
        self,
        pin_id: str,
        request: EngineCoreRequest,
        tier: PrefixPinTier = "gpu",
    ) -> PrefixPinResult:
        request.client_index = self.client_index
        return await self.call_utility_async("pin_prefix", pin_id, request, tier)

    async def unpin_prefix_async(
        self, pin_id: str, expected_request_id: str | None = None
    ) -> bool:
        return await self.call_utility_async(
            "unpin_prefix", pin_id, expected_request_id
        )

    async def pause_prefix_async(self, pin_id: str) -> None:
        await self.call_utility_async("pause_prefix", pin_id)

    async def resume_prefix_async(self, pin_id: str) -> None:
        await self.call_utility_async("resume_prefix", pin_id)

    async def pause_requests_async(self, request_ids: list[str]) -> None:
        await self.call_utility_async("pause_requests", request_ids)

    async def resume_requests_async(self, request_ids: list[str]) -> None:
        await self.call_utility_async("resume_requests", request_ids)

    async def profile_async(
        self, is_start: bool = True, profile_prefix: str | None = None
    ) -> None:
        raise NotImplementedError

    async def reset_mm_cache_async(self) -> None:
        raise NotImplementedError

    async def reset_prefix_cache_async(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        raise NotImplementedError

    async def reset_encoder_cache_async(self) -> None:
        raise NotImplementedError

    async def sleep_async(self, level: int = 1, mode: PauseMode = "abort") -> None:
        raise NotImplementedError

    async def wake_up_async(self, tags: list[str] | None = None) -> None:
        raise NotImplementedError

    async def is_sleeping_async(self) -> bool:
        raise NotImplementedError

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        raise NotImplementedError

    async def add_lora_async(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError

    async def remove_lora_async(self, lora_id: int) -> bool:
        raise NotImplementedError

    async def list_loras_async(self) -> set[int]:
        raise NotImplementedError

    async def pin_lora_async(self, lora_id: int) -> bool:
        raise NotImplementedError

    async def save_sharded_state_async(
        self, path: str, pattern: str | None = None, max_size: int | None = None
    ) -> None:
        raise NotImplementedError

    async def collective_rpc_async(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        raise NotImplementedError

    async def handle_fault(
        self, fault_tolerance_request: FaultToleranceRequest
    ) -> FaultToleranceResult:
        raise NotImplementedError

    async def get_status(self):
        raise NotImplementedError


class InprocClient(EngineCoreClient):
    """
    InprocClient: client for in-process EngineCore. Intended
    for use in LLMEngine for V0-style add_request() and step()
        EngineCore setup in this process (no busy loop).

        * pushes EngineCoreRequest directly into the EngineCore
        * pulls EngineCoreOutputs by stepping the EngineCore
    """

    def __init__(self, *args, **kwargs):
        self.engine_core = EngineCore(*args, **kwargs)

    def get_output(self) -> EngineCoreOutputs:
        outputs, model_executed = self.engine_core.step_fn()
        self.engine_core.post_step(model_executed=model_executed)
        return outputs and outputs.get(0) or EngineCoreOutputs()

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.engine_core.get_supported_tasks()

    def add_request(self, request: EngineCoreRequest) -> None:
        req, request_wave = self.engine_core.preprocess_add_request(request)
        self.engine_core.add_request(req, request_wave)

    def abort_requests(self, request_ids: list[str]) -> None:
        if len(request_ids) > 0:
            self.engine_core.abort_requests(request_ids)

    def shutdown(self, timeout: float | None = None) -> None:
        self.engine_core.shutdown()

    def profile(self, is_start: bool = True, profile_prefix: str | None = None) -> None:
        self.engine_core.profile(is_start, profile_prefix)

    def reset_mm_cache(self) -> None:
        self.engine_core.reset_mm_cache()

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return self.engine_core.reset_prefix_cache(
            reset_running_requests, reset_connector
        )

    def reset_encoder_cache(self) -> None:
        self.engine_core.reset_encoder_cache()

    def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None:
        if mode == "wait":
            raise ValueError("'wait' pause mode is not supported in inproc-engine mode")
        result = self.engine_core.sleep(level, mode)
        assert result is None

    def wake_up(self, tags: list[str] | None = None) -> None:
        self.engine_core.wake_up(tags)

    def is_sleeping(self) -> bool:
        return self.engine_core.is_sleeping()

    def execute_dummy_batch(self) -> None:
        self.engine_core.execute_dummy_batch()

    def set_weight_version(self, weight_version: str) -> None:
        self.engine_core.set_weight_version(weight_version)

    def get_weight_version(self) -> str:
        return self.engine_core.get_weight_version()

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.engine_core.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.engine_core.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.engine_core.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.engine_core.pin_lora(lora_id)

    def save_sharded_state(
        self, path: str, pattern: str | None = None, max_size: int | None = None
    ) -> None:
        self.engine_core.save_sharded_state(path, pattern, max_size)

    def collective_rpc(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        return self.engine_core.collective_rpc(method, timeout, args, kwargs)

    def dp_engines_running(self) -> bool:
        return False


@dataclass
class BackgroundResources:
    """Used as a finalizer for clean shutdown, avoiding
    circular reference back to the client object."""

    ctx: zmq.Context
    # If CoreEngineProcManager, it manages local engines;
    # if CoreEngineActorManager, it manages all engines.
    engine_manager: CoreEngineProcManager | CoreEngineActorManager | None = None
    coordinator: DPCoordinator | None = None
    output_socket: zmq.Socket | zmq.asyncio.Socket | None = None
    input_socket: zmq.Socket | zmq.asyncio.Socket | None = None
    first_req_send_socket: zmq.asyncio.Socket | None = None
    first_req_rcv_socket: zmq.asyncio.Socket | None = None
    stats_update_socket: zmq.asyncio.Socket | None = None
    output_queue_task: asyncio.Task | None = None
    stats_update_task: asyncio.Task | None = None
    shutdown_path: str | None = None
    sync_output_ready: threading.Event | None = None
    sync_output_stopped: threading.Event | None = None

    # Set if any of the engines are dead. Here so that the output
    # processing threads can access it without holding a ref to the client.
    engine_dead: bool = False

    def __call__(self):
        """Clean up background resources."""

        logger.debug_once("[shutdown] MPClient: background resource cleanup start")
        self.engine_dead = True
        primary_error: BaseException | None = None

        def record_cleanup_failure(message: str, error: BaseException) -> None:
            nonlocal primary_error
            if primary_error is None:
                primary_error = error
            else:
                logger.error(
                    "%s: %s: %s",
                    message,
                    type(error).__name__,
                    error,
                )

        if self.engine_manager is not None:
            try:
                self.engine_manager.shutdown(
                    timeout=envs.VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS
                )
            except BaseException as error:
                record_cleanup_failure(
                    "Engine manager background cleanup failed", error
                )
            else:
                self.engine_manager = None
        if self.coordinator is not None:
            try:
                self.coordinator.shutdown()
            except BaseException as error:
                record_cleanup_failure(
                    "DP coordinator background cleanup failed", error
                )
            else:
                self.coordinator = None

        try:
            if isinstance(self.output_socket, zmq.asyncio.Socket):
                # Async case.
                sockets = (
                    self.output_socket,
                    self.input_socket,
                    self.first_req_send_socket,
                    self.first_req_rcv_socket,
                    self.stats_update_socket,
                )

                tasks = (self.output_queue_task, self.stats_update_task)
                running_owner_loops: list[asyncio.AbstractEventLoop] = []
                for task in tasks:
                    task_loop = getattr(task, "_loop", None)
                    if (
                        task_loop is not None
                        and not task_loop.is_closed()
                        and task_loop.is_running()
                        and all(
                            task_loop is not known_loop
                            for known_loop in running_owner_loops
                        )
                    ):
                        running_owner_loops.append(task_loop)
                if len(running_owner_loops) > 1:
                    raise RuntimeError(
                        "Async MP background tasks belong to different "
                        "running event loops"
                    )
                loop = (
                    running_owner_loops[0] if running_owner_loops else None
                )
                cleanup_done = threading.Event()
                cleanup_errors: list[BaseException] = []

                def close_sockets_and_tasks():
                    try:
                        try:
                            close_sockets(sockets)
                        except BaseException as error:
                            cleanup_errors.append(error)
                        for task in tasks:
                            if task is not None and not task.done():
                                try:
                                    task.cancel()
                                except BaseException as error:
                                    cleanup_errors.append(error)
                    finally:
                        cleanup_done.set()

                def close_without_owner_loop() -> None:
                    # ZMQ sockets still need deterministic closure after the
                    # asyncio loop is gone. Do not call Task.cancel() across a
                    # closed-loop/thread boundary.
                    close_sockets(sockets)
                    for task in tasks:
                        if (
                            task is not None
                            and task.done()
                            and not task.cancelled()
                        ):
                            with contextlib.suppress(Exception):
                                task.exception()
                    self.output_queue_task = None
                    self.stats_update_task = None

                if (
                    loop is not None
                    and not loop.is_closed()
                    and loop.is_running()
                ):
                    if in_loop(loop):
                        close_sockets_and_tasks()
                    else:
                        try:
                            loop.call_soon_threadsafe(close_sockets_and_tasks)
                        except RuntimeError:
                            # The loop can close after is_closed() above.
                            close_without_owner_loop()
                        else:
                            timeout = envs.VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS
                            deadline = time.monotonic() + timeout
                            while not cleanup_done.wait(
                                min(0.1, max(0.0, deadline - time.monotonic()))
                            ):
                                if not loop.is_running():
                                    raise RuntimeError(
                                        "Async MP owner loop stopped before "
                                        "socket/task cleanup completed"
                                    )
                                if time.monotonic() >= deadline:
                                    raise TimeoutError(
                                        "Timed out waiting for async MP owner "
                                        "loop to close sockets and tasks"
                                    )
                else:
                    close_without_owner_loop()
                if cleanup_errors:
                    raise cleanup_errors[0]
                self.output_socket = None
                self.input_socket = None
                self.first_req_send_socket = None
                self.first_req_rcv_socket = None
                self.stats_update_socket = None
                self.output_queue_task = None
                self.stats_update_task = None
            else:
                # Sync case.

                # ZMQ context termination can hang if the sockets
                # aren't explicitly closed first.
                close_sockets((self.output_socket, self.input_socket))

                if self.shutdown_path is not None:
                    ready = self.sync_output_ready
                    stopped = self.sync_output_stopped
                    wait_timeout = envs.VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS
                    if stopped is None or not stopped.is_set():
                        if ready is not None and not ready.wait(wait_timeout):
                            raise TimeoutError(
                                "SyncMP output thread did not bind its "
                                "shutdown socket before cleanup"
                            )

                        # The output thread owns its ZMQ socket and must close
                        # it itself. Readiness guarantees the inproc peer is
                        # bound before this zero-linger nonblocking send.
                        shutdown_sender = self.ctx.socket(zmq.PAIR)
                        try:
                            shutdown_sender.setsockopt(zmq.LINGER, 0)
                            shutdown_sender.connect(self.shutdown_path)
                            shutdown_sender.send(b"", flags=zmq.NOBLOCK)
                        finally:
                            shutdown_sender.close(linger=0)

                        if stopped is not None and not stopped.wait(wait_timeout):
                            raise TimeoutError(
                                "SyncMP output thread did not stop after its "
                                "shutdown signal"
                            )

                    # Clear the endpoint only after the owner thread confirms
                    # it has closed the output socket. Failed attempts retain
                    # all state for the next explicit/finalizer retry.
                    self.shutdown_path = None
                    self.sync_output_ready = None
                    self.sync_output_stopped = None
        except BaseException as error:
            record_cleanup_failure("Socket/task background cleanup failed", error)

        logger.debug_once("[shutdown] MPClient: background resource cleanup complete")
        if primary_error is not None:
            raise primary_error

    def validate_alive(self, frames: Sequence[zmq.Frame]):
        if len(frames) == 1 and (frames[0].buffer == EngineCoreProc.ENGINE_CORE_DEAD):
            self.engine_dead = True
            raise EngineDeadError()


@dataclass
class InprocBackgroundResources:
    """Resources owned by an async in-process EngineCore thread."""

    startup_future: Future[None] = field(default_factory=Future)
    lock: threading.RLock = field(default_factory=threading.RLock)
    outputs_queue: asyncio.Queue[EngineCoreOutputs | Exception] = field(
        default_factory=asyncio.Queue
    )
    pending_outputs: deque[EngineCoreOutputs | Exception] = field(
        default_factory=deque
    )
    utility_results: dict[int, asyncio.Future[Any]] = field(default_factory=dict)
    thread: Thread | None = None
    engine_core: "_AsyncInprocEngineCore | None" = None
    loop: asyncio.AbstractEventLoop | None = None
    fatal_error: Exception | None = None
    teardown_error: Exception | None = None
    engine_dead: bool = False
    closing: bool = False
    stopped: bool = False
    terminal_output_published: bool = False
    terminal_output_delivered: bool = False
    dist_ownership_token: object | None = None

    def validate_dist_ownership(self) -> None:
        with self.lock:
            token = self.dist_ownership_token
        if token is None:
            raise RuntimeError(
                "AsyncInprocClient EngineCore owner started without a "
                "torch.distributed ownership reservation"
            )
        _validate_inproc_distributed_ownership(token)

    def release_dist_ownership(self) -> None:
        with self.lock:
            token = self.dist_ownership_token
            self.dist_ownership_token = None
        if token is not None:
            _release_inproc_distributed_ownership(token)

    @staticmethod
    def _dead_error(cause: Exception | None = None) -> EngineDeadError:
        error = EngineDeadError(suppress_context=True)
        if cause is not None:
            error.__cause__ = cause
        return error

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind the bridge lazily to the first asyncio loop using it."""
        with self.lock:
            if self.loop is not None and self.loop is not loop:
                raise RuntimeError(
                    "AsyncInprocClient cannot be used from multiple asyncio loops"
                )
            if self.loop is loop:
                return
            self.loop = loop
            pending = tuple(self.pending_outputs)
            self.pending_outputs.clear()

        # This method runs on ``loop``. Deliver pending owner-thread outputs
        # synchronously so later call_soon_threadsafe deliveries cannot pass them.
        for output in pending:
            self._deliver_output(output)

    def ensure_alive(self) -> "_AsyncInprocEngineCore":
        with self.lock:
            engine_core = self.engine_core
            cause = self.fatal_error
            unavailable = self.engine_dead or self.closing or engine_core is None
        if unavailable:
            raise self._dead_error(cause)
        assert engine_core is not None
        return engine_core

    def register_utility(
        self, call_id: int, future: asyncio.Future[Any]
    ) -> None:
        with self.lock:
            if self.engine_dead or self.closing or self.engine_core is None:
                raise self._dead_error(self.fatal_error)
            self.utility_results[call_id] = future

    def discard_utility(
        self, call_id: int, expected: asyncio.Future[Any] | None = None
    ) -> None:
        with self.lock:
            if expected is None or self.utility_results.get(call_id) is expected:
                self.utility_results.pop(call_id, None)

    def enqueue(self, request_type: EngineCoreRequestType, request: Any) -> None:
        # Hold the state lock across put_nowait so every accepted command is
        # ordered before the shutdown wakeup.
        with self.lock:
            if self.engine_dead or self.closing or self.engine_core is None:
                raise self._dead_error(self.fatal_error)
            self.engine_core.input_queue.put_nowait((request_type, request))

    def enqueue_abort(self, request_ids: list[str]) -> None:
        # Match EngineCoreProc: make aborts visible to an in-flight GPU step and
        # also preserve their FIFO order relative to add requests.
        with self.lock:
            if self.engine_dead or self.closing or self.engine_core is None:
                raise self._dead_error(self.fatal_error)
            self.engine_core.aborts_queue.put_nowait(request_ids)
            self.engine_core.input_queue.put_nowait(
                (EngineCoreRequestType.ABORT, request_ids)
            )

    def publish_output(self, outputs: EngineCoreOutputs) -> None:
        with self.lock:
            if self.stopped:
                return
            loop = self.loop
            if loop is None:
                self.pending_outputs.append(outputs)
                return

        if loop.is_closed():
            return
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(self._deliver_output, outputs)

    def _deliver_output(self, output: EngineCoreOutputs | Exception) -> None:
        if isinstance(output, Exception):
            with self.lock:
                self.terminal_output_delivered = True
            self.outputs_queue.put_nowait(output)
            self._fail_utility_waiters(output)
            return

        if output.utility_output is not None:
            call_id = output.utility_output.call_id
            with self.lock:
                future = self.utility_results.pop(call_id, None)
            if future is not None:
                _process_utility_output(output.utility_output, {call_id: future})
            return

        # Wave/control-only messages belong to DP clients, which this backend
        # rejects during configuration validation.
        if output.outputs or output.scheduler_stats:
            self.outputs_queue.put_nowait(output)

    def attach_engine(self, engine_core: "_AsyncInprocEngineCore") -> None:
        with self.lock:
            self.engine_core = engine_core
            should_stop = self.closing
        if should_stop:
            engine_core.shutdown_state = EngineShutdownState.REQUESTED
            engine_core.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))

    def mark_fatal(self, error: Exception) -> None:
        with self.lock:
            if self.fatal_error is None:
                self.fatal_error = _copy_exception_without_traceback(error)
            self.engine_dead = True
        self._publish_terminal_output()

    def _publish_terminal_output(self) -> None:
        """Publish one ordered terminal marker to the asyncio consumer."""
        with self.lock:
            if self.terminal_output_published:
                return
            self.terminal_output_published = True
            error = self._dead_error(self.fatal_error)
            loop = self.loop
            if loop is None:
                self.pending_outputs.append(error)
                return
        if not loop.is_closed():
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(self._deliver_output, error)
        else:
            with self.lock:
                self.terminal_output_delivered = True
            self._fail_utility_waiters(error)

    def _fail_utility_waiters(self, error: Exception) -> None:
        with self.lock:
            futures = tuple(self.utility_results.values())
            self.utility_results.clear()
        for future in futures:
            if not future.done():
                with contextlib.suppress(RuntimeError):
                    future.set_exception(error)

    def request_shutdown(self) -> None:
        with self.lock:
            if self.closing:
                return
            self.closing = True
            self.engine_dead = True
            engine_core = self.engine_core
        if engine_core is not None:
            engine_core.shutdown_state = EngineShutdownState.REQUESTED
            engine_core.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))

    def mark_stopped(self) -> None:
        with self.lock:
            self.stopped = True
            self.engine_core = None
        self.release_dist_ownership()
        # Wake an already-blocked output consumer on normal shutdown. Fatal
        # paths published the same marker earlier, making this a no-op.
        self._publish_terminal_output()

    def shutdown(
        self, timeout: float | None = None, *, raise_on_error: bool = True
    ) -> None:
        self.request_shutdown()
        thread = self.thread
        if thread is None:
            self.release_dist_ownership()
            return
        if thread is threading.current_thread():
            return
        if thread.ident is None:
            self.release_dist_ownership()
            return

        join_timeout = (
            envs.VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS
            if timeout is None
            else timeout
        )
        thread.join(timeout=join_timeout)
        if thread.is_alive():
            message = (
                "AsyncInprocClient owner thread did not stop within "
                f"{join_timeout}s; an in-process accelerator call may be stuck"
            )
            error = TimeoutError(message)
            self.mark_fatal(error)
            if raise_on_error:
                raise error
            logger.error(message)
            return

        # The owner normally releases this in mark_stopped. Keep an idempotent
        # fallback for fake/test owners and abnormal thread returns.
        self.release_dist_ownership()
        if self.teardown_error is not None and raise_on_error:
            raise self.teardown_error

    def __call__(self) -> None:
        """Best-effort finalizer entry point."""
        logger.debug_once(
            "[shutdown] AsyncInprocClient: background resource cleanup start"
        )
        try:
            self.shutdown(raise_on_error=False)
        except Exception:
            logger.exception(
                "[shutdown] AsyncInprocClient: background resource cleanup failed"
            )
        logger.debug_once(
            "[shutdown] AsyncInprocClient: background resource cleanup complete"
        )


class _InprocEngineOutputQueue:
    """Queue-shaped output adapter for the shared EngineCoreProc loop."""

    def __init__(self, resources: InprocBackgroundResources):
        self.resources = resources
        # Utility results cross the process boundary in AsyncMPClient, where
        # msgpack both validates the result and gives the frontend independent
        # ownership of mutable values. Preserve that contract in-process. A
        # utility Future may complete on a worker thread, so the stateful
        # encoder/decoder pair needs its own synchronization.
        self._utility_codec_lock = threading.Lock()
        self._utility_encoder = MsgpackEncoder()
        self._utility_decoder = MsgpackDecoder(EngineCoreOutputs, share_mem=False)

    def put_nowait(self, item: tuple[int, EngineCoreOutputs] | bytes) -> None:
        if isinstance(item, bytes):
            self.resources.mark_fatal(RuntimeError("EngineCore exited unexpectedly"))
            return
        client_index, output = item
        if client_index != 0:
            self.resources.mark_fatal(
                RuntimeError(
                    "AsyncInprocClient received output for unsupported "
                    f"client_index={client_index}"
                )
            )
            return
        if output.utility_output is not None:
            with self._utility_codec_lock:
                output = self._utility_decoder.decode(
                    self._utility_encoder.encode(output)
                )
        self.resources.publish_output(output)


class _AsyncInprocEngineCore(EngineCoreProc):
    """EngineCoreProc busy-loop semantics without process or ZMQ resources."""

    def __init__(
        self,
        resources: InprocBackgroundResources,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
    ) -> None:
        self.input_queue = queue.Queue[tuple[EngineCoreRequestType, Any]]()
        self.output_queue = _InprocEngineOutputQueue(
            resources
        )  # type: ignore[assignment]

        def executor_fail_callback() -> None:
            self.input_queue.put_nowait((EngineCoreRequestType.EXECUTOR_FAILED, b""))

        self.engine_index = 0
        self.engines_running = False
        self.shutdown_state = EngineShutdownState.RUNNING
        self.has_coordinator = False
        self.publish_dp_lb_stats = False
        self.last_counts = (0, 0)
        self.process_input_queue_block = True
        self.tensor_ipc_receiver = None
        self.enable_fault_tolerance = False

        # The owner thread replaces both the process boundary and ZMQ IO
        # threads, so initialize the shared EngineCore directly.
        EngineCore.__init__(
            self,
            vllm_config,
            executor_class,
            log_stats,
            executor_fail_callback,
            include_finished_set=False,
            inproc_engine=True,
        )

    def _handle_client_request(
        self, request_type: EngineCoreRequestType, request: Any
    ) -> None:
        if request_type == EngineCoreRequestType.ADD and isinstance(
            request, EngineCoreRequest
        ):
            raw_request = request
            try:
                request = self.preprocess_add_request(request)
            except Exception:
                self._handle_request_preproc_error(raw_request)
                return
        super()._handle_client_request(request_type, request)

    def _cleanup_compiled_model_hooks(self) -> None:
        """Drop global bytecode hooks before the executor releases its model."""
        model_executor = getattr(self, "model_executor", None)
        driver_worker = getattr(model_executor, "driver_worker", None)
        model_runner = getattr(driver_worker, "model_runner", None)
        model = getattr(model_runner, "model", None)
        if model is None:
            return

        from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

        for module in model.modules():
            if isinstance(module, TorchCompileWithNoGuardsWrapper):
                module.cleanup()

    def shutdown(self) -> None:
        hook_error: Exception | None = None
        try:
            self._cleanup_compiled_model_hooks()
        except Exception as error:
            logger.exception("AsyncInprocClient failed to clean compiled model hooks")
            hook_error = _copy_exception_without_traceback(error)

        super().shutdown()
        if hook_error is not None:
            raise hook_error


def _run_async_inproc_engine(
    resources: InprocBackgroundResources,
    vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool,
) -> None:
    """Construct, drive, and tear down EngineCore on its owner thread."""
    engine_core = _AsyncInprocEngineCore.__new__(_AsyncInprocEngineCore)
    # Keep partial-construction cleanup in owner-thread mode even if
    # EngineCore.__init__ fails before assigning this field itself.
    engine_core.inproc_engine = True
    initialization_started = False
    startup_complete = False
    primary_error: Exception | None = None

    def normalize_error(exc: BaseException, message: str) -> Exception:
        if isinstance(exc, Exception):
            return exc
        normalized = RuntimeError(message)
        normalized.__cause__ = exc
        return normalized

    try:
        resources.validate_dist_ownership()
        initialization_started = True
        engine_core.__init__(resources, vllm_config, executor_class, log_stats)
        resources.attach_engine(engine_core)
        resources.startup_future.set_result(None)
        startup_complete = True
        engine_core.run_busy_loop()
        busy_loop_error = RuntimeError(
            "AsyncInprocClient EngineCore busy loop returned unexpectedly"
        )
        primary_error = busy_loop_error
        resources.mark_fatal(busy_loop_error)
    except SystemExit as error:
        # EngineCoreProc.run_busy_loop uses SystemExit for its normal shutdown.
        if not startup_complete:
            startup_error = normalize_error(
                error, "AsyncInprocClient EngineCore exited during startup"
            )
            primary_error = startup_error
            resources.mark_fatal(startup_error)
        elif not resources.closing:
            unexpected_exit_error = RuntimeError(
                "AsyncInprocClient EngineCore exited unexpectedly"
            )
            primary_error = unexpected_exit_error
            resources.mark_fatal(unexpected_exit_error)
    except BaseException as error:
        fatal_error = normalize_error(
            error,
            "AsyncInprocClient EngineCore was interrupted by a fatal base exception",
        )
        primary_error = fatal_error
        if not startup_complete and not resources.startup_future.done():
            resources.startup_future.set_exception(
                _copy_exception_without_traceback(fatal_error)
            )
        resources.mark_fatal(fatal_error)
        logger.exception(
            "AsyncInprocClient EngineCore %s.",
            "failed to start" if not startup_complete else "encountered a fatal error",
        )
    finally:
        try:
            # A reservation-validation failure means this process group belongs
            # to the host, and no EngineCore resource exists to clean up.
            if initialization_started:
                engine_core.shutdown()
        except BaseException as error:
            cleanup_error = normalize_error(
                error, "AsyncInprocClient EngineCore teardown was interrupted"
            )
            if primary_error is None:
                primary_error = cleanup_error
                resources.teardown_error = _copy_exception_without_traceback(
                    cleanup_error
                )
                resources.mark_fatal(cleanup_error)
            logger.exception("AsyncInprocClient EngineCore teardown failed")
        finally:
            if not startup_complete and not resources.startup_future.done():
                resources.startup_future.set_exception(
                    _copy_exception_without_traceback(
                        primary_error
                        or RuntimeError(
                            "AsyncInprocClient EngineCore stopped before startup "
                            "completed"
                        )
                    )
                )
            resources.mark_stopped()


@dataclass
class ElasticScalingCache:
    existing_core_engines: list[EngineIdentity]
    num_new_core_engines: int
    pending_notifications: dict[EEPNotificationType, set[int]]
    completion_future: asyncio.Future[None] | None = None


class MPClient(EngineCoreClient):
    """
    MPClient: base client for multi-proc EngineCore.
        EngineCore runs in a background process busy loop, getting
        new EngineCoreRequests and returning EngineCoreOutputs

        * pushes EngineCoreRequests via input_socket
        * pulls EngineCoreOutputs via output_socket

        * AsyncMPClient subclass for AsyncLLM usage
        * SyncMPClient subclass for LLM usage
    """

    def __init__(
        self,
        asyncio_mode: bool,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        client_addresses: dict[str, Any] | None = None,
    ):
        self.vllm_config = vllm_config
        self._shutdown_lock = threading.RLock()
        self._shutdown_complete = False

        # ZMQ setup.
        sync_ctx = zmq.Context(io_threads=2)
        self.ctx = zmq.asyncio.Context(sync_ctx) if asyncio_mode else sync_ctx

        # This will ensure resources created so far are closed
        # when the client is garbage collected, even if an
        # exception is raised mid-construction.
        self.resources = BackgroundResources(ctx=sync_ctx)
        self._finalizer = weakref.finalize(self, self.resources)
        try:
            # State used for data parallel.
            self.engines_running = False
            parallel_config = vllm_config.parallel_config
            # Elastic EP can remove a rank and later add it back with the same
            # identity. The client input ROUTER needs handover to allow the new
            # engine to replace the dead connection.
            enable_input_socket_handover = parallel_config.enable_elastic_ep

            self.stats_update_address: str | None = None
            tensor_queue: Queue | None = None
            if client_addresses:
                # Engines are managed externally to this client.
                input_address = client_addresses["input_address"]
                output_address = client_addresses["output_address"]
                self.stats_update_address = client_addresses.get("stats_update_address")
                # Tensor queues passed via client_addresses for multi-API-server case
                tensor_queue = client_addresses.get("tensor_queue")
                self.input_socket = self.resources.input_socket = make_zmq_socket(
                    self.ctx,
                    input_address,
                    zmq.ROUTER,
                    bind=True,
                    router_handover=enable_input_socket_handover,
                )
                self.resources.output_socket = make_zmq_socket(
                    self.ctx, output_address, zmq.PULL
                )

                # Report bound endpoints back so the parent can forward
                # them to engines (mirrors the DPCoordinator pattern).
                actual_address_pipe: Connection | None = client_addresses.get(
                    "actual_address_pipe"
                )
                if actual_address_pipe is not None:
                    try:
                        actual_input = self.input_socket.getsockopt(
                            zmq.LAST_ENDPOINT
                        ).decode()
                        actual_output = self.resources.output_socket.getsockopt(
                            zmq.LAST_ENDPOINT
                        ).decode()
                        actual_address_pipe.send(
                            {
                                "input_address": actual_input,
                                "output_address": actual_output,
                            }
                        )
                    finally:
                        actual_address_pipe.close()
            else:
                # Engines are managed by this client.
                addresses = get_engine_zmq_addresses(vllm_config)
                self.input_socket = self.resources.input_socket = make_zmq_socket(
                    self.ctx,
                    addresses.inputs[0],
                    zmq.ROUTER,
                    bind=True,
                    router_handover=enable_input_socket_handover,
                )
                self.resources.output_socket = make_zmq_socket(
                    self.ctx, addresses.outputs[0], zmq.PULL
                )

                # Resolve ``tcp://host:0`` placeholders to bound endpoints
                # before engines DEALER-connect. No-op for IPC.
                addresses.inputs[0] = self.input_socket.getsockopt(
                    zmq.LAST_ENDPOINT
                ).decode()
                addresses.outputs[0] = self.resources.output_socket.getsockopt(
                    zmq.LAST_ENDPOINT
                ).decode()

                with launch_core_engines(
                    vllm_config, executor_class, log_stats, addresses
                ) as (engine_manager, coordinator, addresses, tensor_queue):
                    self.resources.coordinator = coordinator
                    self.resources.engine_manager = engine_manager

                self.stats_update_address = addresses.frontend_stats_publish_address
                if coordinator is not None:
                    assert self.stats_update_address == (
                        coordinator.get_stats_publish_address()
                    )

            # Serialization setup with tensor queues for multimodal tensor IPC.
            tensor_ipc_sender: TensorIpcSender | None = None
            model_config = getattr(vllm_config, "model_config", None)
            if model_config is not None and model_config.multimodal_config is not None:
                mm_tensor_ipc = model_config.multimodal_config.mm_tensor_ipc
                if mm_tensor_ipc == "torch_shm" and tensor_queue is not None:
                    tensor_ipc_sender = TensorIpcSender(tensor_queue)

            self.encoder = MsgpackEncoder(oob_tensor_consumer=tensor_ipc_sender)
            self.decoder = MsgpackDecoder(EngineCoreOutputs)

            dp_size = parallel_config.data_parallel_size
            dp_rank = parallel_config.data_parallel_index
            dp_local_size = parallel_config.data_parallel_size_local
            offline_mode = parallel_config.data_parallel_rank_local is not None
            # Client manages local+remote EngineCores in pure internal LB case.
            # Client manages local EngineCores in hybrid and external LB case.
            num_ranks = dp_local_size if parallel_config.local_engines_only else dp_size
            self.engine_ranks_managed = (
                [dp_rank] if offline_mode else list(range(dp_rank, dp_rank + num_ranks))
            )
            assert parallel_config.data_parallel_size_local <= len(
                self.engine_ranks_managed
            )

            # ZMQ identity of each engine that this client will talk to.
            self.core_engines: list[EngineIdentity] = [
                rank.to_bytes(2, "little") for rank in self.engine_ranks_managed
            ]

            # Wait for ready messages from each engine on the input socket.
            identities = set(self.core_engines)
            sync_input_socket = zmq.Socket.shadow(self.input_socket)
            while identities:
                if not sync_input_socket.poll(
                    timeout=VLLM_ENGINE_READY_TIMEOUT_S * 1000  # convert to ms
                ):
                    raise TimeoutError(
                        f"Timed out waiting for engine core processes to "
                        f"start. This is often caused by slow weight loading "
                        f"for large models. Waited "
                        f"{VLLM_ENGINE_READY_TIMEOUT_S}s (configured by "
                        f"VLLM_ENGINE_READY_TIMEOUT_S). To increase the "
                        f"timeout, set the environment variable: "
                        f"VLLM_ENGINE_READY_TIMEOUT_S=<seconds>"
                    )
                identity, payload = sync_input_socket.recv_multipart()
                identities.remove(identity)
                self._apply_ready_response(payload)

            self.core_engine: EngineIdentity = self.core_engines[0]
            self.utility_results: dict[int, AnyFuture] = {}

            # Start monitoring engine core processes for unexpected failures
            self.start_engine_core_monitor()
        except BaseException:
            # Preserve the construction failure as the primary exception. A
            # cleanup failure leaves the finalizer armed so GC can retry the
            # resources still tracked by BackgroundResources.
            try:
                self.resources()
            except BaseException:
                logger.exception(
                    "MPClient cleanup failed during partial construction; "
                    "the finalizer remains armed for retry"
                )
            else:
                self._finalizer.detach()
                self._shutdown_complete = True
            raise

    def shutdown(self, timeout: float | None = None) -> None:
        """Shutdown engine manager under timeout and clean up resources."""
        with self._shutdown_lock:
            if self._shutdown_complete:
                return
            if not self._finalizer.alive:
                self._shutdown_complete = True
                return

            timeout_str = "default" if timeout is None else f"{timeout}s"
            logger.info("[shutdown] MPClient: start timeout=%s", timeout_str)
            self.resources.engine_dead = True
            primary_error: BaseException | None = None
            if self.resources.engine_manager is not None:
                logger.info_once("[shutdown] MPClient: stopping engine manager")
                try:
                    self.resources.engine_manager.shutdown(timeout=timeout)
                except BaseException as error:
                    primary_error = error
                else:
                    # BackgroundResources must not tear down the same Ray
                    # resources a second time after successful direct cleanup.
                    self.resources.engine_manager = None
                    logger.info_once(
                        "[shutdown] MPClient: engine manager stopped"
                    )
            logger.info_once("[shutdown] MPClient: cleaning up background resources")
            try:
                self.resources()
            except BaseException as error:
                if primary_error is None:
                    primary_error = error
                else:
                    logger.error(
                        "Additional MPClient background cleanup failure: "
                        "%s: %s",
                        type(error).__name__,
                        error,
                    )
            logger.info_once("[shutdown] MPClient: complete")
            if primary_error is not None:
                raise primary_error
            # Keep the finalizer armed until every cleanup stage succeeds. If
            # shutdown raises, a later explicit call (or GC) can retry only
            # the resources that BackgroundResources still tracks.
            self._finalizer.detach()
            self._shutdown_complete = True

    def _format_exception(self, e: Exception) -> Exception:
        """If errored, use EngineDeadError so root cause is clear."""
        return (
            EngineDeadError(suppress_context=True) if self.resources.engine_dead else e
        )

    def ensure_alive(self):
        if self.resources.engine_dead:
            raise EngineDeadError()

    def dp_engines_running(self) -> bool:
        return self.engines_running

    def start_engine_core_monitor(self):
        """Start a monitor thread for engine core processes."""
        engine_manager = self.resources.engine_manager
        if engine_manager is None:
            # No engine processes to monitor
            return

        self_ref = weakref.ref(self)

        # Monitor engine core process liveness. If any die unexpectedly,
        # marks the engine as dead, and shuts down the client.
        def monitor_engine_cores():
            try:
                engine_manager.monitor_engine_liveness()
            except BaseException:
                # A manager can discover an engine failure and then fail while
                # killing a process/actor or placement group. Cleanup failure
                # must not prevent the frontend from publishing engine death.
                logger.exception(
                    "EngineCore liveness monitor failed while handling an exit"
                )
            _self = self_ref()
            if not _self or not _self._finalizer.alive:
                return
            _self.resources.engine_dead = True
            logger.warning_once(
                "[shutdown] MPClient: engine core exited unexpectedly; starting cleanup"
            )
            try:
                _self.shutdown()
            except BaseException:
                # MPClient and the owned resources retain their finalizers on
                # cleanup failure, so a later explicit call or GC can retry.
                logger.exception(
                    "MPClient cleanup failed after an EngineCore monitor exit"
                )
            # Note: For MPClient, we don't have a failure callback mechanism
            # like MultiprocExecutor, but we set engine_dead flag which will
            # cause subsequent operations to raise EngineDeadError

        Thread(
            target=monitor_engine_cores, daemon=True, name="MPClientEngineMonitor"
        ).start()

    def _apply_ready_response(self, payload: bytes) -> None:
        """Decode an EngineCoreReadyResponse and sync any post-initialization
        config changes (e.g. auto-fitted max_model_len) back to the frontend."""
        if not payload:
            return
        vllm_config = self.vllm_config
        response = msgspec.msgpack.decode(payload, type=EngineCoreReadyResponse)
        vllm_config.model_config.max_model_len = min(
            vllm_config.model_config.max_model_len, response.max_model_len
        )

        # Setup KV cache config with initialization state from
        # engine core process. Sum num_gpu_blocks from all engines in DP case.
        num_gpu_blocks = vllm_config.cache_config.num_gpu_blocks or 0
        num_gpu_blocks += response.num_gpu_blocks
        vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks

        # Sync block_size: may be enlarged by _align_hybrid_block_size in the
        # worker for hybrid Mamba models.
        cache_config = vllm_config.cache_config
        cache_config.block_size = response.block_size
        # Keep these as per-engine cache_config_info values; do not sum across DP.
        cache_config.kv_cache_size_tokens = (
            getattr(cache_config, "kv_cache_size_tokens", None)
            if getattr(cache_config, "kv_cache_size_tokens", None) is not None
            else response.kv_cache_size_tokens
        )
        cache_config.kv_cache_max_concurrency = (
            getattr(cache_config, "kv_cache_max_concurrency", None)
            if getattr(cache_config, "kv_cache_max_concurrency", None) is not None
            else response.kv_cache_max_concurrency
        )

        # In external DP LB mode, the coordinator address that the
        # front-end procs connect to is obtained by each engine via it's
        # initial handshake with the rank 0 front-end.
        if response.dp_stats_address is not None:
            if self.stats_update_address is None:
                self.stats_update_address = response.dp_stats_address
            else:
                assert response.dp_stats_address == self.stats_update_address


def _process_utility_output(
    output: UtilityOutput, utility_results: dict[int, AnyFuture]
):
    """Set the result from a utility method in the waiting future."""
    future = utility_results.pop(output.call_id, None)
    if future is None:
        # The caller may have been cancelled after admission or the output
        # channel may already have failed every waiter. A late response must
        # not terminate the shared output-processing loop.
        return
    failure_message = output.failure_message
    try:
        if failure_message is not None:
            future.set_exception(Exception(failure_message))
        else:
            assert output.result is not None
            future.set_result(output.result.result)
    except asyncio.InvalidStateError:
        # This can happen if the future is cancelled due to the
        # original calling task being cancelled.
        if failure_message is not None:
            logger.error(
                "Cancelled call to utility method failed with error: %s",
                failure_message,
            )


def _fail_utility_results(
    utility_results: dict[int, AnyFuture], error: Exception
) -> None:
    """Fail and detach every utility waiter after an output-channel fatality."""
    futures = tuple(utility_results.values())
    utility_results.clear()
    for future in futures:
        if future.done():
            continue
        detached = _copy_exception_without_traceback(error)
        with contextlib.suppress(asyncio.InvalidStateError, FutureInvalidStateError):
            future.set_exception(detached)


class _AsyncClientUtilityMixin:
    """Utility proxies for the owner-thread async client."""

    resources: Any

    async def call_utility_async(self, method: str, *args) -> Any:
        raise NotImplementedError

    async def get_supported_tasks_async(self) -> tuple[SupportedTask, ...]:
        return await self.call_utility_async("get_supported_tasks")

    async def pause_scheduler_async(
        self, mode: PauseMode = "abort", clear_cache: bool = True
    ) -> None:
        await self.call_utility_async("pause_scheduler", mode, clear_cache)

    async def resume_scheduler_async(self) -> None:
        await self.call_utility_async("resume_scheduler")

    async def is_scheduler_paused_async(self) -> bool:
        return await self.call_utility_async("is_scheduler_paused")

    async def profile_async(
        self, is_start: bool = True, profile_prefix: str | None = None
    ) -> None:
        await self.call_utility_async("profile", is_start, profile_prefix)

    async def reset_mm_cache_async(self) -> None:
        await self.call_utility_async("reset_mm_cache")

    async def reset_prefix_cache_async(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return await self.call_utility_async(
            "reset_prefix_cache", reset_running_requests, reset_connector
        )

    async def reset_encoder_cache_async(self) -> None:
        await self.call_utility_async("reset_encoder_cache")

    async def sleep_async(self, level: int = 1, mode: PauseMode = "abort") -> None:
        await self.call_utility_async("sleep", level, mode)

    async def wake_up_async(self, tags: list[str] | None = None) -> None:
        await self.call_utility_async("wake_up", tags)

    async def is_sleeping_async(self) -> bool:
        return await self.call_utility_async("is_sleeping")

    async def execute_dummy_batch_async(self) -> None:
        await self.call_utility_async("execute_dummy_batch")

    async def set_weight_version_async(self, weight_version: str) -> None:
        await self.call_utility_async("set_weight_version", weight_version)

    async def get_weight_version_async(self) -> str:
        return await self.call_utility_async("get_weight_version")

    async def add_lora_async(self, lora_request: LoRARequest) -> bool:
        return await self.call_utility_async("add_lora", lora_request)

    async def remove_lora_async(self, lora_id: int) -> bool:
        return await self.call_utility_async("remove_lora", lora_id)

    async def list_loras_async(self) -> set[int]:
        return await self.call_utility_async("list_loras")

    async def pin_lora_async(self, lora_id: int) -> bool:
        return await self.call_utility_async("pin_lora", lora_id)

    async def save_sharded_state_async(
        self, path: str, pattern: str | None = None, max_size: int | None = None
    ) -> None:
        await self.call_utility_async("save_sharded_state", path, pattern, max_size)

    async def collective_rpc_async(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        return await self.call_utility_async(
            "collective_rpc", method, timeout, args, kwargs
        )


class AsyncInprocClient(_AsyncClientUtilityMixin, EngineCoreClient):
    """Asyncio client whose EngineCore is owned by a fixed local thread."""

    resources: InprocBackgroundResources

    @instrument(span_name="AsyncInprocClient init")
    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        client_addresses: dict[str, Any] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ) -> None:
        self._validate_config(
            vllm_config,
            executor_class,
            client_addresses,
            client_count,
            client_index,
        )
        self.vllm_config = vllm_config
        self.client_count = client_count
        self.client_index = client_index
        self.engine_ranks_managed = [0]
        self.core_engines = [int(0).to_bytes(2, "little")]
        self.core_engine = self.core_engines[0]
        self.engines_running = False
        # Preserve the ownership boundary normally provided by msgpack/ZMQ.
        # The owner thread must never receive frontend-owned mutable request
        # containers by reference (streaming updates mutate prompt state).
        self._request_encoder = MsgpackEncoder()
        self._request_decoder = MsgpackDecoder(EngineCoreRequest, share_mem=False)
        self._utility_decoder = MsgpackDecoder(share_mem=False)

        resources = InprocBackgroundResources()
        resources.dist_ownership_token = _reserve_inproc_distributed_ownership()
        try:
            self.resources = resources
            self.outputs_queue = resources.outputs_queue
            self.utility_results = resources.utility_results
            self._finalizer = weakref.finalize(self, resources)
            thread = Thread(
                target=_run_async_inproc_engine,
                args=(resources, vllm_config, executor_class, log_stats),
                daemon=True,
                name="AsyncInprocEngineCore",
            )
        except BaseException:
            resources.release_dist_ownership()
            finalizer = getattr(self, "_finalizer", None)
            if finalizer is not None and finalizer.alive:
                finalizer.detach()
            raise
        resources.thread = thread
        try:
            thread.start()
        except BaseException:
            resources.release_dist_ownership()
            if self._finalizer.alive:
                self._finalizer.detach()
            raise

        try:
            resources.startup_future.result(timeout=VLLM_ENGINE_READY_TIMEOUT_S)
        except TimeoutError as error:
            self._cleanup_failed_startup()
            raise TimeoutError(
                "Timed out waiting for AsyncInprocClient EngineCore thread to "
                f"start after {VLLM_ENGINE_READY_TIMEOUT_S}s"
            ) from error
        except Exception:
            self._cleanup_failed_startup()
            raise

    def _cleanup_failed_startup(self) -> None:
        """Preserve startup failure while retaining teardown retryability."""
        try:
            self.resources.shutdown()
        except BaseException:
            logger.exception(
                "AsyncInprocClient cleanup failed during startup; the "
                "finalizer remains armed for retry"
            )
        else:
            if self._finalizer.alive:
                self._finalizer.detach()

    @staticmethod
    def _validate_config(
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        client_addresses: dict[str, Any] | None,
        client_count: int,
        client_index: int,
    ) -> None:
        unsupported: list[str] = []
        if client_addresses is not None:
            unsupported.append("client_addresses")
        if client_count != 1:
            unsupported.append(f"client_count={client_count}")
        if client_index != 0:
            unsupported.append(f"client_index={client_index}")

        parallel_config = vllm_config.parallel_config
        scalar_defaults = {
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
        }
        for name, expected in scalar_defaults.items():
            value = getattr(parallel_config, name, expected)
            if value != expected:
                unsupported.append(f"{name}={value}")

        boolean_fields = (
            "data_parallel_external_lb",
            "data_parallel_hybrid_lb",
            "enable_elastic_ep",
            "enable_fault_tolerance",
        )
        for name in boolean_fields:
            if getattr(parallel_config, name, False):
                unsupported.append(f"{name}=True")

        if executor_class is not UniProcExecutor:
            name = getattr(executor_class, "__name__", repr(executor_class))
            unsupported.append(f"executor_class={name}")

        if unsupported:
            details = ", ".join(unsupported)
            raise ValueError(
                "AsyncInprocClient only supports one local client and a "
                "single-rank UniProcExecutor; unsupported: " + details
            )

    def _bind_loop(self) -> None:
        self.resources.bind_loop(asyncio.get_running_loop())

    def ensure_alive(self) -> None:
        self.resources.ensure_alive()

    def _format_exception(self, error: Exception) -> Exception:
        if self.resources.engine_dead:
            return self.resources._dead_error(self.resources.fatal_error)
        return error

    def shutdown(self, timeout: float | None = None) -> None:
        timeout_str = "default" if timeout is None else f"{timeout}s"
        logger.info("[shutdown] AsyncInprocClient: start timeout=%s", timeout_str)
        self.resources.shutdown(timeout=timeout)
        if self._finalizer.alive:
            self._finalizer.detach()
        logger.info_once("[shutdown] AsyncInprocClient: complete")

    async def get_output_async(self) -> EngineCoreOutputs:
        self._bind_loop()
        if self.resources.terminal_output_delivered and self.outputs_queue.empty():
            raise self.resources._dead_error(self.resources.fatal_error)
        output = await self.outputs_queue.get()
        if isinstance(output, Exception):
            raise self._format_exception(output) from None
        return output

    async def call_utility_async(self, method: str, *args) -> Any:
        self._bind_loop()
        self.ensure_alive()
        call_id = uuid.uuid1().int >> 64
        future = asyncio.get_running_loop().create_future()
        self.resources.register_utility(call_id, future)
        try:
            owner_payload = self._utility_decoder.decode(
                self._request_encoder.encode(
                    (self.client_index, call_id, method, args)
                )
            )
            self.resources.enqueue(
                EngineCoreRequestType.UTILITY,
                owner_payload,
            )
            return await future
        except BaseException:
            self.resources.discard_utility(call_id, future)
            if not future.done():
                future.cancel()
            raise

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        self._bind_loop()
        request.client_index = self.client_index
        owner_request = self._request_decoder.decode(
            self._request_encoder.encode(request)
        )
        self.resources.enqueue(EngineCoreRequestType.ADD, owner_request)

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        self._bind_loop()
        if not request_ids or self.resources.engine_dead:
            return
        try:
            self.resources.enqueue_abort(list(request_ids))
        except EngineDeadError:
            # Match AsyncMPClient's best-effort abort during teardown.
            if not self.resources.engine_dead:
                raise

    def dp_engines_running(self) -> bool:
        return False

    async def commit_elastic_ep(self) -> None:
        raise ValueError("AsyncInprocClient does not support elastic EP scaling")

    async def prepare_elastic_ep(self, new_data_parallel_size: int) -> None:
        raise ValueError("AsyncInprocClient does not support elastic EP scaling")

    async def handle_fault(
        self, fault_tolerance_request: FaultToleranceRequest
    ) -> FaultToleranceResult:
        raise ValueError("AsyncInprocClient does not support fault tolerance")

    async def get_status(self):
        status = "unhealthy" if self.resources.engine_dead else "healthy"
        return {
            "schema_version": 1,
            "total_engines": 1,
            "engines": [{"id": 0, "status": status}],
        }


class SyncMPClient(MPClient):
    """Synchronous client for multi-proc EngineCore."""

    @instrument(span_name="SyncMPClient init")
    def __init__(
        self, vllm_config: VllmConfig, executor_class: type[Executor], log_stats: bool
    ):
        super().__init__(
            asyncio_mode=False,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=log_stats,
        )

        self.is_dp = self.vllm_config.parallel_config.data_parallel_size > 1
        self.outputs_queue = queue.Queue[EngineCoreOutputs | Exception]()

        # Ensure that the outputs socket processing thread does not have
        # a ref to the client which prevents gc.
        ctx = self.ctx
        out_socket = self.resources.output_socket
        decoder = self.decoder
        utility_results = self.utility_results
        outputs_queue = self.outputs_queue

        shutdown_path = get_open_zmq_inproc_path()
        resources = self.resources
        resources.shutdown_path = shutdown_path
        output_ready = resources.sync_output_ready = threading.Event()
        output_stopped = resources.sync_output_stopped = threading.Event()

        def process_outputs_socket():
            shutdown_socket: zmq.Socket | None = None
            try:
                assert isinstance(out_socket, zmq.Socket)
                shutdown_socket = ctx.socket(zmq.PAIR)
                shutdown_socket.bind(shutdown_path)
                output_ready.set()
                poller = zmq.Poller()
                poller.register(shutdown_socket, zmq.POLLIN)
                poller.register(out_socket, zmq.POLLIN)
                while True:
                    socks = poller.poll()
                    if not socks:
                        continue
                    if len(socks) == 2 or socks[0][0] == shutdown_socket:
                        # shutdown signal, exit thread.
                        break

                    frames = out_socket.recv_multipart(copy=False)
                    resources.validate_alive(frames)
                    outputs: EngineCoreOutputs = decoder.decode(frames)
                    if outputs.utility_output:
                        _process_utility_output(outputs.utility_output, utility_results)
                    else:
                        outputs_queue.put_nowait(outputs)
            except Exception as e:
                # Also release a shutdown waiter when bind/setup itself fails.
                output_ready.set()
                resources.engine_dead = True
                _fail_utility_results(utility_results, e)
                outputs_queue.put_nowait(e)
            finally:
                output_ready.set()
                resources.engine_dead = True
                _fail_utility_results(utility_results, EngineDeadError())
                # Close sockets.
                try:
                    if shutdown_socket is not None:
                        shutdown_socket.close(linger=0)
                finally:
                    try:
                        out_socket.close(linger=0)
                    finally:
                        output_stopped.set()

        # Process outputs from engine in separate thread.
        self.output_queue_thread = Thread(
            target=process_outputs_socket,
            name="EngineCoreOutputQueueThread",
            daemon=True,
        )
        try:
            self.output_queue_thread.start()
        except BaseException:
            output_ready.set()
            output_stopped.set()
            try:
                self.resources()
            except BaseException:
                logger.exception(
                    "SyncMPClient cleanup failed during output-thread "
                    "startup; the finalizer remains armed for retry"
                )
            else:
                self._finalizer.detach()
                self._shutdown_complete = True
            raise

        # The thread takes on responsibility for closing the socket.
        self.resources.output_socket = None

    def get_output(self) -> EngineCoreOutputs:
        # If an exception arises in process_outputs_socket task,
        # it is forwarded to the outputs_queue so we can raise it
        # from this (run_output_handler) task to shut down the server.
        outputs = self.outputs_queue.get()

        if isinstance(outputs, Exception):
            raise self._format_exception(outputs) from None
        if outputs.wave_complete is not None:
            self.engines_running = False
        return outputs

    def _send_input(self, request_type: EngineCoreRequestType, request: Any):
        self.ensure_alive()
        # (Identity, RequestType, SerializedRequest)
        msg = (self.core_engine, request_type.value, *self.encoder.encode(request))
        # Any zero-copy tensor/ndarray frames are kept alive by zmq itself
        # until it's finished sending them (there is a ref chain from the underlying
        # memoryview back to the original owning tensor/ndarray).
        self.input_socket.send_multipart(msg, copy=False)

    def call_utility(self, method: str, *args) -> Any:
        call_id = uuid.uuid1().int >> 64
        future: Future[Any] = Future()
        self.utility_results[call_id] = future
        try:
            self._send_input(EngineCoreRequestType.UTILITY, (0, call_id, method, args))
        except BaseException:
            self.utility_results.pop(call_id, None)
            future.cancel()
            raise

        return future.result()

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.call_utility("get_supported_tasks")

    def add_request(self, request: EngineCoreRequest) -> None:
        if self.is_dp:
            self.engines_running = True
        self._send_input(EngineCoreRequestType.ADD, request)

    def abort_requests(self, request_ids: list[str]) -> None:
        if request_ids and not self.resources.engine_dead:
            self._send_input(EngineCoreRequestType.ABORT, request_ids)

    def profile(self, is_start: bool = True, profile_prefix: str | None = None) -> None:
        self.call_utility("profile", is_start, profile_prefix)

    def reset_mm_cache(self) -> None:
        self.call_utility("reset_mm_cache")

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return self.call_utility(
            "reset_prefix_cache", reset_running_requests, reset_connector
        )

    def reset_encoder_cache(self) -> None:
        self.call_utility("reset_encoder_cache")

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.call_utility("add_lora", lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.call_utility("remove_lora", lora_id)

    def list_loras(self) -> set[int]:
        return self.call_utility("list_loras")

    def pin_lora(self, lora_id: int) -> bool:
        return self.call_utility("pin_lora", lora_id)

    def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None:
        self.call_utility("sleep", level, mode)

    def wake_up(self, tags: list[str] | None = None) -> None:
        self.call_utility("wake_up", tags)

    def is_sleeping(self) -> bool:
        return self.call_utility("is_sleeping")

    def execute_dummy_batch(self) -> None:
        self.call_utility("execute_dummy_batch")

    def set_weight_version(self, weight_version: str) -> None:
        self.call_utility("set_weight_version", weight_version)

    def get_weight_version(self) -> str:
        return self.call_utility("get_weight_version")

    def collective_rpc(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        return self.call_utility("collective_rpc", method, timeout, args, kwargs)

    def save_sharded_state(
        self, path: str, pattern: str | None = None, max_size: int | None = None
    ) -> None:
        self.call_utility("save_sharded_state", path, pattern, max_size)


class AsyncMPClient(MPClient):
    """Asyncio-compatible client for multi-proc EngineCore."""

    @instrument(span_name="AsyncMPClient init")
    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        client_addresses: dict[str, Any] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ):
        super().__init__(
            asyncio_mode=True,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=log_stats,
            client_addresses=client_addresses,
        )

        self.client_count = client_count
        self.client_index = client_index
        self.outputs_queue = asyncio.Queue[EngineCoreOutputs | Exception]()

        # locally-cached engine status
        self._engine_status: dict[int, dict] = {}
        if self.vllm_config.parallel_config.enable_fault_tolerance:
            self._engine_status = {
                rank: {"id": rank, "status": "healthy"}
                for rank in self.engine_ranks_managed
            }
        try:
            # If we are running in an asyncio event loop, start the queue task.
            # Otherwise, it will be started lazily. If it is not started here,
            # we could miss EXECUTOR_FAILED messages from engine core if they
            # occur prior to any requests being sent.
            asyncio.get_running_loop()
            self._ensure_output_queue_task()
        except RuntimeError:
            pass

    def _ensure_output_queue_task(self):
        resources = self.resources
        if resources.output_queue_task is not None:
            return

        # Perform IO in separate task to parallelize as much as possible.
        # Avoid task having direct reference back to the client.
        decoder = self.decoder
        utility_results = self.utility_results
        outputs_queue = self.outputs_queue
        output_handler: (
            Callable[[AsyncMPClient, EngineCoreOutputs], Awaitable[None]] | None
        ) = getattr(self.__class__, "process_engine_outputs", None)
        _self_ref = weakref.ref(self)
        output_socket = resources.output_socket
        assert output_socket is not None

        notification_callback_handler: (
            Callable[[AsyncMPClient, Sequence[Any]], Any] | None
        ) = getattr(self.__class__, "eep_process_engine_core_notification", None)

        async def process_outputs_socket():
            try:
                while True:
                    frames = await output_socket.recv_multipart(copy=False)
                    resources.validate_alive(frames)
                    outputs: EngineCoreOutputs = decoder.decode(frames)
                    if outputs.utility_output:
                        if (
                            outputs.utility_output.call_id == EEP_NOTIFICATION_CALL_ID
                            and notification_callback_handler is not None
                        ):
                            assert _self_ref is not None
                            _self = _self_ref()
                            if not _self:
                                return
                            if outputs.utility_output.result is None:
                                continue
                            notification_data = outputs.utility_output.result.result
                            assert isinstance(notification_data, Sequence)
                            assert len(notification_data) == 2
                            notification_task = asyncio.create_task(
                                notification_callback_handler(
                                    _self, notification_data
                                ),
                                name="ElasticEPNotification",
                            )
                            notification_tasks = getattr(
                                _self, "_elastic_ep_notification_tasks", None
                            )
                            if notification_tasks is None:
                                notification_tasks = set()
                                _self._elastic_ep_notification_tasks = (
                                    notification_tasks
                                )
                            notification_tasks.add(notification_task)

                            def finish_notification(
                                task: asyncio.Task[Any],
                                *,
                                tasks=notification_tasks,
                            ) -> None:
                                tasks.discard(task)
                                if task.cancelled():
                                    return
                                error = task.exception()
                                if error is None:
                                    return
                                resources.engine_dead = True
                                detached = (
                                    _copy_exception_without_traceback(error)
                                    if isinstance(error, Exception)
                                    else RuntimeError(str(error))
                                )
                                _fail_utility_results(
                                    utility_results,
                                    detached,
                                )
                                outputs_queue.put_nowait(detached)

                            notification_task.add_done_callback(
                                finish_notification
                            )
                        elif outputs.utility_output.call_id == FT_STATUS_CALL_ID:
                            _self = _self_ref()
                            if not _self:
                                return
                            if outputs.utility_output.result is not None:
                                _self._engine_status[outputs.engine_index] = (
                                    outputs.utility_output.result.result
                                )
                        else:
                            _process_utility_output(
                                outputs.utility_output, utility_results
                            )
                        continue

                    if output_handler is not None:
                        assert _self_ref is not None
                        _self = _self_ref()
                        if not _self:
                            # Client has been garbage collected, abort.
                            return
                        await output_handler(_self, outputs)

                    if outputs.outputs or outputs.scheduler_stats:
                        outputs_queue.put_nowait(outputs)
            except Exception as e:
                resources.engine_dead = True
                _fail_utility_results(utility_results, e)
                outputs_queue.put_nowait(e)
            except asyncio.CancelledError:
                error = EngineDeadError()
                resources.engine_dead = True
                _fail_utility_results(utility_results, error)
                outputs_queue.put_nowait(error)
            finally:
                resources.engine_dead = True
                _fail_utility_results(utility_results, EngineDeadError())

        resources.output_queue_task = asyncio.create_task(
            process_outputs_socket(), name="EngineCoreOutputQueueTask"
        )

    async def get_output_async(self) -> EngineCoreOutputs:
        self._ensure_output_queue_task()
        # If an exception arises in process_outputs_socket task,
        # it is forwarded to the outputs_queue so we can raise it
        # from this (run_output_handler) task to shut down the server.
        assert self.outputs_queue is not None
        outputs = await self.outputs_queue.get()
        if isinstance(outputs, Exception):
            raise self._format_exception(outputs) from None
        return outputs

    def _send_input(
        self,
        request_type: EngineCoreRequestType,
        request: Any,
        engine: EngineIdentity | None = None,
    ) -> Awaitable[Any]:
        if engine is None:
            engine = self.core_engine

        message = (request_type.value, *self.encoder.encode(request))
        return self._send_input_message(message, engine)

    def _send_input_message(
        self, message: tuple[bytestr, ...], engine: EngineIdentity
    ) -> Awaitable[Any]:
        self.ensure_alive()
        # Any zero-copy tensor/ndarray frames are kept alive by zmq itself
        # until it's finished sending them (there is a ref chain from the underlying
        # memoryview back to the original owning tensor/ndarray).
        return self.input_socket.send_multipart((engine,) + message, copy=False)

    async def call_utility_async(self, method: str, *args) -> Any:
        return await self._call_utility_async(method, *args, engine=self.core_engine)

    async def _admit_utility_async(
        self, method: str, *args, engine: EngineIdentity
    ) -> tuple[int, asyncio.Future[Any]]:
        """Register and send a utility call without awaiting its result."""
        call_id = uuid.uuid1().int >> 64
        future = asyncio.get_running_loop().create_future()
        self.utility_results[call_id] = future
        try:
            message = (
                EngineCoreRequestType.UTILITY.value,
                *self.encoder.encode((self.client_index, call_id, method, args)),
            )
            await self._send_input_message(message, engine)
            self._ensure_output_queue_task()
            return call_id, future
        except BaseException:
            if self.utility_results.get(call_id) is future:
                self.utility_results.pop(call_id, None)
            if not future.done():
                future.cancel()
            raise

    async def _call_utility_async(
        self, method: str, *args, engine: EngineIdentity
    ) -> Any:
        call_id, future = await self._admit_utility_async(
            method, *args, engine=engine
        )
        try:
            return await future
        except BaseException:
            if self.utility_results.get(call_id) is future:
                self.utility_results.pop(call_id, None)
            if not future.done():
                future.cancel()
            raise

    async def get_supported_tasks_async(self) -> tuple[SupportedTask, ...]:
        return await self.call_utility_async("get_supported_tasks")

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        request.client_index = self.client_index
        await self._send_input(EngineCoreRequestType.ADD, request)
        self._ensure_output_queue_task()

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        if request_ids and not self.resources.engine_dead:
            await self._send_input(EngineCoreRequestType.ABORT, request_ids)

    async def pause_scheduler_async(
        self, mode: PauseMode = "abort", clear_cache: bool = True
    ) -> None:
        await self.call_utility_async("pause_scheduler", mode, clear_cache)

    async def resume_scheduler_async(self) -> None:
        await self.call_utility_async("resume_scheduler")

    async def is_scheduler_paused_async(self) -> bool:
        return await self.call_utility_async("is_scheduler_paused")

    async def profile_async(
        self, is_start: bool = True, profile_prefix: str | None = None
    ) -> None:
        await self.call_utility_async("profile", is_start, profile_prefix)

    async def reset_mm_cache_async(self) -> None:
        await self.call_utility_async("reset_mm_cache")

    async def reset_prefix_cache_async(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return await self.call_utility_async(
            "reset_prefix_cache", reset_running_requests, reset_connector
        )

    async def reset_encoder_cache_async(self) -> None:
        await self.call_utility_async("reset_encoder_cache")

    async def sleep_async(self, level: int = 1, mode: PauseMode = "abort") -> None:
        await self.call_utility_async("sleep", level, mode)

    async def wake_up_async(self, tags: list[str] | None = None) -> None:
        await self.call_utility_async("wake_up", tags)

    async def is_sleeping_async(self) -> bool:
        return await self.call_utility_async("is_sleeping")

    async def execute_dummy_batch_async(self) -> None:
        await self.call_utility_async("execute_dummy_batch")

    async def set_weight_version_async(self, weight_version: str) -> None:
        await self.call_utility_async("set_weight_version", weight_version)

    async def get_weight_version_async(self) -> str:
        return await self.call_utility_async("get_weight_version")

    async def add_lora_async(self, lora_request: LoRARequest) -> bool:
        return await self.call_utility_async("add_lora", lora_request)

    async def remove_lora_async(self, lora_id: int) -> bool:
        return await self.call_utility_async("remove_lora", lora_id)

    async def list_loras_async(self) -> set[int]:
        return await self.call_utility_async("list_loras")

    async def pin_lora_async(self, lora_id: int) -> bool:
        return await self.call_utility_async("pin_lora", lora_id)

    async def save_sharded_state_async(
        self, path: str, pattern: str | None = None, max_size: int | None = None
    ) -> None:
        await self.call_utility_async("save_sharded_state", path, pattern, max_size)

    async def collective_rpc_async(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        return await self.call_utility_async(
            "collective_rpc", method, timeout, args, kwargs
        )

    async def handle_fault(
        self, ft_request: FaultToleranceRequest
    ) -> FaultToleranceResult:
        res = await self.call_utility_async(FT_UTILITY_METHOD, ft_request)
        result = msgspec.convert(res, FaultToleranceResult)
        if not result.success:
            status = self._engine_status.get(self.engine_ranks_managed[0])
            if status is not None:
                status["last_ft_request_id"] = result.request_id
                status["ft_error"] = result.reason
        return result

    async def get_status(self):
        return {
            "schema_version": 1,
            "total_engines": len(self.engine_ranks_managed),
            "engines": list(self._engine_status.values()),
        }


class DPAsyncMPClient(AsyncMPClient):
    """Asyncio-compatible client for multi-proc, multi-engine (data parallel)
    EngineCore. Assumes external load-balancing by default."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        client_addresses: dict[str, Any] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ):
        self.current_wave = 0

        super().__init__(
            vllm_config,
            executor_class,
            log_stats,
            client_addresses,
            client_count,
            client_index,
        )

        # List of [waiting, running, kv_cache_usage] per engine.
        # Used only by DPLBAsyncMPClient subclass.
        self.lb_engines: list[list[int | float]] = [
            [0, 0, 0.0] for _ in self.core_engines
        ]

        self.eep_scaling_cache: ElasticScalingCache | None = None

        self.first_req_sock_addr = get_open_zmq_inproc_path()
        self.first_req_send_socket = self.resources.first_req_send_socket = (
            make_zmq_socket(self.ctx, self.first_req_sock_addr, zmq.PAIR, bind=True)
        )
        try:
            # If we are running in an asyncio event loop, start the stats task.
            # Otherwise, it will be started lazily.
            asyncio.get_running_loop()
            self._ensure_stats_update_task()
        except RuntimeError:
            pass

    def _apply_engine_rank_topology(self, ranks: list[int]) -> None:
        """Keep load-balancing and fault-tolerance rank views consistent."""
        self.engine_ranks_managed = ranks
        if self.vllm_config.parallel_config.enable_fault_tolerance:
            previous = self._engine_status
            self._engine_status = {
                rank: previous.get(
                    rank, {"id": rank, "status": "healthy"}
                )
                for rank in ranks
            }

    def _apply_elastic_ep_local_topology(self, new_engine_count: int) -> None:
        """Synchronously update frontend views after an EEP group switch."""
        parallel_config = self.vllm_config.parallel_config
        dp_size = parallel_config.data_parallel_size
        dp_rank = parallel_config.data_parallel_rank
        assert dp_rank == 0
        assert dp_size == new_engine_count
        assert not (
            parallel_config.data_parallel_hybrid_lb
            or parallel_config.data_parallel_external_lb
        )
        self._apply_engine_rank_topology(
            list(range(dp_rank, dp_rank + new_engine_count))
        )
        if len(self.lb_engines) < new_engine_count:
            self.lb_engines.extend(
                [0, 0, 0.0]
                for _ in range(new_engine_count - len(self.lb_engines))
            )
        else:
            self.lb_engines = self.lb_engines[:new_engine_count]

    def _ensure_stats_update_task(self):
        resources = self.resources
        if resources.stats_update_task is not None:
            return

        assert self.stats_update_address is not None
        stats_addr: str = self.stats_update_address
        assert len(self.engine_ranks_managed) > 0

        async def run_engine_stats_update_task():
            with (
                make_zmq_socket(self.ctx, stats_addr, zmq.XSUB, linger=0) as socket,
                make_zmq_socket(
                    self.ctx, self.first_req_sock_addr, zmq.PAIR, bind=False, linger=0
                ) as first_req_rcv_socket,
            ):
                assert isinstance(socket, zmq.asyncio.Socket)
                assert isinstance(first_req_rcv_socket, zmq.asyncio.Socket)
                self.resources.stats_update_socket = socket
                self.resources.first_req_rcv_socket = first_req_rcv_socket
                # Send subscription message.
                await socket.send(b"\x01")

                poller = zmq.asyncio.Poller()
                poller.register(socket, zmq.POLLIN)
                poller.register(first_req_rcv_socket, zmq.POLLIN)

                while True:
                    events = await poller.poll()
                    if (
                        not self.engines_running
                        and len(events) == 2
                        or (events[0][0] == first_req_rcv_socket)
                    ):
                        # Check if this is a regular request notification or
                        # scale up notification
                        buf = first_req_rcv_socket.recv(flags=zmq.NOBLOCK).result()

                        decoded = msgspec.msgpack.decode(buf)
                        if (
                            isinstance(decoded, (list, tuple))
                            and len(decoded) == 2
                            and decoded[0] == "SCALE_ELASTIC_EP"
                        ):
                            # Extract new engine count from the decoded message.
                            # Commit already applies this view synchronously;
                            # marker handling is intentionally idempotent and
                            # also forwards the update to the coordinator.
                            new_engine_count = decoded[1]
                            self._apply_elastic_ep_local_topology(new_engine_count)
                            routing_limit = getattr(
                                self, "_elastic_ep_routing_limit", None
                            )
                            if (
                                routing_limit is not None
                                and new_engine_count <= routing_limit
                            ):
                                # The coordinator has applied the scale-down
                                # marker, so future snapshots already use the
                                # retained topology and no local cap is needed.
                                self._elastic_ep_routing_limit = None
                            # Send scale up notification to coordinator
                            scale_msg = msgspec.msgpack.encode(
                                ("SCALE_ELASTIC_EP", new_engine_count)
                            )
                            await socket.send(scale_msg)
                            continue

                        # we're sending a request while the engines are
                        # paused, so that it can wake the others up
                        # (to run dummy EP loop).
                        assert decoded[0] == "FIRST_REQ"
                        target_eng_index = decoded[1]
                        self.engines_running = True
                        msg = msgspec.msgpack.encode(
                            (target_eng_index, self.current_wave)
                        )
                        await socket.send(msg)

                    buf = None
                    while True:
                        # Drain all stats events (we only care about latest).
                        future: asyncio.Future[bytes] = socket.recv(flags=zmq.NOBLOCK)
                        if isinstance(future.exception(), zmq.Again):
                            break
                        buf = future.result()
                    if buf is None:
                        continue

                    # Update local load-balancing state.
                    counts, wave, running = msgspec.msgpack.decode(buf)
                    self.current_wave = wave
                    self.engines_running = running
                    if counts is not None:
                        # Running and waiting counts are global from the
                        # Coordinator including all EngineCores. Slice to get
                        # just the cores managed by this client.
                        sliced_counts, count_slice = (
                            self._slice_managed_engine_counts(counts)
                        )
                        self.lb_engines = sliced_counts
                        logger.debug(
                            "Received counts: %s (%s)", sliced_counts, count_slice
                        )

        resources.stats_update_task = asyncio.create_task(
            run_engine_stats_update_task()
        )

    def _slice_managed_engine_counts(
        self, counts: list[list[int | float]]
    ) -> tuple[list[list[int | float]], slice]:
        """Slice coordinator counts without crossing an EEP routing gate."""
        ranks = self.engine_ranks_managed
        count_slice = slice(ranks[0], ranks[-1] + 1)
        sliced_counts = counts[count_slice]
        routing_limit = getattr(self, "_elastic_ep_routing_limit", None)
        if routing_limit is not None:
            sliced_counts = sliced_counts[:routing_limit]
        return sliced_counts, count_slice

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        self._ensure_stats_update_task()

        request.current_wave = self.current_wave
        request.client_index = self.client_index

        chosen_engine = self.get_core_engine_for_request(request)
        to_await = self._send_input(EngineCoreRequestType.ADD, request, chosen_engine)
        if not self.engines_running:
            # Notify coordinator that we're sending a request
            req_msg = msgspec.msgpack.encode(("FIRST_REQ", chosen_engine))
            await self.first_req_send_socket.send(req_msg)

        await to_await

        self._ensure_output_queue_task()

    def get_core_engine_for_request(self, request: EngineCoreRequest):
        return self.core_engine


class DPLBAsyncMPClient(DPAsyncMPClient):
    """Asyncio-compatible client for multi-proc, multi-engine (data parallel)
    EngineCore. Load-balances between multiple engine processes."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        client_addresses: dict[str, Any] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ):
        self.client_count = client_count

        # To route aborts to the correct engine.
        self.reqs_in_flight: dict[str, EngineIdentity] = {}

        # Prefix utility calls must stay on the engine that owns the pin.
        # Include the internal request ID so an old, cancelled pin call cannot
        # erase a replacement that reused the same public pin ID.
        self.prefix_pins: dict[str, tuple[EngineIdentity, str]] = {}

        # Exact per-engine count of this client's unfinished requests.
        self.engine_inflight: Counter[EngineIdentity] = Counter()

        super().__init__(
            vllm_config,
            executor_class,
            log_stats,
            client_addresses,
            client_count,
            client_index,
        )

        assert len(self.core_engines) > 1
        self._prepared_elastic_ep: tuple[int, int] | None = None
        # A prepared elastic-EP reconfiguration owns the global lifecycle
        # controls until its commit finishes.  The commit itself is executed
        # under the same lock as public pause/reset/sleep/wake broadcasts, so
        # no lifecycle operation can interleave with the topology switch.
        self._elastic_ep_transaction_active = False
        self._elastic_ep_transaction_pending = False
        self._elastic_ep_transaction_failed = False
        self._elastic_ep_commit_in_progress = False
        self._elastic_ep_prepare_mutated = False
        self._elastic_ep_commit_mutated = False
        self._elastic_ep_prepare_tasks: set[asyncio.Task[None]] = set()
        self._elastic_ep_commit_tasks: set[asyncio.Task[None]] = set()
        self._elastic_ep_fail_stop_tasks: set[asyncio.Task[None]] = set()
        self._dp_lifecycle_broadcast_tasks: set[asyncio.Task[Any]] = set()
        self._dp_fault_recovery_tasks: set[asyncio.Task[Any]] = set()
        self._elastic_ep_notification_tasks: set[asyncio.Task[Any]] = set()
        self._elastic_ep_shutdown_requested = False
        # During scale-down, keep routing pinned to the retained prefix of
        # ``core_engines`` even if the coordinator publishes an old-group stats
        # snapshot while the topology marker is still in flight.
        self._elastic_ep_routing_limit: int | None = None

        self.eng_start_index = (
            len(self.core_engines) * self.client_index
        ) // client_count

    def shutdown(self, timeout: float | None = None) -> None:
        """Stop elastic-EP transactions before tearing down their engines."""
        with self._shutdown_lock:
            if self._shutdown_complete:
                return
            self._request_elastic_ep_shutdown()
            super().shutdown(timeout=timeout)

    def _request_elastic_ep_shutdown(self) -> None:
        """Cancel transaction tasks only for whole-client teardown.

        Public prepare/commit waiters are shielded so caller cancellation cannot
        interrupt a distributed mutation.  Client shutdown is different: every
        owned EngineCore is being destroyed, so the inner tasks must be made
        terminal as well.  Schedule cancellation on their owning event loop
        because shutdown may run in the fail-stop worker thread.
        """
        self._elastic_ep_shutdown_requested = True
        resources = getattr(self, "resources", None)
        if resources is not None:
            # Close admission before clearing the transaction guards. Every
            # send path checks this flag through ensure_alive(), including a
            # lifecycle broadcast that was already waiting for the lock.
            resources.engine_dead = True
        self._elastic_ep_transaction_pending = False
        self._elastic_ep_transaction_active = False
        self._elastic_ep_commit_in_progress = False
        self._prepared_elastic_ep = None

        task_sets = tuple(
            task_set
            for name in (
                "_elastic_ep_prepare_tasks",
                "_elastic_ep_commit_tasks",
                "_elastic_ep_fail_stop_tasks",
                "_dp_lifecycle_broadcast_tasks",
                "_dp_fault_recovery_tasks",
                "_elastic_ep_notification_tasks",
            )
            if (task_set := getattr(self, name, None)) is not None
        )
        tasks = tuple(task for task_set in task_sets for task in tuple(task_set))

        loop: asyncio.AbstractEventLoop | None = next(
            (
                task.get_loop()
                for task in tasks
                if not task.get_loop().is_closed()
                and task.get_loop().is_running()
            ),
            None,
        )
        if loop is None:
            output_task = getattr(
                getattr(self, "resources", None), "output_queue_task", None
            )
            if (
                output_task is not None
                and not output_task.get_loop().is_closed()
                and output_task.get_loop().is_running()
            ):
                loop = output_task.get_loop()

        utility_results = getattr(self, "utility_results", None)
        if loop is None and utility_results is not None:
            loop = next(
                (
                    future.get_loop()
                    for future in utility_results.values()
                    if isinstance(future, asyncio.Future)
                    and not future.get_loop().is_closed()
                    and future.get_loop().is_running()
                ),
                None,
            )

        cancellation_done = threading.Event()

        def cancel_transactions() -> None:
            try:
                # Re-read the sets on the loop so a fail-stop task admitted
                # just before the shutdown flag became visible is included.
                for task_set in task_sets:
                    for task in tuple(task_set):
                        if not task.done():
                            task.cancel()
                    task_set.clear()
                if utility_results is not None:
                    _fail_utility_results(utility_results, EngineDeadError())
            finally:
                cancellation_done.set()

        def detach_closed_loop_state() -> None:
            # Asyncio objects cannot be completed from a foreign thread once
            # their loop is closed. Drop our references and still complete
            # thread-safe concurrent futures used during partial startup.
            for task_set in task_sets:
                task_set.clear()
            if utility_results is None:
                return
            futures = tuple(utility_results.values())
            utility_results.clear()
            for future in futures:
                if isinstance(future, Future) and not future.done():
                    with contextlib.suppress(FutureInvalidStateError):
                        future.set_exception(EngineDeadError())

        if loop is None:
            detach_closed_loop_state()
            return
        if in_loop(loop):
            cancel_transactions()
        else:
            try:
                loop.call_soon_threadsafe(cancel_transactions)
            except RuntimeError:
                # The loop can close after the selection checks above. Never
                # let that race prevent the synchronous physical teardown.
                detach_closed_loop_state()
            else:
                deadline = (
                    time.monotonic()
                    + envs.VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS
                )
                while not cancellation_done.wait(
                    min(0.1, max(0.0, deadline - time.monotonic()))
                ):
                    if not loop.is_running() or time.monotonic() >= deadline:
                        # The callback may have been accepted just before the
                        # loop stopped. It is no longer safe to touch asyncio
                        # objects here; drop ownership and continue physical
                        # engine teardown instead of waiting forever.
                        detach_closed_loop_state()
                        break

    async def _wait_for_elastic_ep_ready_keys(
        self, ready_keys: Sequence[str]
    ) -> None:
        """Wait for readiness with bounded, shutdown-interruptible store calls."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + VLLM_ENGINE_READY_TIMEOUT_S
        last_timeout: BaseException | None = None
        keys = list(ready_keys)
        while True:
            if getattr(self, "_elastic_ep_shutdown_requested", False):
                raise asyncio.CancelledError
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError(
                    "Timed out waiting for elastic-EP EngineCores to become "
                    f"ready after {VLLM_ENGINE_READY_TIMEOUT_S}s"
                ) from last_timeout
            try:
                await asyncio.to_thread(
                    self._coord_store.wait,
                    keys,
                    timedelta(
                        seconds=min(_ELASTIC_EP_STORE_WAIT_SLICE_S, remaining)
                    ),
                )
                return
            except torch.distributed.DistStoreError as error:
                # TCPStore.wait reports an ordinary slice timeout this way.
                # Keep retrying until the global readiness deadline expires.
                last_timeout = error

    def get_core_engine_for_request(self, request: EngineCoreRequest) -> EngineIdentity:
        self.ensure_alive()
        # Engines are in rank order.
        routing_limit = getattr(self, "_elastic_ep_routing_limit", None)
        num_routable_engines = len(self.core_engines)
        if routing_limit is not None:
            num_routable_engines = min(num_routable_engines, routing_limit)
        if num_routable_engines <= 0:
            raise RuntimeError("No data-parallel engine is available for routing")

        eng_index = request.data_parallel_rank
        if eng_index is None:
            eng_index = get_late_interaction_engine_index(
                request.pooling_params, num_routable_engines
            )
        if eng_index is None:
            current_counts = self.lb_engines[:num_routable_engines]
            # TODO use P2C alg for larger DP sizes
            num_engines = len(current_counts)
            min_score: float = sys.maxsize
            eng_index = 0
            for i in range(num_engines):
                # Start from client_index to help with balancing when engines
                # are empty.
                idx = (self.eng_start_index + i) % num_engines
                waiting, running, kv_cache_usage = current_counts[idx]
                # Estimate engine load as the greater of the coordinator's
                # latest (waiting + running) snapshot and this client's own
                # in-flight count (scaled by the number of clients). The
                # in-flight floor is exact and can't be erased by a snapshot
                # rebind, so a burst spreads round-robin even when snapshots
                # race with routing decisions; the snapshot raises the score
                # when other clients or stale requests load the engine.
                inflight = self.engine_inflight[self.core_engines[idx]]
                score: float = max(self.client_count * inflight, waiting + running)
                if waiting:
                    # Waiting requests are penalized in proportion to KV cache
                    # pressure: a queue on a KV-bound engine drains slowly, so
                    # new requests should strongly prefer other engines. With
                    # low KV usage the queue is transient (e.g. mid-burst) and
                    # the penalty stays off, preserving exact round-robin.
                    # Ramps from 0 at <=50% usage to 3x waiting at 100%.
                    score += waiting * 6.0 * max(0.0, kv_cache_usage - 0.5)
                if score < min_score:
                    min_score = score
                    eng_index = idx
            # Increment local waiting count for better balancing between stats
            # updates from the coordinator (which happen every 100ms).
            current_counts[eng_index][0] += self.client_count
            # Rotate the scan start so that ties (equal scores, e.g. right
            # after a coordinator stats reset when engines look equally loaded)
            # don't systematically favor the same engine. This removes the
            # fixed tie-break bias without affecting load-aware decisions when
            # scores actually differ.
            self.eng_start_index = (self.eng_start_index + 1) % num_engines
        elif not 0 <= eng_index < num_routable_engines:
            raise ValueError(
                f"data_parallel_rank {eng_index} is not routable while "
                f"elastic EP exposes ranks [0, {num_routable_engines})"
            )

        chosen_engine = self.core_engines[eng_index]
        # Record which engine is chosen for this request, to handle aborts.
        self.reqs_in_flight[request.request_id] = chosen_engine
        self.engine_inflight[chosen_engine] += 1
        return chosen_engine

    async def _broadcast_dp_lifecycle_utility(
        self,
        method: str,
        args: tuple[Any, ...],
        engines: tuple[EngineIdentity, ...],
    ) -> Any:
        admissions = await asyncio.gather(
            *(
                self._admit_utility_async(method, *args, engine=engine)
                for engine in engines
            ),
            return_exceptions=True,
        )
        admission_errors = [
            result for result in admissions if isinstance(result, BaseException)
        ]
        if admission_errors:
            # At least one peer may already have published a collective
            # descriptor. Mark this frontend unusable and tear down everything
            # it owns. The fixed-cadence partial-descriptor timeout guarantees
            # externally-owned cores cannot remain in a permanent wave.
            error = RuntimeError(
                "Failed to admit a global DP lifecycle operation on every rank"
            )
            self.resources.engine_dead = True
            _fail_utility_results(self.utility_results, error)
            for result in admissions:
                if not isinstance(result, BaseException):
                    with contextlib.suppress(
                        asyncio.CancelledError, asyncio.InvalidStateError
                    ):
                        result[1].exception()
            try:
                await asyncio.to_thread(self.shutdown)
            except BaseException:
                logger.exception(
                    "Failed to tear down engines after partial DP lifecycle "
                    "broadcast admission"
                )
            raise error from admission_errors[0]

        futures = [
            result[1]
            for result in admissions
            if not isinstance(result, BaseException)
        ]
        results = await asyncio.gather(*futures, return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                raise result
        # Fault recovery must aggregate every rank: a structured failure is a
        # successful utility response and therefore would otherwise be hidden
        # by engine zero's result. Other DP utility methods retain the
        # historical engine-zero result contract.
        return results if method == FT_UTILITY_METHOD else results[0]

    def _get_dp_lifecycle_broadcast_lock(self) -> asyncio.Lock:
        lock = getattr(self, "_dp_lifecycle_broadcast_lock", None)
        if lock is None:
            lock = self._dp_lifecycle_broadcast_lock = asyncio.Lock()
        return lock

    def _get_elastic_ep_fault_lock(self) -> asyncio.Lock:
        lock = getattr(self, "_elastic_ep_fault_lock", None)
        if lock is None:
            lock = self._elastic_ep_fault_lock = asyncio.Lock()
        return lock

    def _validate_dp_lifecycle_owner(self) -> None:
        if self.client_count > 1:
            raise RuntimeError(
                "Global DP lifecycle controls are not supported with multiple "
                "API frontends; use a single lifecycle-owner frontend"
            )

    async def _run_serialized_dp_lifecycle_utility(
        self,
        method: str,
        args: tuple[Any, ...],
        engines: tuple[EngineIdentity, ...],
    ) -> Any:
        async with self._get_dp_lifecycle_broadcast_lock():
            self.ensure_alive()
            if (
                getattr(self, "_elastic_ep_transaction_pending", False)
                or getattr(self, "_elastic_ep_transaction_active", False)
                or getattr(self, "_elastic_ep_transaction_failed", False)
            ):
                raise RuntimeError(
                    "Global DP lifecycle controls are unavailable while an "
                    "elastic-EP reconfiguration is pending, active, or failed"
                )
            return await self._broadcast_dp_lifecycle_utility(
                method, args, engines
            )

    async def _call_dp_lifecycle_utility(
        self,
        method: str,
        args: tuple[Any, ...],
        *,
        engines: tuple[EngineIdentity, ...] | None = None,
    ) -> Any:
        self._validate_dp_lifecycle_owner()
        self.ensure_alive()
        if (
            getattr(self, "_elastic_ep_transaction_pending", False)
            or getattr(self, "_elastic_ep_transaction_active", False)
            or getattr(self, "_elastic_ep_transaction_failed", False)
        ):
            raise RuntimeError(
                "Global DP lifecycle controls are unavailable while an "
                "elastic-EP reconfiguration is pending, active, or failed"
            )
        if engines is None:
            engines = tuple(self.core_engines)

        # Caller cancellation must not interrupt rank admission after another
        # engine may already have registered its fixed-cadence descriptor. The
        # lock is acquired inside this shielded task so cancellation cannot
        # release it while the broadcast remains active.
        broadcast = asyncio.create_task(
            self._run_serialized_dp_lifecycle_utility(method, args, engines),
            name=f"DPLBGlobalLifecycle:{method}",
        )
        tasks = getattr(self, "_dp_lifecycle_broadcast_tasks", None)
        if tasks is None:
            tasks = self._dp_lifecycle_broadcast_tasks = set()
        tasks.add(broadcast)

        def finish_broadcast(task: asyncio.Task[Any]) -> None:
            tasks.discard(task)
            if not task.cancelled():
                # Retrieve background failures when the public waiter was
                # cancelled; a non-cancelled waiter will observe the same one.
                task.exception()

        broadcast.add_done_callback(finish_broadcast)
        return await asyncio.shield(broadcast)

    async def _run_serialized_dp_fault_utility(
        self,
        method: str,
        args: tuple[Any, ...],
        engines: tuple[EngineIdentity, ...],
    ) -> Any:
        # Do not use the ordinary lifecycle lock here. A lifecycle operation
        # can be waiting for a faulted EngineCore busy loop, while FT commands
        # are consumed by the EngineCore IO thread and are exactly what lets
        # that busy loop resume. This dedicated lock serializes FT recoveries
        # and makes their admission atomic with elastic-EP prepare.
        async with self._get_elastic_ep_fault_lock():
            self.ensure_alive()
            if (
                getattr(self, "_elastic_ep_transaction_active", False)
                or getattr(self, "_elastic_ep_transaction_failed", False)
            ):
                raise RuntimeError(
                    "Fault recovery is unavailable while an elastic-EP "
                    "reconfiguration is active or failed"
                )
            if method != FT_UTILITY_METHOD or len(args) != 1:
                raise RuntimeError("Invalid DP fault-recovery utility request")
            ft_request = msgspec.convert(args[0], FaultToleranceRequest)
            ranks = tuple(self.engine_ranks_managed)
            if len(ranks) != len(engines):
                failure = (
                    "DP fault recovery topology changed before rank admission"
                )
                self._mark_dp_fault_ranks_unhealthy(
                    ranks, ft_request.request_id, failure
                )
                raise RuntimeError(failure)
            raw_results = await self._broadcast_dp_lifecycle_utility(
                method, args, engines
            )
            return self._aggregate_dp_fault_results(
                ft_request, raw_results, ranks
            )

    def _mark_dp_fault_ranks_unhealthy(
        self,
        ranks: tuple[int, ...],
        request_id: str,
        failure: str,
    ) -> None:
        for rank in ranks:
            status = self._engine_status.setdefault(rank, {"id": rank})
            status["status"] = "unhealthy"
            status["last_ft_request_id"] = request_id
            status["ft_error"] = failure

    def _aggregate_dp_fault_results(
        self,
        ft_request: FaultToleranceRequest,
        raw_results: Any,
        ranks: tuple[int, ...],
    ) -> FaultToleranceResult:
        """Validate every rank and fail closed while holding the FT lock."""
        if not isinstance(raw_results, list):
            failure = "DP fault recovery did not return per-rank results"
            self._mark_dp_fault_ranks_unhealthy(
                ranks, ft_request.request_id, failure
            )
            raise RuntimeError(failure)
        try:
            results = [
                msgspec.convert(result, FaultToleranceResult)
                for result in raw_results
            ]
        except Exception as error:
            failure = "DP fault recovery returned an invalid per-rank result"
            self._mark_dp_fault_ranks_unhealthy(
                ranks, ft_request.request_id, failure
            )
            raise RuntimeError(failure) from error
        if len(results) != len(ranks):
            failure = (
                "DP fault recovery result count does not match the active "
                "engine topology"
            )
            self._mark_dp_fault_ranks_unhealthy(
                ranks, ft_request.request_id, failure
            )
            raise RuntimeError(failure)

        failures: list[str] = []
        for rank, result in zip(ranks, results, strict=True):
            if result.request_id != ft_request.request_id:
                failure = (
                    f"rank {rank}: mismatched request id {result.request_id!r}"
                )
            elif not result.success:
                failure = f"rank {rank}: {result.reason or 'recovery failed'}"
            else:
                continue

            failures.append(failure)
            status = self._engine_status.get(rank)
            if status is not None:
                # A failed or mismatched recovery cannot be treated as a
                # healthy participant in a subsequent elastic-EP group switch.
                status["status"] = "unhealthy"
                status["last_ft_request_id"] = result.request_id
                status["ft_error"] = failure

        return FaultToleranceResult(
            request_id=ft_request.request_id,
            success=not failures,
            reason="; ".join(failures) if failures else None,
        )

    async def _call_dp_fault_utility(
        self,
        method: str,
        args: tuple[Any, ...],
    ) -> Any:
        self._validate_dp_lifecycle_owner()
        self.ensure_alive()
        recovery = asyncio.create_task(
            self._run_serialized_dp_fault_utility(
                method, args, tuple(self.core_engines)
            ),
            name="DPLBFaultRecovery",
        )
        tasks = getattr(self, "_dp_fault_recovery_tasks", None)
        if tasks is None:
            tasks = self._dp_fault_recovery_tasks = set()
        tasks.add(recovery)

        def finish_recovery(task: asyncio.Task[Any]) -> None:
            tasks.discard(task)
            if not task.cancelled():
                task.exception()

        recovery.add_done_callback(finish_recovery)
        return await asyncio.shield(recovery)

    async def handle_fault(
        self, ft_request: FaultToleranceRequest
    ) -> FaultToleranceResult:
        result = await self.call_utility_async(FT_UTILITY_METHOD, ft_request)
        return msgspec.convert(
            result,
            FaultToleranceResult,
        )

    async def call_utility_async(self, method: str, *args) -> Any:
        if method == FT_UTILITY_METHOD:
            return await self._call_dp_fault_utility(method, args)
        if method not in {
            "reset_prefix_cache",
            "pause_scheduler",
            "resume_scheduler",
            "sleep",
            "wake_up",
        }:
            # Only the result from the first engine is returned.
            return (
                await asyncio.gather(
                    *[
                        self._call_utility_async(method, *args, engine=engine)
                        for engine in self.core_engines
                    ]
                )
            )[0]

        return await self._call_dp_lifecycle_utility(method, args)

    async def pin_prefix_async(
        self,
        pin_id: str,
        request: EngineCoreRequest,
        tier: PrefixPinTier = "gpu",
    ) -> PrefixPinResult:
        self.ensure_alive()
        if (
            getattr(self, "_elastic_ep_transaction_pending", False)
            or getattr(self, "_elastic_ep_transaction_active", False)
            or getattr(self, "_elastic_ep_transaction_failed", False)
        ):
            raise RuntimeError(
                "Prefix pins cannot be created while an elastic-EP "
                "reconfiguration is pending, active, or failed"
            )
        if pin_id in self.prefix_pins:
            raise ValueError(f"prefix pin {pin_id!r} already exists")

        request.client_index = self.client_index
        engine = self.get_core_engine_for_request(request)
        reservation = (engine, request.request_id)
        self.prefix_pins[pin_id] = reservation
        try:
            return await self._call_utility_async(
                "pin_prefix", pin_id, request, tier, engine=engine
            )
        except asyncio.CancelledError:
            # The utility is already admitted and continues in EngineCore.
            # Preserve its owner route so the public cancellation cleanup can
            # still release the eventual pin.
            raise
        except BaseException:
            if self.prefix_pins.get(pin_id) == reservation:
                self.prefix_pins.pop(pin_id, None)
            if self.reqs_in_flight.pop(request.request_id, None) == engine:
                self.engine_inflight[engine] -= 1
            raise

    async def unpin_prefix_async(
        self, pin_id: str, expected_request_id: str | None = None
    ) -> bool:
        reservation = self.prefix_pins.get(pin_id)
        if reservation is None:
            return False
        if (
            expected_request_id is not None
            and reservation[1] != expected_request_id
        ):
            return False

        unpin_call = asyncio.create_task(
            self._call_utility_async(
                "unpin_prefix",
                pin_id,
                expected_request_id,
                engine=reservation[0],
            )
        )

        def finish_unpin(task: asyncio.Task[Any]) -> None:
            if task.cancelled():
                return
            try:
                released = task.result()
            except BaseException:
                return
            if released and self.prefix_pins.get(pin_id) == reservation:
                self.prefix_pins.pop(pin_id, None)

        # The core utility remains admitted when its caller is cancelled. Keep
        # consuming its result so the frontend route converges with core state.
        unpin_call.add_done_callback(finish_unpin)
        released = await asyncio.shield(unpin_call)
        if released and self.prefix_pins.get(pin_id) == reservation:
            self.prefix_pins.pop(pin_id, None)
        return released

    async def pause_prefix_async(self, pin_id: str) -> None:
        if reservation := self.prefix_pins.get(pin_id):
            await self._call_utility_async(
                "pause_prefix", pin_id, engine=reservation[0]
            )

    async def resume_prefix_async(self, pin_id: str) -> None:
        if reservation := self.prefix_pins.get(pin_id):
            await self._call_utility_async(
                "resume_prefix", pin_id, engine=reservation[0]
            )

    @staticmethod
    async def process_engine_outputs(
        self: "DPLBAsyncMPClient", outputs: EngineCoreOutputs
    ):
        if outputs.finished_requests and self.reqs_in_flight:
            for req_id in outputs.finished_requests:
                if (engine := self.reqs_in_flight.pop(req_id, None)) is not None:
                    self.engine_inflight[engine] -= 1

    @staticmethod
    async def eep_process_engine_core_notification(
        self: "DPLBAsyncMPClient", notification_data: tuple[str, int]
    ):
        cache = self.eep_scaling_cache
        notification_type_str, dp_rank = notification_data
        try:
            notification_type = EEPNotificationType(notification_type_str)
        except ValueError as e:
            raise ValueError(
                f"Unknown EEP notification type: {notification_type_str}"
            ) from e

        if notification_type == EEPNotificationType.RECONFIGURE_FINISHED:
            from vllm.v1.engine import UtilityResult

            # NOTE(yongji): process a dummy UtilityOutput to resolve the future
            # awaited in _eep_wait_for_setup_switch_complete(), signaling that
            # all engine cores have completed reconfiguration.
            dummy_output = UtilityOutput(
                call_id=EEP_NOTIFICATION_CALL_ID, result=UtilityResult(None)
            )
            _process_utility_output(dummy_output, self.utility_results)
            return
        if cache is None:
            if self.resources.engine_dead:
                return
            raise RuntimeError(
                "Received an elastic-EP shutdown notification without an "
                "active scale-down transaction"
            )
        if notification_type not in cache.pending_notifications:
            cache.pending_notifications[notification_type] = set()
        if dp_rank in cache.pending_notifications[notification_type]:
            raise ValueError(
                f"Duplicate notification {notification_type} from dp_rank {dp_rank}"
            )
        cache.pending_notifications[notification_type].add(dp_rank)
        if len(cache.pending_notifications[notification_type]) >= abs(
            cache.num_new_core_engines
        ):
            engine_manager = self.resources.engine_manager
            assert isinstance(engine_manager, CoreEngineActorManager)
            assert cache.num_new_core_engines < 0
            old_dp_size = len(cache.existing_core_engines)
            new_dp_size = old_dp_size + cache.num_new_core_engines
            try:
                await asyncio.to_thread(
                    engine_manager.scale_down_elastic_ep,
                    old_dp_size,
                    new_dp_size,
                )
            except asyncio.CancelledError:
                raise
            except BaseException as error:
                self.eep_scaling_cache = None
                completion_future = cache.completion_future
                if completion_future is None:
                    raise
                if not completion_future.done():
                    detached = (
                        _copy_exception_without_traceback(error)
                        if isinstance(error, Exception)
                        else RuntimeError(str(error))
                    )
                    completion_future.set_exception(detached)
                return
            self.vllm_config.parallel_config.data_parallel_size_local = len(
                engine_manager.local_engine_actors
            )
            self.eep_scaling_cache = None
            completion_future = cache.completion_future
            if completion_future is not None and not completion_future.done():
                completion_future.set_result(None)

    async def _request_control_async(
        self, method: str, request_ids: list[str]
    ) -> None:
        """Route per-request controls only to each request's owning engine."""
        if self.resources.engine_dead:
            raise EngineDeadError()
        if not request_ids:
            return

        by_engine = defaultdict[EngineIdentity, list[str]](list)
        for request_id in request_ids:
            if engine := self.reqs_in_flight.get(request_id):
                by_engine[engine].append(request_id)

        await asyncio.gather(
            *(
                self._call_utility_async(method, ids, engine=engine)
                for engine, ids in by_engine.items()
            )
        )

    async def pause_requests_async(self, request_ids: list[str]) -> None:
        await self._request_control_async("pause_requests", request_ids)

    async def resume_requests_async(self, request_ids: list[str]) -> None:
        await self._request_control_async("resume_requests", request_ids)

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        if not request_ids or self.resources.engine_dead:
            return

        if len(request_ids) == 1:
            # Fast-path common case.
            if engine := self.reqs_in_flight.get(request_ids[0]):
                await self._abort_requests(request_ids, engine)
            return

        by_engine = defaultdict[EngineIdentity, list[str]](list)
        for req_id in request_ids:
            if engine := self.reqs_in_flight.get(req_id):
                by_engine[engine].append(req_id)
        for engine, req_ids in by_engine.items():
            await self._abort_requests(req_ids, engine)

    async def _abort_requests(
        self, request_ids: list[str], engine: EngineIdentity
    ) -> None:
        await self._send_input(EngineCoreRequestType.ABORT, request_ids, engine)

    def _mark_elastic_ep_transaction_failed(
        self, phase: str, _cause: BaseException
    ) -> RuntimeError:
        error = RuntimeError(
            f"Elastic EP {phase} failed after distributed state mutation; "
            "the EngineCore client cannot safely retry"
        )
        self._elastic_ep_transaction_failed = True
        self._elastic_ep_transaction_pending = False
        self._elastic_ep_transaction_active = False
        self._prepared_elastic_ep = None
        resources = getattr(self, "resources", None)
        if resources is not None:
            resources.engine_dead = True
        utility_results = getattr(self, "utility_results", None)
        if utility_results is not None:
            _fail_utility_results(utility_results, error)
        outputs_queue = getattr(self, "outputs_queue", None)
        if outputs_queue is not None:
            outputs_queue.put_nowait(_copy_exception_without_traceback(error))

        async def teardown() -> None:
            try:
                await asyncio.to_thread(self.shutdown)
            except asyncio.CancelledError:
                if getattr(self, "_elastic_ep_shutdown_requested", False):
                    return
                raise
            except BaseException:
                logger.exception(
                    "Failed to tear down engines after a fatal elastic-EP %s "
                    "failure",
                    phase,
                )

        if getattr(self, "_elastic_ep_shutdown_requested", False):
            return error

        teardown_task = asyncio.create_task(
            teardown(), name=f"DPLBElasticEPFailStop:{phase}"
        )
        tasks = getattr(self, "_elastic_ep_fail_stop_tasks", None)
        if tasks is None:
            tasks = self._elastic_ep_fail_stop_tasks = set()
        tasks.add(teardown_task)

        def finish_teardown(task: asyncio.Task[None]) -> None:
            tasks.discard(task)
            if not task.cancelled():
                task.exception()

        teardown_task.add_done_callback(finish_teardown)
        return error

    def _check_elastic_ep_scale_down_prefix_pins(
        self, new_data_parallel_size: int
    ) -> None:
        removed_engines = set(self.core_engines[new_data_parallel_size:])
        blocking_pins = sorted(
            pin_id
            for pin_id, (engine, _request_id) in getattr(
                self, "prefix_pins", {}
            ).items()
            if engine in removed_engines
        )
        if blocking_pins:
            raise RuntimeError(
                "Cannot scale down elastic EP while prefix pins are owned by "
                f"removed ranks; unpin them first: {blocking_pins}"
            )

    def _elastic_ep_num_redundant_experts(
        self, new_data_parallel_size: int
    ) -> int:
        from vllm.distributed.eplb.eplb_state import MAX_EXPERT_REDUNDANCY

        current_data_parallel_size = len(self.core_engines)
        num_experts = self.vllm_config.model_config.get_num_experts()
        current_redundant = (
            self.vllm_config.parallel_config.eplb_config.num_redundant_experts
        )
        new_redundant = (
            (num_experts + current_redundant)
            * new_data_parallel_size
            // current_data_parallel_size
            - num_experts
        )
        if new_redundant < 0:
            raise ValueError(
                "Elastic EP cannot scale to "
                f"data_parallel_size={new_data_parallel_size}: the target "
                "topology has fewer physical expert slots than logical experts"
            )
        if new_redundant > MAX_EXPERT_REDUNDANCY:
            raise ValueError(
                "Elastic EP cannot scale to "
                f"data_parallel_size={new_data_parallel_size}: "
                f"num_redundant_experts={new_redundant} exceeds the supported "
                f"maximum of {MAX_EXPERT_REDUNDANCY}"
            )

        parallel_config = self.vllm_config.parallel_config
        if getattr(parallel_config, "all2all_backend", None) == "nixl_ep":
            target_ep_size = (
                getattr(parallel_config, "world_size", 1)
                * new_data_parallel_size
            )
            if target_ep_size > envs.VLLM_NIXL_EP_MAX_NUM_RANKS:
                raise ValueError(
                    "Elastic EP target expert-parallel size "
                    f"{target_ep_size} exceeds VLLM_NIXL_EP_MAX_NUM_RANKS="
                    f"{envs.VLLM_NIXL_EP_MAX_NUM_RANKS}"
                )
        return new_redundant

    async def commit_elastic_ep(self) -> None:
        """Commit prepared elastic EP scaling."""
        if getattr(self, "_elastic_ep_transaction_failed", False):
            raise RuntimeError(
                "Elastic EP scaling cannot be retried after a failed "
                "distributed mutation"
            )
        self.ensure_alive()
        prepared = self._prepared_elastic_ep
        if prepared is None:
            raise RuntimeError("Elastic EP scaling has not been prepared")
        if getattr(self, "_elastic_ep_commit_in_progress", False):
            raise RuntimeError("Elastic EP scaling commit is already in progress")
        self._validate_dp_lifecycle_owner()

        new_data_parallel_size, num_redundant_experts = prepared
        self._elastic_ep_transaction_active = True
        self._elastic_ep_commit_in_progress = True
        commit_task = asyncio.create_task(
            self._run_elastic_ep_commit(
                new_data_parallel_size, num_redundant_experts
            ),
            name="DPLBElasticEPCommit",
        )
        tasks = getattr(self, "_elastic_ep_commit_tasks", None)
        if tasks is None:
            tasks = self._elastic_ep_commit_tasks = set()
        tasks.add(commit_task)

        def finish_commit(task: asyncio.Task[None]) -> None:
            tasks.discard(task)
            if not task.cancelled():
                # Retrieve a failure when the public waiter was cancelled.
                task.exception()

        commit_task.add_done_callback(finish_commit)
        # Once commit starts, caller cancellation must not release the
        # lifecycle lock or interrupt a partially switched topology.
        await asyncio.shield(commit_task)

    async def _run_elastic_ep_commit(
        self,
        new_data_parallel_size: int,
        num_redundant_experts: int,
    ) -> None:
        self._elastic_ep_commit_mutated = False
        try:
            async with self._get_dp_lifecycle_broadcast_lock():
                cur_data_parallel_size = len(self.core_engines)
                if new_data_parallel_size > cur_data_parallel_size:
                    await self._commit_scale_up_elastic_ep(new_data_parallel_size)
                else:
                    await self._commit_scale_down_elastic_ep(new_data_parallel_size)
                self.vllm_config.parallel_config.eplb_config.num_redundant_experts = (
                    num_redundant_experts
                )
                async with self._get_elastic_ep_fault_lock():
                    self._prepared_elastic_ep = None
                    self._elastic_ep_transaction_active = False
        except asyncio.CancelledError as error:
            if getattr(self, "_elastic_ep_shutdown_requested", False):
                self._prepared_elastic_ep = None
                self._elastic_ep_transaction_active = False
                raise
            resources = getattr(self, "resources", None)
            if self._elastic_ep_commit_mutated or (
                resources is not None
                and getattr(resources, "engine_dead", False)
            ):
                raise self._mark_elastic_ep_transaction_failed(
                    "commit", error
                ) from error
            # No topology mutation occurred. Preserve the prepared state so a
            # healthy caller can retry commit after an internal cancellation.
            raise
        except BaseException as error:
            resources = getattr(self, "resources", None)
            if self._elastic_ep_commit_mutated or (
                resources is not None
                and getattr(resources, "engine_dead", False)
            ):
                raise self._mark_elastic_ep_transaction_failed(
                    "commit", error
                ) from error
            # A failure before the first topology mutation is retryable.  Keep
            # the prepared transaction and lifecycle guard intact.
            raise
        finally:
            self._elastic_ep_commit_in_progress = False

    async def prepare_elastic_ep(self, new_data_parallel_size: int) -> None:
        """Prepare elastic EP scaling without routing requests to new engines."""
        if getattr(self, "_elastic_ep_transaction_failed", False):
            raise RuntimeError(
                "Elastic EP scaling cannot be prepared after a failed "
                "distributed mutation"
            )
        self.ensure_alive()
        if (
            isinstance(new_data_parallel_size, bool)
            or not isinstance(new_data_parallel_size, int)
            or not 0 < new_data_parallel_size <= _MAX_DP_ENGINE_COUNT
        ):
            raise ValueError(
                "new_data_parallel_size must be an integer in the range "
                f"[1, {_MAX_DP_ENGINE_COUNT}]"
            )
        if (prepared := self._prepared_elastic_ep) is not None:
            if prepared[0] == new_data_parallel_size:
                return
            raise RuntimeError("Elastic EP scaling is already prepared")
        if (
            getattr(self, "_elastic_ep_transaction_pending", False)
            or getattr(self, "_elastic_ep_transaction_active", False)
        ):
            raise RuntimeError("Elastic EP scaling is already in progress")
        if new_data_parallel_size == len(self.core_engines):
            return
        # Reject impossible scale-downs before publishing a transaction guard
        # or touching any distributed group.
        self._elastic_ep_num_redundant_experts(new_data_parallel_size)
        # Pending blocks new lifecycle operations, but deliberately does not
        # block fault recovery: an already-admitted lifecycle operation may
        # need FT to make its EngineCore busy loop responsive again.
        self._validate_dp_lifecycle_owner()
        self._elastic_ep_transaction_pending = True
        prepare_task = asyncio.create_task(
            self._run_elastic_ep_prepare(new_data_parallel_size),
            name="DPLBElasticEPPrepare",
        )
        tasks = getattr(self, "_elastic_ep_prepare_tasks", None)
        if tasks is None:
            tasks = self._elastic_ep_prepare_tasks = set()
        tasks.add(prepare_task)

        def finish_prepare(task: asyncio.Task[None]) -> None:
            tasks.discard(task)
            if not task.cancelled():
                task.exception()

        prepare_task.add_done_callback(finish_prepare)
        await asyncio.shield(prepare_task)

    async def _run_elastic_ep_prepare(
        self, new_data_parallel_size: int
    ) -> None:
        self._elastic_ep_prepare_mutated = False
        try:
            # Activate under the same lock used by public lifecycle controls.
            # Any already-admitted broadcast either finishes first or observes
            # the pending guard when it reaches the lock.
            async with self._get_dp_lifecycle_broadcast_lock():
                lifecycle_states = await asyncio.gather(
                    *(
                        self._call_utility_async(method, engine=engine)
                        for engine in tuple(self.core_engines)
                        for method in ("is_scheduler_paused", "is_sleeping")
                    )
                )
                if any(lifecycle_states):
                    raise RuntimeError(
                        "Elastic EP prepare requires every EngineCore to be "
                        "unpaused and fully awake; resume or wake the engines "
                        "before preparing a topology change"
                    )
                if new_data_parallel_size < len(self.core_engines):
                    self._check_elastic_ep_scale_down_prefix_pins(
                        new_data_parallel_size
                    )
                # The preflight is healthy and no topology mutation has begun.
                # Wait for any in-flight FT recovery, then atomically prevent
                # subsequent recoveries before replacing distributed groups.
                # Lock order is lifecycle -> fault; FT never takes lifecycle.
                async with self._get_elastic_ep_fault_lock():
                    self.ensure_alive()
                    parallel_config = self.vllm_config.parallel_config
                    if getattr(parallel_config, "enable_fault_tolerance", False):
                        unhealthy_ranks = [
                            rank
                            for rank in self.engine_ranks_managed
                            if self._engine_status.get(rank, {}).get("status")
                            != "healthy"
                        ]
                        if unhealthy_ranks:
                            raise RuntimeError(
                                "Elastic EP prepare requires every fault-"
                                "tolerant EngineCore to be healthy; unhealthy "
                                f"ranks: {unhealthy_ranks}"
                            )
                    self._elastic_ep_transaction_active = True
                    self._elastic_ep_transaction_pending = False

            cur_data_parallel_size = len(self.core_engines)
            assert self.vllm_config.parallel_config.data_parallel_backend == "ray", (
                "Only ray DP backend supports scaling elastic EP"
            )
            num_redundant_experts = self._elastic_ep_num_redundant_experts(
                new_data_parallel_size
            )
            if new_data_parallel_size < cur_data_parallel_size:
                await self._prepare_scale_down_elastic_ep(new_data_parallel_size)
            else:
                await self._prepare_scale_up_elastic_ep(
                    new_data_parallel_size, num_redundant_experts
                )
            self._prepared_elastic_ep = (
                new_data_parallel_size,
                num_redundant_experts,
            )
        except asyncio.CancelledError as error:
            if getattr(self, "_elastic_ep_shutdown_requested", False):
                self._prepared_elastic_ep = None
                self._elastic_ep_transaction_pending = False
                self._elastic_ep_transaction_active = False
                raise
            resources = getattr(self, "resources", None)
            if self._elastic_ep_prepare_mutated or (
                resources is not None
                and getattr(resources, "engine_dead", False)
            ):
                raise self._mark_elastic_ep_transaction_failed(
                    "prepare", error
                ) from error
            self._prepared_elastic_ep = None
            self._elastic_ep_transaction_pending = False
            self._elastic_ep_transaction_active = False
            raise
        except BaseException as error:
            resources = getattr(self, "resources", None)
            if self._elastic_ep_prepare_mutated or (
                resources is not None
                and getattr(resources, "engine_dead", False)
            ):
                raise self._mark_elastic_ep_transaction_failed(
                    "prepare", error
                ) from error
            self._elastic_ep_transaction_pending = False
            self._elastic_ep_transaction_active = False
            raise

    def _eep_wait_for_setup_switch_complete(self) -> asyncio.Future:
        """
        Wait for core engines to switch to the new setup.

        In eep_process_engine_core_notification(), a dummy UtilityOutput with
        EEP_NOTIFICATION_CALL_ID will be set when RECONFIGURE_FINISHED
        notification is received from engine 0. We create a future with
        that call_id and wait for it to be resolved.
        """
        future = asyncio.get_running_loop().create_future()
        self.utility_results[EEP_NOTIFICATION_CALL_ID] = future
        self._ensure_output_queue_task()
        return future

    async def _wait_for_new_engine_ready(
        self, new_core_engines: list[bytes]
    ) -> None:
        """Receive Ray actor handshakes without blocking the asyncio loop."""
        new_engine_identities = set(new_core_engines)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + VLLM_ENGINE_READY_TIMEOUT_S
        while new_engine_identities:
            self.ensure_alive()
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError(
                    "Timed out waiting for new engine core processes to "
                    "start. Waited "
                    f"{VLLM_ENGINE_READY_TIMEOUT_S}s (configured by "
                    f"VLLM_ENGINE_READY_TIMEOUT_S). To increase the "
                    f"timeout, set the environment variable: "
                    f"VLLM_ENGINE_READY_TIMEOUT_S=<seconds>"
                )
            events = await self.input_socket.poll(
                timeout=max(1, int(min(1.0, remaining) * 1000))
            )
            if not events:
                continue
            identity, payload = await self.input_socket.recv_multipart()
            new_engine_identities.discard(identity)
            self._apply_ready_response(payload)

    def _setup_elastic_ep_reconfig_bootstrap(self) -> None:
        from vllm.distributed.utils import create_tcp_store
        from vllm.utils.network_utils import get_open_ports_list

        parallel_config = self.vllm_config.parallel_config
        parallel_config._data_parallel_master_port_list = get_open_ports_list(5)
        parallel_config.data_parallel_master_port = (
            parallel_config._data_parallel_master_port_list.pop()
        )

        ip = parallel_config.data_parallel_master_ip
        store = create_tcp_store(
            ip,
            0,
            is_master=True,
            world_size=-1,
            wait_for_workers=False,
        )
        parallel_config._coord_store_port = store.port
        self._coord_store = store

    def _make_reconfig_request(
        self,
        new_data_parallel_size: int,
        rank_type: ReconfigureRankType = ReconfigureRankType.KEEP_CURRENT_RANK,
    ) -> ReconfigureDistributedRequest:
        parallel_config = self.vllm_config.parallel_config
        return ReconfigureDistributedRequest(
            new_data_parallel_size=new_data_parallel_size,
            new_data_parallel_rank=rank_type,
            new_data_parallel_rank_local=ReconfigureRankType.KEEP_CURRENT_RANK,
            new_data_parallel_master_ip=parallel_config.data_parallel_master_ip,
            new_data_parallel_master_port=parallel_config.data_parallel_master_port,
            new_data_parallel_master_port_list=parallel_config._data_parallel_master_port_list,
            coord_store_port=parallel_config._coord_store_port,
        )

    async def _prepare_scale_up_elastic_ep(
        self,
        new_data_parallel_size: int,
        num_redundant_experts: int,
    ) -> None:
        """Prepare scale up by creating new engine cores and reconfiguring
        existing ones."""
        assert isinstance(self.resources.engine_manager, CoreEngineActorManager)
        engine_manager = self.resources.engine_manager

        # Reserve every placement group before any existing rank is told to
        # join the larger world. A partial Ray resource snapshot must fail
        # without stranding old ranks waiting for a rank that was never made.
        try:
            reservation = await asyncio.to_thread(
                engine_manager.reserve_scale_up_elastic_ep,
                self.vllm_config,
                new_data_parallel_size,
            )
        except ElasticEPScaleUpReservationError:
            # Cleanup failure means Ray still owns part of a topology that was
            # never activated. Block retries until fail-stop shutdown removes
            # the manager's tracked resources.
            self._elastic_ep_prepare_mutated = True
            raise
        try:
            self._setup_elastic_ep_reconfig_bootstrap()
            self._elastic_ep_prepare_mutated = True

            # Phase 1: Send reconfig messages to existing engines
            reconfig_futures = []
            for engine in self.core_engines:
                reconfig_request = self._make_reconfig_request(
                    new_data_parallel_size
                )
                coro = self._call_utility_async(
                    "reinitialize_distributed", reconfig_request, engine=engine
                )
                reconfig_futures.append(asyncio.create_task(coro))

            # Phase 2: Create new engines from the complete reservation.
            start_new_worker_future = asyncio.to_thread(
                engine_manager.scale_up_elastic_ep,
                self.vllm_config,
                new_data_parallel_size,
                num_redundant_experts,
                reservation,
            )

            # Phase 3: Wait for new engines to be created and reconfig
            # messages to be received.
            await asyncio.gather(start_new_worker_future, *reconfig_futures)
            ready_keys = [future.result() for future in reconfig_futures]
            ready_keys.extend(
                f"eep_ready/{rank}"
                for rank in range(
                    len(self.core_engines), new_data_parallel_size
                )
            )
            await self._wait_for_elastic_ep_ready_keys(ready_keys)
            logger.info("[Elastic EP] Successfully started new engines")
        except BaseException as error:
            try:
                await asyncio.to_thread(
                    engine_manager.release_scale_up_elastic_ep_reservation,
                    reservation,
                )
            except BaseException as cleanup_error:
                # A leaked reservation blocks a safe retry. Make the outer
                # prepare path fail closed and preserve the original cause.
                self._elastic_ep_prepare_mutated = True
                raise cleanup_error from error
            raise

    async def _commit_scale_up_elastic_ep(self, new_data_parallel_size: int) -> None:
        new_core_engines = [
            rank.to_bytes(2, "little")
            for rank in range(len(self.core_engines), new_data_parallel_size)
        ]

        # The enclosing elastic-EP transaction already owns the lifecycle
        # lock, so use the lock-held broadcast helper to avoid re-entering it.
        await self._broadcast_dp_lifecycle_utility(
            "pause_scheduler", ("keep", False), tuple(self.core_engines)
        )
        self._elastic_ep_commit_mutated = True
        wait_future = self._eep_wait_for_setup_switch_complete()
        finish_futures = [
            asyncio.create_task(
                self._call_utility_async("commit_prepared_elastic_ep", engine=engine)
            )
            for engine in self.core_engines
        ]
        try:
            await asyncio.gather(*finish_futures)
            await asyncio.wait_for(
                wait_future, timeout=VLLM_ENGINE_READY_TIMEOUT_S
            )
            await self._wait_for_new_engine_ready(new_core_engines)
        except BaseException:
            if self.utility_results.get(EEP_NOTIFICATION_CALL_ID) is wait_future:
                self.utility_results.pop(EEP_NOTIFICATION_CALL_ID, None)
            if not wait_future.done():
                wait_future.cancel()
            raise

        self.core_engines.extend(new_core_engines)
        # Update the parallel config
        parallel_config = self.vllm_config.parallel_config
        parallel_config.data_parallel_size = new_data_parallel_size
        if isinstance(self.resources.engine_manager, CoreEngineActorManager):
            parallel_config.data_parallel_size_local = len(
                self.resources.engine_manager.local_engine_actors
            )
        self._apply_elastic_ep_local_topology(new_data_parallel_size)
        # Notify coordinator about scale up through existing
        # stats_update_task connection
        self._ensure_stats_update_task()
        scale_up_marker = msgspec.msgpack.encode(
            ("SCALE_ELASTIC_EP", new_data_parallel_size)
        )
        await self.first_req_send_socket.send(scale_up_marker)

        logger.info(
            "[Elastic EP] Scale up completed, new data parallel size: %s",
            new_data_parallel_size,
        )
        await self._broadcast_dp_lifecycle_utility(
            "resume_scheduler", (), tuple(self.core_engines)
        )

    async def _prepare_scale_down_elastic_ep(self, new_data_parallel_size: int) -> None:
        self._setup_elastic_ep_reconfig_bootstrap()
        self._elastic_ep_prepare_mutated = True

        reconfig_futures = []
        for engine in self.core_engines[:new_data_parallel_size]:
            reconfig_request = self._make_reconfig_request(new_data_parallel_size)
            coro = self._call_utility_async(
                "reinitialize_distributed", reconfig_request, engine=engine
            )
            reconfig_futures.append(asyncio.create_task(coro))

        ready_keys = await asyncio.gather(*reconfig_futures)
        await self._wait_for_elastic_ep_ready_keys(ready_keys)

    async def _quiesce_scale_down_elastic_ep(
        self, new_data_parallel_size: int
    ) -> tuple[list[EngineIdentity], int]:
        old_core_engines = self.core_engines
        self._validate_dp_lifecycle_owner()
        # Recheck at commit because pins may be created after prepare succeeds.
        self._check_elastic_ep_scale_down_prefix_pins(new_data_parallel_size)
        # Stop routing new requests to removing ranks while preserving the old
        # engine list for the unanimous old-group lifecycle broadcast.  Keep a
        # persistent cap because old-group coordinator stats may arrive while
        # the topology marker is in flight and must not re-enable removed ranks.
        previous_routing_limit = getattr(self, "_elastic_ep_routing_limit", None)
        previous_lb_engines = [counts.copy() for counts in self.lb_engines]
        self._elastic_ep_routing_limit = new_data_parallel_size
        self.lb_engines = self.lb_engines[:new_data_parallel_size]
        removed_dp_size = len(old_core_engines) - new_data_parallel_size
        try:
            await self._broadcast_dp_lifecycle_utility(
                "pause_scheduler", ("keep", False), tuple(old_core_engines)
            )
        except BaseException:
            resources = getattr(self, "resources", None)
            if resources is None or not getattr(resources, "engine_dead", False):
                # A common preflight rejection has made no engine mutation, so
                # restore the pre-commit routing view.  Partial admission marks
                # the client dead and intentionally keeps removed ranks gated.
                self._elastic_ep_routing_limit = previous_routing_limit
                self.lb_engines = previous_lb_engines
            raise
        self._elastic_ep_commit_mutated = True
        removed_engines = old_core_engines[new_data_parallel_size:]
        await asyncio.gather(
            *(
                self._call_utility_async(
                    "abort_for_elastic_ep_scale_down",
                    new_data_parallel_size,
                    engine=engine,
                )
                for engine in removed_engines
            )
        )
        self.core_engines = old_core_engines[:new_data_parallel_size]
        return old_core_engines, removed_dp_size

    async def _commit_scale_down_elastic_ep(self, new_data_parallel_size: int) -> None:
        """Scale down the data parallel size by shutting down and
        reconfiguring existing engine cores."""
        cur_data_parallel_size = len(self.core_engines)

        # The old DP group must publish one identical descriptor on every rank.
        # Retain kept-rank requests first, then abort only removed-rank requests
        # through a non-collective utility after the global keep-pause commits.
        old_core_engines, removed_dp_size = (
            await self._quiesce_scale_down_elastic_ep(new_data_parallel_size)
        )
        cleanup_future = asyncio.get_running_loop().create_future()
        self.eep_scaling_cache = ElasticScalingCache(
            existing_core_engines=old_core_engines.copy(),
            num_new_core_engines=new_data_parallel_size - cur_data_parallel_size,
            pending_notifications=dict(),
            completion_future=cleanup_future,
        )
        assert isinstance(self.resources.engine_manager, CoreEngineActorManager)
        self.resources.engine_manager.remove_run_refs_for_scale_down(removed_dp_size)
        wait_future = self._eep_wait_for_setup_switch_complete()
        reconfig_futures = []
        for cur_dp_rank, engine in enumerate(old_core_engines):
            if cur_dp_rank < new_data_parallel_size:
                coro = self._call_utility_async(
                    "commit_prepared_elastic_ep", engine=engine
                )
            else:
                reconfig_request = self._make_reconfig_request(
                    new_data_parallel_size,
                    ReconfigureRankType.SHUTDOWN_CURRENT_RANK,
                )
                coro = self._call_utility_async(
                    "reinitialize_distributed", reconfig_request, engine=engine
                )
            reconfig_futures.append(asyncio.create_task(coro))

        try:
            await asyncio.gather(*reconfig_futures)

            self.vllm_config.parallel_config.data_parallel_size = new_data_parallel_size
            self._apply_elastic_ep_local_topology(new_data_parallel_size)
            self._ensure_stats_update_task()
            scale_down_marker = msgspec.msgpack.encode(
                ("SCALE_ELASTIC_EP", new_data_parallel_size)
            )
            await self.first_req_send_socket.send(scale_down_marker)
            await asyncio.wait_for(
                asyncio.gather(wait_future, cleanup_future),
                timeout=VLLM_ENGINE_READY_TIMEOUT_S,
            )
            await self._broadcast_dp_lifecycle_utility(
                "resume_scheduler", (), tuple(self.core_engines)
            )
        except BaseException:
            self.eep_scaling_cache = None
            if self.utility_results.get(EEP_NOTIFICATION_CALL_ID) is wait_future:
                self.utility_results.pop(EEP_NOTIFICATION_CALL_ID, None)
            for future in (wait_future, cleanup_future):
                if not future.done():
                    future.cancel()
            raise

        logger.info(
            "[Elastic EP] Scale down completed, new data parallel size: %s",
            new_data_parallel_size,
        )
