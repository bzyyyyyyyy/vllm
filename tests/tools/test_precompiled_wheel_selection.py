# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import subprocess
from pathlib import Path

import pytest

from tools import precompiled_wheel_selection as selection

pytestmark = pytest.mark.gcc_extension

_UPSTREAM_COMMIT = "1" * 40
_BASE_COMMIT = "2" * 40
_UPSTREAM_URL = "https://api.github.com/repos/vllm-project/vllm/commits/main"


def test_source_distribution_includes_precompiled_wheel_helper() -> None:
    repo_root = Path(__file__).parents[2]
    manifest_lines = (
        (repo_root / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
    )

    assert "include tools/precompiled_wheel_selection.py" in manifest_lines


@pytest.mark.parametrize(
    ("configured_commit", "expected"),
    [
        (_UPSTREAM_COMMIT.upper(), _UPSTREAM_COMMIT),
        ("NIGHTLY", "nightly"),
    ],
)
def test_resolve_precompiled_wheel_commit_accepts_explicit_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    configured_commit: str,
    expected: str,
) -> None:
    def fail_get_base_commit(**kwargs: object) -> str:
        pytest.fail(f"unexpected base commit lookup: {kwargs}")

    monkeypatch.setattr(
        selection, "get_base_commit_in_main_branch", fail_get_base_commit
    )

    assert (
        selection.resolve_precompiled_wheel_commit(
            configured_commit,
            docker_build_context=False,
            repo_dir=tmp_path,
        )
        == expected
    )


@pytest.mark.parametrize(
    "configured_commit",
    [None, "", "short", "g" * 40, "nightly-ish"],
)
def test_resolve_precompiled_wheel_commit_looks_up_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    configured_commit: str | None,
) -> None:
    calls: list[tuple[bool, str | Path]] = []

    def get_base_commit(*, docker_build_context: bool, repo_dir: str | Path) -> str:
        calls.append((docker_build_context, repo_dir))
        return _BASE_COMMIT

    monkeypatch.setattr(
        selection, "get_base_commit_in_main_branch", get_base_commit
    )

    assert (
        selection.resolve_precompiled_wheel_commit(
            configured_commit,
            docker_build_context=False,
            repo_dir=tmp_path,
        )
        == _BASE_COMMIT
    )
    assert calls == [(False, tmp_path)]


def test_base_commit_lookup_refuses_docker_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail_check_output(*args: object, **kwargs: object) -> bytes:
        pytest.fail(f"unexpected subprocess call: {args}, {kwargs}")

    monkeypatch.setattr(selection.subprocess, "check_output", fail_check_output)

    with pytest.raises(RuntimeError, match="Refusing to fall back"):
        selection.get_base_commit_in_main_branch(
            docker_build_context=True,
            repo_dir=tmp_path,
        )


def test_base_commit_lookup_uses_merge_base_with_head(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    commands: list[list[str]] = []

    def check_output(command: list[str]) -> bytes:
        commands.append(command)
        if command[0] == "curl":
            return f'{{"sha": "{_UPSTREAM_COMMIT}"}}'.encode()
        if "cat-file" in command:
            return b""
        if "merge-base" in command:
            return f"{_BASE_COMMIT}\n".encode()
        pytest.fail(f"unexpected command: {command}")

    def fail_check_call(command: list[str]) -> None:
        pytest.fail(f"unexpected fetch: {command}")

    monkeypatch.setattr(selection.subprocess, "check_output", check_output)
    monkeypatch.setattr(selection.subprocess, "check_call", fail_check_call)

    assert (
        selection.get_base_commit_in_main_branch(
            docker_build_context=False,
            repo_dir=tmp_path,
        )
        == _BASE_COMMIT
    )
    git_cmd = ["git", "-C", str(tmp_path)]
    assert commands == [
        ["curl", "-s", _UPSTREAM_URL],
        [*git_cmd, "cat-file", "-e", _UPSTREAM_COMMIT],
        [*git_cmd, "merge-base", _UPSTREAM_COMMIT, "HEAD"],
    ]


def test_base_commit_lookup_fetches_missing_upstream_commit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("GH_TOKEN", "secret")
    commands: list[list[str]] = []
    fetches: list[list[str]] = []

    def check_output(command: list[str]) -> bytes:
        commands.append(command)
        if command[0] == "curl":
            return f'{{"sha": "{_UPSTREAM_COMMIT}"}}'.encode()
        if "cat-file" in command:
            raise subprocess.CalledProcessError(1, command)
        if "merge-base" in command:
            return f"{_BASE_COMMIT}\n".encode()
        pytest.fail(f"unexpected command: {command}")

    monkeypatch.setattr(selection.subprocess, "check_output", check_output)
    monkeypatch.setattr(
        selection.subprocess,
        "check_call",
        lambda command: fetches.append(command),
    )

    assert (
        selection.get_base_commit_in_main_branch(
            docker_build_context=False,
            repo_dir=tmp_path,
        )
        == _BASE_COMMIT
    )
    git_cmd = ["git", "-C", str(tmp_path)]
    assert commands[0] == [
        "curl",
        "-s",
        _UPSTREAM_URL,
        "-H",
        "Authorization: token secret",
    ]
    assert fetches == [
        [*git_cmd, "fetch", "https://github.com/vllm-project/vllm", "main"]
    ]
    assert commands[-1] == [*git_cmd, "merge-base", _UPSTREAM_COMMIT, "HEAD"]


def test_base_commit_lookup_preserves_failure_as_cause(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    failure = OSError("curl unavailable")

    def raise_failure(_command: list[str]) -> bytes:
        raise failure

    monkeypatch.setattr(selection.subprocess, "check_output", raise_failure)

    with pytest.raises(RuntimeError, match="Refusing to fall back") as exc_info:
        selection.get_base_commit_in_main_branch(
            docker_build_context=False,
            repo_dir=tmp_path,
        )

    assert exc_info.value.__cause__ is failure
