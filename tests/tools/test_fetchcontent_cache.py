# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.gcc_extension

_UTILS_CMAKE = Path(__file__).resolve().parents[2] / "cmake" / "utils.cmake"


def _run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def _make_checkout(source_dir: Path) -> str:
    source_dir.mkdir(parents=True)
    required_file = source_dir / "include" / "api.h"
    required_file.parent.mkdir()
    required_file.write_text("current\n", encoding="utf-8")
    _run(["git", "init"], source_dir)
    _run(["git", "config", "user.email", "test@example.com"], source_dir)
    _run(["git", "config", "user.name", "vLLM Test"], source_dir)
    _run(["git", "config", "commit.gpgsign", "false"], source_dir)
    _run(["git", "add", "include/api.h"], source_dir)
    _run(["git", "commit", "-m", "fixture"], source_dir)
    return _run(["git", "rev-parse", "HEAD"], source_dir).stdout.strip()


def _check_cache(
    tmp_path: Path,
    *,
    cache_root: Path,
    source_dir: Path,
    binary_dir: Path,
    subbuild_dir: Path,
    revision: str,
    check: bool = True,
) -> str:
    result_path = tmp_path / "result.txt"
    script_path = tmp_path / "check_cache.cmake"
    script_path.write_text(
        f"""
include("{_UTILS_CMAKE.as_posix()}")
vllm_prepare_pinned_fetchcontent_checkout(
  REUSE
  CACHE_ROOT "{cache_root.as_posix()}"
  SOURCE_DIR "{source_dir.as_posix()}"
  BINARY_DIR "{binary_dir.as_posix()}"
  SUBBUILD_DIR "{subbuild_dir.as_posix()}"
  GIT_TAG "{revision}"
  REQUIRED_FILE "include/api.h")
file(WRITE "{result_path.as_posix()}" "${{REUSE}}")
""",
        encoding="utf-8",
    )
    completed = subprocess.run(
        ["cmake", "-P", str(script_path)],
        check=check,
        capture_output=True,
        text=True,
    )
    if not check:
        assert completed.returncode != 0
        return ""
    return result_path.read_text(encoding="utf-8")


def test_matching_fetchcontent_checkout_is_reused(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    source_dir = cache_root / "dep-src"
    binary_dir = cache_root / "dep-build"
    subbuild_dir = cache_root / "dep-subbuild"
    revision = _make_checkout(source_dir)
    binary_dir.mkdir()
    subbuild_dir.mkdir()

    result = _check_cache(
        tmp_path,
        cache_root=cache_root,
        source_dir=source_dir,
        binary_dir=binary_dir,
        subbuild_dir=subbuild_dir,
        revision=revision,
    )

    assert result == "TRUE"
    assert source_dir.is_dir()
    assert binary_dir.is_dir()
    assert subbuild_dir.is_dir()


def test_stale_fetchcontent_checkout_is_removed_safely(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    source_dir = cache_root / "dep-src"
    binary_dir = cache_root / "dep-build"
    subbuild_dir = cache_root / "dep-subbuild"
    _make_checkout(source_dir)
    binary_dir.mkdir()
    subbuild_dir.mkdir()
    sibling = cache_root / "unrelated"
    sibling.mkdir()

    result = _check_cache(
        tmp_path,
        cache_root=cache_root,
        source_dir=source_dir,
        binary_dir=binary_dir,
        subbuild_dir=subbuild_dir,
        revision="0" * 40,
    )

    assert result == "FALSE"
    assert not source_dir.exists()
    assert not binary_dir.exists()
    assert not subbuild_dir.exists()
    assert sibling.is_dir()


def test_fetchcontent_cache_rejects_path_outside_root(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    source_dir = tmp_path / "outside-src"
    binary_dir = cache_root / "dep-build"
    subbuild_dir = cache_root / "dep-subbuild"
    _make_checkout(source_dir)

    _check_cache(
        tmp_path,
        cache_root=cache_root,
        source_dir=source_dir,
        binary_dir=binary_dir,
        subbuild_dir=subbuild_dir,
        revision="0" * 40,
        check=False,
    )

    assert source_dir.is_dir()
