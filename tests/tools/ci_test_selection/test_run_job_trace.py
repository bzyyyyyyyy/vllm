# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pybase64 as base64
import pytest

import tools.ci_test_selection.nvtx_test_ranges as nvtx_test_ranges
import tools.ci_test_selection.pytest_trace_plugin as pytest_trace_plugin
import tools.ci_test_selection.run_job_trace as run_job_trace
from tools.ci_test_selection.nvtx_test_ranges import _configured_nvtx
from tools.ci_test_selection.run_job_trace import (
    decode_commands,
    static_build_provenance_reference,
)


def _payload(commands: list[str]) -> str:
    return base64.b64encode(json.dumps(commands).encode()).decode()


def test_decode_commands_round_trip():
    commands = ["pytest -q tests/test_one.py", "python -m pytest tests/test_two.py"]

    assert decode_commands(_payload(commands)) == commands


def test_static_build_provenance_reference_uses_image_file_hashes(
    tmp_path: Path, monkeypatch
):
    graph_dir = tmp_path / "build-graph"
    graph_dir.mkdir()
    (graph_dir / "build-graph.jsonl").write_text('{"graph":1}\n')
    (graph_dir / "kernel-map.jsonl").write_text('{"kernel":1}\n')
    monkeypatch.setenv("BUILDKITE_COMMIT", "b" * 40)
    monkeypatch.setenv("BUILDKITE_BUILD_ID", "build-id")

    reference = static_build_provenance_reference(graph_dir)

    assert reference["publisher_step_key"] == "image-build"
    assert reference["repository_sha"] == "b" * 40
    assert reference["buildkite_build_id"] == "build-id"
    assert reference["files"]["build-graph.jsonl"]["bytes"] > 0
    assert len(reference["files"]["kernel-map.jsonl"]["sha256"]) == 64


def test_top_level_collector_package_avoids_tests_tools_shadowing():
    project_root = Path(__file__).resolve().parents[3]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(project_root / "tools")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import ci_test_selection.run_trace as runner; "
                "print(runner.pytest_command([]))"
            ),
        ],
        cwd=project_root / "tests",
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "ci_test_selection.pytest_trace_plugin" in result.stdout
    assert "ci_test_selection.nvtx_test_ranges" in result.stdout


def test_gpu_wrapper_does_not_prepend_checkout_root_to_pythonpath():
    project_root = Path(__file__).resolve().parents[3]
    wrapper = (
        project_root / "tools" / "ci_test_selection" / "run_traced.sh"
    ).read_text(encoding="utf-8")

    assert 'PYTHONPATH="$REPO_ROOT' not in wrapper


def test_python_only_nvtx_gate_does_not_initialize_cuda(monkeypatch):
    def fail_if_called():
        raise AssertionError("python-only collection touched CUDA")

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=fail_if_called),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("VLLM_CI_TEST_SELECTION_NVTX", "0")

    assert _configured_nvtx() is None


@pytest.mark.parametrize("failure", ["push", "pop"])
def test_nvtx_tooling_failure_does_not_escape_pytest_hook(monkeypatch, failure):
    class BrokenNvtx:
        def range_push(self, _label):
            if failure == "push":
                raise RuntimeError("broken push")

        def range_pop(self):
            if failure == "pop":
                raise RuntimeError("broken pop")

    monkeypatch.setattr(nvtx_test_ranges, "_nvtx", BrokenNvtx())
    wrapper = nvtx_test_ranges._wrap("call", SimpleNamespace(nodeid="test_node"))

    next(wrapper)
    with pytest.raises(StopIteration):
        next(wrapper)


def test_node_export_failure_does_not_change_pytest_result(
    tmp_path: Path, monkeypatch, capsys
):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("block", encoding="utf-8")
    monkeypatch.setenv("VLLM_CI_TEST_SELECTION_NODEIDS", str(blocker / "nodes.json"))

    pytest_trace_plugin.pytest_sessionfinish(SimpleNamespace(), 0)

    assert "node/outcome export failed" in capsys.readouterr().err


@pytest.mark.parametrize("document", [[], [""], {"command": "pytest"}, [1]])
def test_decode_commands_rejects_invalid_documents(document):
    with pytest.raises(SystemExit):
        decode_commands(base64.b64encode(json.dumps(document).encode()).decode())


def test_python_only_job_preserves_pytest_command_and_collects_trace(tmp_path: Path):
    repo = tmp_path / "repo"
    source = repo / "vllm" / "sample.py"
    test_file = repo / "tests" / "test_sample.py"
    source.parent.mkdir(parents=True)
    test_file.parent.mkdir(parents=True)
    (repo / "vllm" / "__init__.py").write_text("", encoding="utf-8")
    source.write_text("def answer():\n    return 42\n", encoding="utf-8")
    test_file.write_text(
        "import os\n\n"
        "from vllm.sample import answer\n\n"
        "def test_answer():\n    assert answer() == 42\n\n"
        "def test_nvtx_is_disabled():\n"
        "    assert os.environ['VLLM_CI_TEST_SELECTION_NVTX'] == '0'\n",
        encoding="utf-8",
    )
    output = tmp_path / "trace"
    project_root = Path(__file__).resolve().parents[3]
    environment = dict(os.environ)
    environment["BUILDKITE_COMMIT"] = "a" * 40
    environment["PYTHONPATH"] = str(project_root)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.ci_test_selection.run_job_trace",
            "--output-dir",
            str(output),
            "--job-key",
            "ci-trace-unit",
            "--represented-job-key",
            "unit",
            "--commands-base64",
            _payload(["pytest -q tests/test_sample.py"]),
            "--repo-root",
            str(repo),
            "--python-only",
        ],
        cwd=repo,
        env=environment,
        check=False,
    )

    assert result.returncode == 0
    shard = output / "commands" / "000"
    trace_rows = [
        json.loads(line)
        for line in (shard / "python-trace.jsonl").read_text().splitlines()
    ]
    assert {row["file"] for row in trace_rows} == {"vllm/sample.py"}
    assert {row["test_id"] for row in trace_rows} == {
        "tests/test_sample.py::test_answer",
    }
    job = json.loads((shard / "job.json").read_text())
    assert job["healthy"] is True
    assert set(job["node_ids"]) == {
        "tests/test_sample.py::test_answer",
        "tests/test_sample.py::test_nvtx_is_disabled",
    }
    summary = json.loads((output / "trace-job.json").read_text())
    assert summary["healthy"] is True
    assert summary["capture_mode"] == "python-only"


@pytest.mark.parametrize(
    "command,expected_status",
    [
        ("echo original-command-ran", 0),
        ("echo original-command-ran && exit 7", 7),
    ],
)
def test_in_place_collection_preserves_original_command_status(
    tmp_path: Path, command: str, expected_status: int
):
    repo = tmp_path / "repo"
    package = repo / "vllm"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    output = tmp_path / "trace"
    project_root = Path(__file__).resolve().parents[3]
    environment = dict(os.environ)
    environment["BUILDKITE_COMMIT"] = "c" * 40
    environment["PYTHONPATH"] = str(project_root)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.ci_test_selection.run_job_trace",
            "--output-dir",
            str(output),
            "--job-key",
            "unit",
            "--represented-job-key",
            "unit",
            "--commands-base64",
            _payload([command]),
            "--repo-root",
            str(repo),
            "--python-only",
            "--preserve-command-exit-code",
        ],
        cwd=repo,
        env=environment,
        check=False,
    )

    assert result.returncode == expected_status
    summary = json.loads((output / "trace-job.json").read_text())
    assert summary["command_results"][0]["command_exit_code"] == expected_status
    assert summary["healthy"] is False


def test_in_place_collection_falls_back_only_when_preflight_never_started_command(
    tmp_path: Path,
):
    repo = tmp_path / "repo"
    package = repo / "vllm"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "raise RuntimeError('preflight failure')\n", encoding="utf-8"
    )
    marker = repo / "original-ran"
    output = tmp_path / "trace"
    project_root = Path(__file__).resolve().parents[3]
    environment = dict(os.environ)
    environment["BUILDKITE_COMMIT"] = "d" * 40
    environment["PYTHONPATH"] = str(project_root)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.ci_test_selection.run_job_trace",
            "--output-dir",
            str(output),
            "--job-key",
            "unit",
            "--represented-job-key",
            "unit",
            "--commands-base64",
            _payload(["touch original-ran"]),
            "--repo-root",
            str(repo),
            "--python-only",
            "--preserve-command-exit-code",
        ],
        cwd=repo,
        env=environment,
        check=False,
    )

    assert result.returncode == 0
    assert marker.is_file()
    summary = json.loads((output / "trace-job.json").read_text())
    assert summary["command_results"][0]["fallback_uninstrumented"] is True
    assert summary["command_results"][0]["command_exit_code"] == 0


def test_finished_command_status_survives_later_collector_crash(
    tmp_path: Path, monkeypatch
):
    repo = tmp_path / "repo"
    repo.mkdir()
    output = tmp_path / "trace"

    def fake_run_command(*_args, command_index, output_dir, **_kwargs):
        command_output = output_dir / "commands" / f"{command_index:03d}"
        command_output.mkdir(parents=True)
        (command_output / "command-status.json").write_text(
            json.dumps(
                {
                    "command_executed": True,
                    "exit_code": 0,
                    "phase": "finished",
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess([], 13)

    def unexpected_fallback(*_args, **_kwargs):
        raise AssertionError("a completed command was rerun")

    monkeypatch.setattr(run_job_trace, "_run_command", fake_run_command)
    monkeypatch.setattr(run_job_trace, "_run_uninstrumented", unexpected_fallback)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_job_trace",
            "--output-dir",
            str(output),
            "--job-key",
            "unit",
            "--represented-job-key",
            "unit",
            "--commands-base64",
            _payload(["pytest tests/test_one.py"]),
            "--repo-root",
            str(repo),
            "--python-only",
            "--preserve-command-exit-code",
        ],
    )

    assert run_job_trace.main() == 0
    summary = json.loads((output / "trace-job.json").read_text())
    assert summary["command_results"][0]["collector_exit_code"] == 13
    assert summary["command_results"][0]["command_exit_code"] == 0
    assert summary["command_results"][0]["fallback_uninstrumented"] is False


def test_started_command_is_not_rerun_when_collector_status_is_unknown(
    tmp_path: Path, monkeypatch
):
    repo = tmp_path / "repo"
    repo.mkdir()
    output = tmp_path / "trace"

    def fake_run_command(*_args, command_index, output_dir, **_kwargs):
        command_output = output_dir / "commands" / f"{command_index:03d}"
        command_output.mkdir(parents=True)
        (command_output / "command-status.json").write_text(
            json.dumps({"command_executed": True, "phase": "started"}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess([], 13)

    def unexpected_fallback(*_args, **_kwargs):
        raise AssertionError("a started command was rerun")

    monkeypatch.setattr(run_job_trace, "_run_command", fake_run_command)
    monkeypatch.setattr(run_job_trace, "_run_uninstrumented", unexpected_fallback)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_job_trace",
            "--output-dir",
            str(output),
            "--job-key",
            "unit",
            "--represented-job-key",
            "unit",
            "--commands-base64",
            _payload(["pytest tests/test_one.py"]),
            "--repo-root",
            str(repo),
            "--python-only",
            "--preserve-command-exit-code",
        ],
    )

    assert run_job_trace.main() == 13
    summary = json.loads((output / "trace-job.json").read_text())
    assert summary["command_results"][0]["command_exit_code"] is None
    assert summary["command_results"][0]["fallback_uninstrumented"] is False


def test_static_provenance_export_failure_does_not_change_command_status(
    tmp_path: Path, monkeypatch
):
    repo = tmp_path / "repo"
    repo.mkdir()
    output = tmp_path / "trace"

    def fake_run_command(*_args, command_index, output_dir, **_kwargs):
        command_output = output_dir / "commands" / f"{command_index:03d}"
        command_output.mkdir(parents=True)
        (command_output / "command-status.json").write_text(
            json.dumps(
                {
                    "command_executed": True,
                    "exit_code": 0,
                    "phase": "finished",
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess([], 0)

    def fail_static_provenance(*_args, **_kwargs):
        raise RuntimeError("broken static provenance")

    monkeypatch.setattr(run_job_trace, "_run_command", fake_run_command)
    monkeypatch.setattr(
        run_job_trace,
        "static_build_provenance_reference",
        fail_static_provenance,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_job_trace",
            "--output-dir",
            str(output),
            "--job-key",
            "unit",
            "--represented-job-key",
            "unit",
            "--commands-base64",
            _payload(["pytest tests/test_one.py"]),
            "--repo-root",
            str(repo),
            "--python-only",
            "--preserve-command-exit-code",
        ],
    )

    assert run_job_trace.main() == 0
    summary = json.loads((output / "trace-job.json").read_text())
    assert summary["healthy"] is False
    assert "broken static provenance" in summary["static_build_provenance_error"]


def test_signal_status_uses_shell_convention(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    output = tmp_path / "trace"

    def fake_run_command(*_args, command_index, output_dir, **_kwargs):
        command_output = output_dir / "commands" / f"{command_index:03d}"
        command_output.mkdir(parents=True)
        (command_output / "command-status.json").write_text(
            json.dumps(
                {
                    "command_executed": True,
                    "exit_code": -9,
                    "phase": "finished",
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess([], -9)

    monkeypatch.setattr(run_job_trace, "_run_command", fake_run_command)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_job_trace",
            "--output-dir",
            str(output),
            "--job-key",
            "unit",
            "--represented-job-key",
            "unit",
            "--commands-base64",
            _payload(["pytest tests/test_one.py"]),
            "--repo-root",
            str(repo),
            "--python-only",
            "--preserve-command-exit-code",
        ],
    )

    assert run_job_trace.main() == 137
    summary = json.loads((output / "trace-job.json").read_text())
    assert summary["command_results"][0]["collector_exit_code"] == 137


@pytest.mark.parametrize("failure", ["write", "parallel-env"])
def test_job_summary_failure_preserves_finished_command_status(
    tmp_path: Path, monkeypatch, capsys, failure: str
):
    repo = tmp_path / "repo"
    repo.mkdir()
    output = tmp_path / "trace"

    def fake_run_command(*_args, command_index, output_dir, **_kwargs):
        command_output = output_dir / "commands" / f"{command_index:03d}"
        command_output.mkdir(parents=True)
        (command_output / "command-status.json").write_text(
            json.dumps(
                {
                    "command_executed": True,
                    "exit_code": 0,
                    "phase": "finished",
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess([], 0)

    monkeypatch.setattr(run_job_trace, "_run_command", fake_run_command)
    if failure == "write":
        monkeypatch.setattr(
            run_job_trace,
            "_atomic_json",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("read-only")),
        )
    else:
        monkeypatch.setenv("BUILDKITE_PARALLEL_JOB", "not-an-integer")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_job_trace",
            "--output-dir",
            str(output),
            "--job-key",
            "unit",
            "--represented-job-key",
            "unit",
            "--commands-base64",
            _payload(["pytest tests/test_one.py"]),
            "--repo-root",
            str(repo),
            "--python-only",
            "--preserve-command-exit-code",
        ],
    )

    assert run_job_trace.main() == 0
    assert "collector job summary failed" in capsys.readouterr().err


def test_image_build_trace_provenance_is_best_effort():
    project_root = Path(__file__).resolve().parents[3]
    image_build = (
        project_root / ".buildkite" / "image_build" / "image_build.sh"
    ).read_text(encoding="utf-8")
    dockerfile = (project_root / "docker" / "Dockerfile").read_text(encoding="utf-8")
    export_dockerfile = (
        project_root / "docker" / "Dockerfile.build-provenance"
    ).read_text(encoding="utf-8")

    assert image_build.count("publish_build_provenance_best_effort") == 3
    assert "affected jobs remain always-run" in image_build
    assert "if ! bash tools/ci_test_selection/export_image_build_provenance.sh" in (
        dockerfile
    )
    assert "COPY --from=source /opt/vllm-ci/build-graph/ /" in export_dockerfile
