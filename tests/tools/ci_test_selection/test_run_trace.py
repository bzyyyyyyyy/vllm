# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from coverage import CoverageData

from tools.ci_test_selection.deep_python_trace import (
    _current_test_id,
    _repository_path,
)
from tools.ci_test_selection.run_trace import (
    _command_environment,
    coverage_rows,
    normalize_repository_path,
    pytest_command,
    validate_import_environment,
)


def test_normalize_repository_path_from_checkout(tmp_path: Path):
    repo = tmp_path / "repo"
    source = repo / "vllm" / "attention" / "ops.py"

    assert normalize_repository_path(str(source), repo) == "vllm/attention/ops.py"


def test_normalize_repository_path_from_site_packages(tmp_path: Path):
    repo = tmp_path / "repo"
    source = tmp_path / "venv" / "site-packages" / "vllm" / "attention" / "ops.py"

    assert normalize_repository_path(str(source), repo) == "vllm/attention/ops.py"


def test_normalize_repository_path_rejects_non_vllm(tmp_path: Path):
    assert (
        normalize_repository_path(str(tmp_path / "torch" / "ops.py"), tmp_path) is None
    )


def test_deep_trace_rejects_python_pseudo_filenames(tmp_path: Path):
    assert _repository_path("<frozen importlib._bootstrap>", tmp_path) is None
    assert _repository_path("<string>", tmp_path) is None


@pytest.mark.parametrize("phase", ["setup", "call", "teardown"])
def test_deep_trace_normalizes_pytest_phase(monkeypatch, phase: str):
    monkeypatch.setenv(
        "PYTEST_CURRENT_TEST", f"tests/kernels/test_ops.py::test_one ({phase})"
    )

    assert _current_test_id() == "tests/kernels/test_ops.py::test_one"


def test_coverage_rows_are_per_test_and_canonical(tmp_path: Path):
    repo = tmp_path / "repo"
    source = repo / "vllm" / "attention" / "ops.py"
    source.parent.mkdir(parents=True)
    source.write_text("one = 1\ntwo = 2\n", encoding="utf-8")
    coverage_file = tmp_path / ".coverage"
    data = CoverageData(basename=str(coverage_file))
    data.set_context("tests/kernels/test_ops.py::test_one|run")
    data.add_lines({str(source): {1, 2}})
    data.set_context("tests/kernels/test_ops.py::test_two|run")
    data.add_lines({str(source): {2}})
    data.write()

    rows = coverage_rows(
        coverage_file,
        repo,
        repository_sha="a" * 40,
        job_key="kernels-ops",
    )

    assert [(row["test_id"], row["line"]) for row in rows] == [
        ("tests/kernels/test_ops.py::test_one", 1),
        ("tests/kernels/test_ops.py::test_one", 2),
        ("tests/kernels/test_ops.py::test_two", 2),
    ]
    assert all(row["file"] == "vllm/attention/ops.py" for row in rows)


def test_pytest_command_loads_python_and_nvtx_plugins():
    command = pytest_command(["tests/kernels/test_ops.py"])

    assert "tools.ci_test_selection.pytest_trace_plugin" in command
    assert "tools.ci_test_selection.nvtx_test_ranges" in command
    assert command[-1] == "tests/kernels/test_ops.py"


def test_deep_environment_disables_pytest_assertion_rewriting(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("PYTEST_ADDOPTS", "-ra")

    environment = _command_environment(
        coverage_file=tmp_path / ".coverage",
        node_file=tmp_path / "nodes.json",
        repo_root=tmp_path,
        auto_load_pytest=True,
        plain_assertions=True,
    )

    assert environment["PYTEST_ADDOPTS"].split() == [
        "-ra",
        "--cov=vllm",
        "--cov-context=test",
        "--cov-report=",
        "--assert=plain",
    ]
    assert environment["VLLM_CI_TEST_SELECTION_PACKAGE"] == ("tools.ci_test_selection")


def test_import_preflight_rejects_checkout_source_for_image_job(tmp_path: Path):
    checkout = tmp_path / "checkout"
    source_package = checkout / "vllm"
    source_package.mkdir(parents=True)
    (source_package / "__init__.py").write_text("", encoding="utf-8")
    output = tmp_path / "import-environment.json"
    environment = {
        "BUILDKITE_BUILD_CHECKOUT_PATH": str(checkout),
        "PYTHONPATH": str(checkout),
    }

    status = validate_import_environment(
        command_cwd=tmp_path,
        environment=environment,
        output_path=output,
        repo_root=tmp_path / "image-workspace",
    )

    document = json.loads(output.read_text(encoding="utf-8"))
    assert status == 1
    assert document["error"] == "image job imported vllm from checkout source"
    assert document["vllm_file"] == str(source_package / "__init__.py")


def test_import_preflight_accepts_installed_package_outside_checkout(tmp_path: Path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    installed_package = tmp_path / "site-packages" / "vllm"
    installed_package.mkdir(parents=True)
    (installed_package / "__init__.py").write_text("", encoding="utf-8")
    output = tmp_path / "import-environment.json"
    environment = {
        "BUILDKITE_BUILD_CHECKOUT_PATH": str(checkout),
        "PYTHONPATH": str(installed_package.parent),
    }

    status = validate_import_environment(
        command_cwd=tmp_path,
        environment=environment,
        output_path=output,
        repo_root=tmp_path / "image-workspace",
    )

    document = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert document["error"] is None
    assert document["vllm_file"] == str(installed_package / "__init__.py")


def test_import_preflight_rejects_broken_pytest_plugin_registration(tmp_path: Path):
    installed_package = tmp_path / "site-packages" / "vllm"
    installed_package.mkdir(parents=True)
    (installed_package / "__init__.py").write_text("", encoding="utf-8")
    output = tmp_path / "import-environment.json"
    environment = {
        "PYTHONPATH": str(installed_package.parent),
        "PYTEST_PLUGINS": "missing_test_selection_plugin",
    }

    status = validate_import_environment(
        command_cwd=tmp_path,
        environment=environment,
        output_path=output,
        repo_root=tmp_path / "image-workspace",
    )

    document = json.loads(output.read_text(encoding="utf-8"))
    assert status != 0
    assert document["error"] == "pytest plugin/options preflight failed"
    assert document["import_exit_code"] == 0
    assert document["pytest_plugin_exit_code"] != 0


def test_import_preflight_rejects_unknown_pytest_hook(tmp_path: Path):
    installed_package = tmp_path / "site-packages" / "vllm"
    installed_package.mkdir(parents=True)
    (installed_package / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "unknown_hook_plugin.py").write_text(
        "def pytest_vllm_ci_unknown_hook():\n    pass\n", encoding="utf-8"
    )
    output = tmp_path / "import-environment.json"
    environment = {
        "PYTHONPATH": os.pathsep.join([str(installed_package.parent), str(tmp_path)]),
        "PYTEST_PLUGINS": "unknown_hook_plugin",
    }

    status = validate_import_environment(
        command_cwd=tmp_path,
        environment=environment,
        output_path=output,
        repo_root=tmp_path / "image-workspace",
    )

    document = json.loads(output.read_text(encoding="utf-8"))
    assert status != 0
    assert document["error"] == "pytest plugin/options preflight failed"
    assert document["import_exit_code"] == 0
    assert document["pytest_plugin_exit_code"] != 0


def test_import_preflight_rejects_unsupported_pytest_option(tmp_path: Path):
    installed_package = tmp_path / "site-packages" / "vllm"
    installed_package.mkdir(parents=True)
    (installed_package / "__init__.py").write_text("", encoding="utf-8")
    output = tmp_path / "import-environment.json"
    environment = {
        "PYTHONPATH": str(installed_package.parent),
        "PYTEST_ADDOPTS": "--vllm-ci-unsupported-option",
    }

    status = validate_import_environment(
        command_cwd=tmp_path,
        environment=environment,
        output_path=output,
        repo_root=tmp_path / "image-workspace",
    )

    document = json.loads(output.read_text(encoding="utf-8"))
    assert status != 0
    assert document["error"] == "pytest plugin/options preflight failed"
    assert document["import_exit_code"] == 0
    assert document["pytest_plugin_exit_code"] != 0


def test_deep_python_trace_records_ordered_repository_calls(tmp_path: Path):
    repo = tmp_path / "repo"
    package = repo / "vllm"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "sample.py").write_text(
        "def inner():\n    return 42\n\ndef outer():\n    return inner()\n",
        encoding="utf-8",
    )
    output = tmp_path / "calls"
    project_root = Path(__file__).resolve().parents[3]
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join([str(repo), str(project_root)]),
            "PYTEST_CURRENT_TEST": "tests/test_sample.py::test_answer (call)",
            "VLLM_CI_TEST_SELECTION_DEEP_TRACE_DIR": str(output),
            "VLLM_CI_TEST_SELECTION_REPO_ROOT": str(repo),
        }
    )

    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from tools.ci_test_selection.deep_python_trace import "
                "install_from_environment; install_from_environment(); "
                "from vllm.sample import outer; assert outer() == 42"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
    )

    rows = [
        json.loads(line)
        for line in next(output.glob("python-calls.*.jsonl")).read_text().splitlines()
    ]
    sample = [
        row
        for row in rows
        if row["file"] == "vllm/sample.py" and not row["function"].endswith(".<module>")
    ]
    assert [(row["event"], row["function"].split(".")[-1]) for row in sample] == [
        ("call", "outer"),
        ("call", "inner"),
        ("return", "inner"),
        ("return", "outer"),
    ]
    assert [row["depth"] for row in sample] == [0, 1, 1, 0]
    assert all(row["test_id"] == "tests/test_sample.py::test_answer" for row in sample)


def test_pytest_plugin_records_main_process_deep_calls(tmp_path: Path):
    repo = tmp_path / "repo"
    package = repo / "vllm"
    tests = repo / "tests"
    package.mkdir(parents=True)
    tests.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "sample.py").write_text(
        "def inner():\n    return 42\n\ndef outer():\n    return inner()\n",
        encoding="utf-8",
    )
    (tests / "test_sample.py").write_text(
        "from vllm.sample import outer\n\n"
        "def test_answer():\n"
        "    assert outer() == 42\n",
        encoding="utf-8",
    )
    output = tmp_path / "calls"
    project_root = Path(__file__).resolve().parents[3]
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join([str(repo), str(project_root)]),
            "PYTEST_PLUGINS": "tools.ci_test_selection.pytest_trace_plugin",
            "VLLM_CI_TEST_SELECTION_DEEP_TRACE": "1",
            "VLLM_CI_TEST_SELECTION_DEEP_TRACE_DIR": str(output),
            "VLLM_CI_TEST_SELECTION_REPO_ROOT": str(repo),
        }
    )

    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_sample.py"],
        cwd=repo,
        env=environment,
        check=True,
    )

    rows = [
        json.loads(line)
        for line in next(output.glob("python-calls.*.jsonl")).read_text().splitlines()
    ]
    sample = [
        row
        for row in rows
        if row["file"] == "vllm/sample.py" and not row["function"].endswith(".<module>")
    ]
    assert [(row["event"], row["function"].split(".")[-1]) for row in sample] == [
        ("call", "outer"),
        ("call", "inner"),
        ("return", "inner"),
        ("return", "outer"),
    ]
    assert all(row["test_id"] == "tests/test_sample.py::test_answer" for row in sample)
