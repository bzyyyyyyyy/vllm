# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import re
import subprocess
from pathlib import Path


_BASE_COMMIT_ERROR = (
    "Failed to determine the upstream base commit for precompiled vLLM "
    "extensions. Refusing to fall back to the nightly wheel because its native "
    "operator schemas may be incompatible. Set VLLM_PRECOMPILED_WHEEL_COMMIT "
    "to a 40-character upstream commit (or explicitly to 'nightly'), or "
    "rebuild with VLLM_USE_PRECOMPILED=0."
)


def resolve_precompiled_wheel_commit(
    configured_commit: str | None,
    *,
    docker_build_context: bool,
    repo_dir: str | Path,
) -> str:
    commit = (configured_commit or "").lower()
    if commit == "nightly" or re.fullmatch(r"[0-9a-f]{40}", commit) is not None:
        return commit
    return get_base_commit_in_main_branch(
        docker_build_context=docker_build_context,
        repo_dir=repo_dir,
    )


def get_base_commit_in_main_branch(
    *,
    docker_build_context: bool,
    repo_dir: str | Path,
) -> str:
    if docker_build_context:
        raise RuntimeError(_BASE_COMMIT_ERROR)

    try:
        curl_cmd = [
            "curl",
            "-s",
            "https://api.github.com/repos/vllm-project/vllm/commits/main",
        ]
        github_token = os.getenv("GH_TOKEN", os.getenv("GITHUB_TOKEN"))
        if github_token:
            curl_cmd += ["-H", f"Authorization: token {github_token}"]
        resp_json = subprocess.check_output(curl_cmd).decode("utf-8")
        upstream_main_commit = json.loads(resp_json)["sha"]
        print(f"Upstream main branch latest commit: {upstream_main_commit}")

        git_cmd = ["git", "-C", str(repo_dir)]
        try:
            subprocess.check_output(
                [*git_cmd, "cat-file", "-e", upstream_main_commit]
            )
        except subprocess.CalledProcessError:
            subprocess.check_call(
                [
                    *git_cmd,
                    "fetch",
                    "https://github.com/vllm-project/vllm",
                    "main",
                ]
            )

        return (
            subprocess.check_output(
                [*git_cmd, "merge-base", upstream_main_commit, "HEAD"]
            )
            .decode("utf-8")
            .strip()
        )
    except Exception as err:
        raise RuntimeError(_BASE_COMMIT_ERROR) from err
