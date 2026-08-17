# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys
from unittest.mock import patch

from vllm.config import VllmConfig
from vllm.transformers_utils.config import maybe_register_config_serialize_by_value


def test_mp_reducer():
    """Test VllmConfig reducer registration without transformers_modules.

    This is a regression test for https://github.com/vllm-project/vllm/pull/19510.
    """
    with (
        patch.dict(sys.modules, {"transformers_modules": None}),
        patch("multiprocessing.reducer.register") as mock_register,
    ):
        maybe_register_config_serialize_by_value()

        assert mock_register.called, (
            "multiprocessing.reducer.register should have been called"
        )

        vllm_config_registered = False
        for call_args in mock_register.call_args_list:
            # Verify that a reducer for VllmConfig was registered
            if len(call_args[0]) >= 2 and call_args[0][0] == VllmConfig:
                vllm_config_registered = True

                reducer_func = call_args[0][1]
                assert callable(reducer_func), "Reducer function should be callable"
                break

        assert vllm_config_registered, (
            "VllmConfig should have been registered to multiprocessing.reducer"
        )
