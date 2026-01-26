"""
Environment test for Swin-VALLR.
Prints a single parseable result line for setup validation.
"""

import json
import traceback
from typing import Tuple

import torch

from logging_utils import setup_logging, log_exception, log_system_info
from artifact_utils import get_useful_dir, save_json


LOGGER = setup_logging("env_test", console_level=None)
try:
    from backend_manager import get_device_info, get_backend, get_device
except Exception as exc:
    print("ENV_TEST_RESULT: " + json.dumps({
        "ok": False,
        "error": f"failed_to_import_backend_manager: {exc}"
    }))
    raise


def _smoke_test(device: torch.device) -> Tuple[bool, str]:
    """Run a small matmul on the target device."""
    try:
        x = torch.randn(2, 2, device=device)
        y = torch.randn(2, 2, device=device)
        z = x @ y
        _ = z.sum().item()
        return True, ""
    except Exception as exc:
        log_exception(LOGGER, "Smoke test failed.")
        return False, str(exc)


def _collect_env_result() -> dict:
    info = get_device_info()
    backend = get_backend()
    device = get_device()

    ok, error = _smoke_test(device)

    result = {
        "ok": ok,
        "backend": backend,
        "device": str(device),
        "backend_name": info.get("backend_name"),
        "mixed_precision": info.get("mixed_precision"),
        "compile_available": info.get("compile_available"),
        "gpu_name": info.get("gpu_name"),
        "torch_version": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "torch_hip_version": torch.version.hip,
        "error": error or None
    }

    return result


if __name__ == "__main__":
    try:
        LOGGER.info("Running env_test.")
        log_system_info(LOGGER)
        result = _collect_env_result()
        try:
            root = get_useful_dir("env_test", "env_result")
            save_json(root / "env_test_result.json", result)
        except Exception:
            log_exception(LOGGER, "Failed to save env_test artifacts.")
        print("ENV_TEST_RESULT: " + json.dumps(result))
    except Exception as exc:
        log_exception(LOGGER, "env_test failed with an exception.")
        print("ENV_TEST_RESULT: " + json.dumps({
            "ok": False,
            "error": f"unhandled_exception: {exc}"
        }))
        traceback.print_exc()
