"""Monkey-patch Vocal2Midi's device_utils for Linux + CUDA.

Vocal2Midi targets Windows + DirectML by default. This patch makes `cuda`
resolve to CUDAExecutionProvider instead of being remapped to `dml`.

Must be imported at module top-level (before any Vocal2Midi inference call) so
that spawn'd ASR worker subprocesses inherit the patch on re-import.
"""

import os
import pathlib

import sys

V2M_DIR = pathlib.Path(os.environ.get(
    "V2M_DIR", "/mnt/d/Project/OpenSourceProject/Vocal2Midi"))
if str(V2M_DIR) not in sys.path:
    sys.path.insert(0, str(V2M_DIR))

from inference import device_utils as du  # noqa: E402

# stop remapping cuda -> dml; keep cuda as cuda
du._DEVICE_ALIASES.update({
    "": "cuda",
    "cuda": "cuda",
    "gpu": "cuda",
})


def _normalize_runtime_device(device, default="cuda"):
    if device is None:
        return default
    return du._DEVICE_ALIASES.get(
        str(device).strip().lower(), str(device).strip().lower())


du.normalize_runtime_device = _normalize_runtime_device


def _resolve_onnx_providers(device, *, label="ONNX"):
    import onnxruntime as ort
    if _normalize_runtime_device(device) == "cpu":
        return "cpu", ["CPUExecutionProvider"]
    avail = set(ort.get_available_providers())
    if "CUDAExecutionProvider" in avail:
        return "cuda", ["CUDAExecutionProvider", "CPUExecutionProvider"]
    print(f"[{label}] CUDA unavailable; using CPU")
    return "cpu", ["CPUExecutionProvider"]


du.resolve_onnx_providers = _resolve_onnx_providers
du.use_dml = lambda device=None: _normalize_runtime_device(device) not in ("cpu",)

# Re-export the pipeline module path for convenience.
V2M_EXPERIMENTS = pathlib.Path(os.environ.get(
    "V2M_EXP_DIR", str(V2M_DIR / "experiments")))

