"""Smoke test: Qwen3-ASR transcribes sws_vocals.wav -> Chinese lyrics."""

import pathlib
import sys
import tempfile

V2M = pathlib.Path("/mnt/d/Project/OpenSourceProject/Vocal2Midi")
sys.path.insert(0, str(V2M))

# --- monkey-patch device_utils for Linux+CUDA ---
from inference import device_utils as du
du._DEVICE_ALIASES.update({"": "cuda", "cuda": "cuda", "gpu": "cuda"})


def _norm(device, default="cuda"):
    if device is None:
        return default
    return du._DEVICE_ALIASES.get(str(device).strip().lower(), str(device).strip().lower())


du.normalize_runtime_device = _norm


def _resolve(device, *, label="ONNX"):
    import onnxruntime as ort
    if _norm(device) == "cpu":
        return "cpu", ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in set(ort.get_available_providers()):
        print(f"[{label}] CUDA")
        return "cuda", ["CUDAExecutionProvider", "CPUExecutionProvider"]
    print(f"[{label}] CPU")
    return "cpu", ["CPUExecutionProvider"]


du.resolve_onnx_providers = _resolve
du.use_dml = lambda device=None: _norm(device) not in ("cpu",)

# --- load audio + slice (reuse GAME slicer for simplicity) ---
import librosa
import numpy as np

AUDIO = "/opt/work/open-source-project/GAME/inputFiles/sws_vocals.wav"
SR = 44100
print(f"[test] loading {AUDIO} ...", flush=True)
wf, _ = librosa.load(AUDIO, sr=SR, mono=True)
# feed the whole waveform as a single chunk (ASR test only)
chunks = [{"waveform": np.asarray(wf, dtype=np.float32)}]
print(f"[test] audio: {len(wf)/SR:.1f}s, {len(chunks)} chunk", flush=True)

# --- load + transcribe ---
from inference.API.asr_api import load_qwen_model, batch_transcribe_asr

MODEL = "/mnt/d/Project/OpenSourceProject/Vocal2Midi/experiments/Qwen3-ASR-1.7B-dml"
print("[test] loading Qwen3-ASR ...", flush=True)
model = load_qwen_model(MODEL, device="cuda", use_cache=False)
print(f"[test] model loaded.", flush=True)

tmp = pathlib.Path(tempfile.mkdtemp(prefix="qwen_asr_"))
print(f"[test] transcribing (temp_dir={tmp}) ...", flush=True)
results, chunk_indices = batch_transcribe_asr(
    chunks=chunks,
    sr=SR,
    asr_model=model,
    temp_dir_path=tmp,
    asr_batch_size=4,
    language="zh",
    force_subprocess=False,
)
print("=== ASR 结果 ===", flush=True)
full_text = " ".join(r.text for r in results if getattr(r, "text", None))
print(full_text, flush=True)
print("=== 完成 ===", flush=True)
