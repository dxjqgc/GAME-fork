"""End-to-end: run Vocal2Midi's full auto-lyric hybrid pipeline on sws_vocals.wav
to produce a lyric-aligned MIDI.

Pipeline: audio -> Qwen3-ASR (lyrics) -> LyricFA (G2P/.lab) -> HubertFA (forced
alignment) -> GAME (known_boundaries constrained extraction) -> lyric MIDI.
"""

import pathlib
import sys

V2M = pathlib.Path("/mnt/d/Project/OpenSourceProject/Vocal2Midi")
sys.path.insert(0, str(V2M))

# --- monkey-patch device_utils for Linux+CUDA (validated in ASR test) ---
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
    avail = set(ort.get_available_providers())
    if "CUDAExecutionProvider" in avail:
        print(f"[{label}] CUDA")
        return "cuda", ["CUDAExecutionProvider", "CPUExecutionProvider"]
    print(f"[{label}] CPU")
    return "cpu", ["CPUExecutionProvider"]


du.resolve_onnx_providers = _resolve
du.use_dml = lambda device=None: _norm(device) not in ("cpu",)


def run():
    from inference.pipeline.auto_lyric_hybrid import auto_lyric_hybrid_pipeline

    EXP = "/mnt/d/Project/OpenSourceProject/Vocal2Midi/experiments"
    auto_lyric_hybrid_pipeline(
        audio_path="/opt/work/open-source-project/GAME/inputFiles/sws_vocals.wav",
        output_filename="sws_vocals_lyric",
        game_model_dir=f"{EXP}/GAME-1.0.3-medium-onnx",
        device="cuda",
        hfa_model_dir=f"{EXP}/1218_hfa_model_new_dict",
        asr_model_path=f"{EXP}/Qwen3-ASR-1.7B-dml",
        ts=[0.0 + i * (1 - 0.0) / 8 for i in range(8)],
        language="zh",
        lyric_output_mode="pinyin",
        original_lyrics="",            # no external lyrics; ASR auto-recognizes
        output_dir=pathlib.Path("/tmp/lyric_midi_out"),
        output_formats=["mid", "txt"],
        slicing_method="default",
        tempo=120,
        quantization_step=0,
        pitch_format="name",
        round_pitch=False,
        quantization_mode="simple",
        seg_threshold=0.2,
        seg_radius=0.02,
        est_threshold=0.2,
        batch_size=4,
        asr_batch_size=4,
        output_lyrics=True,
        output_pitch_curve=False,
        debug_mode=False,
    )
    print("\n=== DONE: check /tmp/lyric_midi_out/ ===", flush=True)


if __name__ == "__main__":
    run()
