"""Flask web service to test GAME ONNX inference quality.

Upload an audio file -> run ONNX inference -> synthesize MIDI to WAV with
fluidsynth -> play original and synthesized audio side by side for A/B
comparison.

Run:  conda activate game && python webapp/app.py
"""

import json
import os
import pathlib
import subprocess
import threading
import uuid

import numpy as np
import soundfile as sf
from flask import Flask, jsonify, request, send_from_directory, render_template

# Make the project root importable so we can import infer_onnx.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from infer_onnx import (  # noqa: E402
    OnnxInfer,
    infer_audio_to_notes,
    _save_file,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SOUNDFONT = os.environ.get(
    "GAME_SOUNDFONT", "/usr/share/sounds/sf2/FluidR3_GM.sf2",
)
PORT = int(os.environ.get("PORT", "5000"))
MAX_UPLOAD_MB = 100
ALLOWED_EXT = {".wav", ".flac", ".mp3", ".aac", ".ogg", ".m4a"}
# Inference defaults (aligned with infer.py).
SEG_THRESHOLD = 0.2
SEG_RADIUS = 0.02
EST_THRESHOLD = 0.2
T0 = 0.0
NSTEPS = 8
TEMPO = 120
# Default model size used at startup and as the <select> default.
DEFAULT_MODEL = os.environ.get("GAME_MODEL", "medium")

BASE_DIR = pathlib.Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# --------------------------------------------------------------------------- #
# Model registry: discover GAME-*-onnx dirs under onnx/, lazy-load on demand.
# --------------------------------------------------------------------------- #
ONNX_ROOT = PROJECT_ROOT / "onnx"


def _discover_models():
    """Return {display_name: abs_dir} for every GAME-*-onnx dir found."""
    out = {}
    if ONNX_ROOT.is_dir():
        for d in sorted(ONNX_ROOT.iterdir()):
            if d.is_dir() and d.name.endswith("-onnx") \
                    and (d / "config.json").is_file() \
                    and (d / "encoder.onnx").is_file():
                out[d.name] = str(d)
    return out


MODELS = _discover_models()
if not MODELS:
    raise SystemExit(f"[app] no ONNX models found under {ONNX_ROOT}")
# {model_key: OnnxInfer}, loaded lazily and cached (each is 200-400MB).
_MODEL_CACHE: dict[str, OnnxInfer] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def get_infer(model_key):
    """Return the OnnxInfer for model_key, loading + caching it if needed."""
    if model_key not in MODELS:
        raise ValueError(
            f"unknown model '{model_key}'. available: {list(MODELS.keys())}")
    with _MODEL_CACHE_LOCK:
        if model_key not in _MODEL_CACHE:
            print(f"[app] loading model {model_key} ...")
            _MODEL_CACHE[model_key] = OnnxInfer(
                MODELS[model_key],
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            print(f"[app] model {model_key} ready, providers="
                  f"{_MODEL_CACHE[model_key].seg.get_providers()}")
    return _MODEL_CACHE[model_key]


def model_params(infer):
    """Per-model inference params derived from the model's config."""
    ts = [T0 + i * (1 - T0) / NSTEPS for i in range(NSTEPS)]
    seg_radius_frames = round(SEG_RADIUS / infer.timestep)
    return ts, seg_radius_frames


# Preload the default model so the first request is fast.
print(f"[app] available models: {list(MODELS.keys())}")
_DEFAULT_KEY = next((k for k in MODELS if DEFAULT_MODEL in k),
                    next(iter(MODELS)))
get_infer(_DEFAULT_KEY)
print(f"[app] default model: {_DEFAULT_KEY}")

# --------------------------------------------------------------------------- #
# Job storage + serialization (single-worker, in-process)
# --------------------------------------------------------------------------- #
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
INFER_LOCK = threading.Lock()  # serialize inference (model is memory-heavy)


def _allowed_file(filename):
    return pathlib.Path(filename).suffix.lower() in ALLOWED_EXT


def _synth_midi(mid_path, out_wav, samplerate):
    """Render MIDI to WAV using fluidsynth CLI."""
    cmd = [
        "fluidsynth", "-ni", "-F", str(out_wav),
        "-r", str(samplerate), SOUNDFONT, str(mid_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if res.returncode != 0:
        raise RuntimeError(
            f"fluidsynth failed (rc={res.returncode}):\n{res.stderr[:500]}")


def _convert_to_wav(src_path, out_wav, samplerate):
    """Make a browser-playable WAV (mono, 44100) from any supported audio."""
    # soundfile handles wav/flac/ogg; for mp3/aac/m4a it may fail -> librosa.
    try:
        data, sr = sf.read(str(src_path), always_2d=True)
        if data.shape[1] > 1:
            data = data.mean(axis=1, keepdims=True)
        data = data[:, 0]
    except Exception:
        import librosa
        data, sr = librosa.load(str(src_path), sr=samplerate, mono=True)
    if sr != samplerate:
        import librosa
        data = librosa.resample(data.astype(np.float32),
                                orig_sr=sr, target_sr=samplerate)
    sf.write(str(out_wav), data.astype(np.float32), samplerate)


def _run_job(job_id, upload_path, language, model_key):
    """Background worker: infer -> save MIDI -> synth WAV."""
    job = JOBS[job_id]
    try:
        job["status"] = "processing"
        job["stage"] = "inferring"
        job["progress"] = 0
        infer = get_infer(model_key)
        ts, seg_radius_frames = model_params(infer)
        samplerate = infer.samplerate
        lang_id = infer.resolve_language(
            language if language != "auto" else None)

        def progress(idx, total, offset, duration, n_notes):
            job["progress"] = int(idx / total * 100) if total else 100
            job["stage_detail"] = f"segment {idx}/{total}, offset={offset:.1f}s"

        with INFER_LOCK:
            notes = infer_audio_to_notes(
                infer, upload_path, lang_id, ts,
                SEG_THRESHOLD, seg_radius_frames, EST_THRESHOLD,
                progress=progress,
            )

        job["stage"] = "synthesizing"
        job["progress"] = 100
        work_dir = STATIC_DIR / job_id
        work_dir.mkdir(parents=True, exist_ok=True)
        mid_path = work_dir / "output.mid"
        # _save_file uses stem of key for filename; pass stem explicitly via
        # a temp key so the file lands at work_dir/output.mid
        _save_file(notes, "output", work_dir, {"mid"}, TEMPO)
        synth_wav = work_dir / "synth.wav"
        _synth_midi(mid_path, synth_wav, samplerate)

        # also write a readable txt
        _save_file(notes, "output", work_dir, {"txt"}, TEMPO)

        # convert original upload to a playable wav
        original_wav = work_dir / "original.wav"
        _convert_to_wav(upload_path, original_wav, samplerate)

        # stats
        pitches = [n.pitch for n in notes]
        onsets = [n.onset for n in notes]
        job["stats"] = {
            "note_count": len(notes),
            "pitch_min": round(min(pitches), 1) if pitches else None,
            "pitch_max": round(max(pitches), 1) if pitches else None,
            "duration_sec": round(max(onsets) if onsets else 0, 2),
        }
        job["status"] = "done"
        job["progress"] = 100
    except Exception as e:
        job["status"] = "error"
        job["error"] = f"{type(e).__name__}: {e}"
        import traceback
        traceback.print_exc()


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    languages = list(get_infer(_DEFAULT_KEY).languages.keys()) or []
    # model display names: full dir name -> short size label
    models = [{"key": k, "name": k} for k in MODELS]
    return render_template("index.html", languages=languages, models=models,
                           default_model=_DEFAULT_KEY)


@app.route("/upload", methods=["POST"])
def upload():
    if "audio" not in request.files:
        return jsonify({"error": "no audio file"}), 400
    f = request.files["audio"]
    if not f.filename or not _allowed_file(f.filename):
        return jsonify({"error": f"unsupported file (allowed: "
                                  f"{sorted(ALLOWED_EXT)})"}), 400
    language = request.form.get("language", "auto")
    model_key = request.form.get("model", _DEFAULT_KEY)
    if model_key not in MODELS:
        return jsonify({"error": f"unknown model '{model_key}'. "
                                  f"available: {list(MODELS)}"}), 400

    job_id = uuid.uuid4().hex[:12]
    work_dir = STATIC_DIR / job_id
    work_dir.mkdir(parents=True, exist_ok=True)
    upload_path = work_dir / f"upload{pathlib.Path(f.filename).suffix.lower()}"
    f.save(str(upload_path))

    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "status": "pending",
            "progress": 0,
            "stage": "queued",
            "stage_detail": "",
            "stats": None,
            "error": None,
        }

    threading.Thread(target=_run_job, args=(job_id, str(upload_path),
                                             language, model_key),
                     daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "unknown job"}), 404
    return jsonify({
        "status": job["status"],
        "progress": job["progress"],
        "stage": job["stage"],
        "stage_detail": job["stage_detail"],
        "error": job["error"],
    })


@app.route("/result/<job_id>")
def result(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "unknown job"}), 404
    if job["status"] != "done":
        return jsonify({"error": f"job not done (status={job['status']})"}), 409
    return jsonify({
        "original_url": f"/audio/{job_id}/original.wav",
        "synth_url": f"/audio/{job_id}/synth.wav",
        "midi_url": f"/audio/{job_id}/output.mid",
        "txt_url": f"/audio/{job_id}/output.txt",
        "stats": job["stats"],
    })


@app.route("/audio/<job_id>/<name>")
def audio(job_id, name):
    # prevent path traversal
    if "/" in name or ".." in name:
        return jsonify({"error": "bad name"}), 400
    d = STATIC_DIR / job_id
    return send_from_directory(str(d), name)


if __name__ == "__main__":
    print(f"[app] serving on http://127.0.0.1:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
