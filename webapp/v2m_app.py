"""Flask micro-service: dry vocal -> [{onset, offset, pitch, lyric}].

Wraps Vocal2Midi's auto_lyric_hybrid_pipeline (Qwen3-ASR + LyricFA + HubertFA
+ GAME) so the GuitarSheetGenerator backend can call it over HTTP without
merging the conflicting torch/tensorflow environments.

Pipeline produces a .txt file (onset\\toffset\\tpitch\\tlyric). We request
pitch_format="number" (raw float MIDI pitch) and round_pitch=False so the
parsed notes are exact, then return JSON. No Vocal2Midi internals are touched.

Run:
    conda activate game
    cd /opt/work/open-source-project/GAME
    python webapp/v2m_app.py

Env overrides:
    V2M_DIR        Vocal2Midi repo root (default /mnt/d/.../Vocal2Midi)
    V2M_EXP_DIR    experiments dir with GAME/HFA/Qwen3-ASR models
    PORT           listen port (default 5010)
    V2M_DEVICE     cuda|cpu (default cuda)
"""

import json
import os
import pathlib
import re
import threading
import time
import uuid

from flask import Flask, jsonify, request

# Make the GAME project root importable so `import webapp.v2m_patch` works
# when run as `python webapp/v2m_app.py` from the repo root.
_GAME_ROOT = pathlib.Path(__file__).resolve().parent.parent
import sys
if str(_GAME_ROOT) not in sys.path:
    sys.path.insert(0, str(_GAME_ROOT))

# device monkey-patch must run before any V2M inference import so spawn'd
# ASR subprocesses inherit the patch on re-import.
import webapp.v2m_patch  # noqa: F401

from inference.pipeline.auto_lyric_hybrid import auto_lyric_hybrid_pipeline  # noqa: E402

PORT = int(os.environ.get("PORT", "5010"))
V2M_EXP_DIR = pathlib.Path(webapp.v2m_patch.V2M_EXPERIMENTS)
V2M_DEVICE = os.environ.get("V2M_DEVICE", "cuda")
# Slicing: "default" (Slicer2 silence detection) has NO max-length cap and
# ignores min/max_len_sec — the bounds monkey-patch in slicer_api only wraps
# smart/heuristic/grid. Spleeter-separated vocals carry residual noise that
# rarely dips below -30dB, so a whole song can collapse into one giant chunk
# (~94s here) and Qwen3-ASR's segmenter attention then tries to allocate
# ~11GB (batch*heads*T^2) and OOMs. "heuristic" hard-splits anything longer
# than slice_max_sec at local energy minima, so the cap is guaranteed.
V2M_SLICE_METHOD = os.environ.get("V2M_SLICE_METHOD", "heuristic")
V2M_SLICE_MIN_SEC = float(os.environ.get("V2M_SLICE_MIN_SEC", "5.0"))
V2M_SLICE_MAX_SEC = float(os.environ.get("V2M_SLICE_MAX_SEC", "10.0"))
MAX_UPLOAD_MB = 100
ALLOWED_EXT = {".wav", ".flac", ".mp3", ".aac", ".ogg", ".m4a"}

BASE_DIR = pathlib.Path(__file__).resolve().parent
WORK_DIR = BASE_DIR / "v2m_jobs"
WORK_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# --------------------------------------------------------------------------- #
# Model dir resolution (validate once at import; models load lazily per call)
# --------------------------------------------------------------------------- #
GAME_MODEL_DIR = V2M_EXP_DIR / "GAME-1.0.3-medium-onnx"
HFA_MODEL_DIR = V2M_EXP_DIR / "1218_hfa_model_new_dict"
ASR_MODEL_PATH = V2M_EXP_DIR / "Qwen3-ASR-1.7B-dml"

for _d, _label in [(GAME_MODEL_DIR, "GAME"), (HFA_MODEL_DIR, "HubertFA"),
                   (ASR_MODEL_PATH, "Qwen3-ASR")]:
    if not _d.exists():
        raise SystemExit(f"[v2m] missing {_label} model dir: {_d}")

# --------------------------------------------------------------------------- #
# Job storage (single-worker, in-process; inference is serialized below)
# --------------------------------------------------------------------------- #
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
INFER_LOCK = threading.Lock()  # serialize heavy pipeline (model memory)

# D3PM sampling schedule (aligned with test_full_pipeline.py).
NSTEPS = 8
T0 = 0.0
TS = [T0 + i * (1 - T0) / NSTEPS for i in range(NSTEPS)]


def _allowed_file(filename: str) -> bool:
    return pathlib.Path(filename).suffix.lower() in ALLOWED_EXT


def _parse_notes_txt(txt_path: pathlib.Path) -> list[dict]:
    """Parse V2M .txt (pitch_format=number) -> list of note dicts.

    Each line: onset\toffset\tpitch(\\tlyric)?
    pitch is raw float MIDI (e.g. 56.63) since round_pitch=False.
    """
    notes: list[dict] = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        try:
            onset = float(parts[0])
            offset = float(parts[1])
            pitch = float(parts[2])
        except ValueError:
            continue
        lyric = parts[3] if len(parts) >= 4 else ""
        notes.append({
            "onset": round(onset, 4),
            "offset": round(offset, 4),
            "pitch": round(pitch, 4),
            "lyric": lyric,
        })
    return notes


def _run_job(job_id: str, audio_path: str, language: str,
             output_lyrics: bool) -> None:
    """Background worker: run the hybrid pipeline, parse notes, store JSON."""
    job = JOBS[job_id]
    try:
        job["status"] = "processing"
        job["stage"] = "inferring"

        out_dir = WORK_DIR / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        output_key = "vocal"

        with INFER_LOCK:
            auto_lyric_hybrid_pipeline(
                audio_path=audio_path,
                output_filename=output_key,
                game_model_dir=str(GAME_MODEL_DIR),
                device=V2M_DEVICE,
                hfa_model_dir=str(HFA_MODEL_DIR),
                asr_model_path=str(ASR_MODEL_PATH),
                ts=TS,
                language=language,
                lyric_output_mode="hanzi" if language == "zh" else "default",
                original_lyrics="",
                output_dir=out_dir,
                # number format -> raw float MIDI pitch in the txt
                output_formats=["txt"],
                slicing_method=V2M_SLICE_METHOD,
                slice_min_sec=V2M_SLICE_MIN_SEC,
                slice_max_sec=V2M_SLICE_MAX_SEC,
                tempo=120.0,
                quantization_step=0,
                pitch_format="number",
                round_pitch=False,
                quantization_mode="simple",
                seg_threshold=0.2,
                seg_radius=0.02,
                est_threshold=0.2,
                batch_size=4,
                asr_batch_size=4,
                output_lyrics=output_lyrics,
                output_pitch_curve=False,
                debug_mode=False,
            )

        txt_path = out_dir / f"{output_key}.txt"
        if not txt_path.is_file():
            raise RuntimeError(f"pipeline produced no txt at {txt_path}")

        notes = _parse_notes_txt(txt_path)
        job["notes"] = notes
        job["note_count"] = len(notes)
        job["status"] = "done"
        job["finished_at"] = time.time()
    except Exception as e:
        import traceback
        job["status"] = "error"
        job["error"] = f"{type(e).__name__}: {e}"
        job["traceback"] = traceback.format_exc()
        job["finished_at"] = time.time()
    finally:
        # The parsed notes live in the in-process JOBS dict; drop the on-disk
        # upload + pipeline artifacts so long-running services don't leak disk
        # (each upload is tens of MB).
        try:
            import shutil
            shutil.rmtree(out_dir, ignore_errors=True)
        except Exception:
            pass


# Keep finished JOBS around long enough for the client to poll /result, then
# drop them so memory does not grow unbounded (each notes list is ~hundreds of
# dicts). 30 min is far longer than any realistic poll cadence.
JOB_TTL_SEC = 30 * 60


def _gc_loop() -> None:
    while True:
        time.sleep(60)
        now = time.time()
        with JOBS_LOCK:
            stale = [jid for jid, j in JOBS.items()
                     if j.get("finished_at") and now - j["finished_at"] > JOB_TTL_SEC]
            for jid in stale:
                JOBS.pop(jid, None)


threading.Thread(target=_gc_loop, daemon=True).start()


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.route("/health")
def health():
    return jsonify({"status": "ok", "device": V2M_DEVICE,
                    "models": {
                        "game": str(GAME_MODEL_DIR),
                        "hfa": str(HFA_MODEL_DIR),
                        "asr": str(ASR_MODEL_PATH),
                    }})


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "no audio file"}), 400
    f = request.files["audio"]
    if not f.filename or not _allowed_file(f.filename):
        return jsonify({"error": f"unsupported file (allowed: "
                                  f"{sorted(ALLOWED_EXT)})"}), 400
    language = request.form.get("language", "zh").lower()
    output_lyrics = request.form.get("output_lyrics", "true").lower() != "false"

    job_id = uuid.uuid4().hex[:12]
    work_dir = WORK_DIR / job_id
    work_dir.mkdir(parents=True, exist_ok=True)
    upload_path = work_dir / f"upload{pathlib.Path(f.filename).suffix.lower()}"
    f.save(str(upload_path))

    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "status": "pending",
            "stage": "queued",
            "notes": None,
            "note_count": None,
            "error": None,
        }

    threading.Thread(target=_run_job, args=(job_id, str(upload_path),
                                             language, output_lyrics),
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
        "stage": job.get("stage"),
        "note_count": job.get("note_count"),
        "error": job.get("error"),
    })


@app.route("/result/<job_id>")
def result(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "unknown job"}), 404
    if job["status"] != "done":
        return jsonify({"error": f"job not done (status={job['status']})",
                        "detail": job.get("error")}), 409
    return jsonify({
        "notes": job["notes"],
        "note_count": job["note_count"],
    })


if __name__ == "__main__":
    print(f"[v2m] serving on http://127.0.0.1:{PORT}  device={V2M_DEVICE}")
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
