"""ONNX inference script for GAME (Generative Adaptive MIDI Extractor).

The repository's `infer.py` only supports PyTorch `.pt` models. This script runs
inference using the exported ONNX models instead, following the workflow
documented in ONNX.md. It does not depend on Lightning/PyTorch.

Pipeline per audio slice (B=1):
    encoder  -> segmenter (D3PM loop) -> bd2dur -> estimator
    waveform    x_seg, maskT            durations, maskN   presence, scores

Audio slicing, MIDI/TXT writing logic are replicated from inference/data.py,
inference/slicer2.py and inference/callbacks.py to match `infer.py`'s behavior.
"""

import argparse
import json
import pathlib
from dataclasses import dataclass

import librosa
import mido
import numpy as np
import onnxruntime as ort

from inference.slicer2 import Slicer


# --------------------------------------------------------------------------- #
# Note bookkeeping (replicated from inference/callbacks.py)
# --------------------------------------------------------------------------- #
@dataclass
class _NoteInfo:
    onset: float   # seconds, absolute (segment offset already added)
    offset: float  # seconds, absolute
    pitch: float   # semitones, A4 = 69


def _process_item(durations, presence, scores, offset, length):
    """Accumulate one slice's output into _NoteInfo list, mirroring
    SaveCombinedFileCallback._process_item (callbacks.py:100-126)."""
    # onset  = cumsum with a leading zero -> start of each note
    # offset = cumsum without leading zero -> end of each note
    d = np.asarray(durations, dtype=np.float64)
    note_onset = np.concatenate([[0.0], d]).cumsum()
    note_offset = d.cumsum()
    # clamp to slice length, then shift to absolute timeline by segment offset
    note_onset = np.clip(note_onset, a_min=None, a_max=length) + offset
    note_offset = np.clip(note_offset, a_min=None, a_max=length) + offset

    presence = np.asarray(presence, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    notes = []
    for onset, off, pitch, valid in zip(
            note_onset.tolist(), note_offset.tolist(),
            scores.tolist(), presence.tolist()):
        if off - onset <= 0:
            continue
        if not valid:  # unvoiced / rest -> dropped (not written)
            continue
        notes.append(_NoteInfo(onset=onset, offset=off, pitch=pitch))
    return notes


def _save_file(notes, key_stem, output_dir, output_formats, tempo):
    """Sort, enforce monotonic time, then flush to disk."""
    sorted_notes = sorted(notes, key=lambda x: (x.onset, x.offset, x.pitch))
    last_time = 0.0
    cleaned = []
    for note in sorted_notes:
        note.onset = max(note.onset, last_time)
        note.offset = max(note.offset, note.onset)
        if note.offset <= note.onset:
            continue
        last_time = note.offset
        cleaned.append(note)

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if "mid" in output_formats:
        _flush_midi(cleaned, output_dir / f"{key_stem}.mid", tempo)
    if "txt" in output_formats:
        _flush_text(cleaned, output_dir / f"{key_stem}.txt", "txt")
    if "csv" in output_formats:
        _flush_text(cleaned, output_dir / f"{key_stem}.csv", "csv")


def _flush_midi(notes, filepath, tempo):
    """Write a MIDI file (callbacks.py:161-185). ticks_per_beat=480 default,
    seconds->ticks = round(sec * tempo * 8)."""
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo), time=0))
    last_time = 0
    for note in notes:
        onset_ticks = round(note.onset * tempo * 8)
        offset_ticks = round(note.offset * tempo * 8)
        if offset_ticks <= onset_ticks:
            continue
        midi_pitch = int(round(note.pitch))
        track.append(mido.Message("note_on", note=midi_pitch,
                                  time=onset_ticks - last_time))
        track.append(mido.Message("note_off", note=midi_pitch,
                                  time=offset_ticks - onset_ticks))
        last_time = offset_ticks
    midi_file = mido.MidiFile(charset="utf8")
    midi_file.tracks.append(track)
    midi_file.save(filepath)


def _flush_text(notes, filepath, file_format):
    """Write txt/csv (callbacks.py:200-236). pitch uses librosa.midi_to_note
    with cents."""
    onset_list = [f"{n.onset:.3f}" for n in notes]
    offset_list = [f"{n.offset:.3f}" for n in notes]
    pitch_list = [librosa.midi_to_note(n.pitch, unicode=False, cents=True)
                  for n in notes]
    if file_format == "txt":
        lines = [f"{o}\t{off}\t{p}"
                 for o, off, p in zip(onset_list, offset_list, pitch_list)]
        filepath.write_text("\n".join(lines), encoding="utf8")
    else:  # csv
        import csv
        with open(filepath, "w", encoding="utf8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["onset", "offset", "pitch"])
            writer.writeheader()
            for o, off, p in zip(onset_list, offset_list, pitch_list):
                writer.writerow({"onset": o, "offset": off, "pitch": p})


# --------------------------------------------------------------------------- #
# ONNX model wrapper
# --------------------------------------------------------------------------- #
class OnnxInfer:
    def __init__(self, model_dir, providers=None):
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.model_dir = pathlib.Path(model_dir)

        def _sess(name):
            return ort.InferenceSession(
                str(self.model_dir / f"{name}.onnx"),
                providers=providers,
            )

        self.enc = _sess("encoder")
        self.seg = _sess("segmenter")
        self.bd2dur = _sess("bd2dur")
        self.est = _sess("estimator")
        # dur2bd not needed for the pure-extraction path (no known durations)

        cfg = json.loads((self.model_dir / "config.json").read_text(encoding="utf8"))
        self.samplerate = int(cfg["samplerate"])
        self.timestep = float(cfg["timestep"])
        self.languages = cfg.get("languages") or {}
        self.supports_loop = bool(cfg.get("loop", False))

        used = set(self.seg.get_providers())
        print(f"[onnx] providers in use: {sorted(used)}")

    def resolve_language(self, language):
        if not language:
            return 0  # universal / unset
        if not self.languages:
            return 0
        if language not in self.languages:
            raise ValueError(
                f"Language '{language}' not supported. Available: "
                f"{', '.join(self.languages.keys())}")
        return int(self.languages[language])

    def run_slice(self, waveform, duration, language_id, ts,
                  seg_threshold, seg_radius, est_threshold):
        """Run the full pipeline on one slice. B=1.

        waveform: float32 [L] -> [1, L]
        duration: float seconds
        Returns (durations[B,N], presence[B,N], scores[B,N]) as numpy.
        """
        wav = np.asarray(waveform, dtype=np.float32)[None, :]        # [1, L]
        dur = np.array([float(duration)], dtype=np.float32)          # [1]
        lang = np.array([int(language_id)], dtype=np.int64)          # [1]

        # ---- encoder ----
        x_seg, x_est, maskT = self.enc.run(
            None, {"waveform": wav, "duration": dur})

        # ---- segmenter (D3PM sampling loop) ----
        T = maskT.shape[1]
        known_boundaries = np.zeros((1, T), dtype=bool)
        boundaries = known_boundaries
        # 0-d ndarrays for scalar inputs (onnxruntime rejects numpy scalars)
        thr = np.array(seg_threshold, dtype=np.float32)
        radius = np.array(seg_radius, dtype=np.int64)
        if self.supports_loop and ts:
            for t_val in ts:
                t = np.array([float(t_val)], dtype=np.float32)      # [1]
                inputs = {
                    "x_seg": x_seg,
                    "language": lang,
                    "known_boundaries": known_boundaries,
                    "prev_boundaries": boundaries,
                    "t": t,
                    "maskT": maskT,
                    "threshold": thr,
                    "radius": radius,
                }
                (boundaries,) = self.seg.run(None, inputs)
        else:
            # completion-style / no loop: single forward
            inputs = {
                "x_seg": x_seg, "language": lang,
                "known_boundaries": known_boundaries,
                "prev_boundaries": known_boundaries,
                "t": np.array([0.0], dtype=np.float32),
                "maskT": maskT, "threshold": thr, "radius": radius,
            }
            (boundaries,) = self.seg.run(None, inputs)

        # ---- bd2dur: boundaries -> durations (seconds) + maskN ----
        durations, maskN = self.bd2dur.run(
            None, {"boundaries": boundaries, "maskT": maskT})

        # ---- estimator: x_est, boundaries, maskT, maskN -> presence, scores
        est_thr = np.array(est_threshold, dtype=np.float32)
        presence, scores = self.est.run(None, {
            "x_est": x_est,
            "boundaries": boundaries,
            "maskT": maskT,
            "maskN": maskN,
            "threshold": est_thr,
        })
        return durations, presence, scores


# --------------------------------------------------------------------------- #
# Audio loading + slicing
# --------------------------------------------------------------------------- #
def load_and_slice(filepath, samplerate):
    """Load audio at samplerate (mono), slice by silence.
    Returns list of (waveform_np, offset_sec, duration_sec)."""
    waveform, _ = librosa.load(str(filepath), sr=samplerate, mono=True)
    slicer = Slicer(
        sr=samplerate,
        threshold=-40.,
        min_length=1000,
        min_interval=200,
        max_sil_kept=100,
    )
    chunks = slicer.slice(waveform)
    out = []
    for c in chunks:
        w = np.asarray(c["waveform"], dtype=np.float32)
        off = float(c["offset"])
        d = w.shape[0] / samplerate
        out.append((w, off, d))
    return out


# --------------------------------------------------------------------------- #
# Reusable inference driver (shared by CLI and web app)
# --------------------------------------------------------------------------- #
def infer_audio_to_notes(infer, filepath, lang_id, ts,
                         seg_threshold, seg_radius_frames, est_threshold,
                         progress=None):
    """Load audio, slice, run inference per slice, return list[_NoteInfo].

    Args:
        infer: OnnxInfer instance.
        filepath: path to audio file.
        lang_id, ts, seg_threshold, seg_radius_frames, est_threshold:
            inference params (see main()).
        progress: optional callable(idx, total, segment_offset, segment_dur,
                     n_notes) called after each slice for progress reporting.

    Returns: list[_NoteInfo] (onset/offset already on absolute timeline).
    """
    slices = load_and_slice(filepath, infer.samplerate)
    all_notes = []
    for idx, (wav, offset, duration) in enumerate(slices):
        durations, presence, scores = infer.run_slice(
            wav, duration, lang_id, ts,
            seg_threshold, seg_radius_frames, est_threshold)
        all_notes.extend(_process_item(
            durations[0], presence[0], scores[0], offset, duration))
        if progress is not None:
            progress(idx + 1, len(slices), offset, duration, durations.shape[1])
    return all_notes


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="ONNX inference for GAME (no PyTorch needed).")
    ap.add_argument("path", type=pathlib.Path,
                    help="Audio file or directory.")
    ap.add_argument("-m", "--model", type=pathlib.Path, required=True,
                    help="ONNX model directory (contains *.onnx + config.json).")
    ap.add_argument("-l", "--language", type=str, default=None,
                    help="Language code (e.g. zh, en, ja, yue).")
    ap.add_argument("--tempo", type=float, default=120,
                    help="Tempo (BPM) for MIDI output. [default: 120]")
    ap.add_argument("--seg-threshold", type=float, default=0.2,
                    help="Boundary decoding threshold. [default: 0.2]")
    ap.add_argument("--seg-radius", type=float, default=0.02,
                    help="Boundary decoding radius in seconds "
                         "(converted to frames). [default: 0.02]")
    ap.add_argument("--est-threshold", type=float, default=0.2,
                    help="Note presence threshold. [default: 0.2]")
    ap.add_argument("--t0", type=float, default=0.0,
                    help="D3PM starting t. [default: 0.0]")
    ap.add_argument("--nsteps", type=int, default=8,
                    help="D3PM sampling steps. [default: 8]")
    ap.add_argument("--output-formats", type=str, default="mid",
                    help="Comma-separated: mid,txt,csv. [default: mid]")
    ap.add_argument("--output-dir", type=pathlib.Path, default=None,
                    help="Output directory (default: beside each input file).")
    ap.add_argument("--glob", type=str, default=None,
                    help="Glob pattern to filter files in a directory.")
    ap.add_argument("--cpu", action="store_true",
                    help="Force CPU execution (disable CUDA).")
    args = ap.parse_args()

    output_formats = {f.strip().lower() for f in args.output_formats.split(",")}
    valid = {"mid", "txt", "csv"}
    if not output_formats.issubset(valid):
        ap.error(f"Unsupported formats: {output_formats - valid}. "
                 f"Supported: {valid}")

    # build filemap {key: path}
    if args.path.is_file():
        filemap = {args.path.name: args.path}
    elif args.path.is_dir():
        if args.glob:
            files = [f for f in args.path.rglob(args.glob) if f.is_file()]
        else:
            exts = {".wav", ".flac", ".mp3", ".aac", ".ogg"}
            files = [f for f in args.path.rglob("*")
                     if f.is_file() and f.suffix.lower() in exts]
        filemap = {f.relative_to(args.path).as_posix(): f for f in files}
        if not filemap:
            ap.error(f"No audio files found in: {args.path}")
    else:
        ap.error(f"Path does not exist: {args.path}")

    providers = (["CPUExecutionProvider"] if args.cpu
                 else ["CUDAExecutionProvider", "CPUExecutionProvider"])
    infer = OnnxInfer(args.model, providers=providers)
    lang_id = infer.resolve_language(args.language)

    # D3PM t schedule (schema.py:343-349): ts = [t0 + i*step for i in range(nsteps)]
    step = (1 - args.t0) / args.nsteps
    ts = [args.t0 + i * step for i in range(args.nsteps)]
    seg_radius_frames = round(args.seg_radius / infer.timestep)

    print(f"[infer] language={args.language!r}->{lang_id}, "
          f"ts={[round(t, 4) for t in ts]}, "
          f"seg_radius={seg_radius_frames} frames, "
          f"samplerate={infer.samplerate}")

    for key, filepath in filemap.items():
        print(f"[infer] processing {key} ...")

        def _progress(idx, total, offset, duration, n_notes):
            print(f"[infer]   segment {idx}/{total}: "
                  f"offset={offset:.2f}s dur={duration:.2f}s "
                  f"notes={n_notes}")

        all_notes = infer_audio_to_notes(
            infer, filepath, lang_id, ts,
            args.seg_threshold, seg_radius_frames, args.est_threshold,
            progress=_progress)

        key_stem = pathlib.Path(key).stem
        out_dir = args.output_dir if args.output_dir is not None \
            else filepath.parent
        _save_file(all_notes, key_stem, out_dir, output_formats, args.tempo)
        print(f"[infer]   saved {len(all_notes)} notes -> {out_dir}/{key_stem}"
              f".{','.join(sorted(output_formats))}")


if __name__ == "__main__":
    main()
