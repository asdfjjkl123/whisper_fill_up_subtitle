#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import shlex
import subprocess
import time
import os
import gc
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime
from faster_whisper import WhisperModel
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

ASS_DIALOGUE_PREFIX = "Dialogue:"
ASS_COMMENT_PREFIX = "Comment:"


def parse_ass_time_to_seconds(t: str) -> float:
    """
    Parse ASS time format: H:MM:SS.cc (centiseconds) or H:MM:SS.mmm (milliseconds)
    Examples: 0:05:41.45, 1:02:03.5 (rare), 0:00:01.234
    """
    t = t.strip()
    m = re.fullmatch(r"(\d+):(\d{1,2}):(\d{1,2})(?:\.(\d+))?", t)
    if not m:
        raise ValueError(f"Unrecognized ASS time: {t!r}")
    h = int(m.group(1))
    mm = int(m.group(2))
    ss = int(m.group(3))
    frac = m.group(4) or "0"

    # ASS常見是2位(centiseconds)，但也有人用3位(milliseconds)
    if len(frac) == 1:
        frac_seconds = int(frac) / 10.0
    elif len(frac) == 2:
        frac_seconds = int(frac) / 100.0
    else:
        # 3+ digits -> treat as milliseconds-like, clamp to 3 digits precision
        frac_seconds = int(frac[:3]) / 1000.0

    return h * 3600 + mm * 60 + ss + frac_seconds


def read_text_file(path: Path) -> List[str]:
    # Try utf-8-sig first (handles BOM), then utf-8, then system default
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return path.read_text(encoding=enc, errors="strict").splitlines()
        except UnicodeDecodeError:
            continue
    return path.read_text(errors="replace").splitlines()


def clear_directory(path: Path) -> None:
    if not path.exists():
        return
    if not path.is_dir():
        raise NotADirectoryError(f"Expected directory path, got file: {path}")

    for child in path.iterdir():
        if child.is_file() and re.fullmatch(r"\d+\.(?:txt|wav)", child.name):
            child.unlink()


def extract_segments_from_ass_events(lines: List[str]) -> List[Tuple[float, float]]:
    """
    Extract (start_sec, end_sec) from ASS events dialogue/comment lines.
    Assumes Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
    So after the prefix, there are 10 fields; we split with maxsplit=9 to keep Text intact.
    """
    segments: List[Tuple[float, float]] = []

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        if line.startswith(ASS_DIALOGUE_PREFIX):
            payload = line[len(ASS_DIALOGUE_PREFIX):].strip()
        elif line.startswith(ASS_COMMENT_PREFIX):
            payload = line[len(ASS_COMMENT_PREFIX):].strip()
        else:
            continue

        parts = payload.split(",", 9)
        if len(parts) < 3:
            continue

        start_str = parts[1].strip()
        end_str = parts[2].strip()

        try:
            start = parse_ass_time_to_seconds(start_str)
            end = parse_ass_time_to_seconds(end_str)
        except ValueError:
            continue

        if end > start:
            segments.append((start, end))

    return segments


def ffprobe_duration_seconds(media_path: Path) -> Optional[float]:
    """
    Return duration seconds (float) using ffprobe, or None if unavailable.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(media_path)
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, check=True)
        s = p.stdout.strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def run_ffmpeg_split_wavs(
    media_path: Path,
    segments: List[Tuple[float, float]],
    out_dir: Path,
    batch_size: int = 24,
    batch_interval: int = 1,
    debug: bool = False,
    debug_dir: Optional[Path] = None,
) -> List[Path]:
    """
    Use ThreadPoolExecutor to run ffmpeg segment-splitting jobs in parallel.
    At most batch_size ffmpeg processes run concurrently.
    Return only after all jobs have finished successfully.

    batch_interval is kept only for compatibility and is unused.
    """
    import concurrent.futures

    if not segments:
        raise RuntimeError("No valid segments parsed from the txt/ass events.")

    out_dir.mkdir(parents=True, exist_ok=True)

    if debug:
        if debug_dir is None:
            debug_dir = out_dir / "_debug_logs"
        debug_dir.mkdir(parents=True, exist_ok=True)

    batch_size = max(1, int(batch_size))

    jobs: List[dict] = []
    wav_paths: List[Path] = []

    for idx, (st, ed) in enumerate(segments, start=1):
        out_wav = out_dir / f"{idx}.wav"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-i", str(media_path),
            "-ss", f"{st:.6f}",
            "-to", f"{ed:.6f}",
            "-vn",
            "-c:a", "pcm_s16le",
            str(out_wav),
        ]

        job = {
            "idx": idx,
            "start": st,
            "end": ed,
            "wav": out_wav,
            "cmd": cmd,
        }
        if debug:
            job["log"] = debug_dir / f"ffmpeg_{idx:04d}.log"

        jobs.append(job)
        wav_paths.append(out_wav)

    total_jobs = len(jobs)

    def _run_single_job(job: dict) -> int:
        idx = job["idx"]
        out_wav: Path = job["wav"]
        cmd = job["cmd"]

        if debug:
            log_path: Path = job["log"]
            with log_path.open("w", encoding="utf-8", errors="replace") as f:
                f.write(f"[START] {datetime.now().isoformat(timespec='seconds')}\n")
                f.write(f"idx={idx}\n")
                f.write(f"segment_start={job['start']:.6f}\n")
                f.write(f"segment_end={job['end']:.6f}\n")
                f.write(f"wav={out_wav}\n")
                f.write("cmd=" + " ".join(shlex.quote(c) for c in cmd) + "\n")
                f.write("-" * 80 + "\n")

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

            with log_path.open("a", encoding="utf-8", errors="replace") as f:
                f.write(f"[END] {datetime.now().isoformat(timespec='seconds')} returncode={result.returncode}\n")
                f.write("-" * 80 + "\n")
                f.write("[STDOUT]\n")
                f.write((result.stdout or "").rstrip() + "\n")
                f.write("-" * 80 + "\n")
                f.write("[STDERR]\n")
                f.write((result.stderr or "").rstrip() + "\n")
                f.write("-" * 80 + "\n")
                f.write(f"[OUTPUT_EXISTS] {out_wav.exists()}\n")
        else:
            result = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

        if result.returncode != 0:
            err_msg = (result.stderr or "").strip()
            raise RuntimeError(
                f"ffmpeg failed for wav #{idx} -> {out_wav}\n"
                f"returncode={result.returncode}\n"
                f"{err_msg}"
            )

        if not out_wav.exists():
            raise RuntimeError(f"ffmpeg finished but output not found: {out_wav}")

        return idx

    completed = 0
    progress_lock = threading.Lock()
    progress_line_len = 0

    def _print_progress() -> None:
        nonlocal progress_line_len
        line = f"[FFmpeg] finished {completed}/{total_jobs}"
        pad = max(0, progress_line_len - len(line))
        print("\r" + line + (" " * pad), end="", flush=True)
        progress_line_len = len(line)

    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
        future_to_job = {
            executor.submit(_run_single_job, job): job
            for job in jobs
        }

        try:
            for future in concurrent.futures.as_completed(future_to_job):
                future.result()
                with progress_lock:
                    completed += 1
                    _print_progress()
        except Exception:
            for future in future_to_job:
                future.cancel()
            raise

    print()
    return wav_paths


def run_whisper_batch(
    wav_paths: List[Path],
    whisper_model,
    output_format: str = "txt",
    task: str = "transcribe",
    language: str = "ja",
    output_dir: Optional[Path] = None,
    debug: bool = False,
    debug_dir: Optional[Path] = None,
    max_parallel: int = 2,
) -> None:

    if not wav_paths:
        return

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    base_dir = output_dir if output_dir is not None else wav_paths[0].parent

    if debug:
        if debug_dir is None:
            debug_dir = base_dir / "_debug_logs"
        debug_dir.mkdir(parents=True, exist_ok=True)

    max_parallel = max(1, int(max_parallel))
    total = len(wav_paths)
    group_size = (total + max_parallel - 1) // max_parallel  # ceil(total / max_parallel)

    indexed_wavs = list(enumerate(wav_paths, start=1))
    groups: List[List[Tuple[int, Path]]] = [
        indexed_wavs[i:i + group_size]
        for i in range(0, total, group_size)
    ]

    progress_lock = threading.Lock()
    progress_state = {gid: 0 for gid in range(1, len(groups) + 1)}
    progress_totals = {gid: len(gitems) for gid, gitems in enumerate(groups, start=1)}
    progress_line_len = 0

    def render_progress_locked(done_suffix: bool = False) -> None:
        nonlocal progress_line_len
        parts = []
        for gid in range(1, len(groups) + 1):
            parts.append(f"[Group {gid}] {progress_state[gid]}/{progress_totals[gid]}")
        line = "[Whisper] " + " ".join(parts)
        if done_suffix:
            line += " done"
        pad = max(0, progress_line_len - len(line))
        print("\r" + line + (" " * pad), end="", flush=True)
        progress_line_len = len(line)

    with progress_lock:
        render_progress_locked(done_suffix=False)

    def _process_group(group_id: int, group_items: List[Tuple[int, Path]]) -> None:
        for local_idx, (idx, wav) in enumerate(group_items, start=1):
            out_dir = output_dir if output_dir is not None else wav.parent
            txt_path = out_dir / f"{idx}.txt"

            log_path = None
            if debug:
                log_path = debug_dir / f"whisper_{idx:04d}.log"
                with log_path.open("w", encoding="utf-8", errors="replace") as f:
                    f.write(f"[START] {datetime.now().isoformat(timespec='seconds')}\n")
                    f.write(f"group={group_id}\n")
                    f.write(f"group_local_idx={local_idx}\n")
                    f.write(f"group_total={len(group_items)}\n")
                    f.write(f"global_idx={idx}\n")
                    f.write(f"wav={wav}\n")
                    f.write("-" * 80 + "\n")

            start = time.time()

            try:
                segments, info = whisper_model.transcribe(
                    str(wav),
                    task=task,
                    language=language,
                    beam_size=5,
                    condition_on_previous_text=False,
                )

                text = " ".join(seg.text.strip() for seg in segments).strip()
                txt_path.write_text(text + "\n", encoding="utf-8")

                if debug and log_path is not None:
                    with log_path.open("a", encoding="utf-8", errors="replace") as f:
                        f.write(f"[TEXT]\n{text}\n")
                        f.write("-" * 80 + "\n")

            except Exception as e:
                if debug and log_path is not None:
                    with log_path.open("a", encoding="utf-8", errors="replace") as f:
                        f.write(f"[ERROR]\n{repr(e)}\n")
                raise

            end = time.time()

            if debug and log_path is not None:
                with log_path.open("a", encoding="utf-8", errors="replace") as f:
                    f.write(f"[END] duration={end-start:.2f}s\n")
                    f.write(f"[OUTPUT] {txt_path}\n")

            with progress_lock:
                progress_state[group_id] = local_idx
                render_progress_locked(done_suffix=False)

    try:
        with ThreadPoolExecutor(max_workers=len(groups)) as ex:
            futures = [
                ex.submit(_process_group, group_id, group_items)
                for group_id, group_items in enumerate(groups, start=1)
            ]
            for fut in as_completed(futures):
                fut.result()
    finally:
        with progress_lock:
            all_done = all(progress_state[gid] == progress_totals[gid] for gid in progress_state)
            render_progress_locked(done_suffix=all_done)
            print()



def merge_whisper_txts(
    wav_paths: List[Path],
    whisper_output_dir: Optional[Path],
    merged_output_dir: Path,
    merged_name: str,
    ass: bool = True,
    txt_events_path: Optional[Path] = None,
) -> Path:
    """
    When ass=False:
      - Merge all whisper *.txt into one txt file.

    When ass=True (default):
      - Generate an .ass-like events file by taking each Dialogue line from txt_events_path (--txt),
        stripping its trailing Text field, and appending the corresponding whisper text.
      - Line i in events corresponds to i.txt (1-based), i.e. first Dialogue -> 1.txt, second -> 2.txt, etc.

    Notes:
      - Only processes lines starting with "Dialogue:" (and "Comment:" if you later extend).
      - For each whisper txt, if it has multiple lines, join into one line using spaces.
    """
    if not wav_paths:
        raise ValueError("wav_paths is empty; cannot infer whisper txt directory.")

    txt_dir = Path(whisper_output_dir) if whisper_output_dir is not None else wav_paths[0].parent
    if not txt_dir.exists():
        raise FileNotFoundError(f"Whisper txt directory not found: {txt_dir}")

    merged_dir = Path(merged_output_dir)

    txt_files = sorted(
        txt_dir.glob("*.txt"),
        key=lambda p: (
            p.stem.isdigit() is False,
            int(p.stem) if p.stem.isdigit() else p.stem
        ),
    )
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found under: {txt_dir}")

    # Read whisper outputs into normalized one-line strings (index aligned to 1..N)
    whisper_lines: List[str] = []
    for p in txt_files:
        raw = p.read_text(encoding="utf-8", errors="replace")
        parts = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        one_line = " ".join(parts).strip()
        one_line = re.sub(r"\s+", " ", one_line)
        whisper_lines.append(one_line)

    # --- ASS generation mode ---
    if ass:
        if txt_events_path is None:
            raise ValueError("ass=True requires txt_events_path (the --txt file path).")

        lines = read_text_file(Path(txt_events_path))

        out_lines: List[str] = []
        event_idx = 0  # counts Dialogue lines only

        for raw in lines:
            s = raw.lstrip()
            if not s.startswith("Dialogue:"):
                # Keep non-Dialogue lines as-is (headers, styles, etc.)
                out_lines.append(raw)
                continue

            event_idx += 1
            if event_idx > len(whisper_lines):
                raise RuntimeError(
                    f"Not enough whisper txt lines: need >= {event_idx}, got {len(whisper_lines)}"
                )

            # Strip trailing text field from Dialogue line:
            # Dialogue: <9 commas fields>,<Text>
            # We split after "Dialogue:" into at most 10 fields, then drop the last (Text) and rejoin.
            payload = s[len("Dialogue:"):].strip()
            parts = payload.split(",", 9)
            if len(parts) < 10:
                # Unexpected format; keep original line and continue
                out_lines.append(raw)
                continue

            prefix_fields = parts[:9]  # up to Effect (9 fields after Layer)
            prefix = "Dialogue: " + ",".join(prefix_fields) + ","
            new_text = whisper_lines[event_idx - 1]
            out_lines.append(prefix + new_text)

        # If there are extra whisper lines (more txt than Dialogue lines), we ignore by default.
        # If you prefer strictness, turn this into an error.
        out_path = merged_dir / merged_name
        out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        return out_path

    # --- TXT merge mode (original behavior) ---
    out_path = merged_dir / merged_name
    out_path.write_text("\n".join(whisper_lines) + "\n", encoding="utf-8")
    return out_path


def main():
    ap = argparse.ArgumentParser(
        description="Parse ASS Events (Dialogue) lines and cut audio into numbered WAVs, then run faster-whisper (whisper-ctranslate2) per WAV."
    )
    ap.add_argument("-i", "--input", required=True, help="Input media file (video/audio).")
    ap.add_argument("-t", "--txt", required=True, help="ASS/SSA txt file containing Events Dialogue lines.")
    ap.add_argument("--output_wav", default=None, help="Output directory for split WAVs. Default: <input_stem>_chunks")
    ap.add_argument("--padding-ms", type=float, default=50.0, help="Padding added to BOTH start and end (milliseconds). Default 50")
    ap.add_argument("--include-comments", action="store_true", help="Also parse Comment: lines (default: only Dialogue:).")
    ap.add_argument("--ass", action="store_true", help="If set, generate an ASS events file by replacing Dialogue text with whisper outputs.")
    ap.add_argument("--ffmpeg-batch-size", type=int, default=24, help="Max parallel ffmpeg processes (default: 24)")
    ap.add_argument("--max-parallel", type=int, default=2, help="Whisper parallel groups. Default: 2")

    # faster-whisper (whisper-ctranslate2) related options
    ap.add_argument("--no-whisper", action="store_true", help="Only split WAVs; do not run faster-whisper.")
    ap.add_argument("--model", default="large-v3", help="Model name. Default: large-v3")
    ap.add_argument("--model-dir", default=None, help="Model cache dir. Default: None")
    ap.add_argument("--device", default="cuda", help="Device. Default: cuda")
    ap.add_argument("--language", default="ja", help="Language. Default: ja")
    ap.add_argument("--output-format", default="txt", help="Output format. Default: txt")
    ap.add_argument("--task", default="transcribe", help="Task. Default: transcribe")
    ap.add_argument("--whisper-output-dir", default=None, help="Optional: directory for whisper outputs (srt). If omitted, use whisper default (same dir as input wav).")

    ap.add_argument("--debug", action="store_true", help="Enable debug logging for ffmpeg and faster-whisper.")
    ap.add_argument("--debug-dir", default=None, help="Optional: directory for debug logs. Default: <out_dir>/_debug_logs or <whisper_out>/_debug_logs")

    args = ap.parse_args()

    in_path = Path(args.input)
    txt_path = Path(args.txt)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")
    if not txt_path.exists():
        raise SystemExit(f"Txt not found: {txt_path}")

    out_dir = Path(args.output_wav) if args.output_wav else in_path.with_name(in_path.stem + "_chunks")
    whisper_out_dir = Path(args.whisper_output_dir) if args.whisper_output_dir else None
    debug_dir = Path(args.debug_dir) if args.debug_dir else None

    clear_directory(out_dir)
    if whisper_out_dir is not None:
        clear_directory(whisper_out_dir)

    lines = read_text_file(txt_path)

    # Optionally ignore comment lines
    if not args.include_comments:
        filtered = []
        for ln in lines:
            s = ln.lstrip()
            if s.startswith(ASS_COMMENT_PREFIX):
                continue
            filtered.append(ln)
        lines = filtered

    segs = extract_segments_from_ass_events(lines)

    pad = max(0.0, args.padding_ms / 1000.0)

    # Clamp with media duration if possible
    dur = ffprobe_duration_seconds(in_path)
    padded: List[Tuple[float, float]] = []
    for st, ed in segs:
        st2 = max(0.0, st - pad)
        ed2 = ed + pad
        if dur is not None:
            ed2 = min(dur, ed2)
        if ed2 > st2:
            padded.append((st2, ed2))

    if not padded:
        raise SystemExit("No segments after padding/clamping.")

    wavs = run_ffmpeg_split_wavs(in_path, padded, out_dir, debug=args.debug, debug_dir=debug_dir, batch_size=args.ffmpeg_batch_size)
    print(f"\nDone split: {out_dir} ({len(wavs)} wav files)\n")

    if not args.no_whisper:
        
        compute_type = "float16" if args.device == "cuda" else "int8"

        print(f"[Whisper] Loading model: {args.model} (device={args.device})")
        whisper_model = WhisperModel(
            args.model,
            device=args.device,
            compute_type=compute_type,
            download_root=args.model_dir,
        )
        print("[Whisper] Model loaded")

        run_whisper_batch(
            wavs,
            whisper_model=whisper_model,
            output_format=args.output_format,
            task=args.task,
            language=args.language,
            output_dir=whisper_out_dir,
            debug=args.debug,
            debug_dir=debug_dir,
            max_parallel=args.max_parallel,
        )
        print("\nDone faster-whisper.")

        merged_name = f"{in_path.stem}_subtitle.txt"
        merged_path = merge_whisper_txts(
            wav_paths=wavs,
            whisper_output_dir=whisper_out_dir,
            merged_output_dir=in_path.parent,
            merged_name=merged_name,
            ass=args.ass,
            txt_events_path=txt_path,
        )
        print(f"Done merge: {merged_path}")

        del whisper_model
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    main()
