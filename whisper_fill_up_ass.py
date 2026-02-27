#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import shlex
import subprocess
import time
import os
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime

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
    batch_size: int = 16,
    batch_interval: int = 1,
    debug: bool = False,
    debug_dir: Optional[Path] = None,
) -> List[Path]:
    """
    Fire-and-forget batching (default):
      - Launch batch_size ffmpeg processes concurrently
      - Immediately sleep batch_interval seconds
      - Launch next batch (no waiting)

    If debug=True:
      - Capture stdout/stderr for EACH ffmpeg job into log files
      - Wait for all ffmpeg jobs to finish before returning (to ensure logs are complete)
    """
    if not segments:
        raise RuntimeError("No valid segments parsed from the txt/ass events.")

    out_dir.mkdir(parents=True, exist_ok=True)

    if debug:
        if debug_dir is None:
            debug_dir = out_dir / "_debug_logs"
        debug_dir.mkdir(parents=True, exist_ok=True)

    wav_paths: List[Path] = []
    procs: List[dict] = []

    batch_count = 0
    for idx, (st, ed) in enumerate(segments, start=1):
        out_wav = out_dir / f"{idx}.wav"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-i", str(media_path),
            "-ss", f"{st:.6f}",
            "-to", f"{ed:.6f}",
            "-vn",
            "-c:a", "pcm_s16le",
            str(out_wav),
        ]

        if debug:
            log_path = debug_dir / f"ffmpeg_{idx:04d}.log"
            p = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            # 先寫入 header，避免中途崩潰沒資訊
            with log_path.open("w", encoding="utf-8", errors="replace") as f:
                f.write(f"[START] {datetime.now().isoformat(timespec='seconds')}\n")
                f.write(f"idx={idx}\n")
                f.write(f"wav={out_wav}\n")
                f.write("cmd=" + " ".join(shlex.quote(c) for c in cmd) + "\n")
                f.write("-" * 80 + "\n")
            procs.append({"proc": p, "wav": out_wav, "cmd": cmd, "log": log_path})
        else:
            # 原本行為：只抓 stderr、不等待
            p = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            procs.append({"proc": p, "wav": out_wav, "cmd": cmd})

        wav_paths.append(out_wav)

        if len(procs) % batch_size == 0:
            batch_count += 1
            print(f"[Batch {batch_count}] Launched {batch_size} ffmpeg processes. Sleeping {batch_interval}s...")
            time.sleep(batch_interval)

    # debug 模式：等待所有 ffmpeg 完成並寫完整 log
    if debug:
        for item in procs:
            p: subprocess.Popen = item["proc"]
            log_path: Path = item["log"]
            stdout, stderr = p.communicate()
            rc = p.returncode

            with log_path.open("a", encoding="utf-8", errors="replace") as f:
                f.write(f"[END] {datetime.now().isoformat(timespec='seconds')} returncode={rc}\n")
                f.write("-" * 80 + "\n")
                f.write("[STDOUT]\n")
                f.write((stdout or "").rstrip() + "\n")
                f.write("-" * 80 + "\n")
                f.write("[STDERR]\n")
                f.write((stderr or "").rstrip() + "\n")
                f.write("-" * 80 + "\n")
                f.write(f"[OUTPUT_EXISTS] {item['wav'].exists()}\n")

    return wav_paths


def run_whisper_batch(
    wav_paths: List[Path],
    model: str = "large-v3",
    device: str = "cuda",
    output_format: str = "srt",
    task: str = "transcribe",
    language: str = "ja",
    whisper_cmd: str = "whisper-ctranslate2",
    model_dir: Optional[str] = None,
    output_dir: Optional[Path] = None,
    max_parallel: int = 2,
    debug: bool = False,
    debug_dir: Optional[Path] = None,
) -> None:
    """
    faster-whisper (whisper-ctranslate2) with a concurrency cap:
      - Keep at most `max_parallel` jobs running
      - As soon as any job finishes, launch the next

    If debug=True:
      - Capture stdout/stderr and write per-wav log file
    """
    if not wav_paths:
        return

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    base_dir = output_dir if output_dir is not None else wav_paths[0].parent
    if debug:
        if debug_dir is None:
            debug_dir = base_dir / "_debug_logs"
        debug_dir.mkdir(parents=True, exist_ok=True)

    def build_cmd(wav: Path) -> List[str]:
        cmd = [
            whisper_cmd,
            str(wav),
            "--model", model,
            "--device", device,
            "--output_format", output_format,
            "--task", task,
            "--language", language,
        ]
        if model_dir:
            cmd += ["--model_dir", model_dir]
        if output_dir is not None:
            cmd += ["--output_dir", str(output_dir)]
        return cmd

    pending = list(wav_paths)
    running: List[dict] = []

    def launch_one(wav: Path, idx: int) -> None:
        cmd = build_cmd(wav)
        print("Launch:", " ".join(shlex.quote(c) for c in cmd))

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"

        log_path = None
        if debug:
            log_path = debug_dir / f"whisper_{idx:04d}.log"
            with log_path.open("w", encoding="utf-8", errors="replace") as f:
                f.write(f"[START] {datetime.now().isoformat(timespec='seconds')}\n")
                f.write(f"idx={idx}\n")
                f.write(f"wav={wav}\n")
                f.write("cmd=" + " ".join(shlex.quote(c) for c in cmd) + "\n")
                f.write("-" * 80 + "\n")

        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE if debug else subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        running.append({"proc": p, "wav": wav, "cmd": cmd, "idx": idx, "log": log_path})

    # prime
    next_idx = 1
    while pending and len(running) < max_parallel:
        launch_one(pending.pop(0), next_idx)
        next_idx += 1

    # run
    while running:
        any_finished = False
        for i in range(len(running) - 1, -1, -1):
            item = running[i]
            p: subprocess.Popen = item["proc"]
            if p.poll() is None:
                continue

            stdout, stderr = p.communicate()
            rc = p.returncode
            any_finished = True
            running.pop(i)

            if debug and item["log"] is not None:
                with Path(item["log"]).open("a", encoding="utf-8", errors="replace") as f:
                    f.write(f"[END] {datetime.now().isoformat(timespec='seconds')} returncode={rc}\n")
                    f.write("-" * 80 + "\n")
                    f.write("[STDOUT]\n")
                    f.write((stdout or "").rstrip() + "\n")
                    f.write("-" * 80 + "\n")
                    f.write("[STDERR]\n")
                    f.write((stderr or "").rstrip() + "\n")

            # launch next
            if pending:
                launch_one(pending.pop(0), next_idx)
                next_idx += 1

        if not any_finished:
            time.sleep(0.2)



def merge_whisper_txts(
    wav_paths: List[Path],
    whisper_output_dir: Optional[Path],
    merged_name: str,
    ass: bool = False,
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
        raise ValueError("wav_paths is empty; cannot infer default txt output directory.")

    txt_dir = Path(whisper_output_dir) if whisper_output_dir is not None else wav_paths[0].parent
    if not txt_dir.exists():
        raise FileNotFoundError(f"Whisper txt directory not found: {txt_dir}")

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
        out_path = txt_dir / merged_name
        out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        return out_path

    # --- TXT merge mode (original behavior) ---
    out_path = txt_dir / merged_name
    out_path.write_text("\n".join(whisper_lines) + "\n", encoding="utf-8")
    return out_path


def main():
    ap = argparse.ArgumentParser(
        description="Parse ASS Events (Dialogue) lines and cut audio into numbered WAVs, then run faster-whisper (whisper-ctranslate2) per WAV."
    )
    ap.add_argument("-i", "--input", required=True, help="Input media file (video/audio).")
    ap.add_argument("-t", "--txt", required=True, help="ASS/SSA txt file containing Events Dialogue lines.")
    ap.add_argument(
        "-o", "--output", default=None,
        help="Output directory for split WAVs. Default: <input_stem>_chunks"
    )
    ap.add_argument("--padding-ms", type=float, default=50.0, help="Padding added to BOTH start and end (milliseconds). Default 50")
    ap.add_argument("--include-comments", action="store_true", help="Also parse Comment: lines (default: only Dialogue:).")
    ap.add_argument("--ass", action="store_true", help="If set, generate an ASS events file by replacing Dialogue text with whisper outputs.")

    # faster-whisper (whisper-ctranslate2) related options
    ap.add_argument("--no-whisper", action="store_true", help="Only split WAVs; do not run faster-whisper.")
    ap.add_argument("--whisper-cmd", default="whisper-ctranslate2", help="Executable in PATH. Default: whisper-ctranslate2")
    ap.add_argument("--model", default="large-v3", help="Model name. Default: large-v3")
    ap.add_argument("--model-dir", default=None, help="Model cache dir. Default: None")
    ap.add_argument("--device", default="cuda", help="Device. Default: cuda")
    ap.add_argument("--language", default="ja", help="Language. Default: ja")
    ap.add_argument("--output-format", default="txt", help="Output format. Default: txt")
    ap.add_argument("--task", default="transcribe", help="Task. Default: transcribe")
    ap.add_argument("--whisper-output-dir", default=None, help="Optional: directory for whisper outputs (srt). If omitted, use whisper default (same dir as input wav).")
    ap.add_argument("--max-parallel", type=int, default=8, help="Max concurrent faster-whisper jobs. Default: 4")

    ap.add_argument("--debug", action="store_true", help="Enable debug logging for ffmpeg and faster-whisper.")
    ap.add_argument("--debug-dir", default=None, help="Optional: directory for debug logs. Default: <out_dir>/_debug_logs or <whisper_out>/_debug_logs")

    args = ap.parse_args()

    in_path = Path(args.input)
    txt_path = Path(args.txt)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")
    if not txt_path.exists():
        raise SystemExit(f"Txt not found: {txt_path}")

    out_dir = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_chunks")
    whisper_out_dir = Path(args.whisper_output_dir) if args.whisper_output_dir else None
    debug_dir = Path(args.debug_dir) if args.debug_dir else None

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

    wavs = run_ffmpeg_split_wavs(in_path, padded, out_dir, debug=args.debug, debug_dir=debug_dir)
    print(f"Done split: {out_dir} ({len(wavs)} wav files)")

    if not args.no_whisper:
        run_whisper_batch(
            wavs,
            model=args.model,
            device=args.device,
            output_format=args.output_format,
            task=args.task,
            language=args.language,
            whisper_cmd=args.whisper_cmd,
            model_dir=args.model_dir,
            output_dir=whisper_out_dir,
            max_parallel=args.max_parallel,
            debug=args.debug,
            debug_dir=debug_dir,
        )
        print("Done faster-whisper.")

        merged_name = f"{in_path.stem}_subtitle.txt"
        merged_path = merge_whisper_txts(
            wav_paths=wavs,
            whisper_output_dir=whisper_out_dir,
            merged_name=merged_name,
            ass=args.ass,
            txt_events_path=txt_path,
        )
        print(f"ass:{args.ass}")
        print(f"Done merge: {merged_path}")


if __name__ == "__main__":
    main()