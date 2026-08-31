#!/usr/bin/env python3
"""Download MOT17 train (SDP folders), encode one MP4 per sequence, summarize GT.

Run on the host (needs wget/curl + ffmpeg), not inside ds8-dev:

  python3 prepare_mot17.py
  python3 prepare_mot17.py --skip-download   # already extracted
"""

from __future__ import annotations

import argparse
import configparser
import json
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MP4 = ROOT / "mp4"
GT = ROOT / "gt"
SUMMARY = ROOT / "mot17_summary.json"
ZIP_PATH = ROOT / "MOT17.zip"
MOT17_URL = "https://motchallenge.net/data/MOT17.zip"
HF_REPO = "Morrison1025/MOT17"

# Official MOT17 duplicates each video three times (DPM / FRCNN / SDP).
# Images+GT on this Hugging Face mirror live under the FRCNN folders.
DETECTOR = "FRCNN"

# Official MOT17 label ids (gt.txt column 8).
CLASS_NAME = {
    1: "pedestrian",
    2: "person_on_vehicle",
    3: "car",
    4: "bicycle",
    5: "motorbike",
    6: "non_motorized_vehicle",
    7: "static_person",
    8: "distractor",
    9: "occluder",
    10: "occluder_ground",
    11: "occluder_full",
    12: "reflection",
    13: "crowd",
}

# TrackEval / MOTChallenge pedestrian protocol.
EVAL_CLASSES = {1}


def train_root() -> Path | None:
    for cand in (
        ROOT / "raw" / "MOT17" / "train",
        ROOT / "raw" / "train",
        ROOT / "train",
    ):
        if cand.is_dir():
            return cand
    return None


def list_sequences() -> list[Path]:
    root = train_root()
    if root is None:
        return []
    return sorted(
        p
        for p in root.iterdir()
        if p.is_dir()
        and p.name.endswith(f"-{DETECTOR}")
        and (p / "img1").is_dir()
        and (p / "gt" / "gt.txt").is_file()
        and (p / "seqinfo.ini").is_file()
    )


def download_from_huggingface() -> None:
    dest = ROOT / "raw"
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Hugging Face snapshot {HF_REPO} (train/*-{DETECTOR} only)")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"]
        )
        from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        local_dir=str(dest),
        allow_patterns=[
            f"train/*-{DETECTOR}/img1/*",
            f"train/*-{DETECTOR}/gt/*",
            f"train/*-{DETECTOR}/seqinfo.ini",
            f"train/*-{DETECTOR}/det/*",
        ],
    )


def download_zip() -> str:
    """Return 'zip' or 'hf' depending on which source landed the files."""
    if list_sequences():
        print(f"already have {len(list_sequences())} SDP train seqs")
        return "existing"
    ROOT.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.is_file() and ZIP_PATH.stat().st_size > 1_000_000_000:
        print(f"exists {ZIP_PATH} ({ZIP_PATH.stat().st_size / 1e9:.2f} GB)")
        return "zip"
    print(f"try official zip {MOT17_URL}")
    try:
        subprocess.check_call(
            ["wget", "-c", "--timeout=8", "--tries=1", "-O", str(ZIP_PATH), MOT17_URL]
        )
        print(f"saved {ZIP_PATH} ({ZIP_PATH.stat().st_size / 1e9:.2f} GB)")
        return "zip"
    except subprocess.CalledProcessError as exc:
        print(f"official zip failed ({exc}); falling back to Hugging Face")
        if ZIP_PATH.is_file() and ZIP_PATH.stat().st_size < 1_000_000:
            ZIP_PATH.unlink()
        download_from_huggingface()
        return "hf"


def extract_sdp_train(source: str) -> None:
    if list_sequences():
        print(f"already extracted {len(list_sequences())} SDP train seqs")
        return
    if source == "hf":
        return
    if not ZIP_PATH.is_file():
        raise FileNotFoundError(ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        members = [m for m in zf.namelist() if "/train/" in m and f"-{DETECTOR}" in m]
        if not members:
            raise RuntimeError("No MOT17/train/*-SDP entries in the zip")
        print(f"extract {len(members)} SDP-train members -> {ROOT / 'raw'}")
        dest = ROOT / "raw"
        dest.mkdir(parents=True, exist_ok=True)
        zf.extractall(dest, members)


def read_seqinfo(seq_dir: Path) -> dict[str, str]:
    ini = seq_dir / "seqinfo.ini"
    cfg = configparser.ConfigParser()
    cfg.read(ini)
    sec = cfg["Sequence"]
    return {k: sec[k] for k in sec}


def encode_mp4(seq_dir: Path, info: dict[str, str]) -> Path:
    MP4.mkdir(parents=True, exist_ok=True)
    base = seq_dir.name.replace(f"-{DETECTOR}", "")
    out = MP4 / f"{base}.mp4"
    img_dir = seq_dir / info.get("imDir", "img1")
    ext = info.get("imExt", ".jpg").lstrip(".")
    fps = info.get("frameRate", "30")
    n = int(info.get("seqLength", "0"))
    first = next(sorted(img_dir.glob(f"*.{ext}")))
    pattern = str(img_dir / f"%0{len(first.stem)}d.{ext}")
    if out.is_file() and out.stat().st_size > 10_000:
        print(f"exists {out}")
        return out
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-start_number",
        str(int(first.stem)),
        "-i",
        pattern,
        "-frames:v",
        str(n),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(out),
    ]
    print(" ".join(cmd))
    subprocess.check_call(cmd)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def parse_gt(gt_path: Path) -> dict:
    rows = []
    with gt_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            frame = int(float(parts[0]))
            tid = int(float(parts[1]))
            x, y, w, h = (float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]))
            conf = int(float(parts[6]))
            cls = int(float(parts[7])) if len(parts) > 7 else 1
            vis = float(parts[8]) if len(parts) > 8 else 1.0
            rows.append(
                {
                    "frame": frame,
                    "id": tid,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "conf": conf,
                    "class": cls,
                    "visibility": vis,
                }
            )

    by_class = Counter(r["class"] for r in rows)
    eval_rows = [r for r in rows if r["class"] in EVAL_CLASSES and r["conf"] == 1]
    eval_ids = {r["id"] for r in eval_rows}
    frames_with_eval = {r["frame"] for r in eval_rows}
    id_len = defaultdict(int)
    for r in eval_rows:
        id_len[r["id"]] += 1

    sample = None
    for r in eval_rows:
        if r["frame"] == 1:
            sample = r
            break
    if sample is None and eval_rows:
        sample = eval_rows[0]

    return {
        "rows_total": len(rows),
        "rows_eval_pedestrian": len(eval_rows),
        "unique_eval_ids": len(eval_ids),
        "frames_with_eval_box": len(frames_with_eval),
        "mean_boxes_per_frame": (len(eval_rows) / len(frames_with_eval)) if frames_with_eval else 0.0,
        "mean_track_len_frames": (sum(id_len.values()) / len(id_len)) if id_len else 0.0,
        "by_class": {CLASS_NAME.get(k, str(k)): v for k, v in sorted(by_class.items())},
        "sample_eval_row": sample,
        "gt_line_format": "frame,id,x,y,w,h,conf,class,visibility",
    }


def copy_gt(seq_dir: Path, info: dict[str, str]) -> Path:
    base = seq_dir.name.replace(f"-{DETECTOR}", "")
    dest = GT / base
    dest.mkdir(parents=True, exist_ok=True)
    src_gt = seq_dir / "gt" / "gt.txt"
    src_ini = seq_dir / "seqinfo.ini"
    (dest / "gt.txt").write_bytes(src_gt.read_bytes())
    (dest / "seqinfo.ini").write_bytes(src_ini.read_bytes())
    return dest / "gt.txt"


def summarize(seqs: list[Path]) -> dict:
    out: dict = {
        "source": MOT17_URL,
        "huggingface_mirror": HF_REPO,
        "detector_folders": DETECTOR,
        "note": (
            "MOT17 ships each video three times (DPM/FRCNN/SDP) for the public-detection "
            "protocol. Images and GT are identical; we keep SDP only. Test has no GT."
        ),
        "eval_protocol": "private-det: use class==1 (pedestrian) and conf==1",
        "class_ids": CLASS_NAME,
        "sequences": [],
    }
    for seq_dir in seqs:
        info = read_seqinfo(seq_dir)
        mp4 = encode_mp4(seq_dir, info)
        gt_copy = copy_gt(seq_dir, info)
        gt_stats = parse_gt(seq_dir / "gt" / "gt.txt")
        rec = {
            "name": seq_dir.name.replace(f"-{DETECTOR}", ""),
            "folder": str(seq_dir),
            "mp4": str(mp4),
            "mp4_mb": round(mp4.stat().st_size / 1e6, 2),
            "gt": str(gt_copy),
            "imWidth": int(info["imWidth"]),
            "imHeight": int(info["imHeight"]),
            "frameRate": float(info["frameRate"]),
            "seqLength": int(info["seqLength"]),
            "duration_s": round(int(info["seqLength"]) / float(info["frameRate"]), 2),
            **gt_stats,
        }
        out["sequences"].append(rec)
        print(
            f"{rec['name']}: {rec['imWidth']}x{rec['imHeight']} "
            f"{rec['seqLength']}f @{rec['frameRate']}fps  "
            f"eval_ids={rec['unique_eval_ids']}  "
            f"eval_boxes={rec['rows_eval_pedestrian']}"
        )
    SUMMARY.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {SUMMARY}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()

    if not args.skip_download:
        source = download_zip()
    else:
        source = "existing"
    if not args.skip_extract:
        extract_sdp_train(source)
    seqs = list_sequences()
    if not seqs:
        print("No SDP train sequences found under data/mot17/raw", file=sys.stderr)
        return 1
    summarize(seqs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
