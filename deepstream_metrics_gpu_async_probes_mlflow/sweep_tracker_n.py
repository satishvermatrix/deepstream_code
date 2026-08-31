#!/usr/bin/env python3
"""Sweep tracker_multisrc over N and write metrics vs N (JSON + optional PNG)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

KEYS = (
    "num_streams",
    "fps_e2e",
    "fps_per_stream",
    "dropped_frames",
    "gpu_util_avg",
    "nvdec_util_avg",
    "nvenc_util_avg",
    "ofa_util_avg",
    "power_w_avg",
    "latency_ds_pgie_ms_avg",
    "latency_ds_tracker_ms_avg",
    "latency_hop_mux_osd_ms_avg",
    "latency_hop_mux_osd_ms_p95",
    "objects_total",
    "frames_osd",
)


def parse_ns(raw: str) -> list[int]:
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        n = int(part)
        if n < 1:
            raise ValueError(f"bad N: {part}")
        out.append(n)
    if not out:
        raise ValueError("empty --streams")
    return out


def run_one(n: int, extra: list[str]) -> dict:
    run_name = f"tracker_n{n}"
    cmd = [
        sys.executable,
        "/workspace/pipeline_tracker_multisrc.py",
        "--num-streams",
        str(n),
        "--sink",
        "fake",
        "--no-log-frames",
        "--run-name",
        run_name,
        *extra,
    ]
    print(" ".join(cmd), flush=True)
    proc = subprocess.run(cmd, check=False)
    metrics_path = Path(f"/workspace/output/logs/{run_name}_metrics.json")
    if proc.returncode != 0 or not metrics_path.is_file():
        raise RuntimeError(f"N={n} failed (exit {proc.returncode})")
    data = json.loads(metrics_path.read_text())
    row = {k: data.get(k) for k in KEYS}
    row["metrics_path"] = str(metrics_path)
    return row


def write_png(rows: list[dict], dest: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skip PNG")
        return

    xs = [r["num_streams"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    series = [
        (axes[0, 0], "fps_e2e", "Total FPS (frames/s)"),
        (axes[0, 1], "fps_per_stream", "FPS per stream"),
        (axes[1, 0], "gpu_util_avg", "SM util avg (%)"),
        (axes[1, 1], "latency_ds_tracker_ms_avg", "Tracker latency avg (ms)"),
    ]
    for ax, key, title in series:
        ys = [r.get(key) or 0 for r in rows]
        ax.plot(xs, ys, marker="o")
        ax.set_title(title)
        ax.set_xlabel("N streams")
        ax.grid(True, alpha=0.3)
        if key == "fps_per_stream":
            ax.axhline(30, color="0.5", linestyle="--", linewidth=1)
    fig.suptitle("YOLO26m + NvDCF vs N (same 1080p file, fakesink, sync=0)")
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=120)
    plt.close(fig)
    print(f"wrote {dest}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--streams", default="1,2,4,8")
    parser.add_argument(
        "-o", "--output", default="/workspace/output/logs/tracker_n_sweep.json"
    )
    parser.add_argument("passthrough", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    extra = list(args.passthrough)
    if extra and extra[0] == "--":
        extra = extra[1:]

    ns = parse_ns(args.streams)
    rows = []
    for n in ns:
        rows.append(run_one(n, extra))
        print(
            f"N={n}  fps={rows[-1]['fps_e2e']:.1f}  "
            f"per_stream={rows[-1]['fps_per_stream']:.1f}  "
            f"gpu={rows[-1]['gpu_util_avg']:.0f}%",
            flush=True,
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pipeline": "tracker_multisrc",
        "sink": "fake",
        "input": "same sample_1080p_h264.mp4 x N",
        "rows": rows,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}")
    write_png(rows, out.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
