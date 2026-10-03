#!/usr/bin/env python
"""Build a frozen date-split Tartan target-domain trajectory protocol.

The official processed Tartan archives do not retain acquisition dates.  This
builder therefore reads the date-keyed raw archive, applies the official
altitude/range filter, resamples each aircraft at one second, and materializes
date-bounded continuous runs.  ``TrajectoryDataset(skip=5)`` then constructs
the registered 136-second windows without creating hundreds of thousands of
tiny files.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import zipfile_deflate64 as zipfile
except ImportError:  # pragma: no cover - useful on a minimal inspection host
    import zipfile  # type: ignore[no-redef]

from causal_measurement_time.information import geodetic_to_local_km
from runway_graph_irl.data_protocol import normalize_aircraft_id


DATE_MEMBER = re.compile(r"^(?P<month>\d{2})-(?P<day>\d{2})-(?P<year>\d{2})/")
REQUIRED_FIELDS = {"ID", "Time", "Date", "Altitude", "Range", "Lat", "Lon"}
OBS_LEN = 16
PRED_LEN = 24
OBS_INTERVAL_SECONDS = 1
PRED_INTERVAL_SECONDS = 5
# TrajectoryDataset's ``pred_len`` argument is the raw-frame duration (120),
# while its output contains 24 points sampled every five seconds.
PRED_DURATION_SECONDS = PRED_LEN * PRED_INTERVAL_SECONDS
RAW_WINDOW_STEPS = OBS_LEN + PRED_DURATION_SECONDS
MAX_GAP_SECONDS = 60.0

AIRPORTS = {
    "KAGC": {
        "archive": ROOT / "dataset/downloads/tartan_kagc_raw_2022.zip",
        "reference": (40.351422, -79.923939),
    },
    "KBTP": {
        "archive": ROOT / "dataset/downloads/tartan_kbtp_raw_2022.zip",
        "reference": (40.777888, -79.949864),
    },
}


@dataclass(frozen=True)
class Track:
    identifier: str
    times: np.ndarray
    positions: np.ndarray


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _timestamp(date_text: str, time_text: str) -> float:
    date_parts = ast.literal_eval(date_text.strip())
    time_parts = ast.literal_eval(time_text.strip())
    year, month, day_value = (int(value) for value in date_parts)
    hour, minute = (int(value) for value in time_parts[:2])
    second = float(time_parts[2])
    return date(year, month, day_value).toordinal() * 86400 + hour * 3600 + minute * 60 + second


def archive_members_by_date(path: Path) -> tuple[dict[str, list[str]], dict[str, int]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    methods: dict[str, int] = defaultdict(int)
    with zipfile.ZipFile(path) as archive:
        for item in archive.infolist():
            member = PurePosixPath(item.filename.replace("\\", "/"))
            if item.is_dir() or member.suffix.lower() != ".csv":
                continue
            match = DATE_MEMBER.match(item.filename.replace("\\", "/"))
            if match is None:
                raise ValueError(f"unexpected raw archive member: {item.filename}")
            value = f"20{match.group('year')}-{match.group('month')}-{match.group('day')}"
            grouped[value].append(item.filename)
            methods[str(int(item.compress_type))] += 1
    return {key: sorted(value) for key, value in sorted(grouped.items())}, dict(sorted(methods.items()))


def _read_day(
    archive: Path,
    date_value: str,
    members: Iterable[str],
    reference: tuple[float, float],
) -> dict[str, list[tuple[float, np.ndarray]]]:
    observations: dict[str, list[tuple[float, np.ndarray]]] = defaultdict(list)
    with zipfile.ZipFile(archive) as container:
        for member_name in members:
            if container.getinfo(member_name).file_size == 0:
                continue
            retained: set[str] = set()
            with container.open(member_name) as binary:
                text = io.TextIOWrapper(binary, encoding="utf-8-sig", newline="")
                reader = csv.DictReader(text)
                if reader.fieldnames is None or not REQUIRED_FIELDS.issubset(reader.fieldnames):
                    raise ValueError(f"unexpected raw schema in {member_name}: {reader.fieldnames}")
                for row in reader:
                    try:
                        if not row["ID"] or not row["Range"] or not row["Bearing"]:
                            continue
                        if int(float(row["Altitude"])) >= 6000 or float(row["Range"]) >= 5:
                            continue
                        key = row["Range"] + row["Bearing"] + row["ID"]
                        if key in retained:
                            continue
                        retained.add(key)
                        timestamp = _timestamp(row["Date"], row["Time"])
                        if date.fromordinal(int(timestamp // 86400)).isoformat() != date_value:
                            raise ValueError(f"member date mismatch for {member_name}")
                        identifier = normalize_aircraft_id(row["ID"])
                        position = geodetic_to_local_km(
                            float(row["Lat"]), float(row["Lon"]), float(row["Altitude"]), *reference
                        )
                        if np.isfinite(position).all():
                            observations[identifier].append((timestamp, position))
                    except (TypeError, ValueError, OverflowError, SyntaxError):
                        continue
    return observations


def _resample_segments(
    identifier: str, observations: list[tuple[float, np.ndarray]]
) -> list[Track]:
    if not observations:
        return []
    values = sorted(observations, key=lambda item: item[0])
    times = np.asarray([item[0] for item in values], dtype=np.float64)
    positions = np.stack([item[1] for item in values]).astype(np.float64)
    # Collapse multiple packets in one second before interpolation.
    seconds = np.floor(times).astype(np.int64)
    unique, inverse = np.unique(seconds, return_inverse=True)
    collapsed = np.zeros((len(unique), 3), dtype=np.float64)
    for index in range(len(unique)):
        collapsed[index] = positions[inverse == index].mean(axis=0)
    boundaries = np.flatnonzero(np.diff(unique) > MAX_GAP_SECONDS) + 1
    segments: list[Track] = []
    for segment_seconds, segment_positions in zip(
        np.split(unique, boundaries), np.split(collapsed, boundaries)
    ):
        if not len(segment_seconds) or int(segment_seconds[-1] - segment_seconds[0] + 1) < RAW_WINDOW_STEPS:
            continue
        # Integer-second interpolation is the same coordinate convention used
        # by the official Tartan preprocessing, with no extrapolation.
        grid = np.arange(segment_seconds[0], segment_seconds[-1] + 1, dtype=np.int64)
        sampled = np.column_stack(
            [np.interp(grid, segment_seconds, segment_positions[:, axis]) for axis in range(3)]
        )
        segments.append(Track(identifier, grid, sampled))
    return segments


def _assign_day_track_ids(tracks: list[Track]) -> list[Track]:
    """Assign compact IDs to continuous tracks without int32 overflow."""
    ordered = sorted(
        tracks,
        key=lambda item: (
            int(item.identifier),
            int(item.times[0]),
            int(item.times[-1]),
        ),
    )
    if len(ordered) >= 2**31:
        raise OverflowError("daily continuous-track count exceeds int32 actor capacity")
    return [
        Track(str(index), track.times, track.positions)
        for index, track in enumerate(ordered, start=1)
    ]


def _split_dates(dates: list[str], seed: int) -> dict[str, list[str]]:
    values = sorted(set(dates))
    train_count = int(len(values) * 0.6)
    development_count = int(len(values) * 0.2)
    return {
        "train": sorted(values[:train_count]),
        "development": sorted(values[train_count : train_count + development_count]),
        "test": sorted(values[train_count + development_count :]),
    }


def _connected_components(tracks: list[Track]) -> list[list[Track]]:
    components: list[list[Track]] = []
    end: int | None = None
    for track in sorted(tracks, key=lambda item: (int(item.times[0]), int(item.times[-1]), item.identifier)):
        start = int(track.times[0])
        if end is None or start > end + 1:
            components.append([track])
            end = int(track.times[-1])
        else:
            components[-1].append(track)
            end = max(end, int(track.times[-1]))
    return components


def _component_lines(tracks: list[Track]) -> list[str]:
    start = min(int(track.times[0]) for track in tracks)
    end = max(int(track.times[-1]) for track in tracks)
    lines: list[str] = []
    for timestamp in range(start, end + 1):
        frame = timestamp - start
        for track in tracks:
            if timestamp < int(track.times[0]) or timestamp > int(track.times[-1]):
                continue
            offset = int(timestamp - int(track.times[0]))
            x, y, z = track.positions[offset]
            # Context columns are retained for TrajectoryDataset compatibility;
            # weather is intentionally not injected into this geometry protocol.
            lines.append(f"{frame} {track.identifier} {x:.9f} {y:.9f} {z:.9f} 0.0 0.0\n")
    return lines


def build_day(
    archive: Path,
    date_value: str,
    members: list[str],
    reference: tuple[float, float],
    output: Path,
    max_windows: int | None = None,
    window_stride: int = 5,
) -> dict[str, int]:
    observations = _read_day(archive, date_value, members, reference)
    tracks = _assign_day_track_ids(
        [segment for identifier, values in observations.items() for segment in _resample_segments(identifier, values)]
    )
    if not tracks:
        return {"scenes": 0, "actors": 0, "rows": 0, "files": 0}
    first = min(int(track.times[0]) for track in tracks)
    last = max(int(track.times[-1]) for track in tracks)
    output.mkdir(parents=True, exist_ok=True)
    scenes = actors = rows = 0
    files = 0
    for index, component in enumerate(_connected_components(tracks)):
        if max_windows is not None and files >= max_windows:
            break
        component_start = min(int(track.times[0]) for track in component)
        component_end = max(int(track.times[-1]) for track in component)
        component_scenes = 0
        component_actors = 0
        for start in range(
            component_start,
            component_end - RAW_WINDOW_STEPS + 2,
            max(1, window_stride),
        ):
            count = sum(
                int(track.times[0]) <= start
                and int(track.times[-1]) >= start + RAW_WINDOW_STEPS - 1
                for track in component
            )
            if count:
                component_scenes += 1
                component_actors += count
        if not component_scenes:
            continue
        lines = _component_lines(component)
        path = output / f"{date_value}_r{index:04d}.txt"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing run: {path}")
        path.write_text("".join(lines), encoding="utf-8")
        scenes += component_scenes
        rows += len(lines)
        actors += component_actors
        files += 1
    return {"scenes": scenes, "actors": actors, "rows": rows, "files": files}


def _qc_scene(path: Path) -> dict[str, int | bool]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) != 7:
                return {"rows": 0, "agents": 0, "contiguous": False, "finite": False}
            rows.append(fields)
    if not rows:
        return {"rows": 0, "agents": 0, "contiguous": False, "finite": False}
    frames = sorted({int(float(row[0])) for row in rows})
    agents = sorted({row[1] for row in rows})
    finite = all(math.isfinite(float(value)) for row in rows for value in row[2:])
    actor_frames = {
        agent: sorted({int(float(row[0])) for row in rows if row[1] == agent})
        for agent in agents
    }
    eligible = any(
        len(values) >= RAW_WINDOW_STEPS
        and values == list(range(values[0], values[-1] + 1))
        for values in actor_frames.values()
    )
    contiguous = frames == list(range(frames[-1] + 1)) and eligible
    return {"rows": len(rows), "agents": len(agents), "contiguous": contiguous, "finite": finite}


def build(args: argparse.Namespace) -> dict[str, object]:
    output_root = args.output.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing target protocol: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    protocol: dict[str, object] = {
        "schema_version": 1,
        "protocol_id": "partc_two_dataset_target_domain_v1",
        "task": "retrospective target-domain trajectory prediction",
        "history": {"points": OBS_LEN, "interval_seconds": OBS_INTERVAL_SECONDS},
        "future": {"points": PRED_LEN, "interval_seconds": PRED_INTERVAL_SECONDS, "horizons_seconds": list(range(5, 121, 5))},
        "raw_window_steps": RAW_WINDOW_STEPS,
        "split": {"unit": "calendar date", "fractions": [0.6, 0.2, 0.2], "rule": "chronological earliest/middle/latest", "seed_field_retained_for_manifest_compatibility": args.seed},
        "window_stride_seconds": args.window_stride,
        "coordinate_frame": "local east,north,up km; per-airport fixed reference",
        "context_columns": "headwind,crosswind placeholders fixed to zero",
        "schema_audit": {
            "raw_archives": {
                "columns": ["ID", "Time", "Date", "Altitude", "Speed", "Heading", "Lat", "Lon", "Age", "Range", "Bearing", "Tail", "AltisGNSS"],
                "date_mapping": "archive member directory",
                "compression": "stored or Deflate64; read with zipfile-deflate64",
                "selected_for_build": True,
            },
            "official_processed_archives": {
                "columns": ["frame", "aircraft_id", "x_km", "y_km", "z_km", "headwind", "crosswind"],
                "delimiter": "comma with blank separator lines",
                "date_mapping": None,
                "selected_for_build": False,
                "reason": "cannot create a leakage-safe calendar-date split",
            },
            "legacy_c29_c30r_npz": {
                "arrays": ["base_features", "history_features", "future_local", "target_residuals", "target_flow", "actor_groups", "anchor_offsets", "packet_available"],
                "future_shape": "N x 4 x 3 at 30/60/90/120 seconds",
                "selected_for_build": False,
                "reason": "privacy-reduced task features cannot reconstruct 16 history plus 24 future coordinates",
            },
        },
        "sources": {},
        "datasets": {},
        "limitations": [
            "raw archive IDs are retained only in scene files for reproducibility; no cross-date actor overlap is used for split assignment",
            "retrospective measurement-time windows; this is not the C29/C30R packet-causal protocol",
            "official processed archives do not expose acquisition dates, so date split is built from raw archives",
        ],
    }
    for airport, spec in AIRPORTS.items():
        archive = Path(args.archive.get(airport, spec["archive"]))
        if not archive.is_file():
            raise FileNotFoundError(archive)
        members_by_date, methods = archive_members_by_date(archive)
        splits = _split_dates(list(members_by_date), args.seed + (0 if airport == "KAGC" else 1))
        protocol["sources"][airport] = {"archive": str(archive), "sha256": sha256(archive), "bytes": archive.stat().st_size, "compression_methods": methods, "dates": len(members_by_date), "reference": list(spec["reference"])}
        airport_summary: dict[str, object] = {"splits": {}, "all_dates": sorted(members_by_date), "split_dates": splits}
        for split_name, dates in splits.items():
            chosen = dates
            if args.max_dates_per_split is not None:
                chosen = chosen[: args.max_dates_per_split]
            split_root = output_root / airport / split_name
            split_root.mkdir(parents=True, exist_ok=True)
            split_summary = {"dates": chosen, "date_count": len(chosen), "scenes": 0, "actors": 0, "rows": 0, "files": 0}
            for date_value in chosen:
                result = build_day(archive, date_value, members_by_date[date_value], spec["reference"], split_root, args.max_windows_per_date, args.window_stride)
                for key in ("scenes", "actors", "rows", "files"):
                    split_summary[key] += result[key]
            airport_summary["splits"][split_name] = split_summary
        protocol["datasets"][airport] = airport_summary
    qc: dict[str, object] = {"files": 0, "bad_files": [], "scenes": 0, "rows": 0, "agents": 0}
    for scene in output_root.glob("*/*/*.txt"):
        result = _qc_scene(scene)
        qc["files"] += 1
        qc["scenes"] += 1
        qc["rows"] += int(result["rows"])
        qc["agents"] += int(result["agents"])
        if not result["contiguous"] or not result["finite"]:
            qc["bad_files"].append(str(scene))
    protocol["qc"] = qc
    manifest = output_root / "manifest.json"
    manifest.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_root / "qc.json").write_text(json.dumps(qc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/partc_two_dataset_20260812/target_domain_data_v1"))
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--max-dates-per-split", type=int, default=None)
    parser.add_argument("--max-windows-per-date", type=int, default=None)
    parser.add_argument("--window-stride", type=int, default=5)
    parser.add_argument("--archive", action="append", default=[], metavar="AIRPORT=PATH")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.archive = dict(value.split("=", 1) for value in args.archive)
    result = build(args)
    print(json.dumps({"output": str(args.output.resolve()), "datasets": result["datasets"], "qc": result["qc"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
