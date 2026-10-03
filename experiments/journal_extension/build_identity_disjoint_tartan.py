"""Build chronological Tartan views with disjoint persistent aircraft IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts import build_tartan_target_domain as base


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path(__file__).with_name("identity_disjoint_tartan_protocol_v1.json")


def identity_hash(values: set[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_protocol() -> dict[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    for key in ("data_manifest", "builder", "KAGC_archive", "KBTP_archive"):
        path = ROOT / protocol["parent"][key]
        if not path.is_file() or base.sha256(path) != protocol["parent"][f"{key}_sha256"]:
            raise RuntimeError(f"identity-disjoint source mismatch: {path}")
    return protocol


def materialize_day(
    *,
    archive: Path,
    date_value: str,
    members: list[str],
    reference: tuple[float, float],
    excluded: set[str],
    output: Path,
    window_stride: int,
) -> tuple[dict[str, int], set[str], set[str]]:
    observations = base._read_day(archive, date_value, members, reference)
    observed = set(observations)
    retained_observations = {
        identifier: values
        for identifier, values in observations.items()
        if identifier not in excluded
    }
    segments = [
        segment
        for identifier, values in retained_observations.items()
        for segment in base._resample_segments(identifier, values)
    ]
    retained_identities = {track.identifier for track in segments}
    tracks = base._assign_day_track_ids(segments)
    if not tracks:
        return {"scenes": 0, "actors": 0, "rows": 0, "files": 0}, observed, retained_identities
    output.mkdir(parents=True, exist_ok=True)
    totals = {"scenes": 0, "actors": 0, "rows": 0, "files": 0}
    for index, component in enumerate(base._connected_components(tracks)):
        component_start = min(int(track.times[0]) for track in component)
        component_end = max(int(track.times[-1]) for track in component)
        scenes = 0
        actors = 0
        for start in range(
            component_start,
            component_end - base.RAW_WINDOW_STEPS + 2,
            max(1, window_stride),
        ):
            count = sum(
                int(track.times[0]) <= start
                and int(track.times[-1]) >= start + base.RAW_WINDOW_STEPS - 1
                for track in component
            )
            if count:
                scenes += 1
                actors += count
        if not scenes:
            continue
        lines = base._component_lines(component)
        path = output / f"{date_value}_r{index:04d}.txt"
        if path.exists():
            raise FileExistsError(path)
        path.write_text("".join(lines), encoding="utf-8")
        totals["scenes"] += scenes
        totals["actors"] += actors
        totals["rows"] += len(lines)
        totals["files"] += 1
    return totals, observed, retained_identities


def build(output: Path, *, max_dates_per_split: int | None = None) -> dict[str, Any]:
    protocol = load_protocol()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    parent = json.loads((ROOT / protocol["parent"]["data_manifest"]).read_text(encoding="utf-8"))
    result: dict[str, Any] = {
        "schema_version": 1,
        "protocol": PROTOCOL.relative_to(ROOT).as_posix(),
        "protocol_sha256": base.sha256(PROTOCOL),
        "formal": max_dates_per_split is None,
        "datasets": {},
    }
    for airport, specification in base.AIRPORTS.items():
        archive = Path(specification["archive"])
        members_by_date, compression = base.archive_members_by_date(archive)
        parent_dates = parent["datasets"][airport]["split_dates"]
        seen: set[str] = set()
        retained_by_split: dict[str, set[str]] = {}
        airport_result: dict[str, Any] = {
            "archive_sha256": base.sha256(archive),
            "compression_methods": compression,
            "splits": {},
        }
        for split in ("train", "development", "test"):
            dates = list(parent_dates[split])
            if max_dates_per_split is not None:
                dates = dates[:max_dates_per_split]
            totals = {"scenes": 0, "actors": 0, "rows": 0, "files": 0}
            observed_split: set[str] = set()
            retained_split: set[str] = set()
            excluded_before = set(seen)
            for date_value in dates:
                day, observed, retained = materialize_day(
                    archive=archive,
                    date_value=date_value,
                    members=members_by_date[date_value],
                    reference=specification["reference"],
                    excluded=excluded_before,
                    output=output / airport / split,
                    window_stride=int(protocol["window"]["window_stride_seconds"]),
                )
                for name in totals:
                    totals[name] += day[name]
                observed_split.update(observed)
                retained_split.update(retained)
            retained_by_split[split] = retained_split
            seen.update(observed_split)
            airport_result["splits"][split] = {
                **totals,
                "dates": dates,
                "date_count": len(dates),
                "observed_identity_count": len(observed_split),
                "retained_identity_count": len(retained_split),
                "excluded_prior_identity_count": len(observed_split & excluded_before),
                "retained_identity_sha256": identity_hash(retained_split),
            }
        intersections = {
            "train_development": len(retained_by_split["train"] & retained_by_split["development"]),
            "train_test": len(retained_by_split["train"] & retained_by_split["test"]),
            "development_test": len(retained_by_split["development"] & retained_by_split["test"]),
        }
        if any(intersections.values()):
            raise RuntimeError(f"retained identity overlap for {airport}: {intersections}")
        if any(airport_result["splits"][split]["scenes"] < 1 for split in ("train", "development", "test")):
            raise RuntimeError(f"identity-disjoint {airport} has an empty split")
        airport_result["retained_identity_intersections"] = intersections
        result["datasets"][airport] = airport_result
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    qc = {
        path.relative_to(output).as_posix(): base._qc_scene(path)
        for path in output.glob("*/*/*.txt")
    }
    if not all(item["contiguous"] and item["finite"] for item in qc.values()):
        raise RuntimeError("identity-disjoint materialization failed QC")
    (output / "qc.json").write_text(json.dumps(qc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-dates-per-split", type=int)
    args = parser.parse_args()
    result = build(args.output.resolve(), max_dates_per_split=args.max_dates_per_split)
    print(json.dumps({"output": str(args.output), "formal": result["formal"], "datasets": result["datasets"]}, indent=2))


if __name__ == "__main__":
    main()
