"""Audit GitHub-facing structure and enforce final metadata for release tags."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PATHS = (
    ".gitattributes",
    ".gitignore",
    ".github/ISSUE_TEMPLATE/bug_report.yml",
    ".github/pull_request_template.md",
    ".github/workflows/ci.yml",
    "CHANGELOG.md",
    "CITATION.cff",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "Dockerfile",
    "README.md",
    "MANIFEST.sha256",
    "SECURITY.md",
    "THIRD_PARTY_NOTICES.md",
    "docs/assets/README.md",
    "constraints/requirements-cpu.txt",
    "docs/ASSET_BOUNDARY.md",
    "docs/ASCENT_NOTICE.md",
    "docs/GITHUB_UPLOAD.md",
    "docs/CODE_AVAILABILITY.md",
    "docs/PROJECT_STRUCTURE.md",
    "docs/RELEASE_AUDIT.md",
    "docs/CODE_MAP.md",
    "docs/DATASETS.md",
    "docs/DATA_AVAILABILITY.md",
    "docs/DATA_SOURCES.md",
    "docs/MODEL_RELEASE.md",
    "docs/PAPER_SCOPE.md",
    "docs/REPRODUCIBILITY.md",
    "docs/THIRD_PARTY_LICENSE_MATRIX.md",
    "docs/assets/measure2act_workflow.png",
    "docs/data_sources.json",
    "docs/dataset_sources.csv",
    "environment.yml",
    "model_release.json",
    "pyproject.toml",
    "requirements-lock.txt",
    "requirements.txt",
    "results/paper_tables/MANIFEST.sha256",
    "results/paper_tables/README.md",
    "results/paper_tables/table3_main_tartan_results.csv",
    "results/paper_tables/table4_probability_controls.csv",
    "results/paper_tables/table5_full_path_calibration.csv",
    "results/paper_tables/table6_equal_budget_control.csv",
    "results/paper_tables/table7_eqmotion_operator_transfer.csv",
    "sbom/python-environment.cdx.json",
    "scripts/fetch_official_data.py",
    "scripts/verify_paper_summaries.py",
)

MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def markdown_files() -> list[Path]:
    documents = [
        ROOT / name
        for name in (
            "README.md",
            "CHANGELOG.md",
            "CITATION.cff",
            "CODE_OF_CONDUCT.md",
            "CONTRIBUTING.md",
            "LICENSE_PENDING.md",
            "SECURITY.md",
            "THIRD_PARTY_NOTICES.md",
        )
    ]
    documents.extend(sorted((ROOT / "docs").rglob("*.md")))
    documents.extend(sorted((ROOT / ".github").rglob("*.md")))
    documents.extend(sorted((ROOT / "results").rglob("*.md")))
    return [path for path in documents if path.is_file()]


def data_source_failures() -> list[str]:
    failures: list[str] = []
    registry_path = ROOT / "docs/data_sources.json"
    csv_path = ROOT / "docs/dataset_sources.csv"
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"invalid data-source registry: {exc}"]

    policy = registry.get("github_data_policy", {})
    if policy.get("third_party_raw_data_hosted") is not False:
        failures.append("data-source policy must forbid hosting third-party raw data")
    if policy.get("third_party_derived_data_hosted") is not False:
        failures.append("data-source policy must forbid hosting third-party derived data")

    datasets = {item.get("id"): item for item in registry.get("datasets", [])}
    trajair = datasets.get("trajair", {})
    if trajair.get("doi") != "10.1184/R1/14866251.v1":
        failures.append("TrajAir must use the official versioned DOI 10.1184/R1/14866251.v1")
    if trajair.get("license", {}).get("name") != "CC BY 4.0":
        failures.append("TrajAir registry must record the official CC BY 4.0 licence")
    expected_trajair = {"111_days.zip", "7days1.zip", "7days2.zip", "7days3.zip", "7days4.zip"}
    actual_trajair = {item.get("name") for item in trajair.get("official_files", [])}
    if actual_trajair != expected_trajair:
        failures.append("TrajAir registry must enumerate 111_days.zip and 7days1-4.zip")
    for item in trajair.get("official_files", []):
        if not item.get("download_url", "").startswith("https://ndownloader.figshare.com/files/"):
            failures.append(f"TrajAir file lacks an official file URL: {item.get('name')}")
        if len(item.get("official_md5", "")) != 32 or len(item.get("sha256", "")) != 64:
            failures.append(f"TrajAir file lacks complete checksums: {item.get('name')}")
    if trajair.get("redistribution_in_this_repository") is not False:
        failures.append("TrajAir redistribution flag must remain false")

    tartan = datasets.get("tartanaviation", {})
    if tartan.get("paper_pinned_commit") != "4065f5bb11c3d8e557dcaf20a56469e6b0738714":
        failures.append("TartanAviation registry has the wrong frozen commit")
    if tartan.get("official_download_script") != "adsb/download.py":
        failures.append("TartanAviation registry must delegate to adsb/download.py")
    if tartan.get("license", {}).get("dataset_payload") != "NOT_EXPLICITLY_STATED_ON_VERIFIED_OFFICIAL_PAGES":
        failures.append("TartanAviation payload licence uncertainty must remain explicit")
    if tartan.get("redistribution_in_this_repository") is not False:
        failures.append("TartanAviation redistribution flag must remain false")
    expected_tartan_assets = {
        "tartan_kagc_processed_official.zip",
        "tartan_kagc_raw_2022.zip",
        "tartan_kbtp_processed_official.zip",
        "tartan_kbtp_raw_2022.zip",
    }
    actual_tartan_assets = {
        item.get("name") for item in tartan.get("study_asset_checksums", [])
    }
    if actual_tartan_assets != expected_tartan_assets:
        failures.append("TartanAviation study-asset checksum names do not match the frozen protocol")

    try:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        failures.append(f"cannot read machine-readable source CSV: {exc}")
    else:
        if len(rows) != 7:
            failures.append(f"dataset_sources.csv must contain seven source rows, found {len(rows)}")
        for row in rows:
            if row.get("hosted_in_github") != "no":
                failures.append(f"source CSV must mark hosted_in_github=no: {row.get('asset')}")
    return failures


def relative_link_failures() -> list[str]:
    failures: list[str] = []
    for document in markdown_files():
        content = document.read_text(encoding="utf-8")
        for raw_target in MARKDOWN_LINK.findall(content):
            target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
            parsed = urlsplit(target)
            if parsed.scheme or target.startswith(("#", "mailto:")):
                continue
            relative = unquote(parsed.path)
            if not relative:
                continue
            resolved = (document.parent / relative).resolve()
            try:
                resolved.relative_to(ROOT)
            except ValueError:
                failures.append(
                    f"link escapes repository in {document.relative_to(ROOT)}: {target}"
                )
                continue
            if not resolved.exists():
                failures.append(
                    f"broken relative link in {document.relative_to(ROOT)}: {target}"
                )
    return failures


def strict_metadata_failures() -> list[str]:
    failures: list[str] = []
    if not (ROOT / "LICENSE").is_file():
        failures.append("approved root LICENSE is missing")
    pending_license = ROOT / ("LICENSE_" + "PENDING.md")
    if pending_license.exists():
        failures.append(f"release gate still present: {pending_license.name}")

    markers = (
        "[DATA_DOI_" + "PENDING]",
        "[SOFTWARE_DOI_" + "PENDING]",
        "[MODEL_DOI_" + "PENDING]",
        "[MODEL_RECORD_URL_" + "PENDING]",
        "[GITHUB_" + "REPOSITORY]",
        "[SOFTWARE_" + "DOI]",
        "LICENSE_" + "PENDING",
        "PENDING_",
    )
    public_metadata = [
        ROOT / "README.md",
        ROOT / "CITATION.cff",
        ROOT / "model_release.json",
        ROOT / "THIRD_PARTY_NOTICES.md",
    ]
    public_metadata.extend(sorted((ROOT / "docs").rglob("*.md")))
    for path in public_metadata:
        content = path.read_text(encoding="utf-8")
        for marker in markers:
            if marker in content:
                failures.append(
                    f"unresolved release metadata in {path.relative_to(ROOT)}: {marker}"
                )

    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    for field in ("repository-code:", "doi:", "date-released:"):
        if not re.search(rf"(?m)^{re.escape(field)}\s*\S+", citation):
            failures.append(f"CITATION.cff missing finalized field: {field[:-1]}")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    if not (ROOT / "LICENSE").is_file():
        failures.append("approved LICENSE is missing from the source tree")
    if not (
        re.search(r'(?m)^license\s*=\s*"[^"]+"\s*$', pyproject)
        or re.search(r'(?m)^license\s*=\s*\{\s*file\s*=\s*"LICENSE"\s*\}\s*$', pyproject)
    ):
        failures.append("pyproject.toml lacks a finalized license declaration")
    ascent_notice = (ROOT / "docs/ASCENT_NOTICE.md").read_text(encoding="utf-8")
    if "https://github.com/a-pru/ascent" not in ascent_notice:
        failures.append("ASCENT notice must identify the architecture reference")
    if "ASCENT-inspired / architecture-informed independently authored implementation" not in ascent_notice:
        failures.append("ASCENT notice must state independent authorship")
    if "not official ASCENT weights" not in ascent_notice:
        failures.append("ASCENT notice must reject an official-weight attribution")
    return failures


def audit(strict: bool) -> list[str]:
    failures = [
        f"missing required GitHub file: {path}"
        for path in REQUIRED_PATHS
        if not (ROOT / path).is_file()
    ]
    if not (ROOT / "LICENSE").is_file() and not (ROOT / "LICENSE_PENDING.md").is_file():
        failures.append("neither an approved LICENSE nor the explicit license gate is present")
    failures.extend(relative_link_failures())
    failures.extend(data_source_failures())
    if strict:
        failures.extend(strict_metadata_failures())
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict", action="store_true", help="require final licence and identifiers"
    )
    args = parser.parse_args()
    failures = audit(args.strict)
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        return 1
    mode = "strict release" if args.strict else "release-candidate"
    print(f"OK: GitHub {mode} presentation verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
