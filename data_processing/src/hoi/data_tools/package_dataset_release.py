"""Package one extracted IKEA recording location into a release-ready tree.

The source tree is treated as strictly read-only. The script only copies files
and creates zip archives in the destination tree (or prints a dry-run plan).

Observed extracted layout:
- most locations contain modules such as gripper/hand/wrist/umi/leica
- direct files inside a module are small metadata files (mostly JSON)
- many recorder folders contain one more level of recording/session folders
- release packaging therefore copies small metadata files directly but archives
  at the deeper sensor-stream boundary where possible, for example
  ``camera_rgb.zip`` inside an Aria session folder
- some folder names vary slightly, for example ``iphone_1 (babyblue)``
  vs. ``iphone_1``; output names are normalized semantically where it is safe

Example:
    python package_dataset_release.py \
        /data/ikea_recordings/extracted/office_1 \
        /data/ikea_recordings/release/office_1 \
        --dry-run
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


LOGGER = logging.getLogger("package_dataset_release")


# Centralized heuristics so the script is easy to adapt later.
DEFAULT_EXPECTED_MODULE_NAMES = ("gripper", "hand", "wrist", "umi", "leica")
DEFAULT_METADATA_SUFFIXES = (".json", ".csv", ".txt", ".md", ".yaml", ".yml")
DEFAULT_METADATA_FILENAMES = ("metadata",)
DEFAULT_SKIPPED_NAMES = (
    ".DS_Store",
    "Thumbs.db",
    "__MACOSX",
    ".Spotlight-V100",
    ".TemporaryItems",
    ".Trashes",
)
DEFAULT_METADATA_MAX_BYTES = 16 * 1024 * 1024
DEFAULT_ZIP_COMPRESSION = zipfile.ZIP_DEFLATED
DEFAULT_ZIP_COMPRESSLEVEL = 6
DEFAULT_ARCHIVE_PROGRESS_EVERY_FILES = 250
DEFAULT_ARCHIVE_PROGRESS_EVERY_PERCENT = 5.0
DEFAULT_7Z_BINARY_CANDIDATES = ("7z", "7zz", "7za")
DEFAULT_7Z_HEARTBEAT_SECONDS = 15.0

TRAILING_PARENTHETICAL_RE = re.compile(r"\s+\([^)]*\)\s*$")
NON_FILENAME_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]+")
MULTIPLE_UNDERSCORES_RE = re.compile(r"_+")


@dataclass(frozen=True)
class ArchiveCandidate:
    """A source directory that should become one output zip archive."""

    source_dir: Path
    output_stem: str
    note: str = ""


@dataclass
class ModuleInspection:
    """Inspection details for one observed module directory."""

    name: str
    path: Path
    metadata_files: List[Path] = field(default_factory=list)
    archive_candidates: List[ArchiveCandidate] = field(default_factory=list)
    skipped_entries: List[str] = field(default_factory=list)
    unexpected_entries: List[str] = field(default_factory=list)


@dataclass
class LocationInspection:
    """Inspection details for one extracted location root."""

    source_root: Path
    root_metadata_files: List[Path] = field(default_factory=list)
    modules: List[ModuleInspection] = field(default_factory=list)
    skipped_root_entries: List[str] = field(default_factory=list)
    unexpected_root_entries: List[str] = field(default_factory=list)
    missing_expected_modules: List[str] = field(default_factory=list)
    extra_modules: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class PackageAction:
    """One packaging action derived from the inspection report."""

    source: Path
    destination: Path
    action_type: str
    note: str = ""


@dataclass(frozen=True)
class PackagedArtifact:
    """One output artifact written to the destination tree."""

    relative_path: Path
    artifact_type: str
    size_bytes: int
    source_relative_path: str
    note: str = ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect one extracted recording location and package a release-ready "
            "copy into a separate destination tree."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Path to one extracted location, for example /data/.../extracted/office_1",
    )
    parser.add_argument(
        "destination",
        type=Path,
        help="Path to the release location output, for example /data/.../release/office_1",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect and print the packaging plan without writing output files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing destination files instead of skipping them.",
    )
    parser.add_argument(
        "--double-zip",
        action="store_true",
        help=(
            "Wrap every archive in a second outer zip so archive outputs become "
            "*.zip.zip. This is useful when an upload target auto-unzips one level."
        ),
    )
    parser.add_argument(
        "--metadata-max-bytes",
        type=int,
        default=DEFAULT_METADATA_MAX_BYTES,
        help="Maximum size for direct metadata files that should be copied as plain files.",
    )
    parser.add_argument(
        "--metadata-suffixes",
        nargs="+",
        default=list(DEFAULT_METADATA_SUFFIXES),
        help="Direct file suffixes that should be copied as metadata when small enough.",
    )
    parser.add_argument(
        "--expected-modules",
        nargs="*",
        default=list(DEFAULT_EXPECTED_MODULE_NAMES),
        help="Expected module names used only for inspection warnings and summaries.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--archive-progress-every-files",
        type=int,
        default=DEFAULT_ARCHIVE_PROGRESS_EVERY_FILES,
        help="Log archive-writing progress at least this often by file count.",
    )
    parser.add_argument(
        "--archive-progress-every-percent",
        type=float,
        default=DEFAULT_ARCHIVE_PROGRESS_EVERY_PERCENT,
        help="Log archive-writing progress at least this often by percent complete.",
    )
    return parser


def configure_logging(level_name: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_name.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
        stream=sys.stdout,
    )


def is_relative_to(path: Path, other: Path) -> bool:
    try:
        path.relative_to(other)
        return True
    except ValueError:
        return False


def is_hidden_or_skipped_name(name: str, skipped_names: Sequence[str]) -> bool:
    return name.startswith(".") or name in skipped_names


def looks_like_metadata_file(
    path: Path,
    metadata_suffixes: Sequence[str],
    metadata_max_bytes: int,
) -> bool:
    if not path.is_file():
        return False

    try:
        if path.stat().st_size > metadata_max_bytes:
            return False
    except OSError as exc:
        LOGGER.warning("Could not stat metadata candidate %s: %s", path, exc)
        return False

    if path.suffix.lower() in metadata_suffixes:
        return True

    if path.name.lower() in DEFAULT_METADATA_FILENAMES:
        return True

    return False


def sanitize_archive_stem(name: str, strip_trailing_parenthetical: bool) -> str:
    candidate = name.strip()
    if strip_trailing_parenthetical:
        candidate = TRAILING_PARENTHETICAL_RE.sub("", candidate).strip() or name.strip()

    candidate = candidate.replace("(", " ").replace(")", " ")
    candidate = NON_FILENAME_CHARS_RE.sub("_", candidate)
    candidate = MULTIPLE_UNDERSCORES_RE.sub("_", candidate).strip("._")
    return candidate or "archive"


def make_unique_stem(preferred: str, used_stems: set[str]) -> str:
    if preferred not in used_stems:
        return preferred

    index = 2
    while f"{preferred}_{index}" in used_stems:
        index += 1
    return f"{preferred}_{index}"


def resolve_archive_candidates(archive_dirs: Sequence[Path]) -> List[ArchiveCandidate]:
    if not archive_dirs:
        return []

    semantic_stems = {
        archive_dir: sanitize_archive_stem(
            archive_dir.name,
            strip_trailing_parenthetical=True,
        )
        for archive_dir in archive_dirs
    }
    raw_stems = {
        archive_dir: sanitize_archive_stem(
            archive_dir.name,
            strip_trailing_parenthetical=False,
        )
        for archive_dir in archive_dirs
    }
    semantic_counts = Counter(semantic_stems.values())
    used_stems: set[str] = set()
    candidates: List[ArchiveCandidate] = []

    # Unique semantic stems get the clean release name first.
    for archive_dir in sorted(archive_dirs, key=lambda item: item.name.casefold()):
        semantic_stem = semantic_stems[archive_dir]
        raw_stem = raw_stems[archive_dir]
        note = ""

        if semantic_counts[semantic_stem] == 1:
            output_stem = make_unique_stem(semantic_stem, used_stems)
            if output_stem != archive_dir.name:
                note = f"normalized from '{archive_dir.name}'"
        else:
            output_stem = make_unique_stem(raw_stem, used_stems)
            note = (
                f"kept a disambiguated stem for '{archive_dir.name}' because "
                f"multiple folders collapse to '{semantic_stem}'"
            )

        used_stems.add(output_stem)
        candidates.append(
            ArchiveCandidate(
                source_dir=archive_dir,
                output_stem=output_stem,
                note=note,
            )
        )

    return candidates


def inspect_module(
    module_dir: Path,
    metadata_suffixes: Sequence[str],
    metadata_max_bytes: int,
    skipped_names: Sequence[str],
) -> ModuleInspection:
    inspection = ModuleInspection(name=module_dir.name, path=module_dir)
    archive_dirs: List[Path] = []

    for child in sorted(module_dir.iterdir(), key=lambda item: item.name.casefold()):
        if is_hidden_or_skipped_name(child.name, skipped_names):
            inspection.skipped_entries.append(child.name)
            continue

        if child.is_symlink():
            inspection.unexpected_entries.append(f"{child.name} (symlink skipped)")
            continue

        if child.is_dir():
            # Only direct child directories of the module become release archives.
            # Example: office_1/gripper/aria_gripper -> office_1/gripper/aria_gripper.zip
            archive_dirs.append(child)
            continue

        if child.is_file():
            if looks_like_metadata_file(child, metadata_suffixes, metadata_max_bytes):
                inspection.metadata_files.append(child)
            else:
                inspection.unexpected_entries.append(
                    f"{child.name} (direct file skipped; not recognized as small metadata)"
                )
            continue

        inspection.unexpected_entries.append(f"{child.name} (unsupported entry type skipped)")

    inspection.archive_candidates = resolve_archive_candidates(archive_dirs)
    return inspection


def collect_visible_entries(
    directory: Path,
    metadata_suffixes: Sequence[str],
    metadata_max_bytes: int,
    skipped_names: Sequence[str],
) -> tuple[List[Path], List[Path], List[Path], List[str], List[str]]:
    metadata_files: List[Path] = []
    non_metadata_files: List[Path] = []
    child_dirs: List[Path] = []
    skipped_entries: List[str] = []
    unexpected_entries: List[str] = []

    for child in sorted(directory.iterdir(), key=lambda item: item.name.casefold()):
        if is_hidden_or_skipped_name(child.name, skipped_names):
            skipped_entries.append(child.name)
            continue

        if child.is_symlink():
            unexpected_entries.append(f"{child.name} (symlink skipped)")
            continue

        if child.is_dir():
            child_dirs.append(child)
            continue

        if child.is_file():
            if looks_like_metadata_file(child, metadata_suffixes, metadata_max_bytes):
                metadata_files.append(child)
            else:
                non_metadata_files.append(child)
            continue

        unexpected_entries.append(f"{child.name} (unsupported entry type skipped)")

    return metadata_files, non_metadata_files, child_dirs, skipped_entries, unexpected_entries


def should_package_immediate_children(
    module_name: str,
    directory: Path,
    metadata_files: Sequence[Path],
    child_dirs: Sequence[Path],
) -> bool:
    if not child_dirs:
        return False

    if metadata_files:
        return True

    if module_name == "leica" and directory.name.isdigit():
        return True

    return False


def append_copy_actions(
    actions: List[PackageAction],
    source_files: Sequence[Path],
    destination_dir: Path,
    action_type: str,
) -> None:
    for source_file in source_files:
        actions.append(
            PackageAction(
                source=source_file,
                destination=destination_dir / source_file.name,
                action_type=action_type,
            )
        )


def append_archive_action(
    actions: List[PackageAction],
    source_path: Path,
    destination_path: Path,
    action_type: str,
    note: str = "",
) -> None:
    actions.append(
        PackageAction(
            source=source_path,
            destination=destination_path,
            action_type=action_type,
            note=note,
        )
    )


def build_stream_parent_actions(
    actions: List[PackageAction],
    source_dir: Path,
    destination_dir: Path,
    archive_suffix: str,
    metadata_suffixes: Sequence[str],
    metadata_max_bytes: int,
    skipped_names: Sequence[str],
) -> None:
    metadata_files, non_metadata_files, child_dirs, skipped_entries, unexpected_entries = collect_visible_entries(
        source_dir,
        metadata_suffixes=metadata_suffixes,
        metadata_max_bytes=metadata_max_bytes,
        skipped_names=skipped_names,
    )

    append_copy_actions(actions, metadata_files, destination_dir, "nested_metadata_copy")

    if skipped_entries:
        LOGGER.info(
            "Skipping hidden/system entries while packaging %s: %s",
            source_dir,
            ", ".join(skipped_entries),
        )

    for entry in unexpected_entries:
        LOGGER.warning("Packaging irregularity in %s: %s", source_dir, entry)

    for extra_file in non_metadata_files:
        LOGGER.warning(
            "Direct non-metadata file found in %s; archiving it separately as %s.zip",
            source_dir,
            extra_file.name,
        )
        append_archive_action(
            actions,
            extra_file,
            destination_dir / f"{extra_file.name}{archive_suffix}",
            "nested_direct_file_archive",
            note="archived direct non-metadata file",
        )

    for archive_candidate in resolve_archive_candidates(child_dirs):
        append_archive_action(
            actions,
            archive_candidate.source_dir,
            destination_dir / f"{archive_candidate.output_stem}{archive_suffix}",
            "nested_stream_archive",
            note=archive_candidate.note,
        )


def inspect_location(
    source_root: Path,
    metadata_suffixes: Sequence[str],
    metadata_max_bytes: int,
    skipped_names: Sequence[str],
    expected_modules: Sequence[str],
) -> LocationInspection:
    inspection = LocationInspection(source_root=source_root)

    for child in sorted(source_root.iterdir(), key=lambda item: item.name.casefold()):
        if is_hidden_or_skipped_name(child.name, skipped_names):
            inspection.skipped_root_entries.append(child.name)
            continue

        if child.is_symlink():
            inspection.unexpected_root_entries.append(f"{child.name} (symlink skipped)")
            continue

        if child.is_dir():
            inspection.modules.append(
                inspect_module(
                    child,
                    metadata_suffixes=metadata_suffixes,
                    metadata_max_bytes=metadata_max_bytes,
                    skipped_names=skipped_names,
                )
            )
            continue

        if child.is_file():
            if looks_like_metadata_file(child, metadata_suffixes, metadata_max_bytes):
                inspection.root_metadata_files.append(child)
            else:
                inspection.unexpected_root_entries.append(
                    f"{child.name} (location-root file skipped; not recognized as small metadata)"
                )
            continue

        inspection.unexpected_root_entries.append(f"{child.name} (unsupported entry type skipped)")

    observed_modules = [module.name for module in inspection.modules]
    inspection.missing_expected_modules = [
        module_name for module_name in expected_modules if module_name not in observed_modules
    ]
    inspection.extra_modules = [
        module_name for module_name in observed_modules if module_name not in expected_modules
    ]
    return inspection


def build_package_actions(
    inspection: LocationInspection,
    destination_root: Path,
    archive_suffix: str,
    metadata_suffixes: Sequence[str],
    metadata_max_bytes: int,
    skipped_names: Sequence[str],
) -> List[PackageAction]:
    actions: List[PackageAction] = []

    for metadata_file in inspection.root_metadata_files:
        actions.append(
            PackageAction(
                source=metadata_file,
                destination=destination_root / metadata_file.name,
                action_type="root_metadata_copy",
            )
        )

    for module in inspection.modules:
        module_destination = destination_root / module.name
        append_copy_actions(actions, module.metadata_files, module_destination, "module_metadata_copy")

        for module_child in module.archive_candidates:
            child_source = module_child.source_dir
            child_destination = module_destination / module_child.output_stem
            child_metadata, child_non_metadata, child_dirs, skipped_entries, unexpected_entries = collect_visible_entries(
                child_source,
                metadata_suffixes=metadata_suffixes,
                metadata_max_bytes=metadata_max_bytes,
                skipped_names=skipped_names,
            )

            if skipped_entries:
                LOGGER.info(
                    "Skipping hidden/system entries while inspecting %s: %s",
                    child_source,
                    ", ".join(skipped_entries),
                )

            for entry in unexpected_entries:
                LOGGER.warning("Packaging irregularity in %s: %s", child_source, entry)

            if not child_dirs:
                append_archive_action(
                    actions,
                    child_source,
                    module_destination / f"{module_child.output_stem}{archive_suffix}",
                    "module_stream_archive",
                    note=module_child.note,
                )
                continue

            if should_package_immediate_children(
                module.name,
                child_source,
                child_metadata,
                child_dirs,
            ):
                build_stream_parent_actions(
                    actions,
                    source_dir=child_source,
                    destination_dir=child_destination,
                    archive_suffix=archive_suffix,
                    metadata_suffixes=metadata_suffixes,
                    metadata_max_bytes=metadata_max_bytes,
                    skipped_names=skipped_names,
                )
                continue

            append_copy_actions(actions, child_metadata, child_destination, "nested_metadata_copy")

            for extra_file in child_non_metadata:
                LOGGER.warning(
                    "Direct non-metadata file found in %s; archiving it separately as %s.zip",
                    child_source,
                    extra_file.name,
                )
                append_archive_action(
                    actions,
                    extra_file,
                    child_destination / f"{extra_file.name}{archive_suffix}",
                    "nested_direct_file_archive",
                    note="archived direct non-metadata file",
                )

            for grandchild_candidate in resolve_archive_candidates(child_dirs):
                grandchild_source = grandchild_candidate.source_dir
                grandchild_destination = child_destination / grandchild_candidate.output_stem
                grandchild_metadata, grandchild_non_metadata, grandchild_dirs, grandchild_skipped, grandchild_unexpected = collect_visible_entries(
                    grandchild_source,
                    metadata_suffixes=metadata_suffixes,
                    metadata_max_bytes=metadata_max_bytes,
                    skipped_names=skipped_names,
                )

                if grandchild_skipped:
                    LOGGER.info(
                        "Skipping hidden/system entries while inspecting %s: %s",
                        grandchild_source,
                        ", ".join(grandchild_skipped),
                    )

                for entry in grandchild_unexpected:
                    LOGGER.warning("Packaging irregularity in %s: %s", grandchild_source, entry)

                if grandchild_dirs:
                    build_stream_parent_actions(
                        actions,
                        source_dir=grandchild_source,
                        destination_dir=grandchild_destination,
                        archive_suffix=archive_suffix,
                        metadata_suffixes=metadata_suffixes,
                        metadata_max_bytes=metadata_max_bytes,
                        skipped_names=skipped_names,
                    )
                else:
                    append_archive_action(
                        actions,
                        grandchild_source,
                        child_destination / f"{grandchild_candidate.output_stem}{archive_suffix}",
                        "nested_stream_archive",
                        note=grandchild_candidate.note,
                    )

    return actions


def human_size(num_bytes: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{num_bytes} B"


def human_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(total_seconds, 60)
    hours, mins = divmod(minutes, 60)
    if hours:
        return f"{hours}h {mins}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def safe_stat_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError as exc:
        LOGGER.warning("Could not stat %s: %s", path, exc)
        return 0


def resolve_7z_binary() -> str:
    for candidate in DEFAULT_7Z_BINARY_CANDIDATES:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise FileNotFoundError(
        "Could not find a 7z binary. Install p7zip-full or a compatible 7z package "
        "and ensure one of these commands is available on PATH: "
        + ", ".join(DEFAULT_7Z_BINARY_CANDIDATES)
    )


def log_inspection_summary(inspection: LocationInspection, destination_root: Path) -> None:
    LOGGER.info("Source location: %s", inspection.source_root)
    LOGGER.info("Destination root: %s", destination_root)

    observed_module_names = [module.name for module in inspection.modules]
    LOGGER.info(
        "Observed modules (%d): %s",
        len(observed_module_names),
        ", ".join(observed_module_names) if observed_module_names else "none",
    )

    if inspection.missing_expected_modules:
        LOGGER.warning(
            "Missing expected modules: %s",
            ", ".join(inspection.missing_expected_modules),
        )
    else:
        LOGGER.info("Missing expected modules: none")

    if inspection.extra_modules:
        LOGGER.warning("Extra non-standard modules: %s", ", ".join(inspection.extra_modules))

    if inspection.root_metadata_files:
        LOGGER.info(
            "Location-root metadata files: %s",
            ", ".join(path.name for path in inspection.root_metadata_files),
        )

    if inspection.skipped_root_entries:
        LOGGER.info(
            "Skipped hidden/system entries at location root: %s",
            ", ".join(inspection.skipped_root_entries),
        )

    if inspection.unexpected_root_entries:
        for entry in inspection.unexpected_root_entries:
            LOGGER.warning("Location-root irregularity: %s", entry)

    for module in inspection.modules:
        LOGGER.info(
            "Module '%s': %d metadata file(s), %d top-level packaging container(s)",
            module.name,
            len(module.metadata_files),
            len(module.archive_candidates),
        )

        if module.metadata_files:
            LOGGER.info(
                "  metadata: %s",
                ", ".join(path.name for path in module.metadata_files),
            )
        else:
            LOGGER.info("  metadata: none")

        if module.archive_candidates:
            container_descriptions = []
            for candidate in module.archive_candidates:
                if candidate.note:
                    container_descriptions.append(
                        f"{candidate.source_dir.name} -> {candidate.output_stem} ({candidate.note})"
                    )
                else:
                    container_descriptions.append(
                        f"{candidate.source_dir.name} -> {candidate.output_stem}"
                    )
            LOGGER.info("  top-level containers: %s", ", ".join(container_descriptions))
        else:
            LOGGER.info("  top-level containers: none")

        if module.skipped_entries:
            LOGGER.info(
                "  skipped hidden/system entries: %s",
                ", ".join(module.skipped_entries),
            )

        if module.unexpected_entries:
            for entry in module.unexpected_entries:
                LOGGER.warning("  %s irregularity: %s", module.name, entry)


def log_action_summary(
    actions: Sequence[PackageAction],
    source_root: Path,
    destination_root: Path,
    dry_run: bool,
) -> None:
    prefix = "Dry-run action" if dry_run else "Planned action"
    if not actions:
        LOGGER.warning("No packaging actions were generated from the inspection.")
        return

    LOGGER.info("%s count: %d", prefix, len(actions))
    for action in actions:
        source_relative = action.source.relative_to(source_root)
        destination_relative = action.destination.relative_to(destination_root)
        note_suffix = f" [{action.note}]" if action.note else ""
        LOGGER.info(
            "%s: %s -> %s (%s)%s",
            prefix,
            source_relative,
            destination_relative,
            action.action_type,
            note_suffix,
        )


def iter_visible_files(root: Path, skipped_names: Sequence[str]) -> Iterable[Path]:
    for current_root, dir_names, file_names in os.walk(root):
        dir_names[:] = sorted(
            [
                name
                for name in dir_names
                if not is_hidden_or_skipped_name(name, skipped_names)
            ],
            key=str.casefold,
        )
        for file_name in sorted(file_names, key=str.casefold):
            if is_hidden_or_skipped_name(file_name, skipped_names):
                continue

            file_path = Path(current_root) / file_name
            if file_path.is_symlink():
                LOGGER.warning("Skipping symlink inside archive source: %s", file_path)
                continue
            if file_path.is_file():
                yield file_path


def write_root_directory_entry(zip_file: zipfile.ZipFile, directory_name: str) -> None:
    info = zipfile.ZipInfo(f"{directory_name.rstrip('/')}/")
    info.external_attr = 0o755 << 16
    zip_file.writestr(info, "")


def list_archive_input_files(source_path: Path, skipped_names: Sequence[str]) -> List[Path]:
    if source_path.is_dir():
        return list(iter_visible_files(source_path, skipped_names))

    if source_path.is_file():
        if is_hidden_or_skipped_name(source_path.name, skipped_names):
            return []
        if source_path.is_symlink():
            LOGGER.warning("Skipping symlink archive source: %s", source_path)
            return []
        return [source_path]

    raise FileNotFoundError(f"Archive source does not exist or is unsupported: {source_path}")


def resolve_archive_member_name(source_path: Path, archive_file: Path) -> Path:
    if source_path.is_dir():
        return archive_file.relative_to(source_path.parent)
    return Path(archive_file.name)


def resolve_inner_archive_name(destination_zip: Path) -> str:
    if destination_zip.name.endswith(".zip.zip"):
        return destination_zip.name[:-4]
    return destination_zip.name


def wrap_archive_in_outer_zip(inner_archive_path: Path, destination_zip: Path) -> int:
    inner_entry_name = resolve_inner_archive_name(destination_zip)

    with tempfile.NamedTemporaryFile(
        prefix=f"{destination_zip.stem}_outer_",
        suffix=".tmp.zip",
        dir=destination_zip.parent,
        delete=False,
    ) as temp_file:
        outer_temp_path = Path(temp_file.name)
    outer_temp_path.unlink(missing_ok=True)

    try:
        LOGGER.info(
            "  wrapping inner archive %s into outer archive %s",
            inner_entry_name,
            destination_zip.name,
        )
        with zipfile.ZipFile(
            outer_temp_path,
            mode="w",
            compression=zipfile.ZIP_STORED,
        ) as zip_file:
            zip_file.write(inner_archive_path, arcname=inner_entry_name)

        os.replace(outer_temp_path, destination_zip)
    except Exception:
        outer_temp_path.unlink(missing_ok=True)
        raise

    return destination_zip.stat().st_size


def create_zip_archive(
    source_path: Path,
    destination_zip: Path,
    skipped_names: Sequence[str],
    overwrite: bool,
    progress_every_files: int,
    progress_every_percent: float,
    double_zip: bool,
) -> int:
    destination_zip.parent.mkdir(parents=True, exist_ok=True)

    if destination_zip.exists():
        if destination_zip.is_dir():
            raise IsADirectoryError(
                f"Destination archive path exists as a directory: {destination_zip}"
            )
        if overwrite:
            LOGGER.warning("Overwriting existing archive: %s", destination_zip)
        else:
            raise FileExistsError(destination_zip)

    archive_files = list_archive_input_files(source_path, skipped_names)
    archive_file_sizes = [safe_stat_size(file_path) for file_path in archive_files]
    total_files = len(archive_files)
    total_input_bytes = sum(archive_file_sizes)
    seven_zip_binary = resolve_7z_binary()

    LOGGER.info(
        "Starting archive %s from %s with %s: %d file(s), %s input",
        destination_zip.name,
        source_path,
        Path(seven_zip_binary).name,
        total_files,
        human_size(total_input_bytes),
    )

    with tempfile.NamedTemporaryFile(
        prefix=f"{destination_zip.stem}_",
        suffix=".tmp.zip",
        dir=destination_zip.parent,
        delete=False,
    ) as temp_file:
        temp_path = Path(temp_file.name)
    temp_path.unlink(missing_ok=True)

    try:
        if not archive_files:
            if not source_path.is_dir():
                raise FileNotFoundError(
                    f"Archive source produced no visible files: {source_path}"
                )
            with zipfile.ZipFile(
                temp_path,
                mode="w",
                compression=DEFAULT_ZIP_COMPRESSION,
                compresslevel=DEFAULT_ZIP_COMPRESSLEVEL,
            ) as zip_file:
                write_root_directory_entry(zip_file, source_path.name)
        else:
            with tempfile.NamedTemporaryFile(
                prefix=f"{destination_zip.stem}_",
                suffix=".list",
                dir=destination_zip.parent,
                delete=False,
                mode="w",
                encoding="utf-8",
                newline="\n",
            ) as list_file:
                list_path = Path(list_file.name)
                processed_input_bytes = 0
                next_percent_log = progress_every_percent if progress_every_percent > 0 else float("inf")
                effective_progress_every_files = max(1, progress_every_files)
                if total_files <= 50:
                    effective_progress_every_files = 1

                for index, (file_path, file_size) in enumerate(
                    zip(archive_files, archive_file_sizes),
                    start=1,
                ):
                    archive_name = resolve_archive_member_name(source_path, file_path)
                    list_file.write(os.fspath(archive_name))
                    list_file.write("\n")
                    processed_input_bytes += file_size

                    percent_complete = 100.0 * index / total_files
                    should_log_progress = False
                    if index == 1 or index == total_files:
                        should_log_progress = True
                    if index % effective_progress_every_files == 0:
                        should_log_progress = True
                    while percent_complete >= next_percent_log:
                        should_log_progress = True
                        next_percent_log += (
                            progress_every_percent if progress_every_percent > 0 else float("inf")
                        )

                    if should_log_progress:
                        LOGGER.info(
                            "  archive input scan %s: %d/%d file(s) (%.1f%%), %s/%s input",
                            destination_zip.name,
                            index,
                            total_files,
                            percent_complete,
                            human_size(processed_input_bytes),
                            human_size(total_input_bytes),
                        )

            try:
                command = [
                    seven_zip_binary,
                    "a",
                    "-tzip",
                    "-mx=1",
                    "-mmt=on",
                    "-bb0",
                    "-bd",
                    "-scsUTF-8",
                    "-y",
                    os.fspath(temp_path),
                    f"@{os.fspath(list_path)}",
                ]
                LOGGER.info(
                    "  invoking 7z for %s from working directory %s",
                    destination_zip.name,
                    source_path.parent,
                )
                LOGGER.info(
                    "  starting 7z compression for %s with fast zip settings (-mx=1, -mmt=on)",
                    destination_zip.name,
                )
                start_time = time.monotonic()
                process = subprocess.Popen(
                    command,
                    cwd=source_path.parent,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )

                while True:
                    try:
                        returncode = process.wait(timeout=DEFAULT_7Z_HEARTBEAT_SECONDS)
                        break
                    except subprocess.TimeoutExpired:
                        LOGGER.info(
                            "  7z compression %s: still running after %s, temp archive size %s",
                            destination_zip.name,
                            human_duration(time.monotonic() - start_time),
                            human_size(safe_stat_size(temp_path)),
                        )

                output = (process.stdout.read() if process.stdout is not None else "") or ""
                if returncode != 0:
                    output = output.strip()
                    if output:
                        LOGGER.error("7z failed while building %s:\n%s", destination_zip.name, output)
                    raise subprocess.CalledProcessError(
                        returncode,
                        command,
                        output=output,
                    )
                LOGGER.info(
                    "  7z compression %s: finished in %s, temp archive size %s",
                    destination_zip.name,
                    human_duration(time.monotonic() - start_time),
                    human_size(safe_stat_size(temp_path)),
                )
            finally:
                list_path.unlink(missing_ok=True)

        if double_zip:
            final_size = wrap_archive_in_outer_zip(temp_path, destination_zip)
        else:
            os.replace(temp_path, destination_zip)
            final_size = destination_zip.stat().st_size
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    finally:
        temp_path.unlink(missing_ok=True)

    if not archive_files:
        LOGGER.warning("Created an empty archive because %s has no visible files.", source_path)
    else:
        LOGGER.info(
            "Finished archive %s: %d file(s) written, output size %s",
            destination_zip.name,
            total_files,
            human_size(final_size),
        )

    return final_size


def copy_metadata_file(source_file: Path, destination_file: Path, overwrite: bool) -> int:
    destination_file.parent.mkdir(parents=True, exist_ok=True)

    if destination_file.exists():
        if destination_file.is_dir():
            raise IsADirectoryError(
                f"Destination metadata path exists as a directory: {destination_file}"
            )
        if overwrite:
            LOGGER.warning("Overwriting existing file: %s", destination_file)
        else:
            raise FileExistsError(destination_file)

    shutil.copy2(source_file, destination_file)
    return destination_file.stat().st_size


def write_manifest(destination_root: Path, artifacts: Sequence[PackagedArtifact]) -> None:
    manifest_path = destination_root / "manifest.csv"
    sizes_path = destination_root / "sizes.txt"

    destination_root.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        prefix="manifest_",
        suffix=".csv.tmp",
        dir=destination_root,
        delete=False,
        mode="w",
        newline="",
        encoding="utf-8",
    ) as temp_manifest:
        temp_manifest_path = Path(temp_manifest.name)
        writer = csv.writer(temp_manifest)
        writer.writerow(
            [
                "relative_path",
                "artifact_type",
                "size_bytes",
                "size_human",
                "source_relative_path",
                "note",
            ]
        )
        for artifact in sorted(artifacts, key=lambda item: item.relative_path.as_posix()):
            writer.writerow(
                [
                    artifact.relative_path.as_posix(),
                    artifact.artifact_type,
                    artifact.size_bytes,
                    human_size(artifact.size_bytes),
                    artifact.source_relative_path,
                    artifact.note,
                ]
            )

    with tempfile.NamedTemporaryFile(
        prefix="sizes_",
        suffix=".txt.tmp",
        dir=destination_root,
        delete=False,
        mode="w",
        encoding="utf-8",
    ) as temp_sizes:
        temp_sizes_path = Path(temp_sizes.name)
        temp_sizes.write("relative_path\ttype\tsize_bytes\tsize_human\tsource_relative_path\n")
        for artifact in sorted(artifacts, key=lambda item: item.relative_path.as_posix()):
            temp_sizes.write(
                "\t".join(
                    [
                        artifact.relative_path.as_posix(),
                        artifact.artifact_type,
                        str(artifact.size_bytes),
                        human_size(artifact.size_bytes),
                        artifact.source_relative_path,
                    ]
                )
                + "\n"
            )

    os.replace(temp_manifest_path, manifest_path)
    os.replace(temp_sizes_path, sizes_path)


def execute_actions(
    actions: Sequence[PackageAction],
    source_root: Path,
    destination_root: Path,
    skipped_names: Sequence[str],
    dry_run: bool,
    overwrite: bool,
    archive_progress_every_files: int,
    archive_progress_every_percent: float,
    double_zip: bool,
) -> List[PackagedArtifact]:
    artifacts: List[PackagedArtifact] = []

    if dry_run:
        return artifacts

    LOGGER.info("Executing %d packaging action(s)...", len(actions))

    for action_index, action in enumerate(actions, start=1):
        source_relative = action.source.relative_to(source_root)
        destination_relative = action.destination.relative_to(destination_root)
        note_suffix = f" [{action.note}]" if action.note else ""

        LOGGER.info(
            "Starting action %d/%d: %s -> %s (%s)%s",
            action_index,
            len(actions),
            source_relative,
            destination_relative,
            action.action_type,
            note_suffix,
        )

        try:
            if action.action_type.endswith("_copy"):
                size_bytes = copy_metadata_file(action.source, action.destination, overwrite)
                artifact_type = "metadata_file"
            elif action.action_type.endswith("_archive"):
                size_bytes = create_zip_archive(
                    source_path=action.source,
                    destination_zip=action.destination,
                    skipped_names=skipped_names,
                    overwrite=overwrite,
                    progress_every_files=archive_progress_every_files,
                    progress_every_percent=archive_progress_every_percent,
                    double_zip=double_zip,
                )
                artifact_type = "zip_archive"
            else:
                LOGGER.warning("Skipping unknown action type: %s", action.action_type)
                continue
        except FileExistsError:
            LOGGER.warning(
                "Skipping existing destination artifact without overwrite for action %d/%d: %s",
                action_index,
                len(actions),
                action.destination,
            )
            continue

        LOGGER.info(
            "Completed action %d/%d: %s -> %s (%s, %s)",
            action_index,
            len(actions),
            source_relative,
            destination_relative,
            artifact_type,
            human_size(size_bytes),
        )
        artifacts.append(
            PackagedArtifact(
                relative_path=destination_relative,
                artifact_type=artifact_type,
                size_bytes=size_bytes,
                source_relative_path=source_relative.as_posix(),
                note=(
                    f"{action.note}; double-zipped"
                    if action.note and artifact_type == "zip_archive" and double_zip
                    else "double-zipped"
                    if artifact_type == "zip_archive" and double_zip
                    else action.note
                ),
            )
        )

    return artifacts


def validate_paths(source_root: Path, destination_root: Path) -> None:
    if not source_root.exists():
        raise FileNotFoundError(f"Source location does not exist: {source_root}")
    if not source_root.is_dir():
        raise NotADirectoryError(f"Source location is not a directory: {source_root}")

    resolved_source = source_root.resolve()
    resolved_destination = destination_root.resolve(strict=False)

    if resolved_source == resolved_destination:
        raise ValueError("Destination must be different from the source location.")
    if is_relative_to(resolved_destination, resolved_source):
        raise ValueError("Destination cannot be inside the source tree.")
    if is_relative_to(resolved_source, resolved_destination):
        raise ValueError("Source cannot be inside the destination tree.")


def warn_if_merging(destination_root: Path) -> None:
    if destination_root.exists():
        if not destination_root.is_dir():
            raise NotADirectoryError(
                f"Destination exists but is not a directory: {destination_root}"
            )
        try:
            has_entries = any(destination_root.iterdir())
        except OSError as exc:
            LOGGER.warning("Could not inspect destination contents at %s: %s", destination_root, exc)
            return

        if has_entries:
            LOGGER.warning(
                "Destination already exists and is not empty; packaging will merge without "
                "deleting anything. Existing files are skipped unless --overwrite is set."
            )


def log_result_summary(destination_root: Path, artifacts: Sequence[PackagedArtifact]) -> None:
    total_bytes = sum(artifact.size_bytes for artifact in artifacts)
    type_counts = Counter(artifact.artifact_type for artifact in artifacts)

    LOGGER.info("Output summary:")
    LOGGER.info("  destination: %s", destination_root)
    LOGGER.info("  artifacts written: %d", len(artifacts))
    LOGGER.info(
        "  metadata files: %d",
        type_counts.get("metadata_file", 0),
    )
    LOGGER.info(
        "  zip archives: %d",
        type_counts.get("zip_archive", 0),
    )
    LOGGER.info("  total output size: %s", human_size(total_bytes))
    LOGGER.info("  manifest: %s", destination_root / "manifest.csv")
    LOGGER.info("  sizes report: %s", destination_root / "sizes.txt")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    source_root = args.source.expanduser()
    destination_root = args.destination.expanduser()
    metadata_suffixes = tuple(suffix.lower() for suffix in args.metadata_suffixes)
    expected_modules = tuple(args.expected_modules)
    skipped_names = tuple(DEFAULT_SKIPPED_NAMES)
    archive_suffix = ".zip.zip" if args.double_zip else ".zip"

    validate_paths(source_root, destination_root)
    warn_if_merging(destination_root)

    if args.double_zip:
        LOGGER.info(
            "Double-zip mode enabled: archive outputs will use the %s suffix.",
            archive_suffix,
        )

    inspection = inspect_location(
        source_root=source_root,
        metadata_suffixes=metadata_suffixes,
        metadata_max_bytes=args.metadata_max_bytes,
        skipped_names=skipped_names,
        expected_modules=expected_modules,
    )
    log_inspection_summary(inspection, destination_root)

    actions = build_package_actions(
        inspection,
        destination_root,
        archive_suffix=archive_suffix,
        metadata_suffixes=metadata_suffixes,
        metadata_max_bytes=args.metadata_max_bytes,
        skipped_names=skipped_names,
    )
    log_action_summary(
        actions,
        source_root=source_root,
        destination_root=destination_root,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        LOGGER.info("Dry-run complete. No files were written.")
        return 0

    artifacts = execute_actions(
        actions=actions,
        source_root=source_root,
        destination_root=destination_root,
        skipped_names=skipped_names,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        archive_progress_every_files=args.archive_progress_every_files,
        archive_progress_every_percent=args.archive_progress_every_percent,
        double_zip=args.double_zip,
    )
    write_manifest(destination_root, artifacts)
    log_result_summary(destination_root, artifacts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
