from pathlib import Path
from typing import Iterator, Tuple, List, Dict
import fnmatch, os, re


class RecordingIndex:
    """
    Fast lookup / iterator for IKEA recordings.
    Builds a 3-level in-memory index:
        location  →  interaction  →  recorder directory
    """

    # ── interaction ↔ recorder tables ────────────────────────────────────
    REC_TYPES_ARIA = (
        ("gripper", "aria_gripper"),
        ("gripper", "aria_human"),
        ("hand",    "aria_human"),
        ("spot",    "aria_spot"),
        ("umi",     "aria_human"),
        ("wrist",   "aria_human"),
        ("wrist",   "aria_wrist"),
    )
    REC_TYPES_IPHONE = (
        ("gripper", "iphone_*"),   # iphone_1 / iphone_2 …
        ("hand",    "iphone_*"),
        ("spot",    "iphone_*"),
        ("umi",     "iphone_*"),
        ("wrist",   "iphone_*"),
    )
    REC_TYPES_SPOT    = (("spot",  "spot"),)
    REC_TYPES_UMI     = (("umi",   "umi_gripper"),)
    REC_TYPES_GRIPPER = (("gripper", "gripper"),
                         ("hand", "gripper"),
                         ("wrist", "gripper"),)

    ALL_REC_TYPES = (
        REC_TYPES_ARIA
        + REC_TYPES_IPHONE
        + REC_TYPES_SPOT
        + REC_TYPES_UMI
        + REC_TYPES_GRIPPER
    )

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        self._index: Dict[str, Dict[str, Dict[str, Path]]] | None = None

    def _build(self) -> None:
        """Populate self._index[location][interaction][recorder] = Path."""
        idx: Dict[str, Dict[str, Dict[str, Path]]] = {}

        for loc_dir in self.root.iterdir():
            if not loc_dir.is_dir():
                continue
            loc = loc_dir.name
            for int_dir in loc_dir.iterdir():
                if not int_dir.is_dir():
                    continue
                interaction = int_dir.name
                for rec_dir in int_dir.iterdir():
                    if not rec_dir.is_dir():
                        continue
                    recorder = rec_dir.name

                    if not any(
                        interaction == pat_int and fnmatch.fnmatch(recorder, pat_rec)
                        for pat_int, pat_rec in self.ALL_REC_TYPES
                    ):
                        continue

                    idx.setdefault(loc, {}) \
                       .setdefault(interaction, {})[recorder] = rec_dir
        self._index = idx

    # ── public helpers ────────────────────────────────────────────────────
    @property
    def index(self):
        if self._index is None:
            self._build()
        return self._index

    @property
    def locations(self) -> List[str]:
        return sorted(self.index.keys())

    def iter_recorders(self) -> Iterator[Tuple[str, str, str, Path]]:
        """Yield (location, interaction, recorder, recorder_dir)."""
        for loc, int_dict in self.index.items():
            for interaction, rec_dict in int_dict.items():
                for recorder, p in rec_dict.items():
                    yield loc, interaction, recorder, p

    # ── VRS utilities ─────────────────────────────────────────────────────
    @staticmethod
    def _is_vrs(fname: str) -> bool:
        return fname.endswith(".vrs") and not fname.startswith("._")

    def vrs_under_recorder(self, recorder_dir: Path) -> List[Path]:
        out: List[Path] = []
        for p, _, fnames in os.walk(recorder_dir):
            out.extend(Path(p) / f for f in fnames if self._is_vrs(f))
        return out
    
    @classmethod
    def _is_valid_recfile(cls, name: str) -> bool:
        """True if the basename begins with the expected 3-field prefix."""
        return bool(cls._NAME_RE.match(Path(name).stem))
    
    # ---------- helper: does a file/dir name follow the canonical pattern?
    _NAME_RE = re.compile(
        r"""^
            [A-Za-z]+           _   # <recLocName>
            [0-9]+[A-Za-z]*     _   # <recLocNumber>  (e.g. 1   or  2p)
            \d+(?:-\d+)*        _   # <cabinetIndex>  (1  or 1-8 or 1-2-4)
        """,
        re.VERBOSE,
    )

    # ── interaction-index parser (third underscore-token) ─────────────────
    @staticmethod
    def _parse_interaction_index(name: str) -> str | None:
        """
        Filenames take the form
            <locName>_<locNumber>_<index>_<anything...>
        Return the 3-rd token if it matches  \\d+(-\\d+)*   else None.
        """
        tokens = Path(name).stem.split('_')
        if len(tokens) < 3:
            return None
        idx_tok = tokens[2]
        return idx_tok if re.fullmatch(r"\d+(?:-\d+)*", idx_tok) else None

    # ── flexible query (one tuple per cabinet-index) ──────────────────────
    def query(
        self,
        location:          str | None = None,
        interaction:       str | None = None,
        recorder:          str | None = None,
        interaction_index: str | None = None,
    ) -> List[Tuple[str, str, str, str | None, Path]]:
        """
        Return tuples **for every cabinet-index present**:

            (location, interaction, recorder, index_or_None, recorder_dir)

        • Pass None for any filter to match all.
        • `recorder` may include wild-cards (fnmatch rules).
        • If `interaction_index` is supplied (may include wild-cards),
          only those indices are returned.
        """
        from fnmatch import fnmatch

        if recorder and not any(c in recorder for c in "*?"):
            recorder += "*"

        # Default: match all indices (“*”), but remember if the caller is filtering
        if interaction_index is None:
            interaction_index = "*"
        elif not any(c in interaction_index for c in "*?"):
            interaction_index += "*"
        idx_filter_on = interaction_index not in ("*", None)

        results: List[Tuple[str, str, str, str | None, Path]] = []

        for loc, inter, rec, rec_dir in self.iter_recorders():
            if location    and loc  != location:    continue
            if interaction and inter != interaction: continue
            if recorder and not fnmatch(rec, recorder): continue

            # collect all index tokens in this recorder directory
            # collect all index tokens only from names that follow the pattern
            tokens = {
                self._parse_interaction_index(e.name)
                for e in rec_dir.iterdir()
                if self._is_valid_recfile(e.name)
            }
            tokens.discard(None)

            # if no index-bearing files, still return one tuple with idx=None
            if not tokens:
                # if not idx_filter_on or fnmatch("", interaction_index):
                #     results.append((loc, inter, rec, None, rec_dir))
                continue

            for tok in sorted(tokens):
                if not idx_filter_on or fnmatch(tok, interaction_index):
                    results.append((loc, inter, rec, tok, rec_dir))

        return results

    # ── gather *.vrs files for any filter combo ───────────────────────────
    def vrs_files(
        self,
        location:    str | None = None,
        interaction: str | None = None,
        recorder:    str | None = None,
        interaction_index: str | None = None,
    ) -> List[Path]:
        files: List[Path] = []
        # Use wildcard index so query always returns tuples ending with rec_dir
        for tpl in self.query(location, interaction, recorder,
                              interaction_index=interaction_index):
            files.extend(self.vrs_under_recorder(tpl[-1]))

        # remove duplicate paths
        files = sorted(set(files))
        return files
    
    def get_all_extracted_data_streams(
        self,
        extraction_path: Path,
        image_streams: bool = False,
        sensor_streams: bool = False
    ) -> List[Path]:
        """
        Returns every sub-directory (at any depth) that contains:
        • a timestamp-named file like 0.png, 1634551234567.jpg, …  OR
        • a file named data.csv

        Optionally filter for specific stream types:
        - image_streams: only return frame/image directories (e.g. 'camera_rgb')
        - sensor_streams: only return sensor directories (e.g. 'imu', 'wrench', etc.)

        Parameters
        ----------
        extraction_path : Path
            Top-level directory to begin the search.
        image_streams : bool, optional
            If True, only return directories named in `image_dirs`.
        sensor_streams : bool, optional
            If True, only return directories named in `sensor_dirs`.

        Returns
        -------
        List[Path]
            Directories or file paths that satisfy the rule (deduplicated, absolute paths).
        """
        import os, re
        from pathlib import Path

        _TS_RE = re.compile(r"^\d+\.(png|jpe?g|npy)$", re.IGNORECASE)
        extraction_path = Path(extraction_path).expanduser().resolve()
        hits: list[Path] = []

        # Define which directory names qualify
        image_dirs = ["camera_rgb", "image_raw_color"]
        sensor_dirs = ["digit", "joint_states", "wrench", "imu", "depth_registered", "telemetry", "camera_depth"]

        for dir_path, _, filenames in os.walk(extraction_path):
            fn_lower = [f.lower() for f in filenames]
            dir_path = Path(dir_path)

            # Apply stream-type filters
            if image_streams and dir_path.name not in image_dirs:
                continue
            if sensor_streams and dir_path.name not in sensor_dirs:
                continue
            if sensor_streams and ("aria" in str(dir_path) and "camera_depth" in str(dir_path)):
                continue
            if sensor_streams and ("umi" in str(dir_path) and "camera_depth" in str(dir_path)):
                continue

            # Skip mixing filters — if both True, require match in either
            if image_streams and sensor_streams:
                if dir_path.name not in image_dirs + sensor_dirs:
                    continue

            # 1. CSV-based data streams
            if not image_streams and "data.csv" in fn_lower:
                hits.append(dir_path / "data.csv")

            # 2. Timestamp-based data (image or numeric)
            if any(_TS_RE.match(name) for name in fn_lower):
                hits.append(dir_path)

        # Deduplicate and return absolute paths
        return sorted(set(p.resolve() for p in hits))
    
    def query_splitting(
        self,
        location: str | None = None,
        interaction: str | None = None,
        interaction_index: str | None = None,
    ) -> List[Tuple[str, str, str, Path]]:
        from fnmatch import fnmatch
        from pathlib import Path

        extracted_root = (self.root.parent / "extracted").expanduser().resolve()
        results: List[Tuple[str, str, str, Path]] = []

        SPLIT_RE = re.compile(r"^interaction_splitting_info_(.+)\.json$")

        # Normalize index filter
        if interaction_index is None:
            interaction_index = "*"
        elif not any(c in interaction_index for c in "*?"):
            interaction_index += "*"

        for loc in self.locations:
            if location and loc != location:
                continue

            loc_dir = extracted_root / loc
            if not loc_dir.is_dir():
                continue

            for inter_dir in loc_dir.iterdir():
                if not inter_dir.is_dir():
                    continue

                inter = inter_dir.name
                if interaction and inter != interaction:
                    continue

                for f in inter_dir.iterdir():
                    if not f.is_file():
                        continue

                    m = SPLIT_RE.match(f.name)
                    if not m:
                        continue

                    idx_token = m.group(1)  # e.g. "1-3-5-7"

                    # Either wildcard match OR exact OR membership match
                    if fnmatch(idx_token, interaction_index):
                        pass
                    else:
                        parts = idx_token.split("-")
                        if not any(fnmatch(p, interaction_index) for p in parts):
                            continue

                    results.append((loc, inter, idx_token, f))

        return results



# ── example usage & quick test ────────────────────────────────────────────
if __name__ == "__main__":
    idx = RecordingIndex("/data/ikea_recordings/raw")

    print("Bedroom – all indices, all recorders")
    for t in idx.query(location="bedroom_1"):
        print("  ", t)

    print("\nBedroom – aria* recorders, all indices")
    for t in idx.query(location="bedroom_1", recorder="aria*"):
        print("  ", t)

    print("\nBedroom – aria* recorders, index 1-8 only")
    for t in idx.query(location="bedroom_1",
                       recorder="aria*",
                       interaction_index="1-8"):
        print("  ", t)

    print("\nFirst five VRS files under bedroom_1 gripper / aria_gripper")
    for p in idx.vrs_files(location="bedroom_1",
                           interaction="gripper",
                           recorder="aria_gripper")[:5]:
        print("  ", p)
