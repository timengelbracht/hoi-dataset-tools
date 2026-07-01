#!/usr/bin/env python3
"""
Lightweight wrapper for the Project Aria MPS CLI (aria_mps).

Usage example
-------------
from pathlib import Path
from mps_wrapper import MPSClient

mps = MPSClient()                   # autodetect 'aria_mps' in PATH
mps.login()                         # or mps.login("user", "pass", save_token=False)

# single-recording request
mps.request_single("/data/run1.vrs", features="SLAM", force=True)

# multi-recording SLAM (shared frame) request
inputs = ["/data/run1.vrs", "/data/run2.vrs"]
mps.request_multi(inputs, output_dir="/tmp/multi_out")

mps.logout()
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from getpass import getpass
from pathlib import Path
from typing import List, Sequence


class MPSClient:

    ARIA_USERNAME = "REDACTED"
    ARIA_PASSWORD = "REDACTED"

    """Thin wrapper around the `aria_mps` command-line tool."""

    def __init__(self, cli_bin: str | Path | None = None):
        self.cli_bin = str(cli_bin or "aria_mps")
        if shutil.which(self.cli_bin) is None:
            raise FileNotFoundError(
                f"aria_mps binary not found in PATH (looked for: {self.cli_bin})"
            )

    # ---- internal ----------------------------------------------------------
    def _run(self, args: Sequence[str]) -> str:
        cmd = [self.cli_bin, *args]
        print("[MPS]", " ".join(shlex.quote(a) for a in cmd))  # simple trace
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        log_file = "/exchange/mps_cli.log"
        output = []
        try:
            for line in proc.stdout:
                print(line, end="")        # stream live output
                output.append(line)
            # with open(log_file, "w", encoding="utf-8") as f:
            #     for line in proc.stdout:
            #         print(line, end="")   # stream live output
            #         f.write(line)         # write to file
            #         f.flush()             # ensure it’s written immediately
            #         output.append(line)
        finally:
            proc.wait()

        if proc.returncode != 0:
            raise RuntimeError(f"aria_mps failed (code {proc.returncode})")

        return "".join(output)

    # ---- auth --------------------------------------------------------------
    def login(
        self,
        username: str | None = None,
        password: str | None = None,
        /,
        *,
        save_token: bool = True,
    ) -> str:
        """Authenticate and optionally store the token under $HOME/.projectaria."""
        if username is None:
            username = input("Project Aria username: ").strip()
        if password is None:
            password = getpass("Project Aria password: ")
        args = ["login", "-u", username, "-p", password]
        if not save_token:
            args.append("--no-save-token")
        return self._run(args)

    def logout(self) -> str:
        """Invalidate the local auth token."""
        return self._run(["logout"])

    # --- single-mode -----------------------------------------------------------
    def request_single(
        self,
        input_path: str | os.PathLike | Sequence[str | os.PathLike],
        *,
        features: str | Sequence[str] | None = None,  
        force: bool = False,
        retry_failed: bool = False,
        no_ui: bool = False,
    ) -> str:
        """
        Submit one recording (or a directory) in *single* mode.

        examples
        --------
        mps.request_single(file, features="SLAM")                  # one feature
        mps.request_single(file, features=["SLAM", "EYE_GAZE"])    # many
        """
        # path = Path(input_path).expanduser().resolve()
        # if not path.exists():
        #     raise FileNotFoundError(path)

        # args = ["single", "--input", str(path), "-u", username, "-p", password]
        # Normalize input_path into a list
        if isinstance(input_path, (str, os.PathLike)):
            input_paths = [input_path]
        else:
            input_paths = list(input_path)

        # Validate
        resolved_paths = []
        for p in input_paths:
            p = Path(p).expanduser().resolve()
            if not p.exists():
                raise FileNotFoundError(p)
            resolved_paths.append(str(p))

        args = ["single"]
        for p in resolved_paths:
            args += ["--input", p]

        args += ["-u", self.ARIA_USERNAME, "-p", self.ARIA_PASSWORD]

        # convert iterable → comma-separated string
        if features:
            if not isinstance(features, str):
                features = list(features)
            args += ["--features", *features]

        if force:
            args.append("--force")
        if retry_failed:
            args.append("--retry-failed")
        if no_ui:
            args.append("--no-ui")

        return self._run(args)


    # --- multi-mode -----------------------------------------------------------
    def request_multi(
        self,
        input_paths: List[str | os.PathLike],
        output_dir: str | os.PathLike | None = None,
        *,
        force: bool = False,
        retry_failed: bool = False,
        no_ui: bool = False,
    ) -> str:

        for path in input_paths:
            path = Path(path).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(path)

        args = ["multi"] + [arg for f in input_paths for arg in ("--input", str(f))]

        if output_dir is not None:
            output_dir = Path(output_dir).expanduser().resolve()
            if not output_dir.exists():
                raise FileNotFoundError(f"Output directory does not exist: {output_dir}")
            args += ["--output", str(output_dir)]
        else:
            raise ValueError("Output directory must be specified for multi requests.")

        if force:
            args.append("--force")
        if retry_failed:
            args.append("--retry-failed")
        if no_ui:
            args.append("--no-ui")

        args += ["-u", self.ARIA_USERNAME, "-p", self.ARIA_PASSWORD]

        return self._run(args)
    
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # Quick smoke-test (replace with your own paths)
    root_vrs = Path("/data/aria/run1.vrs")
    if root_vrs.exists():
        mps = MPSClient()
        try:
            mps.login()  # will prompt
            mps.request_single(root_vrs, features="SLAM", no_ui=True)
        finally:
            mps.logout()
