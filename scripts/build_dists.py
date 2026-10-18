#!/usr/bin/env python3
"""Build the release distributions from one source tree.

The same ``merkle_verify`` package is published under two names:

* ``merkle-weight-verify`` -- the primary distribution (pyproject.toml as-is)
* ``hanfei-fa``            -- an identical copy under the HanFei series name

Both are full distributions. A thin alias that merely depends on
``merkle-weight-verify`` is not safe: both names ship the same files, so
upgrading an existing ``hanfei-fa`` 0.2.0 install to such an alias makes pip
delete the module files the new dependency has just installed.

Usage:
    python scripts/build_dists.py [--outdir dist]
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRIMARY = "merkle-weight-verify"
ALIASES = ["hanfei-fa"]
COPY_IGNORE = shutil.ignore_patterns(
    ".git", "dist", "build", "*.egg-info", "__pycache__", ".pytest_cache",
    ".venv", ".claude", ".internal", "*.merkle.json",
)


def build(src: Path, outdir: Path) -> None:
    subprocess.run([sys.executable, "-m", "build", "--outdir", str(outdir), str(src)], check=True)


def renamed_tree(name: str, tmp: Path) -> Path:
    dst = tmp / name
    shutil.copytree(ROOT, dst, ignore=COPY_IGNORE)
    pyproject = dst / "pyproject.toml"
    text, n = re.subn(
        rf'^name = "{re.escape(PRIMARY)}"$', f'name = "{name}"',
        pyproject.read_text(encoding="utf-8"), flags=re.M,
    )
    if n != 1:
        raise SystemExit(f"could not find the [project] name line in {pyproject}")
    pyproject.write_text(text, encoding="utf-8")
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", default=str(ROOT / "dist"))
    args = ap.parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    build(ROOT, outdir)
    with tempfile.TemporaryDirectory() as tmp:
        for name in ALIASES:
            build(renamed_tree(name, Path(tmp)), outdir)

    for f in sorted(outdir.iterdir()):
        print(f.name)


if __name__ == "__main__":
    main()
