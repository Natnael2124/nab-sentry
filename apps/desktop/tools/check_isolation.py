"""Fail if this branch changed any NAB Sentry V2 file (NAB Find Requirement 12).

Usage::

    venv-desktop\\Scripts\\python.exe apps\\desktop\\tools\\check_isolation.py [--base v2-step1-measure]

Compares the working tree (committed and uncommitted changes, untracked files included) with the
merge base of HEAD and ``--base``. Only paths under ``apps/desktop/`` and ``.kiro/specs/nab-find/``
may differ. Exit code 0 when V2 is untouched, 1 otherwise (the changed V2 paths are printed).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ALLOWED = ("apps/desktop/", ".kiro/specs/nab-find/")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout


def changed_paths(base: str) -> list[str]:
    mb = git("merge-base", "HEAD", base).strip()
    paths = set(git("diff", "--name-only", mb).split())
    paths |= set(git("ls-files", "--others", "--exclude-standard").split())
    return sorted(paths)


def v2_paths(paths: list[str]) -> list[str]:
    return [p for p in paths if not p.startswith(ALLOWED)]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--base", default="v2-step1-measure")
    p.add_argument("--ignore", action="append", default=["config/rules.yaml"],
                   help="untracked local files to ignore (default: config/rules.yaml)")
    args = p.parse_args(argv)
    bad = [x for x in v2_paths(changed_paths(args.base)) if x not in args.ignore]
    if bad:
        print("V2 files changed on this branch:")
        print("\n".join(f"  {x}" for x in bad))
        return 1
    print(f"ok: no V2 file changed since the merge base with {args.base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
