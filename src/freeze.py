"""Freeze the tree, tag it, and launch a long job against it.

The hash caching of 26 October fixed the *labelling* of a run. It does not make
a run reproducible: the Stage 7 rerun was labelled with five different tokens
because the tree moved underneath it, and recovering the code it executed took
a unit-by-unit comparison afterwards. A frozen tree is what actually makes a run
reproducible, so for every long job from here on:

    python -m src.freeze <tag> -- python -m src.stage7 p,s activity,temporal

It refuses to launch if anything tracked is modified, tags the tree, records the
tag and the job's executing-path digest, and prints the rule: do not commit to
this tree while the job runs. The tag is what a reader checks out to reproduce
the rows, and the digest is what the resume logic compares.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone

from src.provenance import JOBS, exec_digest

ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout.strip()


def main() -> int:
    if "--" not in sys.argv[1:]:
        print(__doc__)
        return 2
    cut = sys.argv.index("--")
    tag, command = sys.argv[1:cut], sys.argv[cut + 1:]
    if len(tag) != 1 or not command:
        print("usage: python -m src.freeze <tag> -- <command...>")
        return 2
    tag = tag[0]

    # Outputs are not code. A job already appending to results/ must not block
    # the launch of an unrelated one, and figures are regenerated rather than
    # authored. Everything else being clean is the point.
    OUTPUTS = ("results/", "figures/")
    dirty = "\n".join(
        line for line in git("status", "--porcelain", "--untracked-files=no").splitlines()
        if not any(line[3:].startswith(prefix) for prefix in OUTPUTS)
    )
    if dirty:
        print("Refusing to launch: the tree is not clean.\n")
        print(dirty)
        print("\nCommit or stash first. A run against a moving tree is not reproducible,")
        print("and recovering what it executed costs more than committing does.")
        return 1

    head = git("rev-parse", "--short", "HEAD")
    existing = git("tag", "--list", tag)
    if existing:
        print(f"Refusing to launch: tag {tag} already exists at {git('rev-parse','--short',tag)}.")
        return 1
    git("tag", "-a", tag, "-m", f"Frozen tree for: {' '.join(command)}")

    job = next((name for name in JOBS if name in " ".join(command)), None)
    digest = exec_digest(job) if job else "unknown"
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")

    print(f"tag        {tag} at {head}")
    print(f"job        {job or 'unrecognised'}")
    print(f"digest     {digest}")
    print(f"launched   {started}")
    print(f"command    {' '.join(command)}\n")
    print("Do not commit to this tree while the job runs. If the run has to be")
    print("resumed, check out the tag first:")
    print(f"    git checkout {tag}\n")
    print("Record in docs/pre-registered-decisions.md, with `until` left open until")
    print("the job exits:")
    print(f'    {{"job": "{job}", "since": "{started}", "until": None,')
    print(f'     "launched_from": "{tag}", "digest": "{digest}"}}\n')

    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
