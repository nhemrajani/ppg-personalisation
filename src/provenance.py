"""A digest of the code that produces numbers, so resume is a rule not a call.

A long job appends rows for hours. If the source changes while it runs, or if a
killed job is resumed against edited code, some rows come from one version and
some from another and nothing in the record distinguishes them. The commit hash
does not do this job: it moves whenever a commit is made, which may be unrelated
to the code executing, and it reads `dirty` without saying what was dirty.

So every row carries `src_digest`, a hash over the abstract syntax of the units
on the number-producing path, with comments and formatting normalised away. Two
runs with the same digest ran the same arithmetic. The resume rule is then
mechanical: a configuration is skipped only if it is complete for every subject
**and** its rows carry the current digest. Anything else is rerun whole.

`ACCEPTED` records digests that have been verified equivalent to another by
hand, with the reason and the date. That is the only way to accept rows from a
different digest, and it leaves an audit trail instead of a decision taken at
three in the morning.
"""

from __future__ import annotations

import ast
import functools
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Whole modules whose every line can move a number.
WHOLE = ("src/ridge.py", "src/splits.py", "src/lora.py", "src/preprocess.py",
         "src/papagei.py", "src/embed.py", "src/effective_n.py", "src/data.py",
         "src/quality.py")

# Per job, the units on that job's number-producing path. Scoped per job so that
# adding an arm one job never calls does not invalidate another job's resume.
SHARED = {"src/harness.py": ("Run", "append")}

JOBS = {
    "stage7": {
        "src/stage7.py": ("step_count", "_windows", "_head_from_ridge", "run_one",
                          "select_rates", "RANKS", "IN_BAND", "BUDGETS", "BATCH",
                          "SEED", "EPOCHS", "MIN_STEPS", "LR_GRID",
                          "LR_SELECTION_SUBJECTS", "LR", "TARGET_LAYER",
                          "BUFFER_WINDOWS", "TOLERANCE"),
        "src/arms.py": ("Fitted", "_prepare", "arm_a"),
        **SHARED,
    },
    "stage9": {
        "src/arms.py": ("Fitted", "_prepare", "arm_a", "arm_b", "arm_b_offset",
                        "arm_b_ridge", "_adaptation_predictions", "SHRINKAGE_B",
                        "BUFFER_WINDOWS"),
        **SHARED,
    },
    "c1": {
        "src/arms.py": ("Fitted", "_prepare", "arm_a", "arm_c1", "SHRINKAGE",
                        "BUFFER_WINDOWS"),
        **SHARED,
    },
}

# Verified equivalences. The key is a digest, or a legacy commit token found on
# rows written before digests existed. The value records what it was checked
# against, when, and why. Adding an entry here is the ONLY way to resume over
# rows from a different path, and it leaves an audit trail.
#
# Reproduce the check behind the entries below with:
#   python -m src.provenance stage7 3111aed HEAD
# which reports `select_rates` as the single difference.
ACCEPTED: dict[str, str] = {
    "e669da3": "Stage 7 rerun launched 2026-10-26 from the tree at 3111aed. Verified "
               "2026-10-27: on stage7's path the only difference to HEAD is "
               "select_rates, and only in which ranks it iterates. Its loop body "
               "reads no other rank, so the rate chosen per rank is unchanged. "
               "run_one, step_count, _head_from_ridge, arm_a, Run, append and every "
               "constant are byte-identical, as are all nine whole modules.",
    "e669da3-dirty": "Same run as e669da3. The dirt was arms.py (two new arms stage7 "
                     "never calls), stage9.py, stage7.py's resume logic, harness.py's "
                     "hash caching and CLAUDE.md. None on stage7's path.",
    "ac61595-dirty": "Same run as e669da3.",
    "83e54da-dirty": "Same run as e669da3.",
    "1c8bf6b-dirty": "Same run as e669da3.",
}


def _units(text: str) -> dict[str, str]:
    """Top-level definitions and assignments, normalised through the AST."""
    out: dict[str, str] = {}
    for node in ast.parse(text).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[node.name] = ast.unparse(node)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                names = ([target] if isinstance(target, ast.Name)
                         else getattr(target, "elts", []))
                for name in names:
                    if isinstance(name, ast.Name):
                        out[name.id] = ast.unparse(node)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out[node.target.id] = ast.unparse(node)
    return out


@functools.lru_cache(maxsize=8)
def exec_digest(job: str) -> str:
    """Twelve hex characters identifying one job's number-producing path."""
    digest = hashlib.sha256()
    for path in WHOLE:
        digest.update(path.encode())
        digest.update(ast.unparse(ast.parse((ROOT / path).read_text())).encode())
    for path, names in sorted(JOBS[job].items()):
        found = _units((ROOT / path).read_text())
        digest.update(path.encode())
        for name in names:
            digest.update(name.encode())
            digest.update(found.get(name, "<absent>").encode())
    return digest.hexdigest()[:12]


# Runs that predate digests, identified by the window in which they wrote rows
# rather than by commit. A commit token is the wrong key for this: a run appends
# for hours and picks up whatever commit happens to exist when each row is
# written, so one run spreads over several tokens and a later commit silently
# produces a token nobody recorded. The process, by contrast, loaded its modules
# once at launch and cannot change, so a time window identifies it exactly.
#
# `until` is None while a run is still appending, and is closed to the last row's
# timestamp once it exits.
RUN_WINDOWS: tuple[dict, ...] = (
    {
        "job": "stage7",
        "since": "2026-10-27T14:02:34+00:00",
        "until": "2026-10-27T15:11:12+00:00",
        "launched_from": "armd-2026-10-27",
        "why": "Arm D rerun at the rate reselected under the absolute-error loss, "
               "registered 27 October. Launched through src/freeze.py on an annotated "
               "tag of a clean tree, so this window is belt and braces: every row "
               "carries commit a7fac63 with no -dirty suffix and digest 597c1b602592, "
               "and the code is recoverable as `git checkout armd-2026-10-27`. The "
               "window could not be committed while the job ran, because provenance.py "
               "is itself on stage7's path, so it is recorded on exit.",
    },
    {
        "job": "stage7",
        "since": "2026-10-27T02:49:36+00:00",
        "until": "2026-10-27T07:18:17+00:00",
        "launched_from": "3111aed",
        "why": "Stage 7 rerun. Verified 2026-10-27 that on stage7's path the only "
               "difference between 3111aed and HEAD is select_rates, and only in "
               "which ranks it iterates; its loop body reads no other rank, so the "
               "rate chosen per rank is unchanged. run_one, step_count, "
               "_head_from_ridge, arm_a, Run, append and every constant are "
               "byte-identical, as are all nine whole modules. Reproduce with "
               "`python -m src.provenance stage7 3111aed HEAD`. Window closed to the last row written; the 2,160 rows inside it carried five later commit tokens from the per-row hash defect and were corrected to 3111aed.",
    },
)


def _in_window(job: str, timestamp: str) -> bool:
    for window in RUN_WINDOWS:
        if window["job"] != job or not timestamp:
            continue
        if timestamp < window["since"]:
            continue
        if window["until"] is not None and timestamp > window["until"]:
            continue
        return True
    return False


def acceptable(job: str, token: str, timestamp: str = "") -> bool:
    """Whether rows from this digest, legacy commit or run window may be resumed over.

    Three ways in, in descending strength: the row carries the current digest; it
    falls inside a recorded run window whose equivalence was verified; or its
    token is individually recorded in ACCEPTED. Anything else is rerun whole.
    """
    return (token == exec_digest(job)
            or _in_window(job, timestamp)
            or token in ACCEPTED)


def compare(job: str, rev_a: str, rev_b: str) -> list[str]:
    """Units that differ between two git revisions. Used to justify an ACCEPTED entry."""
    import subprocess

    def at(rev: str, path: str) -> str:
        return subprocess.run(["git", "show", f"{rev}:{path}"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout

    changed = []
    for path in WHOLE:
        if ast.unparse(ast.parse(at(rev_a, path))) != ast.unparse(ast.parse(at(rev_b, path))):
            changed.append(path)
    for path, names in JOBS[job].items():
        a, b = _units(at(rev_a, path)), _units(at(rev_b, path))
        changed += [f"{path}:{n}" for n in names if a.get(n) != b.get(n)]
    return changed


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 4:
        job, a, b = sys.argv[1], sys.argv[2], sys.argv[3]
        differences = compare(job, a, b)
        print(f"{job}: {a} vs {b} on the number-producing path:")
        print("  identical" if not differences
              else "\n".join(f"  CHANGED {d}" for d in differences))
    else:
        for job in JOBS:
            print(f"  {job:8s} {exec_digest(job)}")
