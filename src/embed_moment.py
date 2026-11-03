"""The MOMENT backbone, as a fourth embedding table.

Registered 3 November 2026. MOMENT enters the study exactly as PaPaGei does, as
a cached table of one embedding per window, so every closed-form arm reads it
without knowing which encoder produced it.

This runs in two steps because it straddles two environments. `momentfm` pins
numpy 1.25.2, huggingface-hub 0.24.0 and transformers 4.33.3, none of which can
be installed beside the pinned pyPPG and biobss the preprocessing needs, so
MOMENT lives in `venv-moment/` and the preprocessing stays in `venv/`:

    venv/bin/python -m src.embed_moment prepare          # windows -> cache
    venv-moment/bin/python -m src.embed_moment embed     # cache -> embeddings

The cache is the handoff. `prepare` re-derives the per-window index and asserts
it equals the one `src/embed.py` wrote at Gate 3, so the MOMENT table is aligned
to the same rows as the PaPaGei tables rather than merely the same length.

MOMENT is fed PaPaGei's 1,250-sample windows unchanged, which is 2.4 times its
pre-training context and includes the 125 leading and trailing zeros as real
data. Both are PaPaGei's choices and both are what produced the published 8.82;
see the registration for why they are reported rather than fixed.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS = ROOT / "results" / "embeddings"
CACHE = Path("/private/tmp/claude-501/-Users-neeharikahemrajani-Desktop-Coding-Projects-neeha-som-is/"
             "6dab79f1-4e28-4a05-a08f-dd91b259f4e7/scratchpad/moment_windows")
CHECKPOINT = "AutonLab/MOMENT-1-large"
MOMENT_DIM = 1024
BATCH_SIZE = 64
INDEX_KEYS = ("subject", "heart_rate", "activity", "start_second")


def prepare() -> None:
    """Write the prepared windows, asserting they align to Gate 3's index."""
    from src.data import available_subjects
    from src.preprocess import preprocess_subject

    subjects = available_subjects()
    if not subjects:
        raise SystemExit("No subjects found. Unzip PPG-DaLiA into data/ first.")

    CACHE.mkdir(parents=True, exist_ok=True)
    started = time.time()
    windows, parts = [], {k: [] for k in INDEX_KEYS}
    for subject in subjects:
        prepared = preprocess_subject(subject)
        windows.append(prepared.windows)
        parts["subject"].append(np.full(len(prepared.windows), int(subject[1:])))
        parts["heart_rate"].append(prepared.heart_rate)
        parts["activity"].append(prepared.activity)
        parts["start_second"].append(prepared.start_second)
        print(f"  {subject}: {len(prepared.windows):5d} windows", flush=True)

    stacked = np.concatenate(windows).astype(np.float32)
    index = {k: np.concatenate(v) for k, v in parts.items()}

    # The table must be aligned to the same rows as the PaPaGei tables, not
    # merely be the same length. Gate 3's index is the reference.
    reference = np.load(EMBEDDINGS / "index.npz")
    for key in INDEX_KEYS:
        assert np.array_equal(index[key], reference[key]), \
            f"the prepared windows do not align to Gate 3's index on '{key}'"
    print(f"\nrow order matches results/embeddings/index.npz on all "
          f"{len(INDEX_KEYS)} keys, {len(stacked):,} rows")

    np.save(CACHE / "windows.npy", stacked)
    print(f"wrote {CACHE / 'windows.npy'}: {stacked.shape}, "
          f"{(CACHE / 'windows.npy').stat().st_size / 1e6:.0f} MB, "
          f"{time.time() - started:.0f}s")


def embed() -> None:
    """Embed the cached windows with MOMENT. Runs in venv-moment."""
    import torch
    from momentfm import MOMENTPipeline

    path = CACHE / "windows.npy"
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run the prepare step in venv/ first.")
    windows = np.load(path, mmap_mode="r")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = MOMENTPipeline.from_pretrained(CHECKPOINT,
                                           model_kwargs={"task_name": "embedding"})
    model.init()
    model.eval()
    model.to(device)
    print(f"{CHECKPOINT} on {device}, {len(windows):,} windows of "
          f"{windows.shape[1]} samples, batch {BATCH_SIZE}", flush=True)

    out = np.empty((len(windows), MOMENT_DIM), dtype=np.float32)
    started = time.time()
    with torch.inference_mode():
        for start in range(0, len(windows), BATCH_SIZE):
            batch = np.asarray(windows[start:start + BATCH_SIZE])
            tensor = torch.from_numpy(batch).unsqueeze(1).to(device)
            out[start:start + len(batch)] = (
                model(x_enc=tensor).embeddings.float().cpu().numpy())
            if (start // BATCH_SIZE) % 50 == 0:
                done = start + len(batch)
                rate = done / max(time.time() - started, 1e-9)
                print(f"  {done:6d}/{len(windows)}  {rate:6.0f} windows/s  "
                      f"eta {(len(windows) - done) / max(rate, 1e-9) / 60:5.1f} min",
                      flush=True)

    assert np.isfinite(out).all(), "non-finite MOMENT embeddings"
    target = EMBEDDINGS / "moment_asis.npy"
    np.save(target, out)
    runtime = time.time() - started
    print(f"\nwrote {target.name}: {out.shape}, "
          f"{target.stat().st_size / 1e6:.0f} MB, {runtime / 60:.1f} min")

    meta = EMBEDDINGS / "environment_moment.json"
    meta.write_text(json.dumps({
        "checkpoint": CHECKPOINT,
        "embedding_dim": MOMENT_DIM,
        "input_samples": int(windows.shape[1]),
        "pretraining_context": 512,
        "polarity": "asis only, registered 3 November 2026",
        "device": device,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "machine": platform.machine(),
        "batch_size": BATCH_SIZE,
        "runtime_seconds": round(runtime, 1),
        "rows": int(len(out)),
        "note": "Windows are PaPaGei's 1,250-sample zero-padded inputs, fed "
                "unchanged as their own feature_extraction_moment.py does. That "
                "is 2.4x MOMENT's 512-sample pre-training context and treats the "
                "padding as real data; both are theirs and both are what produced "
                "the published 8.82.",
    }, indent=2))
    print(f"wrote {meta.name}")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    if step == "prepare":
        prepare()
    elif step == "embed":
        embed()
    else:
        raise SystemExit(__doc__)
