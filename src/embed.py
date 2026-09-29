"""Gate 3: cache embeddings for every window, both checkpoints, both polarities.

This is the only expensive step in the study. Everything downstream is
arithmetic on the tables it writes, so it is done once and done completely.

Four tables rather than two. PPG-DaLiA's wrist channel is inverted relative to
the conventional PPG the encoder was pre-trained on, and whether that costs
accuracy is a Gate 4 side experiment. Embedding both polarities now means that
experiment costs minutes later instead of a repeat of this step. The extra
cost here is one more forward pass and about 260 MB.

The signal is not flipped in the pipeline itself. The reproduction has to be
anchored against PaPaGei's behaviour exactly; the flipped table is a clearly
labelled extension.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from src.data import available_subjects
from src.papagei import BACKBONES, EMBEDDING_DIM, embed, load_backbone
from src.preprocess import preprocess_subject

ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS = ROOT / "results" / "embeddings"
BATCH_SIZE = 256
POLARITIES = {"asis": 1.0, "flipped": -1.0}


def device_name() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def embed_windows(model, windows: np.ndarray, device: str) -> np.ndarray:
    """Embeddings for one subject's windows, in batches."""
    out = np.empty((len(windows), EMBEDDING_DIM), dtype=np.float32)
    for start in range(0, len(windows), BATCH_SIZE):
        batch = torch.from_numpy(windows[start : start + BATCH_SIZE]).unsqueeze(1).to(device)
        out[start : start + BATCH_SIZE] = embed(model, batch).cpu().numpy()
    return out


def main() -> None:
    EMBEDDINGS.mkdir(parents=True, exist_ok=True)
    device = device_name()
    subjects = available_subjects()
    if not subjects:
        raise SystemExit("No subjects found. Unzip PPG-DaLiA into data/ first.")

    print(f"device {device}, {len(subjects)} subjects, "
          f"{len(BACKBONES)} checkpoints, {len(POLARITIES)} polarities\n")

    started = time.time()
    models = {name: load_backbone(name, device) for name in BACKBONES}
    tables: dict[tuple[str, str], list] = {(b, p): [] for b in BACKBONES for p in POLARITIES}
    meta: list[dict] = []

    index_parts: dict[str, list] = {k: [] for k in ("subject", "heart_rate", "activity", "start_second")}

    for subject in subjects:
        prepared = preprocess_subject(subject)  # filtering is the slow part; do it once
        windows = prepared.windows
        index_parts["subject"].append(np.full(len(windows), int(subject[1:])))
        index_parts["heart_rate"].append(prepared.heart_rate)
        index_parts["activity"].append(prepared.activity)
        index_parts["start_second"].append(prepared.start_second)
        for polarity, sign in POLARITIES.items():
            signed = windows if sign > 0 else (windows * np.float32(sign))
            for backbone, model in models.items():
                tables[(backbone, polarity)].append(embed_windows(model, signed, device))
        meta.append(
            {
                "subject": subject,
                "windows": int(len(windows)),
                "rejection_rate": prepared.rejection_rate,
                "usable_minutes": prepared.usable_minutes,
                "elapsed_minutes": prepared.elapsed_minutes,
            }
        )
        print(f"  {subject}: {len(windows):5d} windows embedded", flush=True)

    # One row per window, in subject order, with everything needed downstream.
    index = {k: np.concatenate(v) for k, v in index_parts.items()}
    np.savez_compressed(EMBEDDINGS / "index.npz", **index)

    for (backbone, polarity), parts in tables.items():
        stacked = np.concatenate(parts)
        assert len(stacked) == len(index["subject"]), "embedding and index rows disagree"
        assert np.isfinite(stacked).all(), f"non-finite embeddings for {backbone}/{polarity}"
        path = EMBEDDINGS / f"papagei_{backbone}_{polarity}.npy"
        np.save(path, stacked)
        print(f"  wrote {path.name}: {stacked.shape}, {path.stat().st_size / 1e6:.0f} MB")

    environment = {
        "device": device,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "machine": platform.machine(),
        "runtime_seconds": round(time.time() - started, 1),
        "batch_size": BATCH_SIZE,
        "subjects": meta,
        "note": "asis is PaPaGei's pipeline unmodified; flipped negates the window "
                "to test the E4 BVP polarity inversion, and is an extension only.",
    }
    (EMBEDDINGS / "environment.json").write_text(json.dumps(environment, indent=2))
    print(f"\ntotal rows {len(index['subject']):,}, runtime {environment['runtime_seconds']:.0f}s")


if __name__ == "__main__":
    main()
