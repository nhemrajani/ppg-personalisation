# CLAUDE.md

Project context for Claude Code. Read this before writing any code.

---

## What this project is

An independent study at Yale SOM, Fall 2026.

**The question:** wearables run a shared population model and personalise cheaply by normalising each user's readings against their own rolling baseline. Is it worth adapting the model itself per person, how much of that person's data does it need, and what does it cost at scale?

**The output:** an accuracy-versus-cost frontier. Two figures carry the paper.
1. Cost per person (x) against error (y), one point per method. Shows which methods are dominated and where returns diminish.
2. Personal-data budget (x) against error reduction (y). Shows how much data a member must give before adaptation pays.

**Critical framing:** the baseline to beat is *normalisation*, not the raw population model. Most published work compares against the raw model, which overstates the benefit of personalisation. This is the study's main methodological asset — do not let it get lost in code.

---

## Non-negotiable decisions

| Decision | Value |
|---|---|
| Base model | PaPaGei, both checkpoints (`papagei_s.pt`, `papagei_p.pt`) |
| Dataset | PPG-DaLiA, 15 subjects |
| Task | Continuous heart-rate estimation (regression, BPM) |
| Preprocessing | PaPaGei's published pipeline, unmodified |
| Splitting | Subject-level; within-target: adaptation block → buffer → test block |
| Primary metric | MAE in BPM, per subject then averaged, with distribution reported |
| Intervals | Bootstrap, 500 resamples |
| Cost axis | Trainable parameters per person (primary); stored bytes (deployment translation) |

**Reproduction target:** PaPaGei reports MAE 11.53 on PPG-DaLiA heart rate (PaPaGei-S) and 10.92 (PaPaGei-P). Arm A must land near these or the pipeline is wrong.

**Do not rebuild PaPaGei's preprocessing.** Use their code. Deviating makes the reproduction claim unanswerable and costs weeks.

---

## The model

Convolutional ResNet, 512-dim embedding. **The two checkpoints use different classes** (verified against the weights, 2026-09-22):

| | PaPaGei-S | PaPaGei-P |
|---|---|---|
| Class | `ResNet1DMoE`, `n_experts=3` | `ResNet1D`, no experts |
| Trainable parameters, total | 5,785,612 | 4,993,024 |
| Trainable parameters, embedding path | 4,993,024 | 4,993,024 |

**The embedding paths are architecturally identical**: same 266 tensors, same shapes, different weights. Trunk → global average pool → `dense` (512 → 512) → embedding. PaPaGei-S adds mixture-of-experts heads (`expert_layers_*`, `gating_network_*`, 792,588 parameters) that branch off *after* pooling. They were auxiliary pretraining targets and **never feed the embedding**. So RQ5 compares two pretraining objectives on one architecture, which is what it needs.

The embedding is `model(x)[0]` for both classes (the `dense` projection). Loader: `src/papagei.py`.

Config shared by both (`n_experts=3` added for S only):

```python
model_config = {
    'base_filters': 32,
    'kernel_size': 3,
    'stride': 2,
    'groups': 1,
    'n_block': 18,
    'n_classes': 512,   # embedding dimension
    'n_experts': 3
}
```

Weights are **not** in the repo. Download from Zenodo:
```bash
mkdir -p weights
curl -L -o weights/papagei_s.pt \
  "https://zenodo.org/records/13983110/files/papagei_s.pt?download=1"
```

Repo: https://github.com/Nokia-Bell-Labs/papagei-foundation-model
Example notebook: `example_papagei.ipynb`

**Architecture note that matters later:** the encoder is convolutional, not transformer. LoRA, adapters and BitFit were all formulated for transformers. This is an unsolved implementation question for Arm C2 (see below).

---

## Preprocessing (PaPaGei's pipeline — match exactly)

1. Bandpass: 4th-order **Chebyshev Type II** (`cheby2`, 20 dB stopband), 0.5–12 Hz, applied with `filtfilt` (zero-phase). Implemented in `pyPPG.preproc.Preprocess`, called via PaPaGei's `preprocess_one_ppg_signal`
2. Segment: **8-second windows**, 2-second shift (6-second overlap). This is PPG-DaLiA-specific, stated in the paper's dataset appendix (arXiv 2410.20542v2): "we use a 8s window with 6s and 2s overlap and shift". PaPaGei's *general* pipeline uses 10-second windows, which is where the 1,250-sample model input comes from. **The analysis window is 8 s; 1,250 is only the padded input length.**
3. Reject: discard windows >25% flatline
4. Normalise: z-score per window
5. Resample: to 125 Hz (an 8 s window becomes 1,000 samples)
6. Pad to 1,250 samples, the 10 s pretraining length. The paper says only "resample and pad" for PPG-DaLiA. 1,250 and centre padding (half each side) are taken from their example notebook. Since the encoder averages over time, the pad length shifts the embedding, so treat this as an assumption to check if Arm A misses 11.53.

PaPaGei's public repo does **not** contain the PPG-DaLiA segmentation step. Its extraction script loads pre-made windows with `resample=False, normalize=False, fs=64`. Steps 1 to 6 are therefore ours to implement, calling their functions (`preprocess_one_ppg_signal`, biobss flatline detection, `resample_batch_signal`) in the order the paper gives.

**Order, resolved from the paper:** filter → segment → flatline reject → z-score → resample → pad. Two consequences:
- **Filter at the native 64 Hz, before resampling.** pyPPG's `Preprocess` adds a 50 ms moving-average smoothing pass only at ≥75 Hz. Filtering at 64 Hz skips it; filtering after resampling to 125 Hz would silently add it.
- **Z-score per window, after flatline rejection.** Their example notebook z-scores the whole signal *before* filtering, but that notebook is for PPG-BP. Do not copy its order for PPG-DaLiA.

**Cross-check at Gate 2:** PPG-DaLiA's own heart-rate labels are defined on 8 s windows with a 2 s shift, so the label count summed over subjects should be 64,697. Ten-second windows would give one fewer window per subject (64,682), so the exact count does distinguish the two.

**Record the rejection rate per subject.** It varies, and it feeds the "usable minutes vs wear time" analysis.

**PPG-DaLiA loading gotcha:** files are pickled Python 2 objects. Use `encoding='latin1'` or it fails with what looks like corruption.

Dataset facts: 15 subjects (8F, 7M, 21–55), Empatica E4 wrist PPG @ 64 Hz, accelerometer @ 32 Hz, chest RespiBAN ECG ground truth, 8 activities, **64,697 windows**. Fitzpatrick skin type, height, weight, fitness level recorded per subject.

---

## The splitting protocol

This is the hardest part of the design. Get it wrong and every number is invalid.

**The problem:** the population model must never have seen the target subject, AND the target must supply both adaptation data and test data that do not overlap. Windows overlap by 6 seconds, so random splitting leaks.

**For each target subject (rotate through all 15):**

- **Population set** — all windows from the other 14 subjects. Trains the downstream model for Arms A, B, B2.
- **Adaptation block** — contiguous portion of the target's recording. Used by B, B2, C1, C2, D.
- **Buffer** — ≥8 seconds (one window length), discarded entirely. Windows are 8s long, so this guarantees no window spans both blocks. It also covers the filter: `filtfilt` is zero-phase, so each filtered sample depends on raw signal either side of it. At 64 Hz, 99.9% of the filter's response lies within ±1.4 s and it falls below 0.1% of peak beyond ±3.8 s, well inside 8 s.
- **Test block** — later contiguous portion. **Every arm evaluates here.**

**Contiguous and temporally ordered**, for two reasons: random selection within a subject puts near-identical overlapping windows on both sides; and adapting on earlier data while testing on later data is the honest simulation of deployment.

**Activity confound:** PPG-DaLiA runs activities in fixed order, so a naive temporal split confounds adaptation with activity transfer. **Primary protocol:** take an initial portion of *each activity* for adaptation, remainder for test. **Secondary:** naive temporal, reported as a realism check.

**Hyperparameters** are selected on the population set or a validation slice of the adaptation block. Never the test block. Fix search ranges before looking at any test result.

---

## The six arms

| Arm | Description | Cost per person |
|---|---|---|
| A | Frozen encoder, ridge on population set, applied unchanged | 0 |
| B | As A, features rescaled using target's adaptation-block statistics | 2 values |
| B2 | As A, population subjects weighted by embedding similarity to target (unlabelled) | 0 |
| C1 | Frozen encoder, ridge warm-started from population solution, updated toward individual | 513 |
| C2 | Frozen encoder + small trainable components, trained on adaptation block | 10³–10⁴ |
| D | All encoder parameters retrained on adaptation block | 4,993,024 (both backbones, embedding path) |

**Arm B:** statistics come *exclusively* from the adaptation block. The buffer guarantees temporal disjointness. This mitigates the normalisation-inflation effect documented in Otesteanu et al. (2026).

**Arm B2:** adapts using no labels from the target at all. Weight the 14 population subjects by how closely their embedding distributions resemble the target's — requires only the target's unlabelled signal. Principle adapted from GAUL (Kim et al., 2025), reimplemented in-framework so the backbone stays constant and the cost comparison holds. Zero stored parameters, far more sophisticated than rescaling. If it approaches the labelled arms, that is the most deployment-relevant result in the study.

**Arm C1:** warm-start is primary. Fitting 513 coefficients from scratch on minutes of data may underperform Arm A. Warm-started, it equals Arm A at zero data and improves from there — which handles cold start. Report from-scratch as a comparison; the gap quantifies what warm-starting buys.

**Arm C2 — the open problem.** Four options in ascending ambition:
1. Adapt only the final projection layers ← **named fallback, always works**
2. Adapt only the 1×1 convolutions (these *are* matrix multiplications, LoRA applies unchanged)
3. Low-rank decomposition on reshaped convolutional kernels
4. Insert adapter blocks between convolutional blocks

~~Also worth probing: the MoE routing component may be cheaper to adapt.~~ **Not viable:** the MoE heads sit outside the embedding path (see The model), so adapting them cannot change the embedding the ridge head reads. They exist only in PaPaGei-S anyway.

Concrete costs for option 1: the `dense` projection is 512 × 512 + 512 = **262,656** parameters, above the 10³–10⁴ range in the arms table. LoRA on that layer costs 1,024 × rank (rank 1–10 → 10³–10⁴), which fits the range and makes rank the cost axis.

**Arm D:** upper bound, not a deployment candidate. Expected to overfit on minutes of data and possibly underperform Arm A — which is itself a result (adaptation capacity is not monotonically beneficial).

**Budget sweep:** run B, B2, C1, C2, D at 2, 5, 10, 20, 40 minutes and all-available. Budgets taken from the *start* of the adaptation block so smaller budgets are prefixes of larger ones. **Report in usable minutes after rejection, not wear time** — noisier subjects lose more windows. Report both; the gap is a finding.

---

## Repo structure

```
data/        PPG-DaLiA. NEVER committed. Add to .gitignore before first commit.
src/         Reusable code: preprocessing, splitting, arms, evaluation, harness
notebooks/   Exploration and plotting only
results/     Experiment logs, cached embeddings
figures/     Generated plots — every one regenerated by a script
docs/        Proposal, reports, paper
log/         Research log entries, one markdown file per session
```

**Rule:** anything written twice goes in `src/` and gets imported. The failure mode is six slightly different preprocessing functions across four notebooks by November.

`.gitignore` must exclude `venv/`, `data/`, `*.pkl`, `__pycache__/`, `results/embeddings/`, `.ipynb_checkpoints/`.

---

## Build order

Each gate must pass before moving on.

### Gate 1 — PaPaGei runs (target: this week)
Clone, install, download weights, push random noise through the encoder, get 512 numbers out.
```python
x = torch.randn(1, 1, 1250)   # model input: an 8 s window at 125 Hz (1,000 samples) padded to 1,250
with torch.no_grad():
    out = model(x)
```
Meaningless as a prediction. Proves the model loads. **Dependency conflicts are the only unpredictable step in this project — do this first.**

**Status: passed 2026-09-22** on Python 3.12, torch 2.14, CPU and MPS (MPS matches CPU to ~1e-6). Run `python -m src.papagei`.

The Python constraint came from preprocessing dependencies, not PyTorch. `pyPPG==1.0.41` and `biobss` pin their whole 2022 development environments (scipy 1.9, numpy 1.22–1.23, pip, setuptools), which do not install on 3.12. PaPaGei uses one small module from each, so both go in `requirements-nodeps.txt` and are installed with `--no-deps`. A clean environment built this way reproduces Gate 1 exactly.

Open check: filter output under scipy 1.18 vs pyPPG's pinned scipy 1.9.1. Both use `cheby2` + `filtfilt`, expected to agree to floating-point precision, but unverified.

### Gate 2 — Real data through the pipeline
**Status: next.**
Load one subject. Plot 30s raw. Apply PaPaGei preprocessing. Plot filtered vs raw. Detect peaks, count by eye, compare to ECG label. Then one real window → 512 numbers.

Also: synthetic sine wave check. Generate 1.2 Hz at 64 Hz for 30s, filter, detect peaks, confirm 36. Pipeline verified on data where the answer is known.

### Gate 3 — Embeddings cached
All 15 subjects, both checkpoints, saved to disk with subject ID and HR label attached. ~64,697 rows × 512 columns, twice.

Record environment: package versions, hardware, runtime.

**This is the gateway. Everything after is arithmetic on a table.**

### Gate 4 — Arm A reproduces
Ridge on frozen embeddings, PaPaGei's evaluation protocol, bootstrap intervals. Compare to 11.53 / 10.92.

**Open issue: 11.53 is not a leave-one-subject-out number** (verified 2026-09-22). The paper's linear evaluation uses a single fixed subject-level train/val/test split: 80/10/10 for in-domain datasets, **60/20/20 for out-of-domain**. PPG-DaLiA is out-of-domain (unseen in pretraining), so roughly 9/3/3 subjects. PaPaGei's feature-extraction job chunking is consistent with about 3 test subjects. So 11.53 is an MAE over ~3 held-out people, with 500-run bootstrap intervals.

Consequences:
- **Reproduction and Arm A are two different numbers.** Reproduce 11.53 under their 9/3/3 split. Arm A under our leave-one-subject-out rotation will differ and is not expected to equal it.
- **Their split files are not public.** `utilities.py` reads `data/dalia/{train,val,test}.csv`, which are not in the repo. Which 3 subjects were held out is unknown, and with 3 test subjects the MAE depends heavily on which ones. Options: ask the authors for the split, or report the distribution of MAE over all 9/3/3 splits (455 choices of test subjects) and check whether 11.53 falls inside it.
- The reference-numbers table mixes protocols: the classical rows are leave-one-subject-out, the frozen-probe rows are this fixed split. Label them as such before comparing.

If it doesn't match: debug against their published intermediate values. If it still doesn't, document the discrepancy carefully. A documented failed reproduction is a legitimate result.

### Gate 5 — Experiment harness
**Build before the arms multiply, not after.**

One function: configuration in, result row out, every setting recorded.

```
config = {arm, backbone, subject, budget_minutes, split_type, seed, hyperparams}
→ run → append row to results/experiments.csv with config + metrics + timestamp
```

6 arms × 6 budgets × 15 subjects × 2 backbones ≈ 1,000 runs. Computationally trivial, impossible to track by hand.

### Gate 6 — Arms B, B2, C1
Splitting protocol implemented. Three arms with confidence intervals. First real comparison.

### Gate 7 — Arm C2
Resolve the convolutional question. Verify it trains and loss falls before worrying whether it helps. Rank sweep = the cost axis.

### Gate 8 — Arm D, both backbones, figures
Complete frontier. Both graphs. Stratify by activity and by subject.

---

## Conventions

- **British English** in all prose. No em dashes.
- **"Robustness", never "fairness."** With 15 subjects the study can ask whether poorly-served individuals benefit disproportionately — a question about the performance *distribution*. It cannot support demographic claims. Subgroup analysis is illustrative only, with stated power limitations.
- **Commit every working session**, even when the work is bad. Honest messages ("tried filtering, output looks wrong, investigating") are useful. The commit history is part of the deliverable.
- **Every figure regenerated by a script** in the repo. Stated deliverable.
- **Paired comparisons only.** All arms evaluate on identical test sets, so compare per-subject differences, not differences of means. If confidence intervals overlap, write "no detectable difference at this sample size" — do not claim a winner.

---

## Known limitations (state them, don't hide them)

- 15 subjects is thin for a per-person question
- Per-window z-scoring already removes some individual variation, so the "population baseline" is not wholly unpersonalised
- Convolutional adaptation is unresolved, with a named fallback
- Serving cost is unmeasured (per-person adapters break batching), so the frontier is a **lower bound** on production cost
- PaPaGei is mid-table on this task — Moment (8.82), Chronos (9.65), TF-C (9.99) all beat it. The goal is reproducing PaPaGei's number, not showing it is best
- LoRA on PPG is already published (Vision4PPG). The contribution is the per-person framing and the cost comparison

---

## If the schedule slips

Cut arms in order: **D, then C2, then B2.** This preserves points at both ends of the cost axis for as long as possible.

Cut reading before code. Reading recovers over a weekend; code debt compounds.

Never cut: the splitting protocol, the confidence intervals, the reproduction check. Those three are what make this a paper rather than a project.

---

## Reference numbers (PPG-DaLiA heart rate, MAE in BPM)

| Approach | MAE |
|---|---|
| Classical LOSO (Schaeck2017) | ~20.5 |
| Classical LOSO (SpaMa) | ~15.6 |
| Statistical features baseline | 13.1 |
| PaPaGei-S frozen probe | 11.5 |
| PaPaGei-P frozen probe | 10.9 |
| TF-C frozen | 10.0 |
| Chronos frozen | 9.7 |
| Moment frozen | 8.8 |
| Supervised task-specific (Conv-LSTM) | **6.3** |

The gap between frozen probing (~11) and supervised task-specific (6.3) is the headroom. The study asks how much of it per-person adaptation recovers, and at what cost.
