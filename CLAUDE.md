# CLAUDE.md

Project context for Claude Code. Read this before writing any code.

The authoritative method is `docs/methodology.md`, written by the author and transcribed from the proposal. This file is the working summary plus everything verified against PaPaGei's code, paper and weights. Where they disagree, the methodology document wins on design and this file wins on verified facts.

> Disclaimer: work in progress, drafted with Claude Code. Provisional and subject to revision until updated with all the relevant information.

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

**Window length: settled empirically, 2026-09-28.** Run `python -m src.verify_windows`; output in `results/window_length_check.txt`.

The paper states two window lengths and they are not in conflict once located. The 10-second figure is in section 4.1 under **Pre-training**, whose Table 1 lists only VitalDB, MIMIC-III and MESA: 20,751,206 segments over 57,641 hours, which is 9.9998 seconds per segment, so those are 10 s non-overlapping pre-training windows. PPG-DaLiA is an **evaluation** dataset (Table 2, greyed as out-of-domain) and its appendix specifies 8 s windows with a 2 s shift.

Counting PPG-DaLiA's own heart-rate labels confirms it. The label total across the fifteen subjects is **64,697**, exactly PaPaGei's reported sample count, and it matches the 8-second prediction for every individual subject. Ten-second windows give 64,682, fifteen fewer. **The analysis window is 8 s and the buffer rule of ≥8 s is correct.**

Useful by-product, recorded in the same file: per-subject durations run from 87.5 minutes (S6) to 177.5 minutes (S10). Even the shortest subject supports the 40-minute budget with room for buffer and test block, which closes the "is there enough data per subject" risk flagged for Week 4.

**Flatline rejection is inert on this dataset (measured 2026-09-28).** Across all 64,697 windows the flatness fraction is exactly 0.0000%, at minimum flatline durations of 0.5, 1, 2 and 4 s, on raw and filtered signal alike. The cause is a scale mismatch, not clean data: biobss compares consecutive-sample changes against an absolute threshold of 0.01, while the median step in E4 BVP is 4.84 units and the 1st percentile is 0.70. No sample pair in the dataset falls below the threshold. PaPaGei's `is_signal_flat_lined` z-scores the signal and then passes the un-normalised array, which looks unintended; on z-scored input 3 windows would be rejected. Keep their behaviour for fidelity, since it is what produces their 64,697. Run `python -m src.measure_quality`.

**PaPaGei-P was trained to be sign-invariant. PaPaGei-S was trained not to be** (verified 2026-09-28). Their section 4.1: "For augmentations, PaPaGei-P uses cropping (0.50), negation (0.20), flipping (0.20), and scaling (0.40). PaPaGei-S uses cropping (0.25) and Gaussian noise (0.25). PaPaGei-S avoids augmentations that alter PPG's morphology." Their `augmentations.py` defines `Negation.forward` as `return X * -1`, a literal sign flip, applied to one in five training samples for P and never for S.

The published table already satisfies the prediction this makes. Across regression tasks S has the better average MAE, 10.12 against 10.92, and it is the paper's headline model. On PPG-DaLiA heart rate, the one wrist dataset, that reverses: **P 10.92 beats S 11.53**. If the wrist signal is inverted, the sign-invariant model should cope and the morphology-sensitive one should be out of distribution, which is exactly the ordering observed. The authors do not comment on it.

**This makes the polarity experiment a control for Sub Question 3, not a side experiment.** The two checkpoints differ in their treatment of subject identity, which is what Sub Question 3 asks about, but they also differ in sign invariance. On an inverted dataset the second difference is a confound that could account for the entire P-versus-S gap. Running both checkpoints on correctly oriented signal isolates the objective; without it the question is answerable in one sentence by a reviewer. Both polarities are already embedded, so this costs minutes.

**The E4 BVP channel is inverted relative to standard PPG.** Lead with the activity ordering, not the sign. Reading the signal-quality index with inversion assumed, so that more negative means cleaner, the nine activities order as sitting -0.63, working -0.42, lunch -0.29, driving -0.27, transient -0.12, table soccer -0.06, cycling -0.05, walking -0.04, stairs +0.00. That is monotone in physical activity level. The conventional reading would have it that climbing stairs yields cleaner wrist PPG than sitting still. One sign error could be coincidence; a correctly ordered nine-level ranking cannot be. Morphology corroborates it: each ECG R-peak is followed by a sharp downward trough with the dicrotic notch as a secondary bump (`figures/gate2_polarity.png`).

Both load-bearing assumptions are checked. PaPaGei never sign-corrects: no flip, negation or skewness-based correction exists anywhere in their preprocessing or inference path, and skewness appears only as an SQI label and pretraining target. And their pre-training data is conventional polarity, shown by upward systolic peaks in all three datasets in their own Figures 20 and 21, raw and pre-processed.

Consequences: PaPaGei's SQI is skewness, where higher means cleaner on conventional PPG, so its sign is flipped here and clean sitting windows score most negative. Any quality measure on this dataset needs negated skewness. A Gate 4 experiment of about a day, and a control for Sub Question 3 rather than an optional extra: reproduce with the signal as-is and flipped, and compare both against 11.53. It does not displace the frontier. If the effect is material it becomes a discussion section and a named future-work direction. Embeddings are therefore computed in both polarities in the same pass at Gate 3, so this costs minutes later and never needs repeating.

**Standing rule: discoveries do not become pipeline modifications.** Every fix breaks the reproduction. Run their pipeline exactly, report what it does, and put corrections in a clearly labelled extension. There are now four findings about flaws in their pipeline; they belong in a short methods subsection, reported neutrally with evidence. This paper is about the cost frontier, not about what is wrong with PaPaGei.

Undocumented parameters and their effects: `docs/undocumented-parameters.md`.

Full experimental report: `docs/experiment-report.md`.

**Arm B beats Arm C1 at every budget above ten minutes, and loses to it below five** (reran 26 October 2026). Improvement over Arm A at all available data, paired per subject, in all four cells: B gives +2.08, +1.50, +2.97 and +2.19, significant in every one. C1 gives +1.64, +0.80, +1.02 and +0.73, significant in three. C1 minus B is +0.44, +0.69, +1.95 and +1.45, so **Arm B is better than C1 in all four cells and the frontier's knee stays at two parameters.**

**The earlier claim that C1 is not robust was an artefact of a selection defect, now withdrawn.** C1's shrinkage was being scored on rows inside its own fit, so it chose the weakest penalty almost always and ran close to unregularised. Correctly regularised it does not diverge anywhere: its worst cell was 33.22 BPM against an unadapted 11.53, and the worst excess over Arm A is now 3.51. The sentence that two closed-form parameters are stable while 513 are not was measuring the defect, not the arms. Full account in `docs/pre-registered-decisions.md` under the defect notice of 26 October 2026.

**The crossover is the finding that replaces it.** C1 minus B by budget, PaPaGei-P activity-stratified: -4.70, -5.16, +1.33, +0.84, +0.73, +0.44 at 2, 5, 10, 20, 40 and all minutes. C1 is better at the two smallest budgets in all four cells and worse at 20 minutes and above in all four. A regularised warm start shrinks toward the population model when the calibration window is unrepresentative, which is exactly what it was registered to do at cold start; Arm B's two-parameter least-squares fit has no such protection and is wild on 60 windows of sitting still. **So the cheap method wins once there is data and the expensive one wins at cold start**, which is a more useful result than either arm dominating.

**One parameter is safer than two, and the crossover is about regularisation rather than capacity** (Arms B-offset and B-ridge, registered and run 27 October 2026, `python -m src.stage9`). Arm B-offset fixes the slope at 1 and fits only a per-person offset: **one parameter, four bytes.**

| Arm | Params | Worst excess over doing nothing, any cell any budget |
| --- | --- | --- |
| **B-offset** | **1** | **+1.16** |
| C1 | 513 | +3.51 |
| B-ridge | 2 | +7.68 |
| B | 2 | +7.83 |

B-offset beats Arm B at the two smallest budgets in **all eight cell-budget pairs, every one significant**, by up to 7.65 BPM, and it **matches Arm C1 at the smallest budget in all four cells** with the point estimate in its favour everywhere. It costs 0.57 to 0.80 BPM against Arm B at all available data, significantly in all four cells, which is the price of having no slope once a slope is estimable.

So the knee is at **one** parameter at cold start and two once there is twenty minutes of data. The deployment rule is a schedule, not a model-size decision: fit the offset immediately, add the slope later. Arm C1's cold-start advantage was never about capacity; it was about being able to decline to adapt, and one parameter does that better and 513 times more cheaply.

**Arm B-ridge is the instructive failure, and the sharpest form of the result.** Arm B's two parameters, penalised toward slope 1 and offset 0 with the strength selected on held-out adaptation data exactly as C1's is, nests Arm A and Arm B at its endpoints (both asserted in `check_degenerate`). It barely helps: it beats Arm B at the smallest budgets by 0.09 to 1.42 BPM, significant in only 3 of 8, and its worst excess over doing nothing is +7.68 against Arm B's +7.83. The selected penalty sits at the weakest end of the grid even at the two-minute budget. **The reason matters: the held-out slice shares the calibration window's composition, so it is equally unrepresentative of the test block and cannot reveal that the fitted slope will extrapolate badly.** Shrinkage selected on data from an unrepresentative window cannot protect against that window being unrepresentative. Only a structural constraint can, which is what B-offset is. This is a result about what cross-validation can and cannot see, and it belongs in the paper.

Of the five registered predictions, 1, 3 and 5 hold outright, 4 holds at 40 minutes and above but misses the 0.2 BPM bound at 20 minutes in two cells, and **2 holds only directionally**: B-ridge's point estimate beats Arm B in all eight pairs but significantly in three.

**Effective sample size is why.** Windows overlap by six of eight seconds, so N windows carry about N/4 independent samples (`src/effective_n.py`, computed from the union of time spans rather than approximated). Against C1's 513 parameters: 16 independent samples at the 2-minute budget, then 38, 77, 154, 305, and 529 at all available. **C1 only reaches parity with its own parameter count at the largest budget.** That is the cleanest statement of why it needs the penalty and why it cannot beat Arm B on this dataset.

**Arm B2 works only on PaPaGei-S**, +0.34 and +0.30 on the two protocols, with no detectable gain on PaPaGei-P. Reweighting population subjects by similarity helps when the representation does not already encode subject identity, which bears directly on Sub Question 3.

Citations and claims checked against the primary sources: `docs/source-verification.md`. Four citations are wrong, two claims need qualifying, and Bent et al. 2020 is the strongest unused evidence in the bibliography: full Fitzpatrick range, the same Empatica E4, no significant skin-tone effect on heart-rate accuracy, but 30 per cent higher error during activity than at rest.

**Read the worst-windows figure carefully.** `figures/gate2_worst_windows.png` ranks by skewness, which on this dataset ranks by largest single-sample artefact rather than by least usable pulse content. Several of those windows carry clean signal throughout, S9 w2950 and S9 w4056 among them. Nothing in them is dropout: the apparently flat stretches are ordinary pulse of amplitude around 20 crushed by an axis scaled to a spike of 2000. Describe it that way.

**Padding verified against a real window (2026-09-28).** A prepared window is 1,250 samples: 125 leading zeros, a 1,000-sample body which is exactly 8 s at 125 Hz, and 125 trailing zeros. Padding is 20 per cent of every input. It happens after z-scoring, as the paper's step order implies and their example notebook does, so the zeros sit at the body's mean rather than shifting it: body mean -0.0001 and standard deviation 1.0004. The padded window's standard deviation is 0.895, diluted by the zeros but not displaced. Padding before z-scoring would have put the zeros at an offset from the mean and changed every window's statistics.

**Reproduce with their ridge settings, not ours** (read from `linearprobing/regression.py` and `outcome_regression_all.py`): `alpha` grid exactly `[0.1, 1.0, 10.0, 100.0]`, a `StandardScaler` fitted on train and applied to test, `GridSearchCV` with `cv=4` and `scoring='neg_mean_squared_error'` while MAE is what gets reported. The standardisation is the easiest of these to omit and the most likely to move the number. Their `cv=4` is plain KFold, not grouped, so windows from one subject can fall in two folds; that is their protocol and the reproduction matches it rather than improving on it.

Their bootstrap resamples rows of the test set, confirmed in `bootstrap_metric_confidence_interval`, which is windows rather than subjects.

Their `cv=4` leaks by our standards: plain k-fold over pooled windows from nine subjects, so one subject's windows fall in both the fitting and validating folds, and consecutive windows overlap by six seconds. Match it for the reproduction, since fidelity is the point, and note it beside the other pipeline findings. Arm A uses `GroupKFold` grouped by subject instead, and the contrast is worth a sentence.

**The published figures are the best of five pre-training runs, selected per downstream task** (their section 4.1). For PaPaGei-P one model is stated to be best across all tasks; for PaPaGei-S they "choose three models with the highest performance". So 11.53 may not come from the released `papagei_s.pt` at all, and P is the cleaner reproduction target. This is not distinguishable from a pipeline error using the released artefacts, which is why the interpretation is registered in advance in `docs/pre-registered-decisions.md`.

**Their three validation subjects have no stated role.** Alpha comes from `GridSearchCV`'s internal four-fold cross-validation on the training set. Nothing in the paper or code uses the validation split. We hold them out unused rather than training on twelve.

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

| Arm | Description | Trainable per person | Stored per person | Target labels needed |
|---|---|---|---|---|
| A | Frozen encoder, ridge on population set, applied unchanged | 0 | 0 | **none** |
| B | As A, with a per-person affine correction on the predictions | 2 | 8 B | yes |
| B2 | As A, population subjects reweighted by similarity to the target | 0 | **513 values, or refit at inference** | **none** |
| C1 | Frozen encoder, ridge warm-started from the population solution | 513 | ~2 KB | yes |
| C2 | Frozen encoder plus low-rank corrections | 10³-10⁴ | ~4-40 KB | yes |
| D | All encoder parameters on the embedding path retrained | 4,993,024 | ~20 MB | yes |

**Arm B2's storage is not zero, correcting the earlier table.** B2 reweights the population and refits the ridge, which yields a different 513-coefficient model per person. Deploying it means storing those coefficients, which puts it level with C1, or recomputing the fit at inference. What B2 costs nothing of is **labelled data from the user**, which is a different axis and now has its own column. That changes B2's standing for the better: it is not a method that buys 0.16 BPM for free and looks pointless, it is the only method besides the unadapted baseline that asks nothing of the user, and commercially that distinction matters more than parameter count.

**Arm B:** statistics come *exclusively* from the adaptation block. The buffer guarantees temporal disjointness. This mitigates the normalisation-inflation effect documented in Otesteanu et al. (2026).

**Arm B2:** adapts using no labels from the target at all. Weight the 14 population subjects by how closely their embedding distributions resemble the target's — requires only the target's unlabelled signal. Principle adapted from GAUL (Kim et al., 2025), reimplemented in-framework so the backbone stays constant and the cost comparison holds. Zero stored parameters, far more sophisticated than rescaling. If it approaches the labelled arms, that is the most deployment-relevant result in the study.

**Arm C1:** warm-start is primary. Fitting 513 coefficients from scratch on minutes of data may underperform Arm A. Warm-started, it equals Arm A at zero data and improves from there — which handles cold start. Report from-scratch as a comparison; the gap quantifies what warm-starting buys.

**Arm C2 — the open problem.** Four options in ascending ambition:
1. Adapt only the final projection layers ← **named fallback, always works**
2. ~~Adapt only the 1×1 convolutions.~~ **Does not exist in this encoder** (checked 2026-09-28): all 37 Conv1d layers have kernel size 3. The only matrix in the model is the `dense` projection. This option must come out of the proposal's ladder in 4.5.
3. Low-rank decomposition on reshaped convolutional kernels
4. Insert adapter blocks between convolutional blocks

~~Also worth probing: the MoE routing component may be cheaper to adapt.~~ **Not viable:** the MoE heads sit outside the embedding path (see The model), so adapting them cannot change the embedding the ridge head reads. They exist only in PaPaGei-S anyway.

**Option 3 is feasible, spiked 2026-09-28.** `python -m src.spike_c2`. A kernel of shape (out, in, k) is treated as a matrix of shape (out, in*k) and given a low-rank correction, costing r * (out + in*k) per layer. On the final block's two 512-channel convolutions that is 2,048r per layer. Rank 4 on both layers, 16,384 parameters, trained on 512 windows of one subject for 60 steps: loss fell 1.007 to 0.371 against 1.009 to 0.557 for a trainable head alone, gradients reached the adapters, and the corrections moved. This says it runs and trains, not that it helps per person.

Design note for the rank sweep: one convolution at ranks 1 to 4 spans 2,048 to 8,192 parameters, which sits inside the 10³ to 10⁴ band; adapting both convolutions at rank 4 leaves it. Set the sweep on a single convolution, or widen the band in the arms table.

The correction initialises to exactly zero, so an unadapted model is identical to Arm A rather than close to it, which keeps the cost axis starting at a true zero and mirrors C1's warm start.

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
**Status: passed 2026-09-28.** Synthetic 1.2 Hz check returns exactly 36 peaks. Window counts equal the dataset's label counts for all fifteen subjects. Rejection is zero everywhere, for reasons recorded above.
Load one subject. Plot 30s raw. Apply PaPaGei preprocessing. Plot filtered vs raw. Detect peaks, count by eye, compare to ECG label. Then one real window → 512 numbers.

Also: synthetic sine wave check. Generate 1.2 Hz at 64 Hz for 30s, filter, detect peaks, confirm 36. Pipeline verified on data where the answer is known.

### Gate 3 — Embeddings cached
**Status: passed 2026-09-28.** `python -m src.embed`, 76 seconds on MPS. Four tables of 64,697 × 512, both checkpoints in both polarities, 132 MB each, in `results/embeddings/` (not committed; `environment.json` is). Verified: all finite, index aligned to heart rate, re-embedding reproduces cached values to 1.3e-06, and the flipped tables differ from as-is.

**Spike audit** (`python -m src.spike_audit`, `results/spike_audit.csv`). Per-window z-scoring takes its scale from the window's standard deviation, so a single artefact can compress the real pulse. Measured as the largest deviation from the median in interquartile ranges: median 2.3, 95th percentile 6.0, max 67. Beyond 20 IQR there are **71 windows, 0.11%**; beyond 10 IQR, 719 windows, 1.11%. In those 71, the interquartile pulse occupies a median 2.6% of the window range against 24.7% elsewhere, so the mechanism is real but rare. Note it; at this scale it does not need a robustness check, though dropping the 719 and re-running the frontier is minutes of work once the frontier exists.

**This is the gateway. Everything after is arithmetic on a table.**

### Gate 4 — Arm A reproduces
Ridge on frozen embeddings, PaPaGei's evaluation protocol, bootstrap intervals. Compare to 11.53 / 10.92.

**Reported figures, with their intervals:** PaPaGei-S **11.53 [11.40-11.66]**, PaPaGei-P **10.92 [10.80-11.04]** on PPG-DaLiA heart rate.

Those intervals are 500-resample bootstraps over **windows**, not subjects. With three test subjects a subject-level interval would be far wider, so their published interval understates the uncertainty that matters. Our 455-triple distribution measures a different and more honest quantity, and the gap between the two is worth a sentence in the paper.

No seed averaging to account for: the "conducted five times" in their appendix refers to the bootstrap ranking procedure behind Figure 24, not to repeated training runs. The 11.53 is one split.

**The 60/20/20 ratio is confirmed for PPG-DaLiA** (2026-09-28). Their section 4.4 reads: "Initially, we split the in-domain and out-of-domain datasets into training, validation, and test sets at 80/10/10 and 60/20/20 ratios. The splitting is performed at the subject level ensuring no overlap between individuals across the sets." The pairing is by word order rather than an explicit "respectively", so a second argument settles it: PPG-DaLiA is greyed in Table 2 as out-of-domain, and 80/10/10 on fifteen subjects gives 12/1.5/1.5, which does not divide. 60/20/20 gives 9/3/3.

**Open issue: 11.53 is not a leave-one-subject-out number** (verified 2026-09-22). The paper's linear evaluation uses a single fixed subject-level train/val/test split: 80/10/10 for in-domain datasets, **60/20/20 for out-of-domain**. PPG-DaLiA is out-of-domain (unseen in pretraining), so roughly 9/3/3 subjects. PaPaGei's feature-extraction job chunking is consistent with about 3 test subjects. So 11.53 is an MAE over ~3 held-out people, with 500-run bootstrap intervals.

Consequences:
- **Reproduction and Arm A are two different numbers.** Reproduce 11.53 under their 9/3/3 split. Arm A under our leave-one-subject-out rotation will differ and is not expected to equal it.
- **Their split files are not public.** `utilities.py` reads `data/dalia/{train,val,test}.csv`, which are not in the repo. Which 3 subjects were held out is unknown, and with 3 test subjects the MAE depends heavily on which ones. Options: ask the authors for the split, or report the distribution of MAE over all 9/3/3 splits (455 choices of test subjects) and check whether 11.53 falls inside it.
- The reference-numbers table mixes protocols: the classical rows are leave-one-subject-out, the frozen-probe rows are this fixed split. Label them as such before comparing.

If it doesn't match: debug against their published intermediate values. If it still doesn't, document the discrepancy carefully. A documented failed reproduction is a legitimate result.

**Splitting implemented and self-checking (2026-09-28).** `python -m src.splits` asserts every invariant for both protocols and all fifteen subjects: the target never appears in its own population set, no row sits in two blocks, no adaptation window shares a sample with any test window, test follows adaptation, and budgets nest.

Two things the checks caught, both real:
- **Activity runs need a buffer at their boundaries, not only at the internal cut.** Activities alternate with transient periods, and a window straddles the join, so the last test window of one run overlapped the first adaptation window of the next. Splits are now built as labelled time regions with the buffer enforced at every boundary between differently labelled regions.
- **The ordering invariant is per run, not per activity code.** Transient recurs throughout the recording, so an early run's test block legitimately precedes a later run's adaptation block. The first version of the check asserted the wrong thing.

Cost of the stratified protocol: roughly 220 windows per subject go to buffers, about 5 per cent, because each of the sixteen activity runs needs its own. The naive temporal protocol spends 7. Adaptation and test come out near equal at about 2,030 and 2,070 windows.

**Two decisions the splitter makes, now stated rather than implied.** Subjects are stratified over whichever activity runs they actually have, with no special casing: S6's recording stops after about 1.5 hours, so it has no lunch, walking or working and gets 11 runs against 16 for everyone else. And transient periods are runs like any other label rather than being discarded, which matters because they are 26.1 per cent of adaptation windows on average and 43.2 per cent for S6. Both belong in the methodology's section on splitting.

Every subject reaches the full budget sweep. S6 is the binding case at 40.8 usable adaptation minutes against the 40-minute budget, a margin of 48 seconds, so any increase to the largest budget or the buffer would exclude it.

**Status: passed 2026-09-29.** `python -m src.stage4`, 971 rows in `results/experiments.csv`, figure `figures/gate4_reproduction.png`.

Both checkpoints reproduce, which is the clean branch of the registered 2x2. The published figure falls inside our 455-triple distribution for each: PaPaGei-P 10.92 at the 32nd percentile of a distribution with median 11.78, PaPaGei-S 11.53 at the 28th percentile of a median 12.77. Our medians sit about 1 BPM above the published values on both, consistently, which is what selecting a favourable split and the best of five pre-training runs would look like.

**The headline methodological result.** Compared like with like, 95 per cent against 95 per cent: our interval spans **8.75 BPM** for P and 9.47 for S, theirs **0.24** and 0.26. A factor of **36** for both. (At 90 per cent ours spans 7.73 and 7.68, which is where the earlier factor of 32 came from; that comparison was 90 against 95 and understated it.) Their interval bootstraps windows within one fixed choice of three test subjects; ours varies which three subjects those are. On a fifteen-subject benchmark the second is the uncertainty that matters, and it is thirty times the first. This is the sentence that supports Sub Question 2's argument for reporting distributions.

**Arm A, leave-one-subject-out, mean per-subject MAE:** P as-is 11.70, P flipped 11.64, S as-is 13.04, S flipped 13.82.

**The polarity control did its job, and the hypothesis did not survive it.** The sign-invariance asymmetry is confirmed exactly as the training recipe predicts: flipping moves PaPaGei-P by +0.06 BPM, which is nothing, and moves PaPaGei-S by 0.79 BPM. P is sign-invariant, S is not. But the direction is the opposite of the inversion hypothesis. Flipping to restore conventional polarity makes S **worse**, better on only 3 of 15 subjects, so S prefers the wrist signal as it comes. Restoring convention does not recover accuracy, and the inversion does not explain PaPaGei's mid-table showing on this benchmark. Report it as a negative result; the morphological evidence for the inversion still stands, it simply does not have the consequence predicted.

**A residual puzzle, to state rather than smooth over.** PaPaGei-S is morphology-sensitive, it is sign-sensitive as its training recipe predicts, and yet it prefers the wrist signal in the orientation it arrives in rather than the orientation its pre-training corpus was in. Those three do not sit together comfortably. Either the pre-training corpus orientation needs re-checking, or sensitivity to sign does not imply a preference for the orientation seen during pre-training. Report it as an open question.

**Population size: returns diminish, but the earlier "saturates at six" claim was wrong** (`python -m src.population_curve`, 25 draws per size, figure `gate4_population_curve.png`). Mean per-subject MAE, PaPaGei-P:

| Subjects | 2 | 4 | 6 | 8 | 10 | 12 | 14 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MAE | 15.11 | 12.94 | 12.31 | 12.00 | 11.86 | 11.76 | 11.70 |
| Change | | -2.18 | -0.63 | -0.31 | -0.13 | -0.11 | -0.06 |

The paired per-subject comparison of six against fourteen is **+0.604 BPM, 95 per cent interval [0.470, 0.769], worse with six on 15 of 15 subjects**. The earlier figure of 0.19 BPM came from three draws per size and was noise. Twelve to fourteen buys 0.055, so returns do flatten, but nearer **ten to twelve subjects than six**.

This weakens the premise argument without removing it. Going from six subjects to fourteen buys 0.60 BPM, against a per-subject spread running from 7.1 to 23.8. Population size still matters far less than which person you are, but "more population data does not help" is too strong and should not be written.

Note on the figure: sizes below fourteen average 25 draws across 15 subjects, while fourteen is deterministic with one value per subject, so its error bar is not comparable to the others. The paired test is the number to quote.

**Sub Question 2, stress-tested** (`python -m src.difficulty_checks`). Per-subject error under the unadapted population model runs from 7.14 BPM on S7 to 24.46 on S5, a factor of 2.5. Distance of a subject's median heart rate from the population median predicts it, but state the strength carefully: r = 0.87 across all fifteen with a bootstrap interval of [0.52, 0.97], r = 0.89 without S6, and **r = 0.63 explaining 40 per cent** among the thirteen once both extremes are removed. Five predictors were tested on fifteen points. The relationship is real and it survives leverage, but the two outliers make it look stronger than it is.

**The mechanism is largely ridge shrinkage, not physiology.** Decomposing each subject's error into bias and spread: S5 is 92 per cent bias at -22.6 BPM, S6 83 per cent at -14.3, and the well-served subjects 2 to 21 per cent. High heart-rate subjects are under-predicted and low ones over-predicted, which is regression to the training mean. A referee who knows ridge would ask within seconds, so the decomposition belongs beside the correlation. It also sharpens the study rather than weakening it: a systematic offset is what Arm B exists to fix, and the prediction is registered in `docs/pre-registered-decisions.md` before Arm B runs.

**S6 is not a collection artefact.** Its recording is truncated and the missing hour holds lunch, walking and working, whose medians across the other subjects are 80, 96 and 74 BPM, so the concern was reasonable. Restricting every subject to the activities S6 does have, the other fourteen sit at a median of 88 against S6's 121. Truncation accounts for about 4 BPM of S6's 37 BPM distance; the other 33 is the subject.

**A label-free triage signal exists.** Replacing the subject's true median heart rate with the population model's own median prediction on that subject's unlabelled windows gives r = 0.85 across fifteen and 0.62 across thirteen, as good as the labelled version. It needs no ground truth, so it could identify who needs personalisation before any is fitted, and it is the same quantity Arm B2 weights on.

**Skin tone is out of reach for this sample.** Fitzpatrick types run 2 to 4, eleven of fifteen at type 3, with types 5 and 6 absent. The study cannot speak to personalisation across skin tone because the range is not in the data, which is a harder limit than sample size and belongs in the limitations in those terms.

**Sub Question 3 is not confounded by sign invariance.** P beats S on 14 of 15 subjects in both polarities, by 1.34 BPM as-is and 2.18 flipped. The ordering survives the control, so whatever separates the two objectives on this dataset is not their treatment of sign.

### Gate 5 — Experiment harness
**Build before the arms multiply, not after.**

One function: configuration in, result row out, every setting recorded.

```
config = {arm, backbone, subject, budget_minutes, split_type, seed, hyperparams}
→ run → append row to results/experiments.csv recording config + metrics
        + parameter and byte counts + usable and elapsed minutes
        + timestamp + commit hash of the code that produced it
```

6 arms × 6 budgets × 15 subjects × 2 backbones × 2 split protocols, plus the C2 rank sweep and the C1 from-scratch variant, is **on the order of 3,000 runs**. Computationally trivial, impossible to track by hand.

### Gate 6 — Arms B, B2, C1
**Status: passed 2026-09-29.** `python -m src.stage6`, 1,500 rows, 844 seconds. All four degenerate assertions pass: each arm reproduces Arm A exactly when it is doing nothing.

Headline, PaPaGei-P, activity-stratified, all available personal data, mean per-subject MAE:

| Arm | MAE | Against Arm A | Cost per person |
| --- | --- | --- | --- |
| A | 11.72 | | 0 |
| **B** | **9.65** | **-2.08** | **2 values** |
| B2 | 11.57 | -0.16 | 0 |
| C1 | 10.09 | -1.64 | 513 |
| C1-scratch | 10.19 | -1.53 | 513 |

**Two values per person beat 513, and now detectably.** Paired across all fifteen subjects at all available data, C1 minus B is **+0.44 BPM**, B better on 10 of 15. The earlier figure of -0.047 BPM with an interval spanning zero came from the defective C1 and is withdrawn. The claim is now the stronger one: Arm B is better than Arm C1 in every cell measured, not merely indistinguishable from it.

**But do not say the residual is therefore a scalar rescaling.** Arm B's prediction is a linear model, so it lies inside Arm C1's solution space, and if the residual really were just a rescaling then C1 would match B subject by subject. It does not: at all available data the two still trade by up to **-3.09 BPM on S5, where C1 is better, and +2.94 on S8, where B is**, with C1 ahead on 5 of 15. So B wins on average while the two suit different people, and the honest claim is that 513 parameters buy nothing *on average*, not that the other 511 dimensions are empty. (Figures updated 26 October 2026 after the C1 selection fix; the earlier +3.09 on S5 and -2.46 on S8 came from the defective arm.)

~~Checked separately: C1's shrinkage selection lands on the weakest value in the registered range for every subject, so the warm start is not doing work at full data.~~ **Withdrawn 26 October 2026.** That was the symptom of the selection defect, not a property of the warm start. With selection held out of the fit, no cell is pinned to a grid endpoint for all fifteen subjects, in any of the 24 cells. The registered prediction that the penalty would strengthen monotonically as the budget shrinks **failed**: the median selected penalty is roughly flat at 10³ to 10⁴ across budgets. Recorded as a failed prediction rather than quietly dropped.

**The composition hypothesis for S6 is refuted.** The obvious explanation for S6's shortfall, that its adaptation and test blocks differ in activity mix, does not hold. S6's adaptation-to-test gap in mean heart rate is **0.03 BPM, the second smallest of all fifteen**, and across subjects the shortfall correlates with composition mismatch at **r = -0.53**, the wrong sign. Whatever stops the affine correction working on S6 and S11, both of which have large bias and little gain, it is not composition. A candidate worth testing later: the offset those subjects need may vary within the test block, which a single scale and offset cannot follow.

**The registered Arm B prediction is confirmed on its main claim.** Arm B's improvement tracks each subject's bias at **r = 0.84, p = 0.00008**, and tracks distance from the population median at r = 0.61, p = 0.016. S5, the worst-served subject, gains most of anyone at -7.75 BPM. **The prediction was half wrong on S6**, which ranks only 6th of 15 at -1.96 despite having the second-largest bias. A plausible explanation, untested: S6's adaptation block is 43 per cent transient and contains no lunch, walking or working, so the correction fitted there may not transfer to its test block.

**The registered Arm B2 prediction is confirmed.** B2 gains 0.16 BPM, close to nothing, as anticipated from the population curve showing fourteen subjects worth little more than six. Its ceiling is low because its whole mechanism is reweighting subjects whose number barely matters.

**Small budgets make personalisation actively harmful, for a compositional reason.** At 2 and 5 minutes every adaptive arm is far worse than Arm A, and 5 minutes is worse than 2. This is not a sample-size effect and not a bug. Budgets are prefixes in time, as registered, so the first minutes of a recording are sitting still: for S5 the 5-minute budget has a mean heart rate of 94.7 against a test block at 125.5. The calibration window is unrepresentative of what is being predicted. That is precisely the cold-start problem, and it is a deployment-relevant result rather than an artefact, but it must be described as composition rather than data volume.

A secondary budget definition, sampling proportionally across activity runs rather than taking a temporal prefix, would separate composition from volume. It is worth registering and running as a labelled variant.

### Gate 7 — Arm C2
**Status: passed 2026-10-27, both arms.** The rerun under the three amendments registered
on 26 October wrote 144 configurations in 4.4 hours; Arm D was then rerun at a reselected
learning rate, 24 configurations in 69 minutes, launched through `src/freeze.py` on the
tag `armd-2026-10-27`. **That is the first run in this project with clean provenance:**
every one of its 360 rows carries `a7fac63` with no `-dirty` suffix and the executing-path
digest `597c1b602592`. `python -m src.stage7 p,s activity,temporal`.

**The amendments changed the answer, and the earlier headline was wrong.** Under the
withdrawn protocol C2 at rank 16 beat Arm A by 0.60 BPM, no detectable difference. Under
the corrected one it beats Arm A by **2.04 BPM**, and the smoke test's warning was right:
the sentence that everything costlier than two parameters is indistinguishable from doing
nothing does not survive. Most of that gap was optimisation, not capacity, exactly as the
earlier pass's own caveat feared.

**The frontier.** PaPaGei-P, activity-stratified, all available personal data. Paired
per-subject differences against Arm A, 5,000-resample bootstrap intervals.

| Arm | Trainable | Stored | User labels | MAE | vs A | 95% interval |
| --- | --- | --- | --- | --- | --- | --- |
| A, population model | 0 | 0 | none | 11.72 | | |
| **B-offset, offset only** | **1** | **4 B** | yes | 10.22 | +1.50 | [+0.63, +2.54] |
| **B, affine** | **2** | **8 B** | yes | **9.65** | **+2.08** | **[+1.03, +3.33]** |
| B-ridge, shrunk affine | 2 | 8 B | yes | 9.70 | +2.02 | [+0.97, +3.23] |
| B2, unlabelled reweighting | 0 | 2 KB | none | 11.57 | +0.16 | [-0.21, +0.55] |
| C1, output layer | 513 | 2 KB | yes | 10.09 | +1.64 | [+0.49, +3.26] |
| C2 rank 1 | 2,048 | 8 KB | yes | 11.05 | +0.67 | [-1.01, +2.62] |
| C2 rank 4 | 8,192 | 32 KB | yes | 10.18 | +1.55 | [-0.05, +3.39] |
| C2 rank 8 | 16,384 | 64 KB | yes | 9.88 | +1.84 | [+0.35, +3.64] |
| C2 rank 16 | 32,768 | 128 KB | yes | 9.69 | +2.04 | [+0.62, +3.76] |
| D, full fine-tuning | 4,993,024 | 20 MB | yes | 10.77 | +0.96 | [-0.54, +2.71] |

**The answer to the main research question survives, and in a better form.** It is no
longer that costlier adaptation fails. It is that costlier adaptation **works and still
does not beat two parameters**. Paired per-subject, at all available data:

| Contrast | P activity | P temporal | S activity | S temporal |
| --- | --- | --- | --- | --- |
| B vs C2 rank 16 | +0.04 ns | -0.17 ns | **+1.88** | +0.94 ns |
| B vs C2 rank 8 | +0.23 ns | -0.24 ns | **+1.63** | +0.95 ns |
| B vs C1 | +0.44 ns | +0.69 ns | **+1.95** | **+1.45** |
| B vs B-offset | **+0.57** | **+0.80** | **+0.64** | **+0.76** |
| B vs D (4,993,024) | **+1.12** | **+0.69** | **+1.86** | **+1.56** |

Positive means Arm B is better; bold is a detectable difference, `ns` is none at this
sample size. **Arm B is not beaten in any of the twenty contrasts.** Against the 32,768
parameters of C2 rank 16 it ties on PaPaGei-P and wins on PaPaGei-S. Against 513 it ties
on P and wins on S. Against one parameter it wins in all four cells, so the second
parameter earns its keep at full data even though the first does the cold-start work. And
against all 4,993,024 it **wins detectably in all four cells**, which is the cleanest form
of the result the study was built to produce.

**Cold start: only the one-parameter arm is bounded.** Worst excess over doing nothing
across all 24 budget-cells: **B-offset +1.16**, C1 +3.51, B-ridge +7.68, C2 rank 16 +7.29,
B +7.83, D +11.77. Every arm that fits a slope can extrapolate badly from an unrepresentative
calibration window; an offset cannot, because it is bounded by a mean shift. That is the
mechanism behind the crossover, and it now covers C2 as well as C1.

**Registered predictions, scored.** Three of seven failed, and the one whose failure would
have cost most held.

1. C1 never exceeds Arm A by more than 2 BPM at any budget: **failed**, worst +3.51 at
   s/temporal 10 minutes. The defect fix removed the catastrophe, 33.22 down to 3.51, but
   not the overshoot.
2. C1's selected penalty strengthens as the budget shrinks: **failed**, non-monotone, the
   median runs 1e3, 1e2, 1e4, 1e4, 1e3, 1e3. The second half holds: it is no longer the
   weakest value for every subject.
3. C1 lands within 0.5 BPM of Arm B at all available data in all four cells: **failed**,
   holds only on p/activity at +0.44; the others are +0.69, +1.95, +1.45.
4. C1 does not overtake Arm B: **held**, in all four cells. This is the prediction
   registered as the one whose failure would have moved the knee from two parameters to
   513, and it held.
5. C2 improves at every rank and reaches but does not clearly beat Arm B: **held**, all
   twenty rank-cells improve on Arm A and none beats B.
6. Arm D remains at or below Arm A at every budget: **failed**, in both directions. It is
   detectably **worse** than doing nothing at the two smallest budgets in all four cells,
   by 3.15 to 11.28 BPM, and at 10 and 20 minutes on p/temporal. At all available data it
   is 0.63 to 1.11 BPM **better** than Arm A, in none of the four cells detectably. The
   first failure is what the methodology always predicted for this arm; the registered
   prediction was simply stated too strongly.
7. No arm's position on the cost axis moves: **held** trivially.

**Caveat, unchanged and now sharper.** Arms B, B-offset, B-ridge and C1 are closed form;
C2 is trained by gradient descent. Its rate is selected per rank on held-out adaptation
data of two subjects, which is a thin basis, and the selected rate is the most aggressive
in the grid at four of five ranks. Part of the remaining gap to Arm B may still be
optimisation. The honest statement is that under a protocol fixed in advance, two stored
values are not beaten by anything up to 32,768.

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

**Arm D needs a lot of data to be harmless and never becomes worth its price.** Mean
per-subject MAE by budget, with Arm A in brackets:

| | 2 | 5 | 10 | 20 | 40 | all |
| --- | --- | --- | --- | --- | --- | --- |
| p/activity (11.72) | 20.84 | 23.00 | 13.31 | 12.59 | 11.15 | 10.77 |
| p/temporal (10.36) | 17.31 | 19.93 | 22.12 | 12.60 | 10.07 | 9.55 |
| s/activity (13.14) | 17.47 | 17.13 | 14.26 | 13.58 | 12.48 | 12.03 |
| s/temporal (11.53) | 14.68 | 14.94 | 13.36 | 12.25 | 12.26 | 10.90 |

The curve is monotone downward after the five-minute budget in every cell and crosses
Arm A between twenty and forty minutes. So the arm does train, and the earlier reading
that a stability criterion would drive it to its do-nothing limit was wrong: training MAE
falls on 7 to 11 of 15 subjects at the larger budgets against 1 of 15 under the broken
rate. **What it buys at full data is about 1 BPM, roughly half what two parameters buy,
for ten million times the storage.** That is the frontier's far end stated properly, and
it is a better result than the withdrawn one because the arm is working rather than broken.

**Arm D's rate differs by backbone and one cell is weaker than the other three.** PaPaGei-P
runs at 1e-7, the smallest rate in the registered grid, and PaPaGei-S at 3e-7, chosen by
the fallback after the guard reported that no rate reduced training error on both selection
subjects. s/temporal is the one cell where Arm D never beats Arm A at any budget. Treat the
PaPaGei-S Arm D numbers as the least reliable in the study and say so.
