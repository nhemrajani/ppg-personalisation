# Experiment Report

Compute-Efficient Personalisation of PPG Foundation Models. Work completed to
30 September 2026. Every figure quoted is drawn from `results/experiments.csv`,
which holds **7,570 runs**, each carrying its configuration, its metrics, a
timestamp and the commit hash of the code that produced it.

---

## 1. What was built

A reproducible pipeline, in the order it runs.

| Component | Module | What it does |
| --- | --- | --- |
| Data loading | `src/data.py` | Reads PPG-DaLiA, handling the Python 2 pickle encoding |
| Preprocessing | `src/preprocess.py` | PaPaGei's pipeline, assembled from their own functions |
| Quality measurement | `src/quality.py`, `src/measure_quality.py` | Replicates their flatline test exactly; signal quality by activity |
| Encoder | `src/papagei.py` | Loads both checkpoints, pinned submodule, strict weight loading |
| Embeddings | `src/embed.py` | Caches every window, both checkpoints, both polarities |
| Splitting | `src/splits.py` | Population, adaptation, buffer and test blocks, with invariants asserted |
| Ridge | `src/ridge.py` | Gram-matrix ridge, verified against scikit-learn to 1e-8 |
| Arms | `src/arms.py`, `src/stage7.py` | A, B, B2, C1 in closed form; C2 and D trained |
| Record | `src/harness.py` | One row per run, with commit hash |

PaPaGei's source is a git submodule pinned at commit `0c537da`. The environment
rebuilds from two requirements files, the second installed without dependencies
because `pyPPG` and `biobss` pin their entire 2022 development environments.

Two decisions shaped everything downstream. Embeddings are computed **once**, which
took 76 seconds and turned every subsequent experiment into arithmetic on a cached
table. And they were computed in **both signal polarities in the same pass**, which
later made a control experiment cost minutes instead of a repeat of that step.

---

## 2. Method as implemented

**Data.** PPG-DaLiA, fifteen subjects, wrist PPG at 64 Hz with chest ECG ground truth,
eight activities plus labelled transition periods. 64,697 analysis windows.

**Preprocessing.** PaPaGei's published pipeline, unmodified: fourth-order Chebyshev
Type II bandpass between 0.5 and 12 Hz applied at the native rate with zero-phase
filtering, segmentation into 8-second windows at a 2-second shift, flatline rejection,
per-window z-scoring, resampling to 125 Hz and centre padding to 1,250 samples.

**Splitting.** For each target subject: the other fourteen form the population set; the
target's own recording divides into an adaptation block, a discarded buffer of at least
one window, and a later test block. Primary protocol is activity-stratified; the naive
single-cut temporal split is reported as a realism check.

**Arms.** A, unadapted population model. B, per-person affine correction. B2,
population subjects reweighted by similarity to the target using no target labels.
C1, per-person output layer, warm-started and from scratch. C2, low-rank adaptation of
one convolution. D, full fine-tuning.

**Statistics.** MAE in BPM, computed per subject and averaged, with the distribution
reported. Comparisons are paired per subject with bootstrap intervals, since every arm
evaluates on identical test blocks. Where an interval spans zero the finding is
reported as no detectable difference.

**Pre-registration.** Every free parameter was committed before the runs that would
test it: the ridge grid and selection rules, each arm's specification, the degenerate
assertions, the protocol and compute plan, and two falsifiable predictions. Three
amendments were made, each dated, each with its reason, and each before any valid
result for the affected arm existed. Recorded in `docs/pre-registered-decisions.md`.

---

## 3. Results

### 3.1 Reproduction

PaPaGei's published split is not released, so rather than guess, every one of the 455
possible three-subject test sets was run under their protocol.

| | Published | Our median | Published sits at | Inside our range |
| --- | --- | --- | --- | --- |
| PaPaGei-P | 10.92 | 11.78 | 32nd percentile | yes |
| PaPaGei-S | 11.53 | 12.77 | 28th percentile | yes |

Both reproduce. Under the diagnostic registered in advance, this is the branch where
the pipeline is validated and no further explanation is required. Our medians sit
about 1 BPM above theirs on both checkpoints, which is the shape a favourable split
selection would leave.

### 3.2 The frontier

PaPaGei-P, activity-stratified, all available personal data.

| Arm | Trainable per person | Stored | User labels | MAE | vs A |
| --- | --- | --- | --- | --- | --- |
| A, population model | 0 | 0 | none | 11.72 | |
| **B, affine correction** | **2** | **8 B** | yes | **9.65** | **-2.08** |
| B2, unlabelled reweighting | 0 | 513 values | none | 11.57 | -0.16 |
| C1, per-person output layer | 513 | 2 KB | yes | 9.69 | -2.03 |
| C2 rank 1 | 2,048 | 8 KB | yes | 12.53 | +0.80 |
| C2 rank 4 | 8,192 | 32 KB | yes | 11.43 | -0.29 |
| C2 rank 16 | 32,768 | 128 KB | yes | 11.12 | -0.60 |
| D, full fine-tuning | 4,993,024 | 20 MB | yes | 12.12 | +0.40 |

Paired per-subject differences with bootstrap intervals: Arm B beats Arm A by
**+2.08 [1.01, 3.28]**; C2 at rank 16 gives **+0.60 [-0.76, +2.22]**, spanning zero;
Arm D gives **-0.40 [-2.39, +1.73]** and is worse than doing nothing on 8 of 15
subjects. Arm B beats C2 rank 16 by **+1.48 [0.96, 1.95]** and Arm D by
**+2.47 [1.26, 3.71]**.

**Two stored values is the only method that detectably improves on the population
model.** Everything costlier is either indistinguishable from doing nothing or
indistinguishable from two values, and the two most expensive arms are significantly
worse than the cheapest.

### 3.3 Robustness across backbone and protocol

Improvement over Arm A, paired per subject, in all four cells.

| Cell | Arm B | Arm C1 | Arm B2 |
| --- | --- | --- | --- |
| P, activity | **+2.08 [1.01, 3.28]** | +2.03 [0.75, 3.65] | +0.16 [-0.21, 0.56] |
| P, temporal | **+1.50 [0.39, 2.75]** | -0.23 [-1.75, 1.17] | +0.21 [-0.08, 0.55] |
| S, activity | **+2.97 [1.58, 4.54]**, 15/15 | +0.01 [-0.63, 0.79] | **+0.34 [0.12, 0.64]** |
| S, temporal | **+2.19 [0.95, 3.54]** | **-1.34 [-2.35, -0.45]** | **+0.30 [0.14, 0.46]** |

This is the strongest result in the study. **Arm B improves significantly in every
cell.** Arm C1 improves in one, shows no detectable gain in two, and is significantly
**worse than doing nothing** in the fourth. The earlier reading that B and C1 are
equivalent held only in the single cell where both were first measured; across
conditions, two parameters are robust and 513 are not.

The likely mechanism is stability. Arm B estimates two values in closed form; Arm C1
estimates 513 from the same limited data, and when the adaptation block is small or
unrepresentative it overfits.

**Arm B2 is small but real, and only on PaPaGei-S.** It gives a detectable gain on both
protocols for the checkpoint whose pretraining objective is organised by waveform
morphology, and none on the checkpoint that clusters by subject. Reweighting population
subjects by similarity helps when the representation does not already encode identity,
which speaks directly to Sub Question 3.

### 3.4 Why the cheap method wins

Decomposing each subject's error under the population model into bias and spread, the
poorly served subjects are almost entirely bias: **92 per cent for S5** at -22.6 BPM,
83 per cent for S6, against 2 to 21 per cent for the well served. High heart-rate
subjects are under-predicted and low ones over-predicted, which is ridge shrinkage
toward the training mean.

Registered as a prediction before Arm B ran: its improvement should track that bias.
It does, at **r = 0.84, p = 0.00008**. The prediction's second half, that S5 and S6
would gain most, was half right: S5 gained most of any subject at 7.75 BPM, but S6
ranked sixth of fifteen. The obvious explanation, that S6's adaptation and test blocks
differ in activity composition, was tested and **refuted**: S6's adaptation-to-test gap
in mean heart rate is 0.03 BPM, second smallest of fifteen, and across subjects the
shortfall correlates with composition mismatch at r = -0.53, the wrong sign.

### 3.5 Budget sweep

PaPaGei-P, activity-stratified, mean per-subject MAE.

| Arm | 2 min | 5 min | 10 min | 20 min | 40 min | All |
| --- | --- | --- | --- | --- | --- | --- |
| B | 17.40 | 19.55 | 11.69 | 10.44 | 9.91 | 9.65 |
| C1 | 15.42 | 27.72 | 21.38 | 15.20 | 11.69 | 9.69 |
| C2 rank 16 | 16.96 | 18.49 | 13.50 | 12.05 | 11.41 | 11.12 |
| D | 23.89 | 23.77 | 18.09 | 14.11 | 12.25 | 12.12 |

Against Arm A at 11.72. **At two and five minutes every adaptive arm is substantially
worse than no adaptation at all**, and five minutes is worse than two.

This is a composition effect, not a sample-size one. Budgets are prefixes in time, as
registered, so the first minutes of a recording are sitting still: for S5 the
five-minute budget averages 94.7 BPM against a test block at 125.5. Calibrating on
unrepresentative data is worse than not calibrating. A second budget definition
holding composition constant is registered and not yet run.

### 3.6 Per-subject difficulty

Arm A ranges from **7.14 BPM on the best-served subject to 24.46 on the worst**, a
factor of 2.5. Distance of a subject's median heart rate from the population median
predicts that error at r = 0.87 across all fifteen, falling to **r = 0.63, explaining
40 per cent**, once the two extreme subjects are removed. Five predictors were tested
on fifteen points, and that is the honest figure.

A label-free version, substituting the population model's own predictions for true
heart rates, performs as well at r = 0.85. It requires no ground truth from the
subject, so it could identify who needs personalisation before any is fitted.

S5 and S6 are hard for other models too: Reiss et al.'s CNN gives 18.97 and 13.55 on
them, and the Conv-LSTM's per-activity table shows S5 at 66.42 on stairs.

### 3.7 Population size

Mean per-subject MAE against the number of population subjects, 25 random draws per
size.

| Subjects | 2 | 4 | 6 | 8 | 10 | 12 | 14 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PaPaGei-P | 15.11 | 12.94 | 12.31 | 12.00 | 11.86 | 11.76 | 11.70 |
| PaPaGei-S | 15.85 | 14.14 | 13.62 | 13.31 | 13.16 | 13.09 | 13.04 |

Six against fourteen, paired per subject: **+0.604 [0.470, 0.769] on P** and
**+0.585 [0.434, 0.755] on S**, worse with six on 15 of 15 subjects in both. Returns
flatten nearer ten to twelve subjects than six. Population size matters far less than
which subject you are, but the stronger claim that more population data does not help
is not supported.

### 3.8 The polarity experiment

PPG-DaLiA's wrist channel is inverted relative to conventional PPG. The evidence is the
activity ordering: read with inversion assumed, the nine activities order monotonically
by physical intensity, from sitting at the cleanest to stair climbing at the noisiest.
The conventional reading would hold that climbing stairs yields cleaner wrist PPG than
sitting still.

PaPaGei-P trains with negation as an augmentation at probability 0.20; PaPaGei-S
deliberately avoids augmentations that alter morphology. Flipping the signal moves P by
**0.06 BPM** and S by **0.79**, exactly the asymmetry the recipes predict.

The hypothesis that inversion explains PaPaGei's mid-table performance on this
benchmark was **tested and failed**: flipping to restore conventional polarity makes S
worse, not better. Reported as a negative result.

It also served as a control for Sub Question 3. **P beats S on 14 of 15 subjects in
both polarities**, by 1.34 BPM as-is and 2.18 flipped, so the comparison between the
two pretraining objectives is not confounded by sign invariance.

---

## 4. Findings about PaPaGei's published pipeline

Reported neutrally, with evidence, and belonging in a short methods subsection rather
than framed as criticism.

**Published precision is narrower than it appears.** Their 95 per cent interval spans
**0.24 BPM**; ours, over the same comparison, spans **8.75**. A factor of 36. Their
bootstrap resamples windows within one fixed choice of three test subjects; ours varies
which three subjects those are. On a fifteen-subject benchmark the second source
dominates and is invisible in the published figure.

**The flatline rejection step cannot fire on this dataset.** It compares consecutive
samples against an absolute threshold of 0.01 while the median step in the signal is
4.84 units. Zero of 64,697 windows are rejected, at every parameter setting tested.

**A standardiser appears in the code and nowhere in the paper**, and is the single
likeliest cause of a failed reproduction.

**Alpha selection is not grouped by subject**, so one subject's overlapping windows
fall in both cross-validation folds.

**The published figures are the best of five pretraining runs, selected per task.** For
PaPaGei-S they explicitly choose three different models across tasks, so the released
checkpoint may not be the one that produced 11.53, and nothing published allows that to
be determined.

**The Zenodo weights record states no licence.** With the point above, this is a
concrete instance of a model being fully open in weights and code while its headline
number remains unreproducible from what was released.

---

## 5. Corrections and failures

Recorded because the experiment record should show what went wrong as well as what
worked.

**Claims corrected after checking.** An early reading that six population subjects were
worth as much as fourteen came from three random draws and was noise; with 25 draws the
difference is 0.60 BPM and consistent across all subjects. An early reading that
distance from the population median explains 76 per cent of per-subject error was
driven by two high-leverage subjects; among the other thirteen it explains 40 per cent.
An early claim that Arms B and C1 are equivalent held only in the one cell where both
were first measured.

**Hypotheses tested and refuted.** That signal inversion explains PaPaGei's performance
on this dataset. That S6's elevated heart rate is an artefact of its truncated
recording, where truncation accounts for about 4 BPM of a 37 BPM gap. That activity
composition explains where Arm B underperforms.

**Implementation errors, found and fixed.** Arm C2 and Arm D were first run with the
output layer fitted on standardised embeddings but applied to raw ones, giving 331 BPM
for Arm D; a registered assertion requiring each arm to reproduce Arm A before training
would have caught it and had not been implemented. Arm D then diverged at its
registered learning rate of 1e-3, which had never been tested; it was reselected on
training-loss convergence alone and moved to 1e-5. A subsequent patch silently failed
to apply and the run reproduced the broken numbers exactly. No invalid row survives in
the record; each defective run was deleted rather than reported.

---

## 6. Outstanding

- The second budget definition, sampling proportionally across activity runs, which
  separates composition from data volume.
- The two principal figures: the cost-accuracy frontier and the sample-efficiency
  curve.
- Stratified analyses by activity and by subject.
- Arms C2 and D on PaPaGei-S, currently run on PaPaGei-P only.
