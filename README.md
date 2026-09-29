# Compute-Efficient Personalisation of PPG Foundation Models

Independent Study, Fall 2026
Neeharika Hemrajani, Yale School of Management
Supervisor: Professor Sohee Park

Research log (password protected): https://neeha.xyz/ppg-personalisation
Weekly entries, current results, and the standing of each deliverable. The password
is shared separately.

> Disclaimer: this repository is a work in progress and is being drafted with Claude
> Code. Everything here is provisional and subject to revision until I have updated
> it with all the relevant information.

---

## Abstract

Consumer wellness wearables rely on photoplethysmography (PPG) to infer an
individual's cardiovascular health using a shared population model, often
personalised by normalising each user's readings against their own rolling baseline.
As artificial intelligence and machine learning continue to advance, the future of
personalised health will rely on intelligent adaptation to each individual, unlocking
a new era in health and longevity science. Currently, one of the biggest bottlenecks
to achieving this level of personalisation is cost, specifically whether it is worth
adapting wearable models to each person, how much of that person's data this would
require and the feasibility of operating and scaling this technology to millions of
users.

Across the literature on foundation models for healthcare, four families of
per-person adaptation have been demonstrated on physiological signals, but each
relies on a different backbone, a different task, and a different definition of cost
than the current industry standard of using normalisation. The literature offers no
common frontier for practitioners to compare methods and outputs.

This study constructs this frontier to compare and establish the cost-accuracy
trade-off in unlocking personalised mass-market healthcare and wellness. Using
PaPaGei, an open PPG foundation model from Nokia Bell Labs, and PPG-DaLiA, a public
dataset of fifteen subjects, I compare six different methods of per-person adaptation
for continuous heart-rate estimation from zero to approximately five million
trainable parameters per person: an unadapted population model, a per-person affine
correction, unlabelled similarity weighting of population subjects, a warm-started
per-person output layer, parameter-efficient fine-tuning, and full fine-tuning. Each
method is evaluated under one leakage-controlled protocol that divides each target
subject's recording into an adaptation block, a discarded buffer and a later test
block so that no analysis window contributes to both adaptation and evaluation. We
run each method at six personal-data budgets and on both released PaPaGei
checkpoints, whose pretraining objectives encode opposing assumptions about subject
identity.

The study is expected to yield three results. First, an accuracy-versus-cost frontier
identifying which adaptation methods are dominated and where returns diminish.
Second, the personal-data budget at which each method first improves detectably on
normalisation. Third, evidence on whether a representation that already encodes
individual identity leaves more or less headroom for adaptation. Reproduction of
PaPaGei's published heart-rate result anchors the pipeline.

## Problem Statement

The concept of per-person adaptation is not new to the field of machine learning, let
alone across studies of physiological data capture such as with PPG. Each method has
revealed that adapted models beat the population models they were tested against:
domain adaptation using only the individual's unlabelled signal, parameter-efficient
fine-tuning, retrieval, and hypernetworks. However, none of these methods can be
compared with one another as each depends on a different backbone to support the
study; hence a difference in accuracy might be the result of a difference in
pretraining rather than the method of adaptation itself. Some of these studies also
address different downstream tasks, meaning that a method for blood pressure cannot
be assumed to suit heart rate. Moreover, each method defines cost differently. Thus,
the first problem is that a practitioner reading this literature cannot decide
effectively which method of adaptation to deploy, as none of these methods contain a
single axis to define their cost versus accuracy gained.

A second, more specific problem is that personalisation results are almost always
reported as an improvement over an unmodified population model. This is not a
valuable comparison a deployed system actually requires, because commercially
deployed wearables already personalise through normalisation. Therefore, the real
question is not to be found in whether adapting models beats no adaptation, it is
whether adapting these models beats normalisation, a method already running on
millions of wrists. To measure against a raw population model would be to overstate
the accuracy gained by whatever amount normalisation alone already delivers.

Normalisation alone is not an easy baseline to construct, because where the window
used to compute a user's baseline statistics overlaps in time with the data used to
evaluate them, performance is inflated. This is a real constraint when
comparing normalisation to other methods: constructing normalisation with temporal
overlap will flatter the baseline, and every method measured against it would be
unfairly penalised. This is accounted for in this study.

The problem statement this study aims to address is thus as follows: there is
currently a lack in understanding of what per-person adaptation of a PPG foundation
model costs relative to what it delivers, measured against current methods of
personalisation already deployed. Without a common axis, existing results cannot be
compared, and without the right baseline, the comparison cannot be verified. In this
study, I aim to provide both by running every adaptation method on one backbone, one
task, and one dataset so the resulting numbers can be compared in terms of cost
versus accuracy gained.

## Research Questions

Main research question: where does per-person adaptation of a PPG foundation model
stop being worth its cost? For continuous heart-rate estimation on PPG-DaLiA with
PaPaGei as a frozen backbone, what is the relationship between cost per person,
measured in trainable parameters and in stored bytes, and error, measured as mean
absolute error in beats per minute, across methods spanning variable costs in
parameters? I aim to answer this by presenting a frontier to graph the dominant
methods to find the point at which further cost buys no further detectable accuracy.

Sub Question 1: how much of a person's own data does each method need before it pays?
Holding the method fixed and varying the personal-data budget across various minutes
of time, at what budget does each adaptive method first beat the normalisation
baseline detectably, and where does it saturate? I aim to answer this with a
secondary curve of error reduction against personal-data budget.

Sub Question 2: does adaptation help the people the population model serves the
worst? Is the per-subject improvement from adaptation correlated with that subject's
error under the population model? This question is about the distribution of
performance rather than demographics. I understand that fifteen subjects will not
support demographic inference, and hence consider this a study of robustness rather
than fairness. If adaptation compresses the spread of per-subject error, this is an
argument for deploying that is separate from any change in the mean.

Sub Question 3: does the pretraining objective change how much adaptation buys?
PaPaGei releases two checkpoints whose contrastive objectives encode opposite
assumptions about subject identity. Does per-person adaptation buy more on top of a
representation that already encodes identity, or on top of one from which identity
has been removed? Running each experiment arm on both checkpoints will clearly show
how to pretrain the next wearable foundation model.

## Method in Brief

This study design holds everything constant except the thing being measured: one
model, one dataset, one task, one evaluation protocol, and one metric. I vary only
the method of adaptation and the quantity of personal data it is given, so that
differences in error can be attributed to those two factors instead of to differences
in encoder, task or split.

The study runs in four stages. First, I preprocess PPG-DaLiA using PaPaGei's
published pipeline and pass every window through the frozen encoder once, producing a
table of embeddings that is cached to disk. Second, for each of the fifteen subjects
in turn, I divide that table into a population set, an adaptation block, a discarded
buffer and a test block. Third, I fit six adaptation methods on the population set
and the adaptation block, at six personal-data budgets, under both released
checkpoints. Fourth, every method is evaluated on the same test block and placed on a
common cost axis.

| Arm | Description | Trainable parameters per person | Stored bytes per person |
| --- | --- | --- | --- |
| A | Frozen encoder, ridge regression fitted on the population set, applied unchanged | 0 | 0 |
| B | As A, with a per-person affine correction fitted on the target's adaptation block | 2 | 8 |
| B2 | As A, with population subjects weighted by embedding similarity to the target, using no labels | 0 | 0 |
| C1 | Frozen encoder, ridge warm-started from the population solution and updated toward the individual | 513 | ~2 KB |
| C2 | Frozen encoder plus small trainable components, trained on the adaptation block | 10³ to 10⁴ | ~4 to 40 KB |
| D | All encoder parameters retrained on the adaptation block | 4,993,024 | ~20 MB |

The full methodology, covering the splitting protocol, the budgets, the cost axes,
the statistics and the reproduction check, is in [docs/methodology.md](docs/methodology.md).
The bibliography is in [docs/bibliography.md](docs/bibliography.md).

## Reference Numbers

Published mean absolute error for heart-rate estimation on PPG-DaLiA, in BPM. The
protocols differ and the rows are not directly comparable: the classical rows are
leave-one-subject-out, whereas the frozen-probe rows come from PaPaGei's fixed
subject-level split.

| Approach | MAE | Protocol |
| --- | --- | --- |
| Classical (Schaeck 2017) | ~20.5 | Leave-one-subject-out |
| Classical (SpaMa) | ~15.6 | Leave-one-subject-out |
| Statistical features | 13.1 | Leave-one-subject-out |
| PaPaGei-S, frozen probe | 11.5 | Fixed 60/20/20 subject split |
| PaPaGei-P, frozen probe | 10.9 | Fixed 60/20/20 subject split |
| TF-C, frozen | 10.0 | Fixed 60/20/20 subject split |
| Chronos, frozen | 9.7 | Fixed 60/20/20 subject split |
| MOMENT, frozen | 8.8 | Fixed 60/20/20 subject split |
| Supervised, task-specific (Conv-LSTM) | 6.3 | Leave-one-subject-out |

The distance between frozen probing, at roughly 11, and supervised task-specific
training, at 6.3, is the headroom. The study asks how much of it per-person
adaptation recovers, and at what cost.

## Resources

The resources used in this study are all open-source and easily accessible.

Data. I use the PPG-DaLiA dataset, a publicly accessible dataset from the UCI Machine
Learning Repository, which includes fifteen subjects recorded with a wrist-worn
Empatica E4 and a chest RespiBAN ECG device across eight everyday activities plus the
transitions between them, resulting in 64,697 analysis windows. This dataset is freely downloadable and
also records Fitzpatrick skin type, height, weight and self-reported fitness levels
for each subject. Further, this dataset is also the benchmark on which PaPaGei's
published heart-rate result was obtained. I also identified a secondary dataset,
PulseDB, to further check external validity.

Model weights. PaPaGei checkpoints are publicly downloadable from Zenodo and are at
roughly five million parameters each. Their code is released under a BSD 3-Clause
licence; the Zenodo record holding the weights states no licence at all. I use the
model's preprocessing and inference code unmodified.

Computation. The model encoder is frozen in five of the six experimental arms, which
means extracting the 512-dimensional embeddings for all 64k+ windows in PPG-DaLiA
under both checkpoints occurs once and is cached to disk. For experimental arms that
update encoder parameters, the system requires a GPU and runs on minutes of data from
one subject at a time. The experiment harness, which records the configuration and
outcome of every run, is built first.

Software. Python, PyTorch, scikit-learn, NumPy, SciPy, pandas and Matplotlib. The
environment as actually built is described below.

## Status

The work is organised as eight gates, each of which must pass before the next begins.

| Gate | Description | Status |
| --- | --- | --- |
| 1 | PaPaGei loads and returns 512 numbers from random noise | Passed, 22 September 2026 |
| 2 | Real data through the pipeline, verified against known answers | In progress |
| 3 | Embeddings cached for all subjects, both checkpoints | Not started |
| 4 | Reproduction of PaPaGei's published heart-rate result | Not started |
| 5 | Experiment harness | Not started |
| 6 | Arms B, B2 and C1 with confidence intervals | Not started |
| 7 | Arm C2, the convolutional adaptation question | Not started |
| 8 | Arm D, both checkpoints, both figures | Not started |

Graded checkpoints, each falling on a Thursday: proposal on 17 September, midterm
report on 22 October, progress presentation on 19 November, and the final paper,
public repository and presentation on 17 December.

---

## This Repository

```
src/                 Reusable code: data loading, evaluation, model wrappers
external/papagei/    PaPaGei source, as a git submodule pinned to a known commit
weights/             Pretrained checkpoints. Not committed; downloaded from Zenodo.
notebooks/           Exploration and plotting
data/                PPG-DaLiA. Not committed; downloaded separately.
results/             Output tables, saved embeddings, experiment records
figures/             Generated plots, each regenerated by a script
docs/                Written deliverables: methodology, bibliography, paper
log/                 Research log entries in markdown, one file per session
```

This repository is public. Anything genuinely confidential stays out of it,
including the log entries under `log/`, which are readable here whatever the hosted
log does.

### Environment

Runs on Python 3.12, rather than the 3.10 named in the proposal. The constraint came
from two preprocessing dependencies, not from PyTorch: `pyPPG` and `biobss` each pin
their entire 2022 development environments, including versions of numpy, scipy, pip
and setuptools that cannot be installed on current Python. PaPaGei uses one small
module from each, so both are installed without their pins, and their genuine runtime
imports are covered by `requirements.txt`. A clean environment built this way
reproduces Gate 1 exactly.

PaPaGei's source is included as a git submodule pinned to a known commit, so clone
with submodules:

```bash
git clone --recurse-submodules https://github.com/nhemrajani/ppg-personalisation.git
cd ppg-personalisation
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install --no-deps -r requirements-nodeps.txt
```

The pretrained weights are downloaded separately from
[Zenodo](https://zenodo.org/records/13983110) into `weights/`, which is not
committed:

```bash
mkdir -p weights
for m in papagei_s papagei_p; do
  curl -L -o weights/$m.pt "https://zenodo.org/records/13983110/files/$m.pt?download=1"
done
python -m src.papagei
```

The last command loads both encoders, passes random noise through each, and confirms
a 512-dimensional output. It also verifies that the weights load strictly into the
architecture and reports the parameter counts used on the cost axis.

### Getting the Data

PPG-DaLiA is downloaded separately and never committed, being far too large. Unzip it
into `data/`, which leaves the subject files at `data/PPG_FieldStudy/S1/S1.pkl` and so
on for the fifteen subjects.

```bash
python -m src.data
```

That prints the subjects found on disk and the full structure of the first one,
including the shape of every array. The loader handles the one trap in this dataset:
the pickles were written under Python 2 and fail to load without
`encoding='latin1'`.

### Working Rhythm

Anything written twice belongs in `src/` rather than a notebook. The failure mode is
arriving in November with preprocessing existing in six slightly different versions
across four notebooks, and no way to reproduce any of them. Notebooks are for looking
at things; `src/` is for anything reused.

Restart the kernel and run a notebook top to bottom before trusting a result, since
cells run out of order create states that cannot be reproduced.

Commit at the end of each working session, with messages that say what actually
happened. A record of real problems being worked through is a more useful artefact
than a suspiciously clean history.

```bash
git add . && git commit -m "what you did" && git push
```
