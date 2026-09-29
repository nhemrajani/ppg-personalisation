# Running the Study, Start to Finish

Written 28 September 2026. The method is in `methodology.md`; this is the order of
operations, what gets built, and what each step must produce before the next begins.

Four graded checkpoints, each a Thursday: proposal on 17 September (submitted),
midterm report on 22 October, progress presentation on 19 November, and the final
paper, public repository and presentation on 17 December.

---

## Where things stand

| | Status |
| --- | --- |
| Environment, PaPaGei pinned, both checkpoints loading | Done, 22 September |
| Window length settled at 8 seconds against the dataset itself | Done, 28 September |
| Proposal, methodology and bibliography in the repository | Done, 28 September |
| PPG-DaLiA extracted and preprocessed | Not started |

Two things are already known that shorten the work. The encoder runs on the Mac's
GPU, which matters for Arm D. And every subject has between 87.5 and 177.5 minutes of
recording, so the 40-minute budget is feasible for all fifteen.

---

## Stage 1. Preprocessing, and proving it correct

Build `src/preprocess.py`, calling PaPaGei's own functions rather than reimplementing
them: `preprocess_one_ppg_signal` for the filter, biobss for flatline detection, and
`resample_batch_signal` for the rate change. The order is fixed by the paper: filter
at the native 64 Hz, segment into 8-second windows at a 2-second shift, reject
windows more than 25 per cent flat, z-score each survivor, resample to 125 Hz, pad to
1,250 samples.

Four checks, in order of how much they would cost if skipped.

1. A synthetic 1.2 Hz sinusoid at 64 Hz for thirty seconds, filtered and peak
   detected, must return 36 peaks. The answer is known, so any failure is the
   pipeline rather than the data.
2. One real window plotted raw against filtered, with peaks counted by eye and
   compared to the electrocardiogram label for that window.
3. Window counts per subject must equal that subject's label count, which
   `src/verify_windows.py` already established should hold exactly.
4. Rejection rate recorded per subject and written to `results/rejection_rates.csv`.
   This is not diagnostics. It is the input to the usable-minutes analysis, and it
   cannot be recovered later without rerunning everything.

Produces: preprocessed windows on disk, two figures, one rejection table.

## Stage 2. Embeddings, the one expensive step

Build `src/embed.py`. Every window passes once through each frozen checkpoint, on the
GPU, in batches. Each row is stored with its subject, activity label, timestamp and
heart-rate target, so that every later stage is a query rather than a recomputation.

Two tables of roughly 64,697 by 512, about 130 MB each in 32-bit floating point.
Package versions, hardware and runtime are recorded alongside them.

Verify before trusting: row counts match the window counts from Stage 1, no missing
or non-finite values, and a handful of rows re-embedded from scratch reproduce the
cached values exactly.

Nothing downstream can begin until this passes. Everything after it is arithmetic on
a cached table, which is what makes three thousand runs fit in a semester.

## Stage 3. Splitting

Build `src/splits.py`, which implements the protocol in `methodology.md` section 4.4
and nothing else, because every number in the study depends on it being right.

For a target subject it returns the population set (the other fourteen), an
adaptation block, a discarded buffer of at least eight seconds, and a later test
block. Two variants: activity-stratified as primary, naive temporal as the realism
check.

Tested rather than assumed. The tests assert that no window appears in two blocks,
that no adaptation window overlaps a test window in time, that the test block is
strictly later, and that the buffer is never shorter than one window.

## Stage 4. Reproduction

Two separate quantities, which the methodology now keeps apart.

The reproduction proper uses PaPaGei's own protocol: a fixed subject-level split of
roughly nine, three and three, ridge on frozen embeddings, bootstrap intervals from
500 resamples. Their split file is not public, so this reports the distribution of
error across all 455 possible choices of three test subjects and asks whether 11.53
and 10.92 fall inside it.

Arm A is the study's own baseline and rotates through all fifteen subjects. It is a
different number and is not expected to match the published one.

If the published figures fall outside the distribution, the pipeline is debugged
against the authors' intermediate values before anything else proceeds. Padding
length is the first suspect, since the paper does not state it for this dataset.

## Stage 5. The harness

Build `src/harness.py` before the arms multiply. One function takes a configuration
of arm, checkpoint, subject, budget, split type, seed and hyperparameters, runs it,
and appends one row to `results/experiments.csv` recording the configuration, the
metrics, the parameter and byte counts, the usable and elapsed minutes, a timestamp
and the commit hash.

Hyperparameter search ranges are fixed here, written down and committed before any
test result is seen, so that the claim in section 4.4 is checkable.

Every number in the paper traces to one row. Every figure is generated by a script
that reads this file.

## Stage 6. The arms

In the order they should be built, which is also the reverse of the order they would
be cut.

Arm A, the population ridge, is already built by Stage 4.

Arm B is the comparator the study argues with: a per-person affine correction, scale
and offset, fitted on the adaptation block and applied to the predictions. The
statistics must come from the adaptation block alone. Getting this wrong by letting
the correction see test data flatters the baseline and penalises everything else.

Arm B2 weights the fourteen population subjects by how closely their embedding
distributions resemble the target's, using the target's unlabelled signal only, then
fits the ridge on the reweighted population. The distance measure and the weighting
function are the two design choices, and both are fixed in advance.

Arm C1 fits a per-person output layer warm-started from the population solution and
regularised towards it, so that it equals Arm A at zero personal data. The
from-scratch variant runs alongside it, and the gap between them is a result.

Arm C2 is the open question. Start with the fallback, adapting the final projection,
and confirm the loss falls before asking whether it helps. Then attempt low-rank
adaptation of that layer, which costs 1,024 parameters per unit of rank and puts rank
on the cost axis. The mixture-of-experts heads are not a candidate, since they sit
outside the embedding path.

Arm D retrains the whole encoder, 4,993,024 parameters, as an upper bound. It is
expected to overfit on minutes of data and may lose to Arm A, which is itself a
result.

Each arm is run across the six budgets, both checkpoints and both split protocols.

## Stage 7. Analysis and figures

Two figures carry the paper. The frontier plots cost per person against error, one
point per method, showing which methods are dominated and where returns stop. The
sample-efficiency curve plots personal-data budget against error reduction, showing
when each method first beats normalisation.

Three stratified analyses follow. By activity, to see whether gains concentrate in
high-motion conditions, which would suggest adaptation is partly learning a person's
movement rather than their physiology. By subject, to answer Sub Question 2: does
adaptation help those the population model already served worst, and does it compress
the spread of per-subject error. By subject characteristic, reported as illustrative
only, with the power limitation stated.

Comparisons are paired, because every arm evaluates on identical test blocks. Where
intervals overlap, the finding is no detectable difference at this sample size.

## Stage 8. Writing

The literature review is an argument that a specific gap exists, supported by papers,
not a list of them. It draws on the readings already organised by category. The
paragraph stating the gap is written first, and everything before it exists to make
that paragraph inevitable.

The paper is written results first: make the figures, then write the text that
explains them. The methodology section largely exists already.

---

## Calendar

| Week ending | Work | Produces |
| --- | --- | --- |
| 1 October | Stage 1, preprocessing and its four checks | Windows on disk, rejection table, two figures |
| 8 October | Stage 2, embeddings both checkpoints | Two cached tables, environment record |
| 15 October | Stages 3 and 4, splitting and reproduction | Split tests passing, reproduction distribution |
| 22 October | Midterm report | Report, working public repository, reproduction note |
| 29 October | Stage 5, harness, and Arm B | Harness, fixed hyperparameter ranges |
| 5 November | Arms B2 and C1 across budgets | First real comparison with intervals |
| 12 November | Arm C2, fallback first | Rank sweep, cost axis within the arm |
| 19 November | Progress presentation | Slides, both figures in draft, paper outline |
| 26 November | Arm D, stratified analyses. Light week | Complete frontier |
| 3 December | Analysis final, no new experiments after this | Every figure regenerated by script |
| 10 December | Full paper draft, out for comment | Draft to supervisor by Monday |
| 17 December | Final submission | Paper, repository, presentation |

## Weekly rhythm

Before each supervisor meeting, the status document gets achievements, issues and
questions, and plans, with any draft uploaded in advance.

A log entry goes in `log/` for each working session, and a commit at the end of it,
with a message saying what actually happened. The commit history is part of the
deliverable.

## If the schedule slips

Arms are cut in the order D, then C2, then B2, which keeps points at both ends of the
cost axis for as long as possible. Reading is cut before code, since reading recovers
over a weekend and code debt compounds.

Three things are never cut: the splitting protocol, the confidence intervals and the
reproduction check. They are what make this a paper rather than a project.

## Known risks

The reproduction may not land, most likely through the padding length, which the
paper does not state for this dataset. A documented failed reproduction is a
legitimate result, but it costs time.

Parameter-efficient adaptation of a convolutional encoder has no published recipe,
which is why the fallback is named and attempted first.

Fifteen subjects is thin. Sub Question 2 concerns the distribution of performance and
can be answered; demographic claims cannot, and are not made.
