# Methodology

Section 4 of the proposal. Written by Neeharika Hemrajani. Four figures referenced
here are in the proposal document and are not yet reproduced in this repository.

A short list of corrections made to the draft, with evidence, is at the end.

## 4.1 Overview

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

The property that makes this tractable is that the encoder is computed once and
reused. The expensive step happens a single time, and everything after it is
arithmetic on a cached table, which is why a grid of several thousand runs fits
inside a semester.

## 4.2 Data and preprocessing

PPG-DaLiA provides fifteen subjects wearing an Empatica E4 on the wrist, sampling PPG
at 64 Hz and acceleration at 32 Hz, with a chest-worn RespiBAN electrocardiogram
supplying heart-rate ground truth. Each subject performs eight activities in a fixed
order, including sitting, walking, cycling, driving, working and a lunch break, which
is what makes the dataset appropriate for this study: the error rates it produces
reflect ordinary wear rather than laboratory rest.

The files are pickled Python 2 objects and must be loaded with `encoding='latin1'`.
Loaded without it they raise decoding errors that resemble file corruption.

Preprocessing follows PaPaGei's published pipeline without modification. The signal
is bandpass filtered with a fourth-order Chebyshev filter between 0.5 and 12 Hz,
applied forwards and backwards so that no phase distortion is introduced. It is
segmented into 8-second windows at a 2-second shift, so consecutive windows overlap
by 6 seconds. Windows containing more than twenty-five per cent flatline are
discarded as unusable, and each surviving window is z-scored. The window is then
resampled to 125 Hz, giving 1,000 samples, and padded to the 1,250-sample length on
which the encoder was pretrained. This yields 64,697 analysis windows across the
fifteen subjects, which at a 2-second shift corresponds to approximately thirty-six
hours of recording.

I do not reimplement any of this. Rebuilding the preprocessing would make the
reproduction check in section 4.7 impossible to interpret, because a discrepancy
could then be attributed either to my pipeline or to my analysis, with no way to tell
which.

One quantity produced by preprocessing is recorded rather than discarded: the
proportion of windows rejected, per subject. Rejection rates vary between subjects,
and that variation feeds directly into the budget analysis in section 4.6, because a
subject whose signal is noisier supplies fewer usable minutes per minute worn.

## 4.3 Model and embeddings

The base model is PaPaGei's encoder: a one-dimensional convolutional residual network
of eighteen blocks, approximately five million parameters, producing a
512-dimensional embedding for each window. Both released checkpoints are used
throughout, PaPaGei-S and PaPaGei-P, for the reason set out in Sub Question 3.

The two checkpoints are released as different classes. PaPaGei-S is a `ResNet1DMoE`
carrying mixture-of-experts heads and PaPaGei-P a plain `ResNet1D` without them. The
embedding paths are nonetheless identical tensor for tensor, since the
mixture-of-experts heads branch off after global pooling and served as auxiliary
pretraining targets rather than feeding the embedding. Sub Question 3 therefore
compares two pretraining objectives on one architecture, which is what it requires.

PaPaGei was selected over the other open PPG foundation models for three reasons. It
is the only one to release two checkpoints pretrained with deliberately opposing
treatments of subject identity, which is what makes Sub Question 3 possible. At
approximately five million parameters it sits in the range where full per-person
fine-tuning is computationally feasible, so the expensive end of the cost axis can be
anchored rather than assumed; the authors' own finding that their 35 and 139 million
parameter variants were outperformed by the smallest model on nineteen of twenty
tasks indicates that little accuracy is lost by this choice. And it publishes a
heart-rate result on PPG-DaLiA specifically, which supplies the reproduction anchor
described in section 4.7 and without which no downstream comparison could be
verified.

The alternative with the strongest claim is Pulse-PPG, pretrained on field-collected
rather than clinic-collected signal; it releases a single checkpoint and so cannot
support Sub Question 3, but it is the natural third backbone should the primary
experiments conclude early.

Every window is passed through each frozen checkpoint once, and the resulting
embedding is written to disk alongside its subject identifier, activity label,
timestamp and heart-rate target. The result is two tables of roughly 64,697 rows by
512 columns. Package versions, hardware and runtime are recorded with them so that
the cache can be regenerated and verified later. This step is the only part of the
study needing meaningful computation, and it is the gateway: everything downstream
depends on it and nothing downstream can begin until it is complete and checked.

Before extracting anything at scale I verify the pipeline on data where the answer is
known. A synthetic 1.2 Hz sinusoid sampled at 64 Hz for thirty seconds is passed
through the filter and peak detector, which should return thirty-six peaks. A single
real window is then plotted before and after filtering, its peaks counted by eye and
compared against the electrocardiogram label. These checks are trivial, and they
catch the class of error that otherwise survives undetected into the results.

## 4.4 The splitting protocol

This is the part of the design most likely to be wrong, and an error here invalidates
every number in the study, so I set it out in full.

Two requirements pull against each other. The population model must never have seen
the target subject, which is the standard leave-one-subject-out condition. But the
target subject must also supply both the data used to adapt and the data used to
test, and those must not overlap. Because consecutive windows share six of their
eight seconds, random assignment within a subject would place near-identical windows
on both sides of the split, and the reported benefit of adaptation would be an
artefact of that overlap rather than a real effect.

For each target subject in turn, the fourteen remaining subjects form the population
set, which trains the downstream model for the non-adaptive methods. The target's own
recording is then divided in temporal order into an adaptation block, a buffer and a
test block. The buffer is at least one window long, which for 8-second windows means
at least eight seconds, and it is discarded entirely. The test block is later in time
than the adaptation block, and every method is evaluated on it.

The buffer is not a formality. Consider a single cut at minute sixty with no gap. A
window beginning at 59:54 and ending at 60:02 belongs to the adaptation block by its
start time, yet two of its eight seconds are drawn from the test period. Discarding a
gap at least as wide as one window makes it impossible for any window to hold samples
on both sides.

The same buffer also absorbs the reach of the bandpass filter. Because the filter is
applied forwards and backwards over the continuous recording before segmentation,
each filtered sample depends on raw signal on both sides of it. At 64 Hz, 99.9 per
cent of the filter's response lies within 1.4 seconds of the sample and it falls
below 0.1 per cent of its peak beyond 3.8 seconds, comfortably inside the buffer.

The ordering is temporal rather than random for two reasons. The first is leakage, as
above. The second is realism: adapting on earlier data and testing on later data is
what deployment actually looks like, since a device cannot calibrate on a user's
future.

A naive temporal split introduces a confound of its own. Because PPG-DaLiA runs its
activities in a fixed order, cutting the recording at a single point means the
adaptation block and the test block contain different activities, and any apparent
failure of adaptation might be a failure to transfer across activity rather than
across time. The primary protocol therefore takes an initial portion of each activity
for adaptation and the remainder of that activity for test, preserving temporal
ordering within each activity while balancing activity across both blocks. The
secondary protocol is the naive single-cut temporal split, reported as a realism
check. Where the two disagree, the disagreement is reported, because the gap between
them measures how much of adaptation's benefit is activity-specific.

Hyperparameters are selected on the population set, or on a validation slice held out
from within the adaptation block. They are never selected on the test block. Search
ranges are fixed in advance of seeing any test result and recorded in the repository
with a timestamp, so that the claim is checkable rather than merely asserted.

One consequence should be stated plainly: a normalisation baseline constructed with
overlap is flattered, and every adaptive method measured against it is
correspondingly penalised.

## 4.5 The compared methods

Six methods are evaluated, labelled Arms A to D. Arm A is the unadapted population
model and Arm B is the normalisation baseline against which the others are judged;
the remaining four are the adaptive methods.

| Arm | Description | Trainable parameters per person | Stored bytes per person |
| --- | --- | --- | --- |
| A | Frozen encoder, ridge regression fitted on the population set, applied unchanged | 0 | 0 |
| B | As A, with a per-person affine correction fitted on the target's adaptation block | 2 | 8 |
| B2 | As A, with population subjects weighted by embedding similarity to the target, using no labels | 0 | 0 |
| C1 | Frozen encoder, ridge warm-started from the population solution and updated toward the individual | 513 | ~2 KB |
| C2 | Frozen encoder plus small trainable components, trained on the adaptation block | 10³ to 10⁴ | ~4 to 40 KB |
| D | All encoder parameters retrained on the adaptation block | 4,993,024 | ~20 MB |

Stored bytes assume 32-bit floating point, at four bytes per parameter. The
translation matters because storage, not computation, is what decides whether a
method can be deployed across several million members.

Arm A is the reproduction anchor. A ridge regression is fitted from the
512-dimensional embeddings of the population set to heart rate in beats per minute,
then applied unchanged to the target's test block. It carries no per-person cost and
sits at the origin of the frontier.

Arm B is the comparator this study is really arguing with. It is Arm A with a
per-person affine correction, a scale and an offset, fitted on the target's
adaptation block and applied to the predictions. Two stored values per person makes
it the direct analogue of the rolling-baseline normalisation described in the
Introduction. The statistics come exclusively from the adaptation block, never the
test block, and the buffer guarantees they are temporally disjoint.

Arm B2 adapts using no labelled data from the target at all. The fourteen population
subjects are weighted according to how closely their embedding distributions resemble
the target's, which requires only the target's unlabelled signal, and the ridge is
then fitted on the reweighted population. The principle is adapted from GAUL and
reimplemented within this framework rather than run as published, so that the
backbone stays fixed and the cost comparison holds. Its stored cost per person is
zero. If it approaches the labelled arms, that is the most deployment-relevant result
this study can produce, because it removes the cold-start problem entirely: no user
has to supply ground truth, and nothing has to be stored.

Arm C1 fits a per-person output layer, 512 weights and one bias, on the target's
adaptation block. The primary version is warm-started from the population solution
and regularised towards it, which makes it identical to Arm A at zero personal data
and lets it depart from Arm A as data accumulates. This handles cold start by
construction. A from-scratch version is also run, and the gap between the two
quantifies what warm-starting buys. Fitting 513 coefficients from scratch on a few
minutes of data may well underperform Arm A, which is a finding rather than a
failure.

Arm C2 is the open implementation question of this study. Parameter-efficient
fine-tuning methods were formulated for transformers, and PaPaGei's encoder is
convolutional, so the standard recipes do not transfer unchanged. Four options are
available, in ascending order of ambition: adapting only the final projection layers;
adapting only the 1×1 convolutions, which are matrix multiplications and to which
low-rank adaptation applies without modification; low-rank decomposition of reshaped
convolutional kernels; and inserting adapter blocks between convolutional blocks. The
first is a named fallback that always works, so the arm cannot fail outright.
Whichever variant is used, the rank or bottleneck width is swept, and that sweep
produces a cost axis within the arm rather than a single point on it.

Note on the fallback's size: the final projection is a 512 by 512 layer of 262,656
parameters, which is larger than the 10³ to 10⁴ range this arm is designed to
occupy. Low-rank adaptation of that layer costs 1,024 parameters per unit of rank, so
ranks of one to ten fall inside the range and make rank the cost axis.

Arm D retrains the whole encoder on the adaptation block. It is an upper bound rather
than a deployment candidate. It is expected to overfit on minutes of data and may
underperform Arm A, which would itself be a result: adaptation capacity is not
monotonically beneficial, and demonstrating that on a real task is worth the runs it
costs.

## 4.6 Personal-data budgets and the cost axes

Arms B, B2, C1, C2 and D are run at personal-data budgets of two, five, ten, twenty
and forty minutes, and at all available data. Budgets are taken from the start of the
adaptation block, so that each smaller budget is a strict prefix of each larger one
and the resulting curve is monotone in data rather than confounded by which portion
of the recording was sampled.

Budgets are reported in usable minutes after rejection, not in elapsed wear time. A
subject whose recording is noisy loses more windows at the signal-quality stage, so
forty minutes worn is not forty minutes usable, and the discrepancy varies by
subject. Both quantities are reported and the gap between them is treated as a
result, because a product team sizing a calibration period needs the wear-time figure
while a modeller needs the usable-minutes one.

Cost is measured on two axes, stated explicitly because the Problem Statement argues
that leaving cost implicit is what makes the existing literature incomparable. The
primary axis is trainable parameters per person, which is the quantity the machine
learning literature uses and the one on which the six arms are designed to separate.
The secondary axis is stored bytes per person, which decides whether a method is
deployable at a population of several million.

A third axis, serving cost, is not measured. Per-person parameters break inference
batching, and by how much depends on a serving architecture this study does not
assume. Every figure reported here is therefore a lower bound on true production
cost, and the frontier is presented as one.

## 4.7 Metrics, statistics and the reproduction check

The primary metric is mean absolute error in beats per minute, computed per subject
and then averaged across subjects, with the full distribution reported rather than
the mean alone. Reporting the distribution is not decoration: Sub Question 2 asks
whether adaptation helps the worst-served individuals, and that question is
unanswerable from a mean.

Confidence intervals are obtained by bootstrap resampling with 500 resamples. All
arms are evaluated on identical test blocks, which permits paired comparison.
Comparisons are therefore made on per-subject differences rather than on differences
of means, which is both the more powerful test and the correct one given the design.
Where intervals overlap, the finding is reported as no detectable difference at this
sample size. With fifteen subjects this will happen, and saying so is more useful
than declaring a winner the data cannot support.

The reproduction check gates everything else. PaPaGei reports a mean absolute error
of 11.53 for PaPaGei-S and 10.92 for PaPaGei-P on PPG-DaLiA heart rate under linear
probing.

Those figures come from a single fixed subject-level split, not from a
leave-one-subject-out rotation. The paper divides datasets unseen during pretraining
into training, validation and test sets in a 60/20/20 ratio at the subject level,
which for fifteen subjects is roughly nine, three and three. The published error is
therefore an average over about three held-out people. The reproduction is run under
that protocol, and it is a separate quantity from Arm A, which rotates through all
fifteen subjects and is not expected to match it.

The authors' split files are not published, so which three subjects were held out is
unknown, and with three test subjects the figure depends heavily on which. The
reproduction therefore reports the distribution of error across all 455 possible
choices of three test subjects and asks whether the published figure falls inside it.

If the reproduction cannot be reconciled, the discrepancy is documented carefully and
reported, because a carefully documented failed reproduction of an open model is a
legitimate contribution in its own right and considerably more useful to the field
than a quiet substitution of my own numbers.

## 4.8 Experimental harness, risks and fallbacks

Six arms across six budgets, fifteen subjects, two checkpoints and two split
protocols, together with a rank sweep within Arm C2 and a from-scratch variant of Arm
C1, produce on the order of three thousand runs. Each is computationally trivial and
collectively they are impossible to track by hand, so the harness is built before the
arms multiply rather than after.

It exposes a single function taking a configuration of arm, checkpoint, subject,
budget, split type, seed and hyperparameters; it runs that configuration and appends a
row to a results file recording the configuration, the metrics, the parameter and
byte counts, the usable and elapsed minutes, a timestamp and the commit hash of the
code that produced it. Every number appearing in the paper is traceable to one row,
and every figure is generated by a script that reads that file.

---

## Corrections made to the draft

Four changes, each verified against PaPaGei's paper, code or released weights on
22 and 28 September 2026. They should be carried back into the proposal document.

1. **Window length, in 4.2 and 4.4.** The draft gave 10-second windows, an 8-second
   overlap and a buffer of at least 10 seconds. PaPaGei's dataset appendix specifies
   8-second windows with a 2-second shift and 6-second overlap for PPG-DaLiA
   (arXiv 2410.20542v2). The 10-second figure belongs to their general pretraining
   pipeline and survives here only as the 1,250-sample padded input length. The
   leakage example in 4.4 was adjusted accordingly, and the buffer is at least eight
   seconds.

   This was checked against the dataset itself rather than the paper alone, since the
   paper gives both figures. The 10-second statement sits in section 4.1 under
   pre-training, whose table covers only the three pre-training corpora, and the
   arithmetic there (20,751,206 segments over 57,641 hours, or 9.9998 seconds each)
   confirms it describes non-overlapping pre-training windows. PPG-DaLiA is an
   evaluation dataset and its appendix specifies the 8-second window. PPG-DaLiA
   supplies one heart-rate label per analysis window, and those labels total 64,697
   across the fifteen subjects, exactly the sample count PaPaGei reports, matching the
   8-second figure for every subject individually. Ten-second windows would give
   64,682. The check is reproducible with `python -m src.verify_windows`.

2. **Encoder class, in 4.3.** The draft described the encoder as `ResNet1DMoE` for
   both checkpoints. Only PaPaGei-S is; PaPaGei-P is a plain `ResNet1D`. Checking the
   released weights shows the embedding paths are identical tensor for tensor, so the
   comparison in Sub Question 3 is unaffected, but the claim as drafted was wrong.

3. **Mixture-of-experts probing, in 4.5.** The draft suggested the mixture-of-experts
   routing component might be cheaper to adapt than the convolutional trunk. It sits
   outside the embedding path, so adapting it cannot change the embedding the ridge
   head reads, and it exists only in PaPaGei-S. The suggestion was removed.

4. **The reproduction anchor, in 4.7.** The draft stated that Arm A must land near
   11.53. That figure comes from a fixed 60/20/20 subject-level split rather than the
   leave-one-subject-out rotation Arm A uses, so the two are different quantities.
   The section now separates them and proposes reporting the distribution over all
   455 possible test triples, since the authors' split is not public.

Also recorded: Arm D's cost is 4,993,024 trainable parameters, measured from the
released weights rather than rounded to five million, and identical on both
checkpoints.
