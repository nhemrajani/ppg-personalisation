# Pre-registered Decisions

Interpretations and ranges committed before the results that would test them are
seen. The commit history dates each entry, which is the point: a prediction recorded
afterwards is not a prediction.

## The polarity experiment, registered 28 September 2026

PPG-DaLiA's wrist channel is inverted relative to the conventional PPG PaPaGei was
pre-trained on. Embeddings are already cached in both polarities, so the experiment
is to run the reproduction both ways and compare each against the published 11.53 and
10.92, and against the distribution over all 455 possible test triples.

Three outcomes, with what each licenses:

1. **Unflipped lands inside the distribution, flipped does not.** The pipeline is
   correct and the reproduction is anchored. Polarity then becomes a separate claim
   about how well the encoder reads wrist data, reported as a discussion point and a
   named future-work direction.
2. **Flipped lands inside, unflipped does not.** This does not license any conclusion
   about polarity. It would mean PaPaGei flips the signal somewhere we have not
   found, and the response is to go looking again, not to conclude.
3. **Both land inside.** Polarity does not move the number materially. Report it,
   close the thread, move on.

Registered in advance because outcome 2 is confusing if met cold, and the temptation
on meeting it would be to read it as support for the hypothesis when it is the
opposite.

## Hyperparameter ranges, registered 28 September 2026

Committed before the first ridge regression is fitted, and therefore before any
reproduction or Arm A number exists. The claim in the methodology that search ranges
are fixed in advance is only meaningful if the commit that fixes them precedes the
commit that produces the results.

**The grid.** Ridge regularisation strength alpha over `numpy.logspace(-3, 6, 10)`,
that is 1e-3 to 1e6 in ten logarithmic steps. Selection is by mean absolute error in
BPM, the same metric the study reports, computed per subject and averaged so that
selection and evaluation measure the same thing.

**Selection under PaPaGei's protocol, for the reproduction.** Their split is nine
training, three validation and three test subjects. The three validation subjects have
no stated role in their linear evaluation: alpha comes from `GridSearchCV`'s own
four-fold cross-validation within the training set, and nothing in the paper or the
code uses a validation split to select it. They are therefore held out and left
unused here, which matches their stated split rather than quietly training on twelve. Enumerating all 455 test triples
leaves a free choice of which three of the remaining twelve become validation, and
with 455 triples there is enough variation to tune against by accident. The rule,
fixed here: **the three lowest remaining subject identifiers become validation**, the
other nine train. Deterministic, stated in advance, and independent of any result.

**Selection under leave-one-subject-out, for Arm A.** There is no validation set,
since fourteen subjects train and one is held out. Alpha is chosen by grouped
cross-validation within the fourteen, **grouped by subject so that no subject appears
in two folds**, using `GroupKFold` with **seven folds of two subjects each**, which
divides evenly. The held-out subject takes no part in selection.

**Amendment, same day, before any fit.** PaPaGei's own code fixes the settings for the
reproduction, and matching them is the entire point of a reproduction, so a wider grid
would be a different experiment. The reproduction therefore uses their configuration
exactly: `alpha` over `[0.1, 1.0, 10.0, 100.0]`, a `StandardScaler` fitted on the
training subjects and applied to test, `GridSearchCV` with `cv=4` and scoring on
negative mean squared error, with MAE reported. Their cross-validation is not grouped
by subject; the reproduction keeps that rather than correcting it.

The grid and rules registered above continue to govern **our own arms**, A through D,
where there is no obligation to match anyone. Our arms also standardise the embedding
features, with the scaler fitted on the population set alone and applied unchanged to
the target's blocks, for the same reason their pipeline does it.

**Where each experiment runs.** The 455-triple reproduction runs on unflipped signal
only, because that is their protocol and doubling it would blur what the reproduction
means. Polarity is tested at Arm A instead: leave-one-subject-out over all fifteen
subjects, both checkpoints, flipped and unflipped, sixty runs. That gives a paired
comparison across every subject, which is a stronger design for the polarity question
than comparing two distributions, and it is the form Sub Question 3 needs.

**Applies to every arm that fits a ridge**, which is A, B and B2 directly, and C1 as
its warm start. Arm C1's shrinkage strength and Arm C2's rank are separate ranges and
are registered before those arms run, not here.

## Reclassification of the polarity experiment, 28 September 2026

Registered as a side experiment earlier today. It is now a **control for Sub Question
3**, and the reason is recorded here so that it too is dated.

Sub Question 3 attributes the difference between PaPaGei-P and PaPaGei-S to their
treatment of subject identity. The two checkpoints also differ in sign invariance: P
trains with negation at probability 0.20, and S deliberately avoids augmentations that
alter morphology. On a dataset whose signal is inverted, sign invariance is a confound
for the identity comparison and could account for the entire gap. Running both
checkpoints on correctly oriented signal isolates the objective.

It still does not displace the frontier, and it still costs minutes because both
polarities are already embedded. What changes is that Sub Question 3 cannot be
answered without it.

## A limit on what the reproduction can establish, recorded 28 September 2026

Their section 4.1: "We performed five iterations of pre-training and selected the
best-performing model for each downstream task. For SimCLR and PaPaGei-P, a single
model consistently achieves the best performance across all tasks. For BYOL, we select
two models that perform best across all tasks. Similarly, for TF-C and PaPaGei-S, we
choose three models with the highest performance."

So the published figures are the best of five pre-training runs, chosen per downstream
task. For PaPaGei-S that is explicitly not one model across tasks, which means the
11.53 on PPG-DaLiA heart rate may come from a checkpoint other than the single
`papagei_s.pt` released on Zenodo. PaPaGei-P is the cleaner target, since one model is
stated to be best on everything.

**The two explanations separate after all, because P is a control for S.** The
pipeline is shared: same preprocessing, embeddings, standardisation, ridge, split
rule and grid. Nothing in it is S-specific. So PaPaGei-P is the primary reproduction
gate and PaPaGei-S the secondary, and the reading is registered here before either
runs:

| | P reproduces | P misses |
| --- | --- | --- |
| **S reproduces** | Everything is fine. | Not expected. Would point at something P-specific, and there is nothing in our pipeline that qualifies, so it would mean looking again. |
| **S misses** | The pipeline is exonerated by P. Model selection across pre-training runs stands as the explanation for S. | Pipeline bug. The shared machinery is the only thing that could fail for both. |

"Reproduces" means the published figure falls inside our distribution over the 455
test triples; "misses" means it falls outside.

**Polarity is not a candidate explanation for a reproduction miss** in any branch.
PaPaGei obtained 11.53 on the same inverted wrist data we have, so the inversion
affects them and us equally. Polarity is tested at Arm A, not here.

## Arm B's effect on the poorly-served subjects, registered 29 September 2026

Registered before Arm B is implemented or run.

Under the unadapted population model, error is dominated by systematic bias for the
subjects furthest from the population centre: S5 shows a bias of -22.6 BPM within a
24.5 BPM error, 92 per cent of it, and S6 -14.3 of 17.2. Low heart-rate subjects are
over-predicted and high ones under-predicted, which is what ridge shrinkage toward
the training mean produces.

Arm B is a per-person affine correction, a scale and an offset, which is the textbook
fix for exactly that kind of offset. So:

1. **Arm B should help S5 and S6 far more than it helps the other thirteen.**
2. **The improvement should track the bias**, and therefore the distance measure,
   across subjects.
3. **If the improvement does not track bias, the mechanism is not shrinkage** and the
   correlation between distance and error needs another explanation.

This is falsifiable and the result does not yet exist. If it holds, it also means the
distance measure is a triage signal for who needs personalisation at all, and the
label-free version, using the population model's own median prediction on the
subject's unlabelled windows, correlates nearly as well at r = 0.85 across fifteen
subjects, which connects it directly to Arm B2's weighting.

## Limits of the sample on skin tone, recorded 29 September 2026

PPG-DaLiA's fifteen subjects span Fitzpatrick skin types 2 to 4, with eleven of them
at type 3. Types 5 and 6 are absent entirely. The study therefore cannot speak to the
question of whether personalisation helps across skin tone, not because fifteen
subjects is a small sample but because the range does not exist in the data. This is
stated here so that it appears in the limitations as a hard constraint rather than a
hedge, and it is the concrete reason the study measures the distribution of
performance rather than making demographic claims.

## Arm specifications, registered 29 September 2026

Registered before any row exists for Arms B, B2, C1, C2 or D, which the experiment
record confirms at the commit that carries this text.

### Arm B, per-person affine correction

Scale and offset are fitted by **ordinary least squares of the adaptation block's
predictions against the adaptation block's labels**, then applied to the test block's
predictions. Two stored values. The obvious choice, written down rather than assumed.

### Arm B2, unlabelled similarity weighting

The primary version follows GAUL's actual mechanism, which operates on predicted heart
rate in one dimension rather than in embedding space:

1. Fit the population ridge on the population set.
2. Predict heart rate on the target's **unlabelled** windows.
3. Fit a univariate Gaussian to those predictions, giving a mean and a standard
   deviation.
4. Weight each population subject by the density of **its own mean predicted heart
   rate** under that Gaussian.
5. Normalise the weights to sum to one.
6. Refit the ridge with those subject weights.

The bandwidth is the target's own predicted standard deviation, so there is no
temperature parameter to tune. **A single pass is primary**; two and three passes are
run as a sensitivity check.

Registered as an optional secondary experiment: GAUL's **sample-level** weighting,
down-weighting individual population windows whose predicted heart rate is far from
the target's range, entering as row weights in weighted least squares.

A variant weighting by **embedding-distribution similarity** is an extension of ours
rather than GAUL's method, and will be labelled as such rather than attributed to Kim
et al. If it is run, its distance measure and bandwidth are further registered
choices, recorded before it runs.

### Arm C1, warm-started per-person output layer

The per-person solution minimises squared error plus a penalty on departure from the
population solution. The shrinkage strength is swept over
`numpy.logspace(-3, 6, 10)`, the same range as the ridge penalty. At the strong end
the solution equals the population model, which is what makes the arm identical to
Arm A at zero personal data.

### Arm C2, low-rank adaptation

Adapted layer: the **second convolution of the final residual block**, a single
512-channel convolution. A rank-r correction there costs 2,048r parameters.

**Primary sweep: ranks 1, 2 and 4**, costing 2,048, 4,096 and 8,192 parameters, all
inside the 10³ to 10⁴ band the methodology's arms table states. **Ranks 8 and 16 are
run as a labelled extension** at 16,384 and 32,768, outside that band, to show where
the curve goes. The primary sweep and the table agree; the extension is reported as
beyond the stated range.

### Arm D, full fine-tuning

Adam at a learning rate of 1e-3, batch size 32, **a fixed 200 steps with no early
stopping**. The step count is fixed in advance precisely because this is the only arm
that trains a full model and therefore the one most exposed to tuning. No stopping
criterion consults the test block, at any point.

The output layer is initialised from Arm A's ridge solution and trained jointly with
the encoder, so that the arm begins from the population model rather than from noise.

## Predictions, registered 29 September 2026

Falsifiable statements about results that do not yet exist.

**Arm B should help S5 and S6 disproportionately.** Ridge shrinks predictions toward
the training mean, and the error of subjects far from the population centre is
overwhelmingly systematic bias: 92 per cent for S5, 83 per cent for S6, against 2 to
21 per cent for the well-served. A per-person scale and offset is the textbook
correction for exactly that. The prediction is that Arm B's improvement tracks each
subject's distance from the population median heart rate, and that the two outlying
subjects gain most. If the improvement does not track bias, the mechanism is not
shrinkage.

**Arm B2's ceiling may be low.** The population curve shows fourteen subjects are
worth little more than six. B2's entire mechanism is making better use of the same
fourteen, so there may be little headroom to exploit. If B2 beats the baseline anyway,
the gain comes from **which** subjects are used rather than how many, which is the
more interesting of the two outcomes.

## Correctness assertions, registered 29 September 2026

Each arm must reduce to Arm A under the setting where it does nothing. These are
asserted in code and halt the run if violated, in the same spirit as the splitting
invariants.

| Arm | Degenerate setting | Must equal |
| --- | --- | --- |
| B | scale 1, offset 0 | Arm A |
| B2 | uniform weights | Arm A |
| C1 | zero personal data, full shrinkage | Arm A |
| C2 | low-rank correction at its zero initialisation | Arm A |

If any fails, the run stops rather than proceeding.

## Protocol and compute plan, registered 29 September 2026

Arms B, B2 and C1 run under **both** split protocols, activity-stratified and naive
temporal, since they cost minutes.

Arms C2 and D run under the **activity-stratified protocol only**, with a
**single-budget check** under the naive temporal protocol. The methodology presents
the two protocols as a genuine comparison, so this asymmetry is recorded here rather
than discovered later. The reason is compute: C2 and D train, and running both
protocols would roughly double an already overnight job.

## A second budget definition, registered 29 September 2026

Registered after seeing the temporal-prefix budget curve but before running the
alternative, and recorded as such rather than presented as the original plan.

The registered budget is a **prefix in time**: the first n minutes of the adaptation
block. Under the activity-stratified protocol those minutes are whatever the subject
did first, which in PPG-DaLiA is sitting still. For S5 the five-minute budget averages
94.7 BPM against a test block at 125.5. Every adaptive arm is therefore worse than no
adaptation at small budgets, and five minutes is worse than two because it is more
purely sedentary.

That is a real deployment finding and the temporal prefix stays as the primary
definition, because it answers the question a device actually faces: what happens when
calibration data is whatever the user did first.

A **second definition is added as a clearly labelled variant**: sample the budget
proportionally across the activity runs within the adaptation block, preserving
nesting so that smaller budgets remain prefixes within each run. This holds
composition roughly constant and so answers the question Sub Question 1 actually asks,
which is how much data is needed rather than which data happens to arrive first.

**The gap between the two curves is the cost of unrepresentative calibration**, and it
is a better result than either curve alone. Neither replaces the other and both are
reported.

## Amendment to Arm D's learning rate, registered 29 September 2026

Registered before any valid Arm D result exists. The earlier rows were produced by a
defective implementation and were deleted rather than reported.

Arm D was registered at a learning rate of 1e-3, fixed in advance precisely because it
is the arm most exposed to tuning. At that rate it **diverges**: training loss on the
adaptation block rises from under 1 to the order of 10⁶ within a few steps and never
recovers. The registered setting does not produce a result to report; it produces a
broken model.

The rate was therefore reselected **on training-loss convergence on the adaptation
block alone**, across two subjects, at 1e-3, 3e-4, 1e-4, 3e-5 and 1e-5. Only 3e-5 and
1e-5 converge; 1e-5 reaches the lowest final training loss on both. **Arm D runs at
1e-5.** No test-block quantity was consulted at any point in that choice, and the
criterion was whether the arm trains at all rather than how well it scores.

**Arm C2 keeps the registered 1e-3**, which converges on both subjects and gives the
lowest final training loss of the rates tested.

**Implementation clarification for both arms.** The regression target is heart rate in
BPM, so a squared-error loss starts in the hundreds and its gradients scale with it.
Targets are standardised using the **population** training statistics, and the scaling
is folded into the output layer so the arm still begins exactly at Arm A. Predictions
are returned to BPM before scoring. This is a numerical detail rather than a design
choice, but it is recorded because it is the difference between an arm that trains and
one that does not.
