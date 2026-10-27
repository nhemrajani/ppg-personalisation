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

## Defect Notice: Arm C1's Shrinkage Was Selected In Sample, Recorded 26 October 2026

This is a defect notice rather than an amendment. It records that the implementation did
not do what was registered, and that the registered behaviour has been restored. No
decision changes.

Arm C1's shrinkage strength was registered as a sweep over `numpy.logspace(-3, 6, 10)`
selected on a validation slice of the adaptation block. The implementation built the
Gram matrix and the right-hand side from the **whole** adaptation block and then scored
each candidate on the final quarter of that same block. The scoring rows were inside
the fit, so the comparison measured in-sample error, which falls as the penalty weakens.
The loop therefore chose the weakest penalty in the grid for nearly every subject, and
Arm C1 has been running close to unregularised at every budget in all four cells.

This explains a symptom already recorded in CLAUDE.md and previously left unexplained:
that C1's selection lands on the weakest value in the registered range for every
subject. It was read at the time as evidence that the warm start does no work at full
data. The correct reading is that the selection was not functioning.

The test block was never touched, so no reported evaluation was contaminated. The defect
affects which model Arm C1 deployed, not how it was scored, and it affects no other arm.

**Restored behaviour.** The adaptation block is cut at 75 per cent. The model is fitted
on the first part, each candidate penalty is scored on the held-out tail, and the
selected penalty is then refitted on the whole budget. A **four-window buffer is
discarded at the cut**, because windows overlap by six of their eight seconds and
without it the last fitting window and the first scoring window share signal. Every
other block boundary in this study already enforces a buffer for that reason; this one
was missing because the slice was not previously held out at all.

The selected penalty is now recorded on every row. It was not recorded before, which is
why the defect survived a reading of the results.

**Known limitation of the restored version, stated rather than fixed.** Budgets are
temporal prefixes, so the held-out tail is the latest quarter of the adaptation block
and under the activity-stratified protocol it is the latest activity runs rather than a
balanced sample of all of them. Selection therefore happens on a composition that
differs from the test block. A slice taken from the tail of each activity run would
match composition better, and is not done here because it is a design change rather than
a restoration. It is noted as a candidate for the second budget definition.

All previously recorded C1 and C1-scratch rows were deleted rather than reported.

## Amendment: The Trained Arms Optimise The Reported Metric, Registered 26 October 2026

Registered before any rerun of Arms C2 and D. The rows produced under the previous
setting were deleted rather than reported.

**What was registered.** The study reports MAE, and the closed-form arms select on MAE.
No loss function was registered for Arms C2 and D. The implementation used mean squared
error, and the implementation clarification of 29 September refers to a squared-error
loss, so MSE was the assumed default rather than a stated decision.

**What changes.** Arms C2 and D now train on mean absolute error, so that the quantity
optimised is the quantity reported.

**Why.** The per-subject error distribution on this dataset is heavy tailed: S5 sits at
23.8 BPM under Arm A against 7.08 for S7. Squared error weights the worst windows
quadratically, so it can reduce MSE while raising MAE. Both trained arms begin exactly at
Arm A by construction and both moved away from it in the direction of worse MAE, which is
the signature this mismatch would produce. Aligning the loss with the metric is a change
to a previously unregistered setting and is recorded as an amendment rather than a fix.

MSE and MAE on the adaptation block are logged at the start and end of training on every
run, so the direction each moved is visible in the record.

## Amendment: Training Amount Is Fixed In Epochs, Not Steps, Registered 26 October 2026

**What was registered.** A fixed 200 steps at batch size 32, with no early stopping, for
both trained arms. The step count was fixed in advance because these are the arms most
exposed to tuning.

**What changes.** The step count is `max(10, round(10 * n / 32))` where `n` is the number
of windows in the budget, so every budget receives ten passes over its own data.

**Why.** 200 steps at batch 32 is 6,400 sample draws whatever the budget. The
two-minute budget holds about 60 windows, so it received roughly 107 passes; the
all-available budget holds about 3,000, so it received about two. The budget sweep for
C2 and D therefore varied data volume and training amount together, and its
non-monotonicity cannot be attributed to either. The confound was not anticipated by the
registration.

**Why ten.** The registered 200 steps corresponds to 10.7 passes at the 20-minute
budget, the middle of the sweep. Fixing ten passes leaves the middle of the sweep close
to what was registered and changes only the ends, which is the smallest change that
removes the confound. The floor of ten steps exists so that Adam's moment estimates are
not evaluated after three updates; it binds at no budget in this sweep.

The registration's intent is preserved. The step count was fixed so that the arms most
exposed to tuning could not be tuned on results, and scaling it with the size of the
adaptation block keeps that property, because nothing in the rule consults the test
block. The value was chosen before any affected run and is not revised afterwards.

`steps`, `epochs` and the number of adaptation windows are recorded on every row.

## Amendment: Arm C2's Learning Rate Is Selected Per Rank, Registered 26 October 2026

**What was registered.** Arm C2 keeps a single learning rate of 1e-3 for every rank.

**What changes.** The rate is selected per rank from the grid `[3e-3, 1e-3, 3e-4, 1e-4]`,
registered here before selection.

**Why.** One rate covers ranks 1 to 16, which span 2,048 to 32,768 trainable parameters.
Each rank therefore sits at a different point on its own training curve, so the rank
sweep is not a controlled sweep over capacity and the extrapolation from it to higher
ranks is not safe. That extrapolation is currently load bearing: it is the basis for the
claim that no affordable rank reaches Arm B.

**How, and the one deviation from the brief that prompted this.** The rate is selected
once per rank on the **held-out tail of the adaptation block** of two subjects, S1 and
S7, at the all-available budget, by adaptation-block MAE. It is not selected per subject
or per budget. Selecting per row would multiply the compute by the size of the grid, and
the precedent set by the Arm D amendment of 29 September is selection on a small number
of subjects using adaptation-block quantities only. No test-block quantity is consulted
at any point, which is the property that matters.

The selected rate is recorded on every row.

## Predictions For The Rerun, Registered 26 October 2026

Falsifiable, and registered before the affected runs exist.

1. **Arm C1 stops diverging at middling budgets.** Its worst cell currently reaches
   33.22 BPM against an unadapted 11.53. With the penalty actually selected, no cell
   exceeds Arm A by more than 2 BPM at any budget.
2. **Arm C1's selected penalty moves toward the strong end of the grid as the budget
   shrinks**, and is no longer the weakest value for every subject at every budget.
3. **Arm C1 lands within 0.5 BPM of Arm B at all available data** in all four cells.
4. **Arm C1 does not overtake Arm B.** 513 parameters correctly regularised should
   approach two parameters, not beat them, because what both arms correct is the same
   per-person bias.
5. **Arm C2 improves at every rank** once the loss matches the metric, and reaches but
   does not clearly beat Arm B.
6. **Arm D remains at or below Arm A** at every budget.
7. **No arm's position on the cost axis moves**, since no change here touches a
   parameter count.

If prediction 3 holds and prediction 4 fails, the central claim of the study weakens:
the knee in the frontier would sit at 513 parameters rather than two. That outcome is
recorded here as the one that would cost the most, so that it cannot be reinterpreted
later.

## Two Further Closed-Form Arms, Registered 27 October 2026

Registered before either arm is fitted. They exist to test the mechanism claimed for the
crossover found on 26 October, that Arm C1 beats Arm B at small budgets because it is the
only arm in the sweep able to decline to adapt. If that is right the crossover is a
statement about regularisation rather than about capacity, and a cheap arm with shrinkage
should reproduce it.

Arm B's failure mode at cold start is specific. Two minutes of sitting still spans a few
BPM of heart rate, least squares fits a slope across that range, and the test block spans
forty, so a small slope error becomes a large extrapolation error at the extremes. Arm B
is not merely unhelpful there; it is worse than doing nothing.

### Arm B-offset, one parameter

The slope is fixed at 1 and only the offset is fitted:
`offset = mean(y_adapt - predicted_adapt)`, applied to the test predictions.

One parameter, four stored bytes per person. No slope means no extrapolation risk
whatever the calibration window contains; the worst case is an unhelpful constant shift.

This arm should arguably have been in the original design. The proposal's mechanism for
per-person error is ridge shrinkage toward the training mean, which predicts a per-person
**bias**, not a per-person gain. On that argument B-offset is the arm the theory points
at and Arm B is already one parameter more than the theory requires. It is registered now
rather than retrofitted into the original table, and the record shows it was added after
Arm B's result was known.

Degenerate case: an empty budget gives an offset of zero, which is Arm A exactly.

### Arm B-ridge, two parameters shrunk toward the null correction

The same two parameters as Arm B, penalised toward slope 1 and offset 0:

```
minimise  sum (y_i - a p_i - b)^2  +  lambda * n * [ sigma^2 (a - 1)^2 + b^2 ]
```

where `p` are the population model's predictions on the adaptation block, `sigma` is their
standard deviation and `n` the number of windows. Solved as a two by two system,
`theta = (X'X + lambda n P)^-1 (X'y + lambda n P theta0)` with `P = diag(sigma^2, 1)` and
`theta0 = (1, 0)`.

`SHRINKAGE_B = numpy.logspace(-4, 4, 9)`, registered here before use. Selected on a slice
held out of the fit under the same discipline as the Arm C1 fix of 26 October, including
the four-window discard at the cut, and refitted on the whole budget at the selected
value. The selected value is recorded on every row.

**Deviation from the brief that prompted this, with its reason.** The brief gives
`a = (s_xy + lambda) / (s_xx + lambda)` with the offset left unpenalised, and states that
at strong shrinkage this is Arm A exactly. It is not. With the offset profiled out, the
limit is slope 1 with the offset equal to the mean residual, which is **Arm B-offset**,
checked numerically before registering. Penalising both parameters gives the limit the
brief asks for and nests the two existing arms as its endpoints: at the strongest
shrinkage Arm B-ridge is Arm A, and at the weakest it is Arm B. That yields two degenerate
assertions rather than one, which is a better instrument than either version.

The penalty on the offset is scaled by `sigma^2` so that both terms carry units of BPM
squared, the same as the data term, and a single `lambda` is dimensionless. Without that
scaling one `lambda` would constrain an offset of tens of BPM and a slope of order one by
wildly different amounts.

### Predictions, Registered Before Either Arm Runs

1. **B-offset beats Arm B at the two smallest budgets in all four cells.**
2. **B-ridge beats Arm B at the two smallest budgets in all four cells.**
3. **At least one of the two matches Arm C1 at the smallest budget**, their paired
   difference's bootstrap interval containing zero.
4. **At twenty minutes and above B-ridge converges on Arm B**, within 0.2 BPM, because the
   selected shrinkage should weaken as the calibration window becomes representative.
5. **B-offset is worse than Arm B at all available data** in at least three of four cells,
   because by then there is enough data to estimate a slope and a real per-person gain
   term exists.

If prediction 3 holds, the crossover is about regularisation and not capacity: the knee
sits at one or two parameters across the entire budget sweep, and the deployment
recommendation becomes a shrinkage schedule rather than a choice of model size. If it
fails, Arm C1 earns its place at cold start on capacity and the frontier has two genuine
regimes. Both are reportable; the second is the more interesting and the first is better
for the study's headline, which is the reason for registering the predictions first.

## Limitation Recorded, Not Fixed: Arm C2's Rate Selection, 27 October 2026

Arm C2's learning rate is selected on two subjects at the all-available budget. The rates
chosen for ranks 1, 2 and 4 were 3e-3, 1e-3 and 3e-3, which is not monotone in capacity.
With two subjects that is as likely to be selection noise as a real pattern, and a rate
chosen at the all-available budget need not be right at a two-minute one. The compute
argument for selecting this way stands, roughly twenty hours against sixty-five, and no
test-block quantity is consulted at any point. Recorded as a limitation of the trained
arms rather than corrected.

## Record Integrity Note: Commit Provenance On The Stage 7 Rerun, 27 October 2026

The repository conventions state that every row carries the commit hash of the code that
produced it. For the Stage 7 rerun launched on 26 October that is **not true**, and the
rows are being corrected rather than left to be read as accurate.

`commit_hash()` read `git HEAD` once per row. The rerun appends rows for about four
hours, and commits were made during it, so its rows carry three different hashes
(`e669da3`, `ac61595`, `83e54da`), most suffixed `-dirty`, all of them **later** than the
code executing. The running process loaded its modules at launch and was unaffected by
the edits, so the numbers are sound; only the provenance field is wrong.

The code that produced every row of that run is the working tree at **`3111aed`**, the
state at launch. The affected rows' commit field is corrected to `3111aed` once the run
finishes, with a note recording the correction, because the file cannot be rewritten
while a job is appending to it.

**Prevented from recurring:** `commit_hash()` is now cached for the life of the process,
so a hash refers to the run rather than to whenever a row happened to be written.

## Code Identity Of The Stage 7 Rerun, Verified 27 October 2026

The run launched 26 October appends rows for about four hours and the tree moved
underneath it, so its rows carry five different commit tokens. The question is whether
the code producing numbers changed. It did not.

Compared unit by unit between `3111aed`, the tree at launch, and HEAD, over the nine
whole modules on the path and the named units of the three that changed. Reproduce with
`python -m src.provenance stage7 3111aed HEAD`.

**Identical:** `run_one`, `step_count`, `_head_from_ridge`, `_windows`, and every
constant including `BATCH`, `SEED`, `EPOCHS`, `MIN_STEPS`, `LR`, `LR_GRID`,
`TARGET_LAYER` and `RANKS`; `arm_a`, `Fitted`, `_prepare`, `SHRINKAGE`; `Run` and
`append`; and `ridge.py`, `splits.py`, `lora.py`, `preprocess.py`, `papagei.py`,
`embed.py`, `effective_n.py`, `data.py`, `quality.py`.

**Changed:** `select_rates` alone, from `for rank in RANKS` to
`for rank in sorted(ranks) if ranks else RANKS`. The loop body reads no rank other than
its own, so per-rank selection is independent and the rate chosen for a given rank is
unchanged whether the set is all five ranks or a subset. The 101 added lines in
`arms.py` are Arms B-offset and B-ridge and the extended degenerate check, none of which
Stage 7 calls. The change in `harness.py` is the hash caching, which touches the
provenance field only.

**Conclusion: resume is safe and the provenance correction is cosmetic.** No completed
configuration needs rerunning.

### Was The Launch Tree Clean

Yes, for everything that matters, and the evidence is in the record rather than asserted.

`3111aed` was committed at 02:49:36 UTC and the job launched immediately after. Its
first row was written at 02:55:23 UTC, after learning-rate selection, and carries
`e669da3` **without** the `-dirty` suffix. Between launch and that row the only commits
were `73415dc` and `e669da3`, which touch `CLAUDE.md` and `docs/experiment-report.md`
and no source file. So at the moment the first number was produced, `src/` on disk was
clean and identical to `3111aed`.

A second argument closes it independently of the timestamps. Every edit made during the
run was subsequently committed, and the executing path at HEAD is identical to
`3111aed`'s. Had there been an uncommitted edit to that path at launch, HEAD would now
differ from `3111aed` there. It does not. **So the code that produced these rows is
fully recoverable as `git show 3111aed:src/<file>`,** and the `-dirty` tokens from
02:56 onward record edits to files off the path.

The one thing this cannot exclude is an edit made and reverted before any commit. None
was made, but the repository cannot prove that, which is the general reason for the
freeze discipline below rather than a specific doubt about this run.

### The Rule, In Code Rather Than In Judgement

`src/provenance.py` digests the units on each job's number-producing path, scoped per
job. Stage 7 skips a configuration only when it is complete for all fifteen subjects
**and** every row carries an acceptable path identity; otherwise the configuration is
rerun whole. Rows now record `src_digest`.

Acceptance is keyed on the **run window**, not the commit, because a commit token
identifies when a row was written rather than what wrote it. `RUN_WINDOWS` holds the
window for this run with the verification above as its reason, left open until the job
exits and closed to the last row's timestamp afterwards.

### Freeze Discipline For The Remaining Long Jobs

For budget definition 2 and the MOMENT backbone, launch through
`python -m src.freeze <tag> -- <command>`. It refuses to start unless the tree is clean,
ignoring `results/` and `figures/` which are outputs, tags the tree, prints the job's
digest and the window entry to record here, and states that no commit is to be made to
that tree while the job runs. Hash caching fixed the labelling of a run; a frozen tree
and a tag are what make it reproducible.
