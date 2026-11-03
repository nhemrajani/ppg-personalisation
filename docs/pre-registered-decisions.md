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

## Defect Notice: Arm D's Learning Rate Was Selected Under A Loss That No Longer Applies, Recorded 27 October 2026

The amendment of 29 September selected Arm D's learning rate of 1e-5 by training-loss
convergence on the adaptation block, across two subjects, at a time when the trained
arms optimised **squared error**. The amendment of 26 October changed the loss to
**absolute error**, so that the trained arms optimise the metric the study reports. It
reselected Arm C2's rate per rank under the new loss. It did not reselect Arm D's.
`LR["D"] = 1e-5` stayed hardcoded in `src/stage7.py` while `select_rates` ran only for
the ranked arm. That is an oversight in applying the amendment, not a design decision.

**The consequence is that Arm D does not train.** On the rerun's 360 rows the training
MAE on the adaptation block **rises** rather than falls, by a median of 13 to 55 BPM
depending on budget, and more than doubles on 218 of 360 runs. Loss traces oscillate
upward, for example 1.40, 11.42, 18.04, 11.32 at the smallest budget. Test MAE reaches
174.7 BPM against an unadapted 11.72. These are the numbers of a diverged optimisation,
not of an overfitted model.

**Why the loss change breaks a rate that previously worked.** Targets are standardised,
so residuals start near 1. A squared-error gradient is proportional to the residual and
therefore shrinks as the fit improves, which anneals the effective step size without
anyone choosing to. An absolute-error gradient has constant magnitude whatever the
residual, so it does not anneal. A rate validated under the first loss carries no
guarantee under the second, and for 4,993,024 parameters it fails.

**The instrumentation registered in the 26 October amendment is what caught this.**
Logging training-set MAE before and after adaptation, and a four-point loss trace, was
added so that a trained arm's number could not be reported without evidence that the
arm had trained. It did that job. Without it the 62.48 BPM figure would have been
reported as evidence that full fine-tuning overfits, which it is not.

### Amendment: Arm D's Rate Is Reselected Under The Absolute-Error Loss

Registered before the affected runs exist. The procedure is **the one already
registered on 29 September, applied to the loss now in force**: the rate is chosen by
training-loss convergence on the adaptation block alone, on subjects 1 and 7, at
1e-5, 3e-6, 1e-6, 3e-7 and 1e-7, extending the grid downward because 1e-5 is now its
divergent end. The criterion is whether the arm trains, not how well it scores. No
test-block quantity is consulted at any point. This is the same code path
`select_rates` already uses for Arm C2, extended to cover the unranked arm.

**The 360 Arm D rows produced at 1e-5 under absolute-error loss are deleted rather
than reported**, as the September amendment's rows were, and for the same reason: they
record a broken optimisation rather than a result. The deletion is its own commit.

**What this does and does not rescue.** If Arm D converges at a lower rate, the study
regains its upper-bound arm and prediction 6 can be scored on a trained model. If it
does not converge at any rate in the extended grid, the honest report is that full
fine-tuning of this encoder on minutes of per-person data could not be made to train
under a protocol fixed in advance, which is a weaker claim than overfitting and must be
written as such. Either way the frontier's knee is unaffected, because Arm D sits at
the far end of the cost axis and no arm's parameter count changes.

**A standing lesson, recorded because it will recur.** An amendment that changes the
objective invalidates every hyperparameter previously selected against the old one. The
26 October entry should have said so and reselected both rates. Any future change to
the loss, the target scaling or the step budget reselects every rate in the same commit.

## Outcome Of Arm D's Rate Reselection, Recorded 27 October 2026

Run before any Arm D row exists, on subjects 1 and 7, scored on held-out adaptation
data only. Mean MAE on the held-out tail, with `t` marking a rate that reduced training
error on both subjects:

| Rate | 1e-5 | 3e-6 | 1e-6 | 3e-7 | 1e-7 |
| --- | --- | --- | --- | --- | --- |
| PaPaGei-P | 35.63 | 41.04 | 19.52 | 8.78 | **6.91 t** |
| PaPaGei-S | 24.82 | 12.77 | 8.21 | 8.21 | 9.31 |

**On PaPaGei-P only the smallest rate in the grid trains, and the sequence is monotone
toward it.** On PaPaGei-S no rate reduced training error on both subjects; the guard
reported that and the arm falls back to the best-scoring rate, 3e-7.

**The monotonicity is the finding, and extending the grid downward would not change it.**
As the rate goes to zero Arm D goes to Arm A, because the correction initialises at zero
and the head is in the same optimiser at the same rate. A selection criterion that
rewards stability therefore drives Arm D toward doing nothing, and on PaPaGei-P it goes
all the way to the grid's edge. Chasing the edge with a wider grid would confirm the
limit rather than find an interior optimum, so the grid is **not** extended. That
decision is recorded here before the runs so it cannot be read later as having been made
to protect a result.

**What Arm D can and cannot be said to show.** Under the protocol registered in advance,
which is absolute-error loss, ten epochs, Adam, no gradient clipping and no learning-rate
schedule, Arm D is unstable at every rate that moves 4,993,024 parameters appreciably and
stable only where it barely moves them. So Arm D will contribute a frontier point at
roughly Arm A's error and 20 MB of storage per person. **That point is honest as a
statement about this protocol and must not be written as a statement about full
fine-tuning.** The study cannot distinguish "full fine-tuning does not help here" from
"full fine-tuning could not be optimised here", and the limitation is to be stated in
those words.

**Why the protocol is not changed to rescue it.** Gradient clipping or a schedule would
very likely stabilise Arm D. Applying either only to Arm D would make the arms
incomparable, and applying it to all trained arms would invalidate the 1,800 Arm C2 rows
that have just been produced under the registered settings. Neither is worth doing
on the strength of a result already seen. If a stabilised variant is wanted it is
registered as a **separate labelled arm** with its own entry, run alongside rather than
in place of Arm D, so that the comparison it enters is stated rather than silent.

## Arm D's Result, Recorded 27 October 2026

360 rows, 24 configurations, 69 minutes, launched through `src/freeze.py` on the
annotated tag `armd-2026-10-27` at commit `a7fac63` on a clean tree. Every row carries
that commit with no `-dirty` suffix and the executing-path digest `597c1b602592`. The run
window is recorded in `src/provenance.py` on exit rather than at launch, because
`provenance.py` is itself on stage7's path and committing to it would have broken the
freeze.

**The prediction I made in the entry above was wrong, and the record should say so
plainly.** I wrote that a stability criterion would drive Arm D toward doing nothing and
that it would contribute a frontier point at roughly Arm A's error. It does not. At the
reselected rate Arm D trains: adaptation-block MAE falls on 7 to 11 of 15 subjects at the
larger budgets, against 1 of 15 under the mismatched rate, and its budget curve is
monotone downward after five minutes in all four cells. The reasoning that led me wrong
was sound as far as it went, that the arm tends to Arm A as the rate tends to zero, but
1e-7 over several hundred steps is not close enough to zero for that limit to bite.

**The result.** Arm D beats Arm A at all available data by 0.63 to 1.11 BPM, detectably
in none of the four cells, and is detectably **worse** than Arm A at the two smallest
budgets in all four cells, by 3.15 to 11.28 BPM. Arm B beats Arm D at all available data
in all four cells, **detectably in every one**, by 0.69 to 1.86 BPM. So the far end of the
cost axis is occupied by an arm that works, costs 20 MB per person, and delivers about
half of what eight bytes delivers.

**The limitation stated earlier stands in weakened form.** The study still cannot claim
that full fine-tuning has no more to give on this task, because the protocol was fixed in
advance and a schedule or gradient clipping would probably improve it. What it can now
claim is stronger than before: Arm D trained, under a protocol registered before the
result was seen, and lost to two parameters in every cell. That is a comparison rather
than a non-result.

**One cell to treat with suspicion.** PaPaGei-S runs at 3e-7 selected by the fallback
after the guard reported that no rate reduced training error on both selection subjects,
and s/temporal is the single cell where Arm D never beats Arm A at any budget. These are
the least reliable numbers in the study and are to be labelled as such wherever quoted.

## Correction Notice: The Correlation Comparison For Arm B Cannot Measure What It Was Read As Measuring, Recorded 3 November 2026

Two observations prompted the floor baselines registered below. One holds and one does
not, and the one that does not should be withdrawn before it is built on.

**Withdrawn.** It was read from the record that Arm B leaves the prediction-truth
correlation essentially unchanged, 0.617 against Arm A's 0.626, and that this is not what
a genuine accuracy improvement looks like. The premise is right and the inference is not
available. Pearson's r is invariant under a strictly increasing affine map of either
variable, and Arm B's prediction is exactly such a map of Arm A's: all 360 fitted scales
in the record are positive, the smallest being 0.018, so no row flips sign. Arm B
therefore **cannot** change the correlation, whatever it does to the error. Checked
rather than argued: paired by subject and cell across all 1,080 rows of Arms B, B-offset
and B-ridge, the largest difference from Arm A's correlation is 0.000000.

The quoted gap of 0.617 against 0.626 is an artefact of averaging unpaired row sets. Arm
A has 120 rows, one per subject per cell, because it has no budget dimension; Arm B has
360, six budgets per cell. The Arm B mean therefore includes the small-budget cells where
the fitted scale falls to 0.018, and the Arm A mean does not. Comparing a 360-row mean to
a 120-row mean violates the study's own paired-comparison convention.

**This strengthens the case for a floor baseline rather than weakening it.** Correlation
is structurally blind to the entire class of corrections Arms B, B-offset and B-ridge
belong to, so it cannot be used to ask whether those arms are improving the estimate or
substituting a constant. The only instrument that can answer that question is a baseline
that substitutes a constant on purpose, which is what Arm Z1 is.

**Upheld.** The fitted scale averages 0.777 across the four cells at all available data
and runs as low as 0.207, on S5, where the offset reaches +104.23 BPM. At that scale the
correction retains about a fifth of the model's variation and supplies the rest as a
constant. That is a real observation about the mechanism and it is what Arm Z1 is
registered to measure. The earlier figure of 0.78 for the mean scale is confirmed at all
available data; over all budgets and cells the mean is 0.692 and the minimum 0.018.

## Two Floor Baselines, Registered 3 November 2026

Registered and committed before either arm is fitted. Both are closed form and consult
no embedding and no model output whatsoever.

The study has measured six adaptation methods against the frozen population model and
against each other. It has never measured any of them against not having a model. Without
that floor, the quantity every arm's improvement is expressed in is unanchored: a
reviewer can ask what the foundation model contributes over knowing a person's own
average heart rate, and nothing in the record answers it.

### Arm Z0, global constant

Predict the mean heart rate of the population set, the fourteen subjects other than the
target, for every test window. Zero parameters per person, zero bytes, no labels from the
target. The floor for knowing nothing at all.

Z0 does not depend on the target's budget, so it is recorded **once per cell** with the
budget field blank, exactly as Arm A is, rather than as six identical rows. This is a
stated deviation from running it across all six budgets; it writes the same information
without putting duplicate rows in the record, and the budget figures broadcast from the
single row in analysis.

### Arm Z1, per-person constant

Predict the mean heart rate of the target's own adaptation budget for every test window.
One parameter per person, four bytes, labels required. Recorded across all six budgets,
so Z1 at two minutes sees exactly the two minutes Arm B-offset sees. The model is never
consulted.

**This is the load-bearing arm.** It costs exactly what Arm B-offset costs, one parameter
and four bytes, and it uses no model output at all. The difference between Arm Z1 and the
model-based arms is therefore the honest measure of what the foundation model contributes
on this task, with the per-person mean held constant between them.

### Correctness assertions

Neither arm reduces to Arm A, so the degenerate check that every other arm carries does
not apply. Two different assertions replace it, checked in code rather than assumed:

- Z0's and Z1's predictions are constant across the test block, to floating point.
- Z1's constant equals the mean label over the budget rows, and Z0's equals the mean
  label over the population rows.

### Scale and offset promoted to columns

The fitted scale and offset of Arms B and B-ridge move from the hyperparameters JSON
string to real columns, `fitted_scale` and `fitted_offset`, and the existing rows are
backfilled from their own hyperparameters strings rather than refitted. This is a
change of storage only: no number changes, and the backfill is asserted to agree with
the string it was read from.

### Predictions, registered before either arm runs

1. **Arm Z1 lands between Arm A and Arm B, closer to Arm A.** On PaPaGei-P,
   activity-stratified, at all available data, Arm A is 11.72 and Arm B is 9.65, so this
   predicts Z1 in roughly 10.5 to 11.7 with the upper half of that range more likely.
2. **If Arm Z1 beats Arm A, the frozen probe contributes nothing on this task**, and that
   is the most important sentence in the paper. Registered as the outcome that would
   force the headline to move, before it is known whether it happens.
3. **Arm B still beats Arm Z1**, because the model contributes something beyond the
   person's own mean. If this fails, every arm in the study is a reparameterisation of
   the per-person mean and the frontier is measuring the wrong thing.
4. **Arm Z0 is worse than Arm A** in all four cells. A population-mean constant should
   lose to a population model; if it does not, the probe is not using the embedding.
5. **Arm Z1 beats Arm Z0** at all available data in all four cells, since the per-person
   mean is the single largest component of per-subject error, which the bias
   decomposition of 29 September already established.
6. **Arm Z1 is worse than Arm A at the two smallest budgets** in most cells, for the same
   compositional reason every adaptive arm is: the first minutes of a recording are
   sitting still, so the per-person mean estimated there is not the test block's mean.

Prediction 3 is the one whose failure would be most expensive, and prediction 2 is the one
whose success would most change the paper. Both are recorded here with the arms unfitted.

## Outcome Of The Floor Baselines, Recorded 3 November 2026

420 rows, `python -m src.stage10`, 0.1 seconds, commit `aad675c` on a clean tree. The
registered assertions pass for all fifteen subjects in both protocols: both arms are
constant across the test block, each equals the mean of the rows it is entitled to see,
each is priced at the registered cost, and Arm Z1 with an empty budget reduces to Arm Z0
exactly. Both floors are also identical across the two backbones in all 210 cells, as
they must be, since neither reads an embedding.

**Five of the six predictions held. The one that failed, failed in the direction that
favours the model.**

Mean per-subject MAE at all available personal data:

| Arm | p/activity | p/temporal | s/activity | s/temporal |
| --- | --- | --- | --- | --- |
| Z0, global constant | 19.35 | 16.27 | 19.35 | 16.27 |
| Z1, per-person constant | 14.24 | 11.83 | 14.24 | 11.83 |
| A, population model | 11.72 | 10.36 | 13.14 | 11.53 |
| B, affine | 9.65 | 8.86 | 10.16 | 9.35 |

1. **Arm Z1 lands between Arm A and Arm B, closer to Arm A: failed.** Z1 is 14.24 on
   p/activity, which is 2.52 BPM **worse** than Arm A rather than between it and Arm B.
   The registered range was 10.5 to 11.7 and the answer is outside it in the direction
   that makes the foundation model look better. The error came from estimating the floor
   as a person's mean heart rate; what Arm Z1 actually has is an *estimate* of that mean
   taken from an earlier and differently composed stretch of the same recording, which is
   the honest version and is worse.
2. **If Arm Z1 beats Arm A, the frozen probe contributes nothing: antecedent false.** Arm
   A is better than Arm Z1 in all four cells on the point estimate, so the headline does
   not have to move. It must be stated with its qualification, below.
3. **Arm B still beats Arm Z1: held**, detectably in all four cells, by 2.49 to 4.59 BPM,
   better on 13 to 15 of 15 subjects. This was the prediction whose failure would have
   been most expensive and it held with room to spare.
4. **Arm Z0 is worse than Arm A in all four cells: held**, detectably, by 4.74 to 7.62.
5. **Arm Z1 beats Arm Z0 at all available data in all four cells: held**, detectably, by
   4.44 to 5.10.
6. **Arm Z1 is worse than Arm A at the two smallest budgets: held**, detectably in all
   four cells at both 2 and 5 minutes, by 6.47 to 16.19 BPM.

### The qualification that has to travel with prediction 2

Arm A beats Arm Z1 in all four cells, but **detectably in only one of them**: +2.52
[+0.05, +4.84] on p/activity, against +1.48, +1.11 and +0.30 elsewhere with intervals
spanning zero. On PaPaGei-S under naive temporal splitting the frozen probe is 0.30 BPM
better than predicting the person's own average, which is nothing.

So the floor does not overturn the study, and it does sharply reduce what the probe can
be said to contribute on its own. The defensible sentence is that a frozen PaPaGei
embedding with a ridge probe beats a per-person constant by 0.3 to 2.5 BPM depending on
backbone and protocol, detectably in one of four cells, while the same embedding plus two
stored values beats that constant by 2.5 to 4.6 BPM detectably in all four. **The
adaptation is doing more of the work than the representation is**, and that is a finding
the study is now equipped to state rather than an objection it has to absorb.

### Arm Z1 confirms the compositional explanation, with no model to blame

Arm Z1's budget curve on p/activity runs 26.11, 27.91, 15.23, 14.90, 14.80, 14.24. At two
and five minutes a one-parameter arm carrying no model output at all is 12 to 16 BPM worse
than doing nothing, and five minutes is worse than two. That is the same shape every
adaptive arm shows at small budgets, and CLAUDE.md attributes it to composition: budgets
are temporal prefixes, so the first minutes are sitting still and the mean estimated there
is not the test block's mean. Arm Z1 has no encoder, no probe and no slope to extrapolate,
so composition is the only available explanation. **The cold-start problem is now
demonstrated on an arm that cannot have any other cause.**

## The MOMENT Backbone: Two Checks, Then The Registration, 3 November 2026

The objection the floor baselines create is that the estimator is weak, so of course cheap
corrections beat capacity. Published purpose-built methods reach 4 to 8 BPM on PPG-DaLiA
and this study sits at 11 to 13, because linear probing is a representation benchmark and
not an estimator. A stronger backbone is the test of whether the finding survives, and
MOMENT at a reported 8.82 is the strongest frozen probe in the reference table.

Two things were checked before any code was written, as required.

### Check 1: MOMENT's input-length convention against the 1,250-sample windows

**Settled empirically, and the answer is not the one the configuration implies.**
`MOMENT-1-large` reports `seq_len = 512`, `patch_len = 8`, `patch_stride_len = 8` and
`d_model = 1024`. Feeding it PaPaGei's 1,250-sample zero-padded windows nonetheless runs
without error and returns finite embeddings of width 1,024, because the embedding path
takes its patch count from the tensor rather than the configuration.

It also **reads the whole window rather than truncating to its context**, which is the
part that mattered and could not be read off the source. Two inputs identical in their
first 512 samples and random thereafter give embeddings that differ by 0.036, and a single
spike placed at sample 1248, 1249 or 1250 moves the embedding by 0.07 to 0.10. Had the
embeddings matched, PaPaGei's MOMENT baseline would have been computed on 4.1 of every 8
seconds and the reference figure would have needed re-reading.

**Caveat to carry into the paper.** MOMENT is therefore being run at 2.4 times its
pre-training context, 1,250 samples against 512, and its revolution-invariant
normalisation sees the 125 leading and 125 trailing zeros as real data because `input_mask`
defaults to all ones. Both are PaPaGei's choices, not ours, and matching them is what makes
the comparison to 8.82 like for like. Some of MOMENT's accuracy may be left on the table
by them. **The standing rule applies: this is reported, not fixed.** A 512-sample variant
is a clearly labelled extension if it is wanted, never a change to the reproduction.

**Embedding width differs, 1,024 against PaPaGei's 512.** Arm C1's per-person cost was
hardcoded at 513 parameters and 2,052 bytes and now derives from the embedding width, so
MOMENT's Arm C1 is correctly priced at 1,025 and 4,100. Asserted to still return exactly
513 and 2,052 on 512 dimensions, so no row written before today changes.

### Check 2: how the reference figure of 8.82 was produced

**By PaPaGei themselves, with their own probe.** Their repository contains
`linearprobing/feature_extraction_moment.py`, which loads `AutonLab/MOMENT-1-large` with
`task_name="embedding"`, feeds the same stored segments with no resampling or truncation,
and takes `model(x).embeddings`. The downstream fit is their own `regression.py` and
`outcome_regression_all.py`, the same code path that produced 11.53 and 10.92.

So the protocol to match is not invented, it is the one already implemented in
`src/stage4.py`: the 60/20/20 subject split over all 455 test triples, `StandardScaler`
fitted on train, `GridSearchCV` with `cv=4` plain k-fold and `scoring='neg_mean_squared_error'`,
and the alpha grid exactly `[0.1, 1.0, 10.0, 100.0]`. MOMENT enters as a fourth embedding
table and nothing downstream changes.

### What transfers, and what does not

Arms A, B, B-offset, B-ridge, B2, C1, Z0 and Z1 are arithmetic on a cached embedding table
and do not know which encoder produced it. Arms C2 and D are architecture-specific and are
**not** run on MOMENT; the cost axis there would not be comparable in any case.

**One polarity only, `asis`.** The flipped table exists for PaPaGei because the two
checkpoints differ in sign invariance, which is a PaPaGei-specific control for Sub
Question 3. It answers no registered question about MOMENT and would double the most
expensive step in the study, so it is not computed. Stated rather than silently skipped.

### Predictions, registered before MOMENT is embedded

1. **MOMENT's Arm A beats PaPaGei-P's Arm A** on the leave-one-subject-out protocol, since
   it beats it by 2.1 BPM on PaPaGei's own fixed-split probe.
2. **Arm B's gain over Arm A shrinks as the backbone improves.** What Arm B corrects is a
   per-person bias and a better representation should leave less of it. Three points
   ordered by baseline quality, Arm A at 13.14 for PaPaGei-S, 11.72 for PaPaGei-P and an
   expected 9 to 11 for MOMENT, against Arm B gains of +2.98 and +2.08. This predicts a
   MOMENT gain **below +2.08**.
3. **The gap between Arm Z1 and Arm A widens on MOMENT**, because a better model should
   beat the person's own average by more than a worse one. Arm Z1 is 14.24 on p/activity
   and does not depend on the backbone at all, so this is a prediction purely about Arm A.
   Against +2.52 on PaPaGei-P, this predicts **more than +2.52** on MOMENT.
4. **Arm B's fitted scale moves closer to 1 on MOMENT.** Now stated with its mechanism,
   which the decomposition of 3 November supplies: the scale is exactly
   `r * sd(truth) / sd(prediction)`, so it can only approach 1 if the correlation does.
   The mean adaptation-block correlation at all available data is 0.650 on PaPaGei-P, and
   this predicts **above 0.650 on MOMENT**, carrying the scale up with it. This is the
   sharpest prediction available, because it turns the scale distribution from an awkward
   detail into a diagnostic of backbone quality.
5. **Arm B still beats Arm C1 at all available data.** The study's central claim is that
   two stored values are not beaten by 513. If a better representation reverses that, the
   knee moves and the headline changes.

Predictions 2 and 3 pull in opposite directions on purpose and both can hold: Arm B's
gain over Arm A can shrink while Arm A's gain over a constant grows, since the first is
about residual bias and the second about representation quality.

**Prediction 5 is the one whose failure would be most expensive**, and prediction 4 the
one whose success would most strengthen the paper.

## Outcome Of The MOMENT Backbone, Recorded 3 November 2026

The table is 64,697 by 1,024, embedded in 25.4 minutes on MPS from windows whose row order
was asserted equal to Gate 3's index on all four keys. 455 reproduction triples, then
1,140 arm rows in 835 seconds. Every degenerate and floor assertion passes on the new
table, and the floor rows match Stage 10's PaPaGei rows in all 210 shared cells, which
confirms the splits did not move between backbones.

### The reproduction gate passes, and its shape corroborates an earlier reading

MOMENT's published 8.82 sits at the **37.6th percentile** of our 455-triple distribution,
median 9.35, range 5.98 to 16.87. All three backbones now reproduce with one signature:
our median above theirs, their figure in the lower third.

| Backbone | Published | Our median | Gap | Percentile |
| --- | --- | --- | --- | --- |
| MOMENT-1-large | 8.82 | 9.35 | 0.53 | 37.6 |
| PaPaGei-P | 10.92 | 11.78 | 0.86 | 32 |
| PaPaGei-S | 11.53 | 12.77 | 1.24 | 28 |

**MOMENT's gap is the smallest, which is what the registered reading of the PaPaGei gap
predicts.** PaPaGei's published figures are the best of five pre-training runs selected
per downstream task; MOMENT is one released checkpoint with no such selection, so it
should show less of a gap than either PaPaGei row, and it does. This was not registered in
advance, so it is corroboration and not a test, and it should be reported as such.

### The predictions, scored

Mean per-subject MAE at all available personal data, activity-stratified:

| Arm | Cost | MOMENT | PaPaGei-P | PaPaGei-S |
| --- | --- | --- | --- | --- |
| Z1, per-person constant | 1 | 14.24 | 14.24 | 14.24 |
| A, population model | 0 | 9.39 | 11.72 | 13.14 |
| B-offset | 1 | 8.76 | 10.22 | 10.81 |
| B, affine | 2 | 8.17 | 9.65 | 10.16 |
| B-ridge | 2 | 8.08 | 9.70 | 10.21 |
| B2, unlabelled | 0 | 10.20 | 11.57 | 12.79 |
| C1, output layer | 513 / 1,025 | **7.92** | 10.09 | 12.11 |

1. **MOMENT's Arm A beats PaPaGei-P's: held.** 9.39 against 11.72 on activity, 8.22
   against 10.36 on temporal.
2. **Arm B's gain over Arm A shrinks as the backbone improves: held, monotonically in
   both protocols.** Activity: S +2.97, P +2.08, MOMENT +1.22. Temporal: S +2.19, P +1.50,
   MOMENT +0.80. The registered bound was below +2.08 and the result is +1.22. On
   moment/temporal the gain stops being detectable at all, [-0.14, +1.87].
3. **The gap between Arm Z1 and Arm A widens on MOMENT: held.** The registered bound was
   above +2.52 and the result is +4.85 on activity and +3.62 on temporal, **both
   detectable**, where on PaPaGei-P only one cell of four was. The floor objection is much
   less sharp on a good backbone, which is the right way round.
4. **Arm B's fitted scale moves closer to 1 on MOMENT: split, and the decomposition says
   why.** The mechanism's own half holds cleanly: the mean adaptation-block correlation
   rises monotonically across all three backbones in both protocols, 0.621, 0.650, 0.727
   on activity and 0.667, 0.699, 0.763 on temporal, and both MOMENT values clear the
   registered 0.650. The scale itself moves toward 1 on activity, 0.689 to 0.736, and
   **away from it on temporal**, 0.863 to 0.853.

   This is a failure of the prediction as written and not of the mechanism. The scale is a
   product of two factors and **both improved**: the spread ratio also moved toward 1,
   from 1.264 to 1.137 on temporal. Because that ratio sits above 1, its improvement pulls
   the scale down while the correlation's pushes it up, and on temporal the first is
   larger. **The diagnostic is the pair, not its product.** Measured as the distance of
   `(r, spread ratio)` from `(1, 1)`, the ordering is monotone in backbone quality in both
   protocols: 0.400, 0.364, 0.277 on activity and 0.454, 0.400, 0.274 on temporal. The
   prediction should have been registered on the pair, and is restated that way for any
   future backbone.
5. **Arm B still beats Arm C1: failed on the point estimate, not detectably.** This was
   registered as the prediction whose failure would be most expensive, and it is the one
   that failed.

### Arm B minus Arm C1, which is the finding

| Protocol | PaPaGei-S | PaPaGei-P | MOMENT |
| --- | --- | --- | --- |
| activity | **+1.95** | +0.44 | **-0.25** |
| temporal | **+1.45** | +0.69 | **-0.12** |

Positive means Arm B is better; bold is detectable. **The advantage of two stored values
over 513 decays monotonically with backbone quality and crosses zero at the best backbone
tested.** Neither MOMENT cell is detectable, [-1.26, +0.41] and [-1.38, +0.95], and Arm B
is still better on 9 and 10 of fifteen subjects, so the knee has not provably moved and
the study must not claim it has.

**What it does say is better than what the prediction assumed.** Two parameters beating
513 was partly a property of a weak estimator. On a 385M-parameter encoder that reaches
9.39 BPM unadapted, the two methods are indistinguishable, and the trend across three
backbones says which way the next one would go. The honest sentence for the paper is that
the cheap correction's dominance is **conditional on the representation**, demonstrated
over a 3.75 BPM range of baseline quality, rather than a general law. That is a claim the
study can defend and the earlier one was not.

### One result that was not registered

**Arm B2 is detectably worse than doing nothing on MOMENT**, -0.81 [-1.34, -0.37], better
on 4 of 15 subjects. Across backbones its gain runs S +0.34, P +0.16 and MOMENT -0.81, so
the pattern CLAUDE.md already records, that reweighting helps only when the representation
does not already encode subject identity, extends monotonically and past zero. The
unlabelled arm does not merely stop helping on a better representation; it starts hurting.
Reported as an observation rather than a test, since no prediction was registered for it.

### Record integrity note

`src/stage4.py` was edited during the reproduction run, to record MOMENT's published
interval read from the paper's Table 3. The repository's own rule is not to touch the tree
while a job runs and it was broken here. The edit changed `PUBLISHED_INTERVAL`, a reporting
constant that `reproduction()` never reads, and the running process had already imported
the module, so no recorded number is affected. Noted rather than passed over, because the
value of the rule is that breaches are reported even when they turn out to be harmless.
