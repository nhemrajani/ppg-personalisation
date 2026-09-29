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
where there is no obligation to match anyone.

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

Registered in advance so that the interpretation of a miss is fixed before the result
exists: if the reproduction lands outside the 455-triple distribution on S but inside
on P, model selection across pre-training runs is a live explanation alongside a
pipeline error, and the two are not distinguishable with the released artefacts.
