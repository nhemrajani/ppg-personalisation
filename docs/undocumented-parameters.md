# Undocumented Parameters

Values PaPaGei's pipeline requires but neither their paper nor their released code
states. Each is set explicitly here rather than left as an implicit default, with the
evidence for the choice and what is known about its effect.

| Parameter | Value used | Basis | Effect if changed |
| --- | --- | --- | --- |
| Minimum flatline duration | 2 s | Their code takes it as a command-line argument and publishes no default | None. Swept over 0.5, 1, 2 and 4 s: flatness stays exactly 0.0000% and no window is rejected at any value. The result is set by the change threshold, not this parameter. |
| Padded window length | 1,250 samples | Matches the 10 s pre-training window, and their example notebook pads a shorter recording to this length | Unknown, and load bearing. The encoder averages over time, so padding changes the embedding. First suspect if the reproduction misses 11.53. |
| Padding placement | Centred, half each side | Their example notebook | Unknown, same reasoning as above |
| Flatline change threshold | 0.01, their default | Their code | Inert on this dataset. The threshold is absolute while the median step between consecutive E4 BVP samples is 4.84 units, so no sample pair qualifies. On z-scored input 3 of 64,697 windows would be rejected. |
| Subject membership of the 9/3/3 split | Unknown | Their code reads `data/dalia/{train,val,test}.csv`, which is not released | Material. With three test subjects the published error depends heavily on which three. Handled by reporting the distribution over all 455 possible test triples. |
| Heart-rate range filter | None. All 64,697 windows are scored | Resolved 28 September, see below | Settled, not open |
| Ridge regularisation strength | To be fixed before Gate 5 | Ours to choose | Selected on the population set or a validation slice of the adaptation block, never the test block, with the range committed before any test result is seen. |

## Note on the flatline check

Two separate issues combine, and neither changes what we do, which is to run their
pipeline exactly.

The threshold is compared against raw amplitude, so its units do not match the
signal. PaPaGei's `is_signal_flat_lined` computes a z-scored copy of the window and
then passes the un-normalised array to the detector, which appears unintended. Were
the z-scored array passed, 3 windows would be rejected, which is negligible for this
study but worth reporting as a reproduction detail.

Their detector also excludes any sample equal to the window maximum or minimum, so a
perfectly constant window scores zero flatness rather than one. This is replicated
faithfully in `src/quality.py`, verified case by case against their function.


## Note on the 60 to 150 BPM range

PaPaGei's PPG-DaLiA appendix says the dataset "captures a wide range of heart rates,
varying from 60 to 150 beats per minute". The labels actually run from 42 to 187, and
7.45 per cent of windows fall outside the stated range, so the sentence either
describes the data loosely or reports a filter they applied. The distinction matters:
had they filtered, we would be scoring on roughly three thousand windows they
excluded, and those sit at the extremes where error is largest, so our figure would
come out worse than theirs for reasons unrelated to correctness.

It is descriptive. Two pieces of evidence. Their evaluation code contains no
heart-rate filtering of any kind; the only selection is by subject identifier. And
their own Figure 19 histogram of PPG-DaLiA heart rate spans roughly 42 to 185 with
visible mass below 60 and above 150, matching our distribution in range, in the
bimodal peak near 75 and 88, and in the long right tail. The figure is drawn from the
same unfiltered labels.

No windows are excluded on this basis.
