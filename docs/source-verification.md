# Source Verification

Claims and citations checked against the primary sources held in the project's
reading folder, 29 September 2026. Recorded so that corrections are traceable and so
that the same checks are not repeated.

## Verified correct

| Claim | Source | Status |
| --- | --- | --- |
| Black patients had nearly three times the frequency of occult hypoxemia undetected by pulse oximetry | Sjoding et al. 2020, NEJM | Correct. Michigan cohort 11.7% against 3.6%, multicentre 17.0% against 6.2% |
| Conv-LSTM reaches 6.3 BPM on PPG-DaLiA | Wilkosz and Szczesna 2021, Sensors | Correct, 6.28 BPM, leave-one-subject-out |
| Classical baselines at roughly 15.6 and 20.5 BPM | Same, quoting Reiss et al. | Correct. SpaMa 15.56, Schaeck2017 20.45 |
| Skewness is the best signal quality index for PPG | Elgendi 2016, Bioengineering | Correct. Eight indices compared, skewness best at F1 86.0, 87.2 and 79.1 per cent |
| Normalisation overlapping the evaluation window inflates performance | Tognotti, Otesteanu et al. 2026, Frontiers in Digital Health | Correct as a mechanism, with caveats below |

## Corrections needed

**Four citations are wrong.**

1. The normalisation-inflation paper is **Tognotti, Otesteanu, Anceschi and Menon**. The
   bibliography lists it as "Otesteanu, C., et al.", but Otesteanu is the second author.
2. The Conv-LSTM paper is **Wilkosz and Szczesna**, *Sensors* 21(15), 5212. The
   bibliography lists it as "Rescic, N., et al." at 21(8), 2719. Wrong authors, wrong
   issue, wrong article number.
3. The on-device paper is by **Simon A. Lee** and colleagues at Samsung Research
   America, titled *Towards On-device Foundation Models for Wearable Signals*. The
   bibliography gives "Lee, J. Y." and adds "Raw" to the title, and omits one author.
4. The nine-minute cuffless blood pressure paper has no authors listed at all. It is
   **Mekonnen, Lu, Hsieh, Chu and Yang (2024)**, *Scientific Reports*.

**Two claims need qualifying.**

The normalisation-inflation result comes from a **classification** study of anxiety
detection on 52 participants using the PhysioNet Spider Fear dataset, not from
heart-rate regression on PPG. The mechanism transfers and justifies the buffer; the
magnitudes do not transfer and should not be quoted as though they did.

The headroom argument compares frozen probing at roughly 11 BPM against supervised
task-specific training at 6.3. Those are not like for like: the Conv-LSTM uses **PPG
and accelerometer**, while the foundation-model probes use PPG alone. Part of the gap
is the motion channel rather than anything adaptation could recover. The frozen-probe
rows, MOMENT, Chronos and TF-C, are directly comparable to PaPaGei; the supervised row
is not.

## Worth adding, because it strengthens the argument

**Bent et al. 2020**, *npj Digital Medicine*, is the most directly transferable
evidence available and is currently in the bibliography but unused in the text. Fifty
three participants spanning **the full Fitzpatrick range 1 to 6**, six devices
including the **Empatica E4**, the same device PPG-DaLiA uses. They found **no
statistically significant difference in heart-rate accuracy across skin tones**, and
none for heart-rate variability either, but large and significant differences between
devices and between activities: **absolute error during activity averaged 30 per cent
higher than at rest**.

That supports three things at once. It justifies treating skin tone as an open
question rather than an established effect. It is independent evidence for the
activity-based analysis, since activity, not pigmentation, is what moved error in a
study designed to find a pigmentation effect. And its reported error range of roughly
10 to 15 BPM brackets this study's Arm A figure of 11.70.

**Mekonnen et al. 2024** sets a reference point for the budget sweep. Personalised
cuffless blood pressure from **nine minutes** of personal calibration data, where
prior work required "tens to hundreds of minutes", and performance degrades
significantly below nine. The budget grid of 2, 5, 10, 20 and 40 minutes brackets
that figure. Two caveats: the task is blood pressure, not heart rate, and their method
mixes personal data into a general training set, so it is closer to a warm-started
per-person model than to normalisation. It currently sits under "Normalisation and
Cheap Personalisation" in the bibliography, which is the wrong category.

**Lee et al. 2025** gives the cost axis a memorable anchor. Their entire on-device
foundation model occupies **3.6 MB**, against **20 MB per person** for full
fine-tuning in this study. A per-person copy costs several times the whole model. It
also benchmarks PaPaGei directly and finds it competitive, and its axis is model size
for everyone rather than cost per person, which is exactly the distinction the
proposal's Scale of Adaptation section draws.

## Not verified

Koerber et al. 2023, the systematic review behind the "four of ten studies" sentence,
is cited but is not in the reading folder, so the claim is unchecked.

Opened and confirmed by title and role only, not read in full: Charlton's 2023
roadmap, Ismail's motion-artefact review, the ACM signal-quality assessment paper, the
Monte Carlo skin-tone modelling, Hardt's equality of opportunity, Green AI, and Guo's
calibration paper.
