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

## Scope

Polarity is a side experiment of about a day. It does not displace the frontier. If
the effect is material it becomes a discussion section, not a change of direction.
