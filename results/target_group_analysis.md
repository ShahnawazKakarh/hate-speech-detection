# Target-group analysis on HateXplain (DistilBERT)

Per-group breakdown of the DistilBERT-HateXplain test-set predictions, default decision threshold (0.50). `hate %` is the share of posts in the group with majority-vote `hatespeech`; `flagged %` is the share predicted as hate. Large gaps between `recall (hate)` across groups indicate the model is systematically missing hate aimed at some targets.

| Group | n | n (hate) | hate % | flagged % | recall (hate) | F1 (hate) | F1 (macro) |
|---|---|---|---|---|---|---|---|
| Overall | 1924 | 594 | 30.9% | 30.7% | 0.754 | 0.757 | 0.824 |
| Race | 544 | 286 | 52.6% | 55.3% | 0.843 | 0.821 | 0.806 |
| Religion | 351 | 208 | 59.3% | 57.0% | 0.760 | 0.775 | 0.731 |
| Gender | 177 | 17 | 9.6% | 6.2% | 0.471 | 0.571 | 0.767 |
| Sexual Orientation | 187 | 54 | 28.9% | 23.5% | 0.537 | 0.592 | 0.723 |
| Other | 288 | 28 | 9.7% | 8.0% | 0.429 | 0.471 | 0.710 |
| None | 377 | 1 | 0.3% | 2.9% | 0.000 | 0.000 | 0.492 |
