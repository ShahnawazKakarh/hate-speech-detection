# Two-stage cascade sweep (v0.2.2)

Stage 1 = TF-IDF + LR (cheap prefilter). Stage 2 = fine-tuned DistilBERT (expensive verifier). Samples with stage-1 `p_hate >= t1` are re-scored by stage 2; the rest use the stage-1 score. Final decision threshold is the standard 0.5. Latency and throughput are extrapolated from `results/inference_cost.json`.

## davidson

| Stage-1 t1 | % → stage 2 | F1 (hate) | F1 (macro) | AUC | Latency ms / sample | Throughput samples/s |
|---|---|---|---|---|---|---|
| 0.05 | 93.0% | 0.3682 | 0.6681 | 0.8881 | 18.89 | 324.0 |
| 0.10 | 73.5% | 0.3682 | 0.6681 | 0.8276 | 14.99 | 409.4 |
| 0.15 | 53.7% | 0.3682 | 0.6681 | 0.7850 | 11.05 | 557.9 |
| 0.20 | 40.0% | 0.3682 | 0.6681 | 0.7551 | 8.32 | 744.9 |
| 0.25 | 29.2% | 0.3682 | 0.6681 | 0.7348 | 6.17 | 1013.6 |
| 0.30 | 22.8% | 0.3682 | 0.6681 | 0.7166 | 4.88 | 1291.6 |
| 0.35 | 18.0% | 0.3697 | 0.6690 | 0.7344 | 3.93 | 1622.7 |
| 0.40 | 14.8% | 0.3713 | 0.6699 | 0.7552 | 3.31 | 1945.4 |
| 0.45 | 12.0% | 0.3777 | 0.6735 | 0.7648 | 2.74 | 2382.2 |
| 0.50 | 10.2% | 0.3777 | 0.6735 | 0.7878 | 2.39 | 2757.1 |

**Reference points**

- DistilBERT alone: F1-hate = 0.3682, F1-macro = 0.6681, AUC = 0.8966, throughput = 303.9 samples/s.
- **No-loss cascade** (F1-hate ≥ DistilBERT alone): t1 = 0.50, stage-2 rate = 10.2%, F1-hate = 0.3777, throughput = 2757.1 samples/s (**9.1× DistilBERT alone**).
- Best-F1 operating point: t1 = 0.45, F1-hate = 0.3777 at 2382.2 samples/s.

## hatexplain

| Stage-1 t1 | % → stage 2 | F1 (hate) | F1 (macro) | AUC | Latency ms / sample | Throughput samples/s |
|---|---|---|---|---|---|---|
| 0.05 | 98.8% | 0.7568 | 0.8243 | 0.9077 | 17.89 | 226.9 |
| 0.10 | 94.5% | 0.7568 | 0.8243 | 0.9029 | 17.13 | 237.1 |
| 0.15 | 87.3% | 0.7568 | 0.8243 | 0.8967 | 15.86 | 256.4 |
| 0.20 | 79.0% | 0.7574 | 0.8249 | 0.8869 | 14.38 | 283.2 |
| 0.25 | 69.4% | 0.7591 | 0.8266 | 0.8757 | 12.68 | 321.8 |
| 0.30 | 60.6% | 0.7596 | 0.8274 | 0.8656 | 11.11 | 368.2 |
| 0.35 | 52.3% | 0.7580 | 0.8267 | 0.8572 | 9.64 | 425.6 |
| 0.40 | 45.5% | 0.7546 | 0.8253 | 0.8530 | 8.43 | 488.3 |
| 0.45 | 38.8% | 0.7458 | 0.8206 | 0.8671 | 7.23 | 572.0 |
| 0.50 | 33.1% | 0.7346 | 0.8149 | 0.8724 | 6.22 | 667.9 |

**Reference points**

- DistilBERT alone: F1-hate = 0.7568, F1-macro = 0.8243, AUC = 0.9088, throughput = 225.7 samples/s.
- **No-loss cascade** (F1-hate ≥ DistilBERT alone): t1 = 0.35, stage-2 rate = 52.3%, F1-hate = 0.7580, throughput = 425.6 samples/s (**1.9× DistilBERT alone**).
- Best-F1 operating point: t1 = 0.30, F1-hate = 0.7596 at 368.2 samples/s.
