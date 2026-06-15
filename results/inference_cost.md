# Inference-cost benchmark

Measured on the *test* split of each model's training dataset, single thread, no GPU (CPU/MPS only). Latency is per single sample (median + 5/95 percentile). Throughput is samples per second at batch size 32. RAM delta is resident set size increase when the model is loaded into memory.

| Model | Dataset | Size on disk (MB) | RAM load Δ (MB) | Latency p50 (ms) | Latency p05/p95 (ms) | Throughput @ bs=32 (samples/s) |
|---|---|---|---|---|---|---|
| TF-IDF + LR | davidson | 1.4 | 81.1 | 0.35 | 0.32 / 0.48 | 39183.0 |
| TF-IDF + LR | hatexplain | 2.0 | 26.0 | 0.34 | 0.31 / 0.43 | 33212.7 |
| Doc2Vec + LR | davidson | 30.8 | 40.1 | 0.50 | 0.32 / 0.90 | 2124.5 |
| Doc2Vec + LR | hatexplain | 30.4 | 21.5 | 0.71 | 0.32 / 1.40 | 1412.8 |
| DistilBERT | davidson | 256.1 | 173.5 | 19.93 | 16.24 / 32.11 | 303.9 |
| DistilBERT | hatexplain | 256.1 | 19.2 | 17.78 | 14.68 / 21.81 | 225.7 |
