# Multi-seed results (binary classification)

Mean ± std across seeds {42, 7, 1337}, identical preprocessing, identical splits, default 0.5 decision threshold. Seeds were chosen before any results were inspected.

| Model | Dataset | n | Acc | F1 (hate) | F1 (macro) | AUC |
|---|---|---|---|---|---|---|
| TF-IDF + LR | davidson | 3 | 0.904 ± 0.000 | 0.405 ± 0.002 | 0.677 ± 0.001 | 0.857 ± 0.000 |
| TF-IDF + LR | hatexplain | 3 | 0.828 ± 0.000 | 0.731 ± 0.000 | 0.802 ± 0.000 | 0.892 ± 0.000 |
| Doc2Vec + LR | davidson | 3 | 0.554 ± 0.006 | 0.169 ± 0.003 | 0.432 ± 0.004 | 0.722 ± 0.004 |
| Doc2Vec + LR | hatexplain | 3 | 0.701 ± 0.003 | 0.617 ± 0.004 | 0.686 ± 0.003 | 0.797 ± 0.007 |
| DistilBERT | davidson | 3 | 0.942 ± 0.003 | 0.395 ± 0.029 | 0.682 ± 0.015 | 0.895 ± 0.001 |
| DistilBERT | hatexplain | 3 | 0.848 ± 0.002 | 0.761 ± 0.005 | 0.825 ± 0.002 | 0.910 ± 0.001 |
