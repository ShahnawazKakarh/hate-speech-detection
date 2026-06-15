# Threshold-tuning ablation

Decision threshold selected by maximizing F1 on the hate class on the *validation* set, then applied unchanged to the test set. Default = 0.5 baseline shown for reference.

| Model | Dataset | t* | F1 (hate) @ 0.5 | F1 (hate) @ t* | Δ F1 (hate) | F1 (macro) @ t* |
|---|---|---|---|---|---|---|
| TF-IDF + LR | davidson | 0.60 | 0.4030 | **0.4304** | +0.0274 | 0.6958 |
| TF-IDF + LR | hatexplain | 0.49 | 0.7311 | **0.7316** | +0.0005 | 0.8011 |
| Doc2Vec + LR | davidson | 0.93 | 0.1689 | **0.2158** | +0.0469 | 0.5479 |
| Doc2Vec + LR | hatexplain | 0.71 | 0.6061 | **0.6174** | +0.0113 | 0.7064 |
| DistilBERT | davidson | 0.30 | 0.3682 | **0.4626** | +0.0944 | 0.7152 |
| DistilBERT | hatexplain | 0.50 | 0.7568 | **0.7568** | +0.0000 | 0.8243 |
