# Three-way classification results

Hate / offensive / neither-or-normal, default `argmax` decision rule. Per-class F1 plus macro-F1 and weighted-F1.

| Model | Dataset | Acc | F1 hate | F1 offensive | F1 neither | F1 macro | F1 weighted |
|---|---|---|---|---|---|---|---|
| TF-IDF + LR | davidson | 0.8725 | 0.4360 | 0.9203 | 0.8562 | 0.7375 | 0.8816 |
| TF-IDF + LR | hatexplain | 0.6684 | 0.7326 | 0.5243 | 0.7170 | 0.6580 | 0.6669 |
| Doc2Vec + LR | davidson | 0.5143 | 0.1772 | 0.6265 | 0.5211 | 0.4416 | 0.5828 |
| Doc2Vec + LR | hatexplain | 0.5577 | 0.6234 | 0.3972 | 0.6043 | 0.5416 | 0.5512 |
| DistilBERT | davidson | 0.9137 | 0.4202 | 0.9484 | 0.9064 | 0.7583 | 0.9108 |
| DistilBERT | hatexplain | 0.7001 | 0.7706 | 0.5230 | 0.7560 | 0.6832 | 0.6941 |
