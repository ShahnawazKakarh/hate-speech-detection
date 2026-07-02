# Gemini LLM-as-classifier results (v0.2.1)

Zero-shot and 4-shot evaluation of `gemini-2.5-flash` on the same Davidson and HateXplain test splits used in v0.1. DistilBERT v0.1 numbers included for direct comparison. Decision threshold 0.5; `p_hate` returned by the model serves as the score for ROC-AUC.

## davidson

| Method | Acc | P | R | F1 (hate) | F1 (macro) | AUC |
|---|---|---|---|---|---|---|
| DistilBERT (v0.1) | 0.9391 | 0.4583 | 0.3077 | 0.3682 | 0.6681 | 0.8966 |
| Gemini 2.5 Flash, zero-shot | 0.9129 | 0.3270 | 0.4825 | 0.3898 | 0.6715 | 0.7974 |
| Gemini 2.5 Flash, 4-shot | 0.8814 | 0.2773 | 0.6573 | 0.3900 | 0.6622 | 0.8411 |

## hatexplain

| Method | Acc | P | R | F1 (hate) | F1 (macro) | AUC |
|---|---|---|---|---|---|---|
| DistilBERT (v0.1) | 0.8503 | 0.7593 | 0.7542 | 0.7568 | 0.8243 | 0.9088 |
| Gemini 2.5 Flash, zero-shot | 0.7204 | 0.5264 | 0.9394 | 0.6747 | 0.7148 | 0.8860 |
| Gemini 2.5 Flash, 4-shot | 0.7801 | 0.5918 | 0.9276 | 0.7226 | 0.7703 | 0.8903 |
