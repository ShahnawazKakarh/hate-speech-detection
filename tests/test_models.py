"""Tests for model wrappers (smoke tests on tiny synthetic data)."""

import pytest
from hsd.models.tfidf import TfidfConfig, build_tfidf_pipeline


def _toy_data():
    texts = [
        "i hate that group they are awful",
        "you people are disgusting",
        "have a great day everyone",
        "what a lovely sunset",
        "kill all of them",
        "the weather is nice today",
        "they should be wiped out",
        "happy birthday my friend",
    ] * 4  # 32 samples
    labels = [1, 1, 0, 0, 1, 0, 1, 0] * 4
    return texts, labels


def test_tfidf_pipeline_fits_and_predicts():
    texts, labels = _toy_data()
    pipe = build_tfidf_pipeline(TfidfConfig(max_features=200, min_df=1))
    pipe.fit(texts, labels)
    preds = pipe.predict(texts)
    assert len(preds) == len(texts)
    assert set(preds.tolist()) <= {0, 1}


@pytest.mark.slow
def test_doc2vec_pipeline_fits_and_predicts():
    from hsd.models.doc2vec import Doc2VecClassifier, Doc2VecConfig

    texts, labels = _toy_data()
    cfg = Doc2VecConfig(vector_size=20, epochs=3, min_count=1, infer_steps=5, workers=1)
    model = Doc2VecClassifier(cfg).fit(texts, labels)
    preds = model.predict(texts[:4])
    assert len(preds) == 4
