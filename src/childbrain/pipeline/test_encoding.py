from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from . import encoding


def test_trf_scores_to_df_averages_folds_then_chapters() -> None:
    scores = pd.DataFrame(
        dict(
            subject="s",
            dim=0,
            channel="ch",
            timeline=["c0", "c0", "c0", "c1"],
            split=[0, 1, 2, 0],
            score=[0.1, 0.2, 0.3, 0.6],
        )
    )
    row = encoding.trf_scores_to_df(scores, "f").iloc[0]
    assert row.score == pytest.approx(0.4)
    assert (row.n_chapters, row.feat_dim, row.channel) == (2, "f0", "ch")
    assert 0 < row.pvalue < 0.1


def test_permutation_null_detects_prediction() -> None:
    rng = np.random.default_rng(0)
    actual = rng.normal(size=(200, 2))
    pred = np.stack([actual[:, 0], rng.normal(size=200)], axis=1)
    extractor = SimpleNamespace(frequency=20.0, aggregation="sum")
    data = SimpleNamespace(neuro=extractor, extractor=extractor)
    trf = encoding.TRFEncoding.model_construct(
        data=data, tmin=0.0, tmax=0.1, alphas=[1.0], n_permutations=99  # type: ignore[arg-type]
    )
    observed, pvalues = trf._permutation_null([("c0", pred, actual)])
    assert observed[0] > 0.99
    assert pvalues[0] == 0.01
    assert pvalues[1] > 0.01
