from types import SimpleNamespace

import numpy as np
import pandas as pd

from . import base, decoding


def test_stack_neuro_at_time() -> None:
    neuro = np.arange(12).reshape(1, 2, 6)
    stacked = decoding._stack_neuro_at_time(neuro, 0, radius=1)
    np.testing.assert_array_equal(stacked, [[0, 6, 1, 7]])
    assert decoding._stack_neuro_at_time(neuro, 3, radius=1).shape == (1, 6)


def test_sliding_window_decodes_signal() -> None:
    rng = np.random.default_rng(0)
    extractor = rng.normal(size=(60, 2))
    neuro = rng.normal(size=(60, 3, 5))
    neuro[:, 0, 2] += 3 * extractor[:, 0]
    sw = decoding.SlidingWindow.model_construct(
        data=SimpleNamespace(start=-1.0, stop=1.0),  # type: ignore[arg-type]
        model=base.RidgeCVConfig(alphas=[1.0, 10.0], alpha_per_target=False),
        cv=base.Cv(n_splits=5),
        scoring=base.SpearmanR(),
    )
    events = pd.DataFrame({"subject": ["s"] * 60})
    df = decoding.scores_to_df(sw.score_arrays(neuro, extractor, events, "s"), "f", ["a", "b"])
    peak = df.set_index(["feature", "time"]).score
    assert peak["a", 0.0] > 0.8
    assert abs(peak["b", 0.0]) < 0.4
    assert df.set_index(["feature", "time"]).pvalue["a", 0.0] < 0.05
