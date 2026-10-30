import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

from . import data, stats, style

RELEASE = Path(os.environ.get("CHILDBRAIN_RELEASE", "release"))
needs_release = pytest.mark.skipif(not (RELEASE / "participants.tsv").exists(),
                                   reason="set CHILDBRAIN_RELEASE to the released data")


def test_fdr_bh() -> None:
    p = np.random.default_rng(0).random(50) ** 3
    np.testing.assert_allclose(stats.fdr_bh(p), multipletests(p, method="fdr_bh")[1])
    assert np.isnan(stats.fdr_bh(np.array([0.01, np.nan]))[1])


def test_cluster_mask() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 60))
    x[:, 20:30] += 2
    mask = stats.cluster_mask(x, n_permutations=200)
    assert mask[20:30].all() and not mask[:15].any()


def test_rank_biserial() -> None:
    assert stats.rank_biserial(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == 1.0
    assert stats.rank_biserial(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == -1.0


def test_format_tick() -> None:
    assert [style.format_tick(v) for v in (0, 0.05, -0.25, 1.5)] == ["0", ".05", "-.25", "1.5"]


@needs_release
def test_release_tables() -> None:
    data.set_release(RELEASE)
    part = data.participants()
    assert len(part) == 65 and list(part.columns) == ["age_range", "age_group", "has_soz_ranking"]
    assert not data.has_exact_age()
    el = data.electrodes()
    assert el.participant_id.nunique() == 65 and el[el.x.notna()].participant_id.nunique() == 50
    assert el.vertex.notna().sum() == el.x.notna().sum()
    trf = data.trf_linguistic()
    assert set(trf.model) == {data.TRF_PHONETIC, data.TRF_LEXICAL, data.TRF_SUPRA}
    counts = data.roi_means(data.trf_scores(trf, data.TRF_LEXICAL)).groupby("roi").participant.nunique()
    # ROI participant counts of the paper's ROI coverage table.
    expected = dict(whole_brain=65, STG=43, STS=42, ATL=22, IPL=35, OCC=19)
    expected["L/IFC"] = 46
    assert counts.to_dict() == expected
    assert isinstance(data.decoding_levels(levels=["lexical"]), pd.DataFrame)
