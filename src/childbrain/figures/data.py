"""Read the released data (``release/``, see its README).

The release folder is ``$CHILDBRAIN_RELEASE`` (default: ``./release``), or set with
:func:`set_release`. Tables are read once and cached.

The release gives age ranges only. Analyses of exact age (and of the seizure onset zone)
need the full participants table (same ids, with ``age``, ``soz_hemisphere``,
``soz_lobe``), set with ``$CHILDBRAIN_PARTICIPANTS`` or :func:`set_participants`; without
it they are skipped (:func:`has_exact_age`).
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import numpy as np
import pandas as pd

#: Age groups, youngest first (``participants.tsv`` ``age_group``).
GROUPS = ["1-5", "6-11", "12+"]
#: Released age ranges (``participants.tsv`` ``age_range``), youngest first.
AGE_RANGES = ["1-2", "3-5", "6-8", "9-11", "12-17", "18+"]
#: ROIs in panel order; ``whole_brain`` is every electrode.
ROIS = ["whole_brain", "STG", "STS", "ATL", "IPL", "L/IFC", "OCC"]
ROI_TITLES = {"whole_brain": "Whole brain", **{r: r for r in ROIS[1:]}}
#: The "language ROIs" (Fig. 6 training gain).
LANGUAGE_ROIS = ["STG", "STS", "L/IFC", "IPL", "ATL"]
#: Linguistic levels of the decoding and encoding figures.
LEVELS = ["phonetic", "lexical", "supralexical"]
LEVEL_TITLES = {"acoustic": "Acoustic", "phonetic": "Phoneme", "lexical": "Lexical",
                "supralexical": "Supralexical"}
#: Nested linguistic TRFs (``encoding/trf_linguistic.csv.gz`` ``model``).
TRF_PHONETIC = "phonetic"
TRF_LEXICAL = "phonetic+lexical"
TRF_SUPRA = "phonetic+lexical+supralexical"

_RELEASE: Path | None = None
_PARTICIPANTS: Path | None = None


def set_release(path: str | Path) -> None:
    global _RELEASE
    _RELEASE = Path(path)
    _clear()


def set_participants(path: str | Path) -> None:
    """Use the full participants table (exact ages, seizure onset zone)."""
    global _PARTICIPANTS
    _PARTICIPANTS = Path(path)
    _clear()


def _clear() -> None:
    for f in (participants, electrodes, decoding_linguistic, decoding_embeddings,
              decoding_layerwise, stimulus_decoding, trf_linguistic, trf_embeddings,
              trf_splits, trf_predictions, evoked, predictor_correlation, decoding_pvalues):
        f.cache_clear()


def release_dir() -> Path:
    path = _RELEASE or Path(os.environ.get("CHILDBRAIN_RELEASE", "release"))
    if not (path / "participants.tsv").exists():
        raise FileNotFoundError(
            f"no release at {path.resolve()}: set CHILDBRAIN_RELEASE or pass --release")
    return path


def _read(rel: str, **kwargs) -> pd.DataFrame:
    path = release_dir() / rel
    if rel.endswith((".tsv", ".tsv.gz")):
        return pd.read_csv(path, sep="\t", na_values="n/a", **kwargs)
    return pd.read_csv(path, **kwargs)


@functools.cache
def participants() -> pd.DataFrame:
    """One row per participant, indexed by ``participant_id``; ``age_group`` is ordered.

    Released columns: ``age_range``, ``age_group``, ``has_soz_ranking``; the
    full table adds ``age`` (years), ``soz_hemisphere`` and ``soz_lobe``.
    """
    df = _read("participants.tsv").set_index("participant_id")
    path = _PARTICIPANTS or os.environ.get("CHILDBRAIN_PARTICIPANTS")
    if path:
        full = pd.read_csv(path, sep="\t", na_values="n/a").set_index("participant_id")
        assert full.index.sort_values().equals(df.index.sort_values()), f"{path}: other participants"
        assert full.age_group.reindex(df.index).equals(df.age_group), f"{path}: other age groups"
        df = full.reindex(df.index)
    df["age_group"] = pd.Categorical(df.age_group, categories=GROUPS, ordered=True)
    return df


def has_exact_age() -> bool:
    """Whether exact ages are available."""
    return "age" in participants().columns


def skip_without_exact_age(what: str) -> bool:
    """True (and a message) when ``what`` needs exact ages that are not available."""
    if has_exact_age():
        return False
    print(f"  skipped {what}: needs exact ages (full participants table, "
          "CHILDBRAIN_PARTICIPANTS / --participants)")
    return True


@functools.cache
def electrodes() -> pd.DataFrame:
    """``electrodes.tsv`` with ``vertex`` (closest fsaverage pial vertex) and ``medial``.

    Electrode positions are those of their closest pial vertex, so the vertex is found
    exactly; ``medial``: |x| of the vertex below 2 cm.
    """
    df = _read("electrodes.tsv")
    from .brain import pial_surfaces

    df["vertex"] = pd.array([pd.NA] * len(df), dtype="Int64")
    df["medial"] = False
    for hemi, surf in zip(("left", "right"), pial_surfaces()):
        rows = (df.hemisphere == hemi) & df.x.notna()
        if not rows.any():
            continue
        from scipy.spatial import cKDTree

        dist, vert = cKDTree(surf).query(df.loc[rows, ["x", "y", "z"]].to_numpy())
        assert dist.max() < 1e-4, f"{hemi}: electrode {dist.max() * 1e3:.2f} mm from the pial surface"
        df.loc[rows, "vertex"] = vert
        x = surf[vert, 0]
        df.loc[rows, "medial"] = x > -0.02 if hemi == "left" else x < 0.02
    return df


def located(df: pd.DataFrame) -> pd.DataFrame:
    """Rows of ``df`` (with participant and electrode ids) whose electrode has a position."""
    el = electrodes()
    keys = set(zip(el.participant_id[el.x.notna()], el.electrode_id[el.x.notna()]))
    mask = [k in keys for k in zip(df.participant, df.electrode)]
    return df[mask]


def roi_electrodes(roi: str) -> pd.DataFrame:
    """participant, electrode of the ROI (``whole_brain``: every electrode)."""
    el = electrodes()
    if roi != "whole_brain":
        el = el[el.roi == roi]
    return el[["participant_id", "electrode_id"]].rename(
        columns={"participant_id": "participant", "electrode_id": "electrode"})


def add_age(df: pd.DataFrame, col: str = "participant") -> pd.DataFrame:
    """Add ``age`` and ``group`` (ordered age group). ``age`` is the exact age (years) when
    available (:func:`has_exact_age`), otherwise the released age range (e.g. ``"6-8"``)."""
    part = participants()
    df = df.copy()
    df["age"] = df[col].map(part.age if has_exact_age() else part.age_range)
    df["group"] = pd.Categorical(df[col].map(part.age_group), categories=GROUPS, ordered=True)
    return df


# ── decoding ──────────────────────────────────────────────────────────────────


@functools.cache
def decoding_linguistic() -> pd.DataFrame:
    """``decoding/brain_linguistic.csv.gz`` (participant, roi, level, feature_group,
    feature, time, score, n_dims), with age and group."""
    return add_age(_read("decoding/brain_linguistic.csv.gz"))


def decoding_levels(roi: str = "whole_brain", levels: list[str] | None = None) -> pd.DataFrame:
    """One time course per participant and level: mean over the level's features (each
    feature first averaged over its dimensions)."""
    df = decoding_linguistic().query("roi == @roi")
    if levels is not None:
        df = df[df.level.isin(levels)]
    out = df.groupby(["participant", "level", "time"], as_index=False).score.mean()
    return add_age(out)


def decoding_features(roi: str = "whole_brain") -> pd.DataFrame:
    """One time course per participant and feature (level kept)."""
    df = decoding_linguistic().query("roi == @roi")
    out = df.groupby(["participant", "level", "feature_group", "feature", "time"],
                     as_index=False).score.mean()
    return add_age(out)


@functools.cache
def decoding_embeddings() -> pd.DataFrame:
    """``decoding/brain_embeddings.csv.gz``, with ``model``, ``depth``, ``untrained``."""
    df = _read("decoding/brain_embeddings.csv.gz")
    parts = df.embedding.str.extract(r"^(?P<model>[a-z0-9]+)_L(?P<depth>[0-9.+]+)(?P<untrained>_untrained)?$")
    df["model"] = parts.model
    df["depth"] = parts.depth
    df["untrained"] = parts.untrained.notna()
    return add_age(df)


@functools.cache
def decoding_layerwise() -> pd.DataFrame:
    return add_age(_read("decoding/brain_layerwise.csv.gz"))


@functools.cache
def stimulus_decoding() -> pd.DataFrame:
    return _read("decoding/stimulus_feature_to_feature.csv.gz")


@functools.cache
def decoding_pvalues() -> pd.DataFrame:
    """``decoding/brain_linguistic_pvalues.csv.gz``: whole-brain p-value of each feature's
    decoding at each time (participant, level, feature_group, feature, time, pvalue)."""
    return add_age(_read("decoding/brain_linguistic_pvalues.csv.gz"))


# ── encoding ──────────────────────────────────────────────────────────────────


@functools.cache
def trf_linguistic() -> pd.DataFrame:
    """``encoding/trf_linguistic.csv.gz`` (participant, electrode, model, r, perm_p,
    n_segments), with age and group."""
    return add_age(_read("encoding/trf_linguistic.csv.gz"))


@functools.cache
def trf_embeddings() -> pd.DataFrame:
    return add_age(_read("encoding/trf_embeddings.csv.gz"))


@functools.cache
def trf_splits() -> pd.DataFrame:
    return add_age(_read("encoding/trf_phonetic_splits.csv.gz", dtype={"segment": str}))


@functools.cache
def trf_predictions() -> pd.DataFrame:
    return add_age(_read("encoding/trf_example_predictions.csv"))


def trf_scores(trf: pd.DataFrame, model: str, baseline: str | None = None) -> pd.DataFrame:
    """Per electrode: participant, electrode, age, group, ``score`` = r(model) [- r(baseline)]."""
    keys = ["participant", "electrode"]
    out = trf[trf.model == model].set_index(keys)
    score = out.r
    if baseline is not None:
        score = score - trf[trf.model == baseline].set_index(keys).r.reindex(score.index)
    out = out[["age", "group"]].assign(score=score).dropna(subset=["score"])
    return out.reset_index()


def fdr_bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (NaN kept)."""
    from .stats import fdr_bh as _fdr

    return _fdr(p)


def trf_significant(trf: pd.DataFrame, model: str, baseline: str | None = None) -> set[tuple[str, str]]:
    """(participant, electrode) significant for ``model``: permutation p, BH-FDR across
    every electrode; for a gain over ``baseline``, the gain must also be positive."""
    full = trf[trf.model == model].set_index(["participant", "electrode"])
    sig = pd.Series(fdr_bh(full.perm_p.to_numpy()) < 0.05, index=full.index)
    if baseline is not None:
        delta = trf_scores(trf, model, baseline).set_index(["participant", "electrode"]).score
        sig &= delta.reindex(sig.index).gt(0)
    return set(sig.index[sig])


def roi_means(values: pd.DataFrame, rois: list[str] = ROIS, col: str = "score",
              located_only: bool = False) -> pd.DataFrame:
    """Per ROI and participant: mean of ``col`` over the participant's ROI electrodes.

    ``whole_brain`` is every electrode of ``values`` (``located_only``: only those with a
    position, as the brain maps).
    """
    rows = []
    for roi in rois:
        sub = values
        if roi == "whole_brain":
            if located_only:
                sub = located(values)
        else:
            sub = values.merge(roi_electrodes(roi), on=["participant", "electrode"])
        m = sub.groupby("participant", as_index=False)[col].mean().assign(roi=roi)
        rows.append(m)
    return add_age(pd.concat(rows, ignore_index=True))


# ── evoked, stimulus ──────────────────────────────────────────────────────────


@functools.cache
def evoked(event: str = "word") -> pd.DataFrame:
    """``evoked/{word,phoneme}_onset.csv.gz`` with age and group."""
    return add_age(_read(f"evoked/{event}_onset.csv.gz"))


@functools.cache
def predictor_correlation() -> pd.DataFrame:
    return _read("stimulus/predictor_correlation.csv")
