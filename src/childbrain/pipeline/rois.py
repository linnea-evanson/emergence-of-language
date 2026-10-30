"""Anatomical ROIs of the figures, as sets of bipolar channels.

An ROI is the set of bipolar channels of the electrode table (``electrodes.tsv``)
whose midpoint falls in the ROI's parcels: Destrieux parcels (``destrieux``) for
:data:`ROIS_DETAILED`, Desikan parcels (``desikan``) for :data:`ROIS_BRAIN_AREA`.

``IeegExtractor.picks`` applies to the raw before bipolar referencing, so the
extractor picks the monopolar endpoints (:func:`roi_contacts`) and then keeps the
ROI's bipolar channels only (:func:`roi_referenced_channels`).
"""

import functools
import typing as tp
from pathlib import Path

import pandas as pd
import yaml

from .study import Evanson2023

ROIS_DETAILED: dict[str, list[str]] = {
    "stc": [
        "G_temp_sup-Lateral", "G_temp_sup-Plan_polar", "G_temp_sup-Plan_tempo",
        "G_temp_sup-G_T_transv", "G_temporal_sup", "S_temporal_transverse",
    ],
    "sts": ["S_temporal_sup"],
    "tp": ["Pole_temporal"],
    # Supramarginal + Angular Gyrus
    "sma": ["G_pariet_inf-Supramar", "S_interm_prim-Jensen", "G_pariet_inf-Angular"],
    "occipital_visual": [
        "G_occipital_sup", "G_occipital_middle", "G_and_S_occipital_inf",
        "S_calcarine", "G_cuneus", "S_oc_middle_and_Lunatus",
        "S_oc_sup_and_transversal", "Pole_occipital",
    ],
}

ROIS_BRAIN_AREA: dict[str, list[str]] = {
    "ilfc": [
        "lateralorbitofrontal", "frontalpole", "parsopercularis",
        "parstriangularis", "parsorbitalis",
    ],
}

#: ROI -> figure label, in panel order.
FIGURE_ROIS: dict[str, str] = {
    "stc": "STG",
    "sts": "STS",
    "tp": "ATL",
    "sma": "IPL",
    "ilfc": "L/IFC",
    "occipital_visual": "OCC",
}


def in_roi(electrodes: pd.DataFrame, roi: str) -> pd.Series:
    """Rows of ``electrodes`` (with ``destrieux`` / ``desikan`` parcels) inside ``roi``."""
    if roi in ROIS_DETAILED:
        return electrodes["destrieux"].isin(ROIS_DETAILED[roi])
    if roi in ROIS_BRAIN_AREA:
        return electrodes["desikan"].isin(ROIS_BRAIN_AREA[roi])
    raise ValueError(f"unknown ROI {roi!r}; available: {sorted(FIGURE_ROIS)}")


@functools.lru_cache(maxsize=4)
def load_electrodes(table: Path) -> pd.DataFrame:
    """``electrodes.tsv``: one row per bipolar channel, e.g. ``P01``, ``EEG A1-EEG A2``."""
    return pd.read_csv(table, sep="\t", na_values="n/a")


def roi_electrodes(rois: tp.Sequence[str], table: Path) -> pd.DataFrame:
    """Electrode rows belonging to the union of ``rois``."""
    df = load_electrodes(table)
    mask = pd.Series(False, index=df.index)
    for roi in rois:
        mask |= in_roi(df, roi)
    return df.loc[mask].reset_index(drop=True)


def _bipolar_endpoints(channel: str) -> tuple[str, str]:
    """Split ``"EEG A1-EEG A2"`` into its two monopolar contact names."""
    anode, _, cathode = channel.partition("-")
    return anode.strip(), cathode.strip()


# The study renames every channel to "<name>_<participant>" (Evanson2023._load_raw)
# while the electrode table stores bare names, hence the suffixes below.


def roi_contacts(rois: tp.Sequence[str], table: Path) -> dict[str, tuple[str, ...]]:
    """Monopolar contacts per participant, e.g. ``{"P01": ("EEG A1_P01", …)}``."""
    out: dict[str, set[str]] = {}
    for row in roi_electrodes(rois, table).itertuples():
        anode, cathode = _bipolar_endpoints(row.channel)
        subject = str(row.participant_id)
        contacts = out.setdefault(subject, set())
        contacts.add(f"{anode}_{subject}")
        contacts.add(f"{cathode}_{subject}")
    return {subject: tuple(sorted(chans)) for subject, chans in out.items()}


def roi_referenced_channels(
    rois: tp.Sequence[str], table: Path
) -> dict[str, tuple[str, ...]]:
    """Bipolar channels per participant after referencing, e.g. ``"EEG A1_P01-EEG A2_P01"``."""
    out: dict[str, set[str]] = {}
    for row in roi_electrodes(rois, table).itertuples():
        anode, cathode = _bipolar_endpoints(row.channel)
        subject = str(row.participant_id)
        out.setdefault(subject, set()).add(f"{anode}_{subject}-{cathode}_{subject}")
    return {subject: tuple(sorted(chans)) for subject, chans in out.items()}


@functools.lru_cache(maxsize=None)
def _recording_channels(data: Path, subject: str) -> tuple[str, ...]:
    """Channel names of a participant's first chapter, as the study names them."""
    path = data / "annotations" / f"{subject}.yaml"
    if not path.exists():
        return ()
    info = yaml.safe_load(path.open())
    chapters = [k for k, v in info.items() if isinstance(v, dict) and "filename" in v]
    if not chapters:
        return ()
    study = Evanson2023(path=data)
    try:
        raw = study._load_raw({"subject": subject, "chapter": chapters[0]})
    except Exception:  # unreadable / missing recording: no coverage
        return ()
    return tuple(raw.ch_names)


def _same_probe(a: str, b: str) -> bool:
    """Probe test of ``IeegExtractor._apply_bipolar_ref``."""
    return [c for c in a if not c.isdigit()] == [c for c in b if not c.isdigit()]


def roi_covered_subjects(rois: tp.Sequence[str], table: Path, data: Path) -> frozenset[str]:
    """Participants whose recording yields at least one ROI bipolar channel.

    Replays the extractor: pick the ROI contacts present in the recording (in
    recording order), pair neighbours on the same probe, keep the ROI's pairs.
    A contact listed in the table may be absent from the recording, and a
    bipolar channel needs both endpoints.
    """
    contacts = roi_contacts(rois, table)
    pairs = roi_referenced_channels(rois, table)
    covered = set()
    for subject in pairs:
        wanted = set(contacts.get(subject, ()))
        picked = [ch for ch in _recording_channels(data, subject) if ch in wanted]
        derived = {f"{a}-{b}" for a, b in zip(picked, picked[1:]) if _same_probe(a, b)}
        if derived & set(pairs[subject]):
            covered.add(subject)
    return frozenset(covered)
