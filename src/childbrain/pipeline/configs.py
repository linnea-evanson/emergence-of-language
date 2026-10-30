"""Configurations of the paper's analyses: study chain, extractors, presets.

Paths and compute come from the environment:

* ``CHILDBRAIN_DATA``: the raw data folder (``participants.tsv``, ``annotations/``,
  ``neuro/``, the stimulus);
* ``CHILDBRAIN_CACHE``: where every intermediate result is cached;
* ``CHILDBRAIN_ELECTRODES``: electrode table of the ROIs (default
  ``$CHILDBRAIN_DATA/electrodes.tsv``);
* ``CHILDBRAIN_CLUSTER``: ``slurm`` to submit jobs (default: run in-process),
  with ``CHILDBRAIN_SLURM_PARTITION`` / ``CHILDBRAIN_SLURM_ACCOUNT`` if needed.

Extractors are plain dicts, instantiated by :mod:`childbrain.pipeline.run`.
"""

import os
import typing as tp
from copy import deepcopy
from pathlib import Path

import numpy as np

from .extractors import PHONEME_FEATURE_COLS
from .transforms import POS_TAGS

DATA = Path(os.environ.get("CHILDBRAIN_DATA", "data"))
CACHE = Path(os.environ.get("CHILDBRAIN_CACHE", "cache"))
ELECTRODES = Path(os.environ.get("CHILDBRAIN_ELECTRODES", DATA / "electrodes.tsv"))
CLUSTER: str | None = os.environ.get("CHILDBRAIN_CLUSTER") or None

LLAMA_MODEL_ID = "meta-llama/Llama-3.1-8B"
WAV2VEC_MODEL_ID = "facebook/wav2vec2-base"
FREQUENCY = 20.0
N_SPLITS = 10
#: Timelines (chapters) with fewer words are left out.
MIN_WORDS = 50

# ============== infra ==============

_slurm = {
    key: value
    for key, value in {
        "slurm_partition": os.environ.get("CHILDBRAIN_SLURM_PARTITION"),
        "slurm_account": os.environ.get("CHILDBRAIN_SLURM_ACCOUNT"),
    }.items()
    if value
}
infra_study = {"backend": "Cached", "folder": str(CACHE / "study"), "mode": "retry"}
infra_timelines = {
    "folder": str(CACHE / "study"),
    "cluster": CLUSTER,
    **_slurm,
    "cpus_per_task": 4,
    "timeout_min": 60,
    "min_samples_per_job": 10,
    "max_jobs": 30,
    "mode": "cached",
}
infra_syntax = {"folder": str(CACHE / "study"), "mode": "cached"}
infra_features = {
    "folder": str(CACHE / "features"),
    "cluster": CLUSTER,
    **_slurm,
    "gpus_per_node": 1,
    "cpus_per_task": 20,
    "timeout_min": 60,
    "min_samples_per_job": 200,
    "mode": "cached",
}
infra_neuro = {"folder": str(CACHE / "neuro"), "mode": "cached"}
infra_llama = {
    "folder": str(CACHE / "features"),
    "cluster": CLUSTER,
    **_slurm,
    "gpus_per_node": 1,
    "cpus_per_task": 40,
    "timeout_min": 60,
    "min_samples_per_job": 200,
}
#: Wav2VecLayer runs the whole book in one call, so that its PCA sees every chapter.
infra_wav2vec = {**infra_features, "timeout_min": 180}
#: One decoding / encoding / evoked job per subject; ``folder`` is set per analysis.
infra_exp = {
    "cluster": CLUSTER,
    **_slurm,
    "cpus_per_task": 40,
    "timeout_min": 180,
    "max_jobs": 150,
    "min_samples_per_job": 1,
    "mode": "cached",
}

# ============== study ==============

steps: dict[str, dict] = {
    "study": {
        "name": "Evanson2023",
        "path": str(DATA),
        "infra_timelines": infra_timelines,
        "infra": infra_study,
    },
    "CleanTextNewlines": {"name": "CleanTextNewlines", "infra": infra_study},
    "NormalizeNumbers": {"name": "NormalizeNumbers", "language": "french", "infra": infra_study},
    "AddSentenceToWords": {
        "name": "AddSentenceToWords",
        "max_unmatched_ratio": 0.05,
        "infra": infra_study,
    },
    "RemoveMissingSentenceChar": {"name": "RemoveMissingSentenceChar", "infra": infra_study},
    "AddConcatenationContext": {
        "name": "AddConcatenationContext",
        "sentence_only": False,
        "max_context_len": 1024,
        "split_field": "chapter",
        "infra": infra_study,
    },
    "RemoveMissing": {"name": "RemoveMissing", "infra": infra_study},
    "RemoveMissingSentence": {"name": "RemoveMissing", "field": "sentence", "infra": infra_study},
    "SyntaxEnhancer": {
        "name": "SyntaxEnhancer",
        "benepar_model": "benepar_fr2",
        "benepar_device": "cuda",
        "filter_failed": True,
        "infra": infra_study,
        "timeline_infra": infra_syntax,
    },
    "AddPOS": {"name": "AddPOS", "language": "french", "infra": infra_study},
    "RemoveMissingPOS": {
        "name": "RemoveMissing",
        "event_types": "Word",
        "field": "pos_tag",
        "infra": infra_study,
    },
}


def chain(study: str = "Evanson2023") -> dict:
    """The study chain; ``EvansonStimSingle`` is the stimulus alone (one 'subject')."""
    return {
        "name": "Chain",
        "steps": {**steps, "study": {**steps["study"], "name": study}},
        "infra": infra_study,
    }


# ============== neuro ==============

ieeg = {
    "name": "RoiIeegExtractor",
    "frequency": FREQUENCY,
    "filter": [0.05, 20.0],
    "notch_filter": 50.0,
    "baseline": None,
    "scaler": "StandardScaler",
    "reference": "bipolar",
    "picks": ["seeg"],
    "channel_order": "original",
    "aggregation": "mean",
    "infra": infra_neuro,
}
#: Evoked responses: notch at 50 Hz harmonics (those above Nyquist are skipped), in volts.
ieeg_evoked = {**ieeg, "notch_filter": [50.0, 100.0, 150.0, 200.0], "scaler": None}

# ============== linguistic features ==============

melspec_base = {"name": "MelSpectrum", "frequency": FREQUENCY, "infra": infra_features}
phoneme_features_base = {"name": "PhonemeFeatures", "frequency": FREQUENCY, "aggregation": "trigger"}
wordfreq_base = {
    "name": "ContentWordFrequency",
    "event_types": "Word",
    "aggregation": "trigger",
    "language": "french",
    "frequency": FREQUENCY,
    "infra": infra_features,
}
pos_base = {"name": "POSFeatures", "aggregation": "trigger", "frequency": FREQUENCY}
treedepth_base = {"name": "DependencyDepthExtractor", "aggregation": "trigger", "frequency": FREQUENCY}
wordpos_base = {"name": "WordPositionExtractor", "aggregation": "trigger", "frequency": FREQUENCY}
nclosing_base = {"name": "NClosingExtractor", "aggregation": "trigger", "frequency": FREQUENCY}
nopening_base = {"name": "NOpeningExtractor", "aggregation": "trigger", "frequency": FREQUENCY}
deplength_base = {"name": "DependencyLengthExtractor", "aggregation": "trigger", "frequency": FREQUENCY}


def _static(event_type: str, extractor: dict, dim_names: list[str] | None = None) -> dict:
    """Decoding target: the extractor's value at each ``event_type`` onset."""
    out: dict[str, tp.Any] = {"event_type": event_type, "extractor": {**extractor, "frequency": 0.0}}
    if dim_names is not None:
        out["dim_names"] = dim_names
    return out


linguistic_features = {
    "melspec": {
        "event_type": "Word",
        "extractor": {
            "name": "ToStatic",
            "event_types": "Word",
            "frequency": FREQUENCY,
            "extractor": melspec_base,
        },
    },
    "phoneme_features": _static("Phoneme", phoneme_features_base, PHONEME_FEATURE_COLS),
    "wordfreq": _static("Word", wordfreq_base),
    "pos": _static("Word", pos_base, POS_TAGS),
    "treedepth": _static("Word", treedepth_base),
    "wordpos": _static("Word", wordpos_base),
    "nclosing": _static("Word", nclosing_base),
    "nopening": _static("Word", nopening_base),
    "deplength": _static("Word", deplength_base),
}

# ============== model embeddings ==============
#
# wav2vec 2.0: "0.1" is conv block 2, "0.8" transformer hidden state 8; the
# book-level StandardScaler + IncrementalPCA(50) of Wav2VecLayer.
# Llama 3.1: hidden states 3 ("0.1") and 26 ("0.8") of 33, i.e. layers=0.1/0.8
# of cache_n_layers=11 equidistant ones, so that every depth shares one cache.
# Random networks are seeded so that every job builds the same one.

N_LAYERS = 11
LAYER_GRID = [i / 10 for i in range(N_LAYERS)]
W2V_LAYERS: dict[str, tuple[str, int]] = {"1": ("conv", 2), "8": ("trans", 8)}
LLAMA_LAYERS: dict[str, float] = {"1": 0.1, "8": 0.8}
_W2V_N_CONV = 7
_W2V_N_BLOCKS = 20


def _tag(depth: float) -> str:
    return str(int(round(depth * 10)))


def _suffix(pretrained: bool) -> str:
    return "" if pretrained else "_untrained"


def _w2v_block_from_depth(depth: float) -> tuple[str, int]:
    """Relative depth over the 20 blocks (7 conv, 13 transformer states): 0.1 → conv 2."""
    idx = int(round(depth * (_W2V_N_BLOCKS - 1)))
    return ("conv", idx) if idx < _W2V_N_CONV else ("trans", idx - _W2V_N_CONV)


def wav2vec(
    block_layers: list[tuple[str, int]],
    *,
    pretrained: bool = True,
    n_components: int | None = 50,
    frequency: float | str = FREQUENCY,
) -> dict:
    return {
        "name": "Wav2VecLayer",
        "model_name": WAV2VEC_MODEL_ID,
        "block_layers": block_layers,
        "pretrained": pretrained,
        "n_components": n_components,
        "frequency": frequency,
        "infra": infra_wav2vec,
    }


def llama(layer: float, *, pretrained: bool = True, **overrides: tp.Any) -> dict:
    ext = {
        "name": "HuggingFaceText",
        "model_name": LLAMA_MODEL_ID,
        "frequency": FREQUENCY,
        "contextualized": True,
        "layers": layer,
        "layer_aggregation": None,
        "cache_n_layers": N_LAYERS,
        "device": "accelerate",
        "infra": infra_llama,
        "aggregation": "mean",
        **overrides,
    }
    if not pretrained:
        ext.update(name="SeededHuggingFaceText", pretrained=False, init_seed=0)
    return ext


def _wav2vec_feature(
    block_layer: tuple[str, int], pretrained: bool = True, frequency: float | str = "native"
) -> dict:
    """PCA'd embedding at the word onset, at the block's own rate or at 20 Hz."""
    inner = wav2vec([block_layer], pretrained=pretrained, frequency=frequency)
    if frequency != "native":
        inner["allow_missing"] = True  # same cache as the TRF regressor
    return {
        "event_type": "Word",
        "extractor": {
            "name": "ToStatic",
            "extractor": inner,
            "event_types": "Audio",
            "frequency": FREQUENCY,
        },
    }


def _llama_feature(layer: float, pretrained: bool = True) -> dict:
    """Word embedding, StandardScaler + PCA(50) over the participant's words."""
    return {
        "event_type": "Word",
        "extractor": {
            "name": "ScaledExtractorPCA",
            "extractor": llama(layer, pretrained=pretrained, aggregation="trigger", frequency=0.0),
            "n_components": 50,
            "whiten": False,
            "infra": {"folder": str(CACHE / "features")},
        },
    }


model_features = {
    "wav2vec_pca1": _wav2vec_feature(W2V_LAYERS["1"]),
    "wav2vec_pca1_untrained": _wav2vec_feature(W2V_LAYERS["1"], pretrained=False),
    "wav2vec_pca8": _wav2vec_feature(W2V_LAYERS["8"]),
    "llama_pca1": _llama_feature(0.1),
    "llama_pca1_untrained": _llama_feature(0.1, pretrained=False),
    "llama_pca8": _llama_feature(0.8),
}
#: wav2vec at the rate of the neural data; the paper's wav2vec decoding.
wav2vec_20hz_features = {
    f"wav2vec_pca{tag}{_suffix(pretrained)}_20hz": _wav2vec_feature(
        block_layer, pretrained=pretrained, frequency=FREQUENCY
    )
    for tag, block_layer in W2V_LAYERS.items()
    for pretrained in (True, False)
}
layer_features = {
    **{
        f"wav2vec_depth{_tag(d)}_20hz": _wav2vec_feature(
            _w2v_block_from_depth(d), frequency=FREQUENCY
        )
        for d in LAYER_GRID
    },
    **{f"llama_depth{_tag(d)}": _llama_feature(d) for d in LAYER_GRID},
}
#: Decoding targets of the stimulus-side (feature-to-feature) analysis.
hierarchy_features = {
    **linguistic_features,
    "llama_pca1": model_features["llama_pca1"],
    "llama_pca8": model_features["llama_pca8"],
    "wav2vec_pca1_20hz": wav2vec_20hz_features["wav2vec_pca1_20hz"],
    "wav2vec_pca8_20hz": wav2vec_20hz_features["wav2vec_pca8_20hz"],
}

# ============== decoding ==============

EPOCH_START, EPOCH_STOP = -2.0, 3.0
#: Ridge input at t: the neuro at t-1, t, t+1 (2 samples at the epoch edges).
NEURO_TIME_RADIUS = 1
ridge = {"alphas": np.logspace(-2, 7, 10).tolist(), "alpha_per_target": False}

#: Decoding feature sets; "_20hz" and "best" sets complete the wav2vec rows and the ROIs.
DECODING_SETS: dict[str, dict[str, dict]] = {
    "linguistic": linguistic_features,
    "layers": layer_features,
    "models": model_features,
    "models_20hz": wav2vec_20hz_features,
    "models_best": {k: model_features[k] for k in ("wav2vec_pca8", "llama_pca1")},
    "models_best_20hz": {"wav2vec_pca8_20hz": wav2vec_20hz_features["wav2vec_pca8_20hz"]},
    "hierarchy": hierarchy_features,
}


def _as_neuro(extractor: dict) -> dict:
    """A feature as a 20 Hz time series, the "neuro" of stimulus-side decoding."""
    neuro = deepcopy(extractor)
    if neuro.get("aggregation") == "trigger":
        neuro["aggregation"] = "mean"
    return neuro


#: Decoding sources: the recordings, or a stimulus feature (on ``EvansonStimSingle``).
neuros: dict[str, dict] = {
    "ieeg": ieeg,
    "ieeg_evoked": ieeg_evoked,
    "melspec": melspec_base,
    "phoneme_features": _as_neuro(phoneme_features_base),
    "wordfreq": _as_neuro(wordfreq_base),
    "pos": _as_neuro(pos_base),
    "treedepth": _as_neuro(treedepth_base),
    "wordpos": _as_neuro(wordpos_base),
    "nclosing": _as_neuro(nclosing_base),
    "nopening": _as_neuro(nopening_base),
    "deplength": _as_neuro(deplength_base),
    **{
        f"wav2vec_pca{tag}": {**wav2vec([block_layer]), "allow_missing": True}
        for tag, block_layer in W2V_LAYERS.items()
    },
    **{
        f"llama_pca{tag}": {
            "name": "ScaledExtractorPCA",
            "extractor": llama(layer),
            "n_components": 50,
            "whiten": False,
            "infra": {"folder": str(CACHE / "features")},
        }
        for tag, layer in LLAMA_LAYERS.items()
    },
}
IEEG_NEUROS = ("ieeg", "ieeg_evoked")

# ============== encoding (TRF) ==============

TRF_TMIN, TRF_TMAX = -0.2, 1.0
#: Chapters are cut into contiguous chunks so that the non-shuffled KFold over
#: chunks splits each chapter into contiguous blocks of time.
TRF_SEGMENT_DURATION = 10.0
TRF_ALPHA_HOLDOUT = 0.2
TRF_N_PERMUTATIONS = 1000

TRF_DIM_NAMES: dict[str, list[str]] = {
    "phoneme_features": list(PHONEME_FEATURE_COLS),
    "wordfreq": ["zipfreq"],
    "pos": list(POS_TAGS),
    "treedepth": ["treedepth"],
    "wordpos": ["wordpos"],
    "nclosing": ["n_closing"],
    "nopening": ["n_opening"],
    "deplength": ["deplength"],
}
#: Nested feature sets, each fitted as one design matrix.
TRF_FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "phoneme_word": ("phoneme_features", "wordfreq", "pos"),
    "phoneme_word_supra_parser": (
        "phoneme_features", "wordfreq", "pos",
        "treedepth", "wordpos", "nclosing", "nopening", "deplength",
    ),
}


def _onset_impulse(extractor: dict) -> dict:
    """The feature vector at the event onset sample, zero elsewhere."""
    inner = {**deepcopy(extractor), "frequency": FREQUENCY, "aggregation": "sum", "allow_missing": True}
    return {
        "name": "OnsetImpulse",
        "extractor": inner,
        "frequency": FREQUENCY,
        "aggregation": "sum",
        "allow_missing": True,
    }


def _concat(extractors: list[dict]) -> dict:
    return {
        "name": "ConcatExtractor",
        "extractors": extractors,
        "extractor_aggregation": "cat",
        "frequency": FREQUENCY,
        "aggregation": "sum",
        "allow_missing": True,
    }


def _trf_features() -> dict[str, dict]:
    """TRF regressors (``event_type`` is metadata: segments are cut from Audio)."""
    out: dict[str, dict] = {"melspec": {"event_type": "Audio", "extractor": melspec_base}}
    for key in TRF_DIM_NAMES:
        cfg = linguistic_features[key]
        out[key] = {"event_type": cfg["event_type"], "extractor": _onset_impulse(cfg["extractor"])}
    for name, members in TRF_FEATURE_SETS.items():
        out[name] = {"event_type": "Word", "extractor": _concat([out[m]["extractor"] for m in members])}

    # wav2vec: continuous from the audio onset; Llama: each word's embedding held
    # over the word. chapter_pca: StandardScaler + (Incremental)PCA(50) per chapter.
    for tag, block_layer in W2V_LAYERS.items():
        for pretrained in (True, False):
            out[f"wav2vec_pca{tag}{_suffix(pretrained)}"] = {
                "event_type": "Audio",
                "extractor": {**wav2vec([block_layer], pretrained=pretrained), "allow_missing": True},
            }
    out["wav2vec_pca1_8"] = {
        "event_type": "Audio",
        "extractor": {**wav2vec(list(W2V_LAYERS.values()), n_components=None), "allow_missing": True},
        "chapter_pca": {"n_components": 50, "incremental": True},
    }

    def llama_regressor(layer: float, pretrained: bool = True) -> dict:
        return {**llama(layer, pretrained=pretrained, layer_aggregation="mean"), "allow_missing": True}

    llama_pca = {"n_components": 50, "incremental": False}
    for tag, layer in LLAMA_LAYERS.items():
        for pretrained in (True, False):
            out[f"llama_pca{tag}{_suffix(pretrained)}"] = {
                "event_type": "Word",
                "extractor": llama_regressor(layer, pretrained),
                "chapter_pca": llama_pca,
            }
    out["llama_pca1_8"] = {
        "event_type": "Word",
        "extractor": _concat([llama_regressor(0.1), llama_regressor(0.8)]),
        "chapter_pca": llama_pca,
    }
    return out


trf_features = _trf_features()
TRF_SETS: dict[str, list[str]] = {
    "linguistic": ["phoneme_features", "phoneme_word", "phoneme_word_supra_parser"],
    "models": [
        f"{model}_pca{tag}"
        for model in ("wav2vec", "llama")
        for tag in ("1", "1_untrained", "8", "8_untrained", "1_8")
    ],
}

# ============== evoked responses ==============

EVOKED_TMIN, EVOKED_TMAX = -1.0, 2.0
#: Decoding feature whose events are epoched (its values are not used).
EVOKED_EVENTS: dict[str, str] = {"word": "wordpos", "phoneme": "phoneme_features"}
