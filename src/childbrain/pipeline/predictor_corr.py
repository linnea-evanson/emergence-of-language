"""Cross-correlation between the TRF predictors, on the stimulus alone.

The TRF regressors of the 9 chapters (``EvansonStimSingle``) at 20 Hz, pooled
over chapters. Multi-dimensional families are collapsed where their
dimensions are not of interest:

* Mel spectrogram: mean over the 40 bands (broadband envelope);
* wav2vec 2.0: first component of its book-level PCA;
* Llama 3.1: first component of StandardScaler + PCA over the pooled chapters
  (averaging components would not do: their signs are arbitrary).
"""

from pathlib import Path

import neuralset as ns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import configs

#: (TRF key, family label, how its dimensions are collapsed)
PREDICTORS: list[tuple[str, str, str]] = [
    ("melspec", "Mel spectrogram", "mean"),
    ("phoneme_features", "Phoneme", "all"),
    ("wordfreq", "Zipf freq.", "all"),
    ("pos", "Part of speech", "all"),
    ("treedepth", "Tree depth", "all"),
    ("nopening", "Opening nodes", "all"),
    ("nclosing", "Closing nodes", "all"),
    ("deplength", "Dep. length", "all"),
    ("wordpos", "Word position", "all"),
    ("wav2vec_pca1", "wav2vec 2.0 0.1", "pc1"),
    ("wav2vec_pca8", "wav2vec 2.0 0.8", "pc1"),
    ("llama_pca1", "Llama 3.1 0.1", "pca"),
    ("llama_pca8", "Llama 3.1 0.8", "pca"),
]
DIM_LABELS = {
    "approximant": "Approx.", "fricative": "Fricative", "high": "High", "labial": "Labial",
    "low": "Low", "mid": "Mid", "nasal": "Nasal", "plosive": "Plosive", "trill": "Trill",
    "velar": "Velar", "voice": "Voice", "vowel": "Vowel",
    "ADJ": "Adj", "ADP": "Adp", "ADV": "Adv", "CONJ": "Conj", "DET": "Det",
    "NOUN": "Noun", "PRON": "Pron", "VERB": "Verb",
}


def compute(out: Path) -> None:
    """Save ``R``, ``labels``, ``families`` and ``n_samples`` to ``out`` (npz)."""
    chain = configs.chain("EvansonStimSingle")
    events = ns.Chain(steps=chain["steps"], infra=chain["infra"]).run()
    segmenter = ns.dataloader.Segmenter(
        start=0.0,
        duration=None,
        trigger_query='type=="Audio"',
        extractors={key: configs.trf_features[key]["extractor"] for key, _, _ in PREDICTORS},
        drop_incomplete=False,
    )
    dset = segmenter.apply(events)
    timelines = dset.triggers.timeline.to_numpy()
    # once: a second prepare() fails once the Llama extractors hold tensors
    dset.prepare()

    chapters: dict[str, list[np.ndarray]] = {key: [] for key, _, _ in PREDICTORS}
    for timeline in sorted(set(timelines)):
        batch = dset.select(timelines == timeline).load_all(num_workers=1)
        n_times = {v.shape[-1] for v in batch.data.values()}
        if len(n_times) != 1:
            raise RuntimeError(f"{timeline}: extractors disagree on n_times {n_times}")
        for key in chapters:
            chapters[key].append(batch.data[key].numpy()[0].astype(np.float32))

    rows, labels, families = [], [], []
    for key, family, collapse in PREDICTORS:
        X = np.concatenate(chapters[key], axis=1)
        if collapse == "mean":
            X, names = X.mean(0, keepdims=True), [family]
        elif collapse == "pc1":
            X, names = X[:1], [f"{family} (PC1)"]
        elif collapse == "pca":
            X = PCA(1).fit_transform(StandardScaler().fit_transform(X.T)).T
            names = [f"{family} (PC1)"]
        elif len(X) == 1:
            names = [family]
        else:
            names = [DIM_LABELS[name] for name in configs.TRF_DIM_NAMES[key]]
        rows.append(X)
        labels += names
        families += [family] * len(X)
    X = np.concatenate(rows)
    keep = X.std(1) > 0
    X = X[keep]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        R=np.corrcoef(X),
        labels=np.array(labels)[keep],
        families=np.array(families)[keep],
        n_samples=X.shape[1],
    )
