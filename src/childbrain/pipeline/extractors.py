"""Feature and neuro extractors of the paper, on top of neuralset's."""

import logging
import tempfile
import typing as tp
from pathlib import Path

import exca
import librosa
import mne
import neuralset as ns
import numpy as np
import pandas as pd
import pydantic
import scipy.signal
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from mne._fiff.pick import _VALID_CHANNEL_TYPES
from neuralset import utils as nsutils
from neuralset.base import TimedArray
from neuralset.extractors.base import BaseStatic
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import gen_batches
from tqdm import tqdm
from transformers import (
    AutoConfig,
    Wav2Vec2Config,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
)

from .rois import roi_contacts, roi_referenced_channels
from .transforms import POS_TAGS, pos_onehot

logger = logging.getLogger(__name__)

# ============================================================
# Linguistic features
# ============================================================


class DependencyDepthExtractor(BaseStatic):
    """Extracts the dependency-tree depth for each Word event.

    Parses ``event.sentence`` with spaCy, locates the token by
    ``event.sentence_char`` (character offset), and returns its depth in
    the dependency tree (0 = root). Raises ``ValueError`` if
    ``sentence_char`` is missing or matches no token.
    """

    event_types: tp.Literal["Word"] = "Word"
    language: str = "french"

    _nlp_cache: tp.ClassVar[dict[str, tp.Any]] = {}
    _doc_cache: tp.ClassVar[dict[tuple[str, str], tp.Any]] = {}

    @staticmethod
    def _token_depth(token: tp.Any) -> int:
        depth = 0
        while token.head != token:
            token = token.head
            depth += 1
        return depth

    def _nlp(self, language: str) -> tp.Any:
        if language not in self._nlp_cache:
            self._nlp_cache[language] = nsutils.get_spacy_model(language=language)
        return self._nlp_cache[language]

    def _doc(self, language: str, sentence: str) -> tp.Any:
        key = (language, sentence)
        if key not in self._doc_cache:
            nlp = self._nlp(language)
            disable = [c for c in ("ner", "lemmatizer") if c in nlp.pipe_names]
            self._doc_cache[key] = nlp(sentence, disable=disable)
        return self._doc_cache[key]

    @staticmethod
    def _find_token_by_char(doc: tp.Any, char_idx: int) -> tp.Any:
        """Return the token at ``char_idx``, or the token whose span contains it."""
        for tok in doc:
            if tok.idx == char_idx:
                return tok
        for tok in doc:
            if tok.idx < char_idx < tok.idx + len(tok.text):
                return tok
        return None

    def get_static(self, event: tp.Any) -> torch.Tensor:
        if event.sentence_char is None:
            raise ValueError(f"Word {event.text!r} has no sentence_char")
        language = getattr(event, "language", None) or self.language
        sentence = event.sentence or event.text
        doc = self._doc(language, sentence)
        token = self._find_token_by_char(doc, int(event.sentence_char))
        if token is None:
            raise ValueError(
                f"No token at or containing sentence_char={event.sentence_char!r} "
                f"in sentence {sentence!r} for word {event.text!r}."
            )
        return torch.tensor([float(self._token_value(token))], dtype=torch.float32)

    def _token_value(self, token: tp.Any) -> float:
        return self._token_depth(token)


class DependencyLengthExtractor(DependencyDepthExtractor):
    """Distance, in tokens, from each Word to its dependency head (0 = root)."""

    def _token_value(self, token: tp.Any) -> float:
        return abs(token.i - token.head.i)


class WordPositionExtractor(BaseStatic):
    """Normalised character position of a word in its sentence, ``sentence_char / len(sentence)``."""

    event_types: tp.Literal["Word"] = "Word"

    def get_static(self, event: tp.Any) -> torch.Tensor:
        if event.sentence_char is None:
            raise ValueError(f"Word {event.text!r} has no sentence_char")
        sentence = event.sentence or event.text
        n = max(len(sentence), 1)
        pos = float(event.sentence_char) / n
        return torch.tensor([pos], dtype=torch.float32)


CONTENT_WORDS_QUERY: str = "pos_tag.isin(['NOUN', 'VERB', 'ADJ', 'ADV'])"


class ContentWordFrequency(ns.extractors.WordFrequency):
    """Zipf word frequency, of content words only.

    ``event_query`` filters the trigger events of the decoding epochs.
    """

    event_query: str | None = CONTENT_WORDS_QUERY


class POSFeatures(BaseStatic):
    """8-d binary POS features (``pd.get_dummies`` of the collapsed tags)."""

    event_types: tp.Literal["Word"] = "Word"
    feature_names: list[str] = POS_TAGS
    pos_field: str = "pos_tag"

    def get_static(self, event: tp.Any) -> torch.Tensor:  # type: ignore[override]
        tag = event._get_field_or_extra(self.pos_field)
        return torch.from_numpy(pos_onehot(tag))


PHONEME_FEATURE_COLS: list[str] = [
    "low",
    "high",
    "mid",
    "labial",
    "velar",
    "vowel",
    "plosive",
    "fricative",
    "nasal",
    "approximant",
    "trill",
    "voice",
]


class PhonemeFeatures(BaseStatic):
    """Return the 12-d articulatory feature vector for a Phoneme event."""

    event_types: tp.Literal["Phoneme"] = "Phoneme"
    feature_names: list[str] = PHONEME_FEATURE_COLS

    def get_static(self, event: tp.Any) -> torch.Tensor:  # type: ignore[override]
        values = [
            float(event._get_field_or_extra(feat) or 0.0)
            for feat in self.feature_names
        ]
        return torch.tensor(values, dtype=torch.float32)


class NClosingExtractor(BaseStatic):
    """Scalar ``n_closing`` from Word events (set by ``SyntaxEnhancer`` in the chain)."""

    event_types: tp.Literal["Word"] = "Word"
    _field: tp.ClassVar[str] = "n_closing"

    def get_static(self, event: tp.Any) -> torch.Tensor:
        val = event._get_field_or_extra(self._field)
        if val is None or pd.isna(val):
            raise ValueError(f"Word {event.text!r} has no {self._field}")
        return torch.tensor([float(val)], dtype=torch.float32)


class NOpeningExtractor(NClosingExtractor):
    """Scalar ``n_opening`` from Word events: constituents this word opens."""

    _field: tp.ClassVar[str] = "n_opening"


# ============================================================
# iEEG
# ============================================================

_CHANNEL_TYPE_PICKS = {"all", "data", "meg", "ref_meg"} | set(_VALID_CHANNEL_TYPES)


class RoiIeegExtractor(ns.extractors.IeegExtractor):
    """iEEG, optionally restricted to the bipolar channels of anatomical ROIs.

    Selection happens in two steps, because ``picks`` applies to the monopolar
    raw while the ROI is defined on bipolar channels:

    1. ``_picks_for_subject`` picks the monopolar endpoints of the subject's
       ROI bipolar channels.
    2. ``_apply_bipolar_ref`` keeps only the ROI's bipolar channels. Without
       this, referencing bridges gaps in the ROI (contacts ``A3`` and ``A8``
       with nothing picked in between give a spurious ``A3-A8`` whose midpoint
       lies outside the ROI).

    With ``rois=()`` this is ``IeegExtractor``.
    """

    rois: tuple[str, ...] = ()
    roi_table: Path | None = None

    _roi_subject: str | None = pydantic.PrivateAttr(None)

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if self.rois:
            if self.roi_table is None or not self.roi_table.exists():
                raise ValueError(f"electrode table not found: {self.roi_table}")
            if self.reference != "bipolar":
                raise ValueError(f"ROIs need reference='bipolar', got {self.reference!r}")

    def _picks_for_subject(self, subject: str) -> tuple[str, ...]:
        if not self.rois:
            return self.picks
        assert self.roi_table is not None
        contacts = roi_contacts(self.rois, self.roi_table).get(subject, ())
        if not contacts:
            raise ValueError(f"{subject} has no contacts in ROI(s) {self.rois}")
        return contacts

    def _preprocess_raw(  # type: ignore[override]
        self, raw: mne.io.BaseRaw, event: tp.Any
    ) -> ns.extractors.neuro.MneTimedArray:
        subject = event.subject.rsplit("/", 1)[-1]
        self._roi_subject = subject
        raw = self._pick_channels(raw, self._picks_for_subject(subject))

        if self.drop_bads:
            raw.load_data()
            raw = raw.drop_channels(raw.info["bads"])

        if self.reference == "bipolar":
            raw.load_data()
            raw = self._apply_bipolar_ref(raw)

        # MneRaw picks again: keep the (bipolar) channels derived above
        original_picks = self.picks
        object.__setattr__(self, "picks", tuple(raw.ch_names))
        try:
            return ns.extractors.neuro.MneRaw._preprocess_raw(self, raw, event)
        finally:
            object.__setattr__(self, "picks", original_picks)

    def _pick_channels(  # type: ignore[override]
        self, raw: mne.io.BaseRaw, picks: tuple[str, ...]
    ) -> mne.io.BaseRaw:
        """Pick channel types, or the listed channels present, in the raw's own order.

        ``raw.pick(names)`` orders by ``names``, which are string-sorted
        (``A1, A10, A2``), while ``_apply_bipolar_ref`` pairs neighbours and
        assumes probe-ascending order.
        """
        if all(p in _CHANNEL_TYPE_PICKS for p in picks):
            return raw.pick(list(picks), verbose=False)
        index = {ch: i for i, ch in enumerate(raw.ch_names)}
        present = [ch for ch in picks if ch in index]
        if not present:
            raise ValueError(f"No channels matched picks {picks!r}")
        raw = raw.pick(present, verbose=False)
        ordered = sorted(raw.ch_names, key=index.__getitem__)
        if ordered != list(raw.ch_names):
            raw = raw.reorder_channels(ordered)
        return raw

    def _apply_bipolar_ref(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        if self.rois and len(raw.ch_names) < 2:
            raise ValueError(
                f"{self._roi_subject}: only {raw.ch_names} survived "
                f"picking for ROI(s) {self.rois}, need >=2 for a bipolar channel"
            )
        raw = super()._apply_bipolar_ref(raw)
        if not self.rois:
            return raw
        assert self.roi_table is not None and self._roi_subject is not None
        keep_set = set(
            roi_referenced_channels(self.rois, self.roi_table).get(self._roi_subject, ())
        )
        keep = [ch for ch in raw.ch_names if ch in keep_set]
        if not keep:
            raise ValueError(
                f"{self._roi_subject}: no ROI bipolar channel "
                f"survived referencing for ROI(s) {self.rois}"
            )
        return raw.pick(keep, verbose=False)


# ============================================================
# TRF regressors
# ============================================================


class ConcatExtractor(ns.extractors.meta.AggregatedExtractor):
    """Concatenate extractors into one regressor with a concrete frequency.

    A feature set's columns are fitted in a single design matrix.
    ``AggregatedExtractor`` already concatenates, but reports
    ``frequency="native"``, while the TRF requires the extractor and the neuro
    to declare the same frequency.
    """

    frequency: float  # type: ignore[assignment]

    def _get_timed_arrays(
        self, events: tp.Any, start: float, duration: float
    ) -> tp.Iterable[TimedArray]:
        raise NotImplementedError(
            "ConcatExtractor combines whole extractor outputs, so it cannot be "
            "wrapped in OnsetImpulse; concatenate OnsetImpulse extractors instead."
        )


class OnsetImpulse(ns.extractors.base.BaseExtractor):
    """Wrap an extractor so each event becomes a one-sample impulse at its onset.

    A zero signal with the event's feature vector written at the onset sample
    only, ``feature_signal[onsets] = values``. Without it, a static extractor
    spans ``event.duration``, i.e. a box-car over the whole word.
    """

    extractor: ns.extractors.base.BaseExtractor
    event_types: str | tuple[str, ...] = "Event"
    aggregation: tp.Literal["sum", "mean"] = "sum"

    def model_post_init(self, context: tp.Any) -> None:
        self.event_types = self.extractor.event_types
        super().model_post_init(context)
        if not self.frequency:
            raise ValueError("OnsetImpulse needs a non-zero frequency")

    def prepare(self, obj: tp.Any) -> None:
        self.extractor.prepare(obj)
        # Also populates ``_missing_default``, the zeros of a segment without
        # events, which each DataLoader worker would otherwise lack.
        super().prepare(obj)

    def _get_timed_arrays(
        self, events: tp.Any, start: float, duration: float
    ) -> tp.Iterable[TimedArray]:
        step = 1.0 / float(self.frequency)
        stop = start + duration
        emitted = False
        template = None

        for tarray in self.extractor._get_timed_arrays(
            events=events, start=start, duration=duration
        ):
            if template is None:
                template = tarray
            # A word overlapping this segment but starting before it belongs to
            # the segment holding its onset.
            if not start <= tarray.start < stop:
                continue
            tarray.duration = step
            emitted = True
            yield tarray

        if not emitted:
            if template is None:
                return
            # Every onset is elsewhere: all zeros, with the feature shape.
            yield TimedArray(
                frequency=0,
                start=start,
                duration=step,
                data=np.zeros_like(template.data),
            )


# ============================================================
# Model embeddings
# ============================================================


class SeededHuggingFaceText(ns.extractors.HuggingFaceText):
    """HuggingFaceText with a seeded random initialisation (``pretrained=False``).

    Seeding keeps the random network identical across jobs filling one cache.
    """

    init_seed: int = 0

    def load_model(self) -> torch.nn.Module:
        if self.pretrained:
            return super().load_model()
        # Weights are initialised on the CPU before the model is moved to its device.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.init_seed)
            return self._load_random_model()

    def _load_random_model(self) -> torch.nn.Module:
        hf_config = self.hf_config
        Model = hf_config.model_cls(self.model_name)
        model_kwargs = hf_config.model_kwargs or {}
        config_kwargs: dict[str, tp.Any] = {
            "output_hidden_states": True,
            "local_files_only": True,
        }
        if "revision" in model_kwargs:
            config_kwargs["revision"] = model_kwargs["revision"]
        config = AutoConfig.from_pretrained(self.model_name, **config_kwargs)
        if hasattr(Model, "from_config"):
            model = Model.from_config(config, **model_kwargs)
        else:
            model = Model._from_config(config, **model_kwargs)
        if isinstance(self.dtype, str) and self.dtype != "auto":
            model.to(dtype=getattr(torch, self.dtype))
        if self.device == "accelerate":
            model.to("cuda" if torch.cuda.is_available() else "cpu")
        else:
            model.to(self.device)
        model.eval()
        return model

    def _aggregate_layers(self, latents: np.ndarray) -> np.ndarray:
        # the cached latents are a memmap, which must be loaded before indexing
        return super()._aggregate_layers(np.asarray(latents))


class ScaledExtractorPCA(ns.extractors.meta.ExtractorPCA):
    """ExtractorPCA with a StandardScaler before the PCA, static or dynamic.

    ``make_pipeline(StandardScaler(), PCA(50, whiten=False))``. A dynamic
    extractor's time points are independent samples of the same space.
    """

    def prepare(self, obj: tp.Any) -> None:
        pca_events = self._prepare_pca_uid(obj)
        if not pca_events:
            return
        events = list(pca_events.values())
        self.extractor.prepare(events)
        start = min(e.start for e in events)
        duration = max(e.start + e.duration for e in events) - start
        tas = list(
            self.extractor._get_timed_arrays(events, start=start, duration=duration)
        )
        pca = make_pipeline(
            StandardScaler(), PCA(n_components=self.n_components, whiten=self.whiten)
        )
        if any(ta.frequency for ta in tas):
            feat_dim = tas[0].data.shape[:-1]
            flat_feat = int(np.prod(feat_dim))
            pca.fit(np.concatenate([ta.data.reshape(flat_feat, -1).T for ta in tas]))
            pca_data = [pca.transform(ta.data.reshape(flat_feat, -1).T).T for ta in tas]
        else:
            pca_data = list(pca.fit_transform(np.array([ta.data.ravel() for ta in tas])))
        done = set()
        with self.infra.cache_dict.writer() as w:
            for (i_uid, event), d in zip(pca_events.items(), pca_data):
                if i_uid not in done:
                    w[i_uid] = TimedArray(
                        data=d,
                        frequency=float(tas[0].frequency),
                        start=event.start,
                        duration=event.duration,
                    )
                    done.add(i_uid)

    def _get_timed_arrays(
        self, events: list, start: float, duration: float
    ) -> tp.Iterable[TimedArray]:
        yield from self._get_data([self._to_uid(e) for e in events])  # type: ignore[misc]


#: wav2vec 2.0 conv strides; each block's native rate is 16 kHz divided by the
#: strides up to it, and every transformer layer runs at the last conv's rate
#: (conv_2: 800 Hz, trans_*: 50 Hz).
_W2V_CONV_STRIDES: tuple[int, ...] = (5, 2, 2, 2, 2, 2, 2)
_W2V_AUDIO_FS: int = 16_000


def _w2v_native_rate(block: str, layer: int) -> int:
    n = layer + 1 if block == "conv" else len(_W2V_CONV_STRIDES)
    return int(_W2V_AUDIO_FS / np.prod(_W2V_CONV_STRIDES[:n]))


def _resample(x: np.ndarray, native_rate: int, frequency: float) -> np.ndarray:
    """(time, dims) at ``native_rate`` → ``frequency``."""
    n_samples = int(x.shape[0] / native_rate * frequency)
    return scipy.signal.resample(x, n_samples, axis=0)


def _audiobook_chapters(filepath: str) -> list[str]:
    """Every chapter wav next to ``filepath``, in chapter order."""
    folder = Path(filepath).parent
    files = sorted(
        folder.glob("Le_Petit_Prince_*.wav"),
        key=lambda p: int(p.stem.rsplit("_", 1)[1]),
    )
    return [str(p) for p in files] or [str(filepath)]


def _fit_incremental_pca(chunks: list[np.ndarray], n_components: int) -> IncrementalPCA:
    """``IncrementalPCA().fit(np.concatenate(chunks))`` without the concatenation.

    Uses the same minibatches as ``IncrementalPCA.fit`` (``5 * n_features``
    rows, via ``gen_batches``), so the fit is identical; a batch that spans two
    chunks is assembled from both.
    """
    ipca = IncrementalPCA(n_components=n_components, whiten=False)
    bounds = np.cumsum([0] + [len(c) for c in chunks])
    batch_size = 5 * chunks[0].shape[1]
    for batch in gen_batches(int(bounds[-1]), batch_size, min_batch_size=n_components):
        parts = [
            c[max(batch.start, lo) - lo : min(batch.stop, hi) - lo]
            for c, lo, hi in zip(chunks, bounds[:-1], bounds[1:])
            if max(batch.start, lo) < min(batch.stop, hi)
        ]
        ipca.partial_fit(np.concatenate(parts), check_input=False)
    return ipca


class Wav2VecLayer(ns.extractors.base.BaseExtractor):
    """wav2vec 2.0 activations of the audiobook.

    * each chapter is cut into non-overlapping ``chunk_duration`` pieces (60 s)
      that are run separately and concatenated;
    * a conv block is read at its ``.conv`` output (before norm and GELU), via a
      forward hook on ``feature_extractor.conv_layers[i].conv``; a transformer
      layer is ``hidden_states[i]``;
    * with ``n_components``, a ``StandardScaler`` + ``IncrementalPCA`` is fitted
      once on all chapters concatenated, at the native rate, so every
      participant gets the same components;
    * resampling to ``frequency`` is ``scipy.signal.resample`` over the whole
      chapter; ``frequency="native"`` keeps the block's own rate.

    ``block_layers`` with two entries: each layer is resampled to ``frequency``,
    trimmed to the shorter one and concatenated; its PCA is fitted per chapter
    by the TRF (``chapter_pca_components``), so ``n_components`` must be None.

    The random network is ``Wav2Vec2Model(Wav2Vec2Config())`` seeded with
    ``init_seed``, in eval mode.

    One call computes the whole book, so that the PCA sees every chapter.
    """

    event_types: str | tuple[str, ...] = "Audio"
    model_name: str = "facebook/wav2vec2-base"
    block_layers: list[tuple[tp.Literal["conv", "trans"], int]] = [("conv", 2)]
    pretrained: bool = True
    init_seed: int = 0
    n_components: int | None = 50
    chunk_duration: float = 60.0
    frequency: float | tp.Literal["native"] = 20.0
    device: tp.Literal["auto", "cpu", "cuda"] = "auto"
    infra: exca.MapInfra = exca.MapInfra(version="1")

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if not self.block_layers:
            raise ValueError("block_layers must name at least one layer")
        if len(self.block_layers) > 1:
            if self.n_components is not None:
                raise ValueError("several layers: set n_components=None (PCA in the TRF)")
            if self.frequency == "native":
                raise ValueError("several layers need a common frequency")

    def _exclude_from_cache_uid(self) -> list[str]:
        return super()._exclude_from_cache_uid() + ["device"]

    def _download_snapshot(self) -> None:
        """Fetch the model on the submitting node; compute nodes load it offline."""
        try:
            snapshot_download(repo_id=self.model_name, local_files_only=True)
        except LocalEntryNotFoundError:
            snapshot_download(repo_id=self.model_name)

    def prepare(self, obj: tp.Any) -> None:
        events = self._event_types_helper.extract(obj)
        if not events:
            return
        self._download_snapshot()
        list(self._get_data(_audiobook_chapters(str(events[0].filepath))))  # type: ignore[attr-defined]
        # As BaseExtractor.prepare: run once to learn the output shape.
        self(events[0], start=events[0].start, duration=0.001, trigger=events[0])

    @infra.apply(item_uid=str)
    def _get_data(self, filepaths: list[str]) -> tp.Iterator[TimedArray]:  # type: ignore[override]
        book = _audiobook_chapters(filepaths[0])
        book += [f for f in filepaths if f not in book]
        arrays = self._compute_book(book)
        for filepath in filepaths:
            yield arrays[filepath]

    def _get_timed_arrays(
        self, events: list, start: float, duration: float
    ) -> tp.Iterable[TimedArray]:
        paths = [str(e.filepath) for e in events]
        for event, ta in zip(events, self._get_data(paths)):
            yield ta.with_start(event.start).overlap(start=start, duration=duration)

    def _load(self) -> tuple[tp.Any, tp.Any, str]:
        processor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.model_name, local_files_only=True
        )
        model: tp.Any
        if self.pretrained:
            model = Wav2Vec2Model.from_pretrained(self.model_name, local_files_only=True)
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.init_seed)
                model = Wav2Vec2Model(Wav2Vec2Config())
        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return model.to(device).eval(), processor, device

    def _chapter_activations(
        self, filepath: str, model: tp.Any, processor: tp.Any, device: str
    ) -> tuple[list[np.ndarray], float]:
        """(time, dims) at the native rate for each of ``block_layers``, and the
        chapter's duration in seconds."""
        audio, _ = librosa.load(filepath, sr=_W2V_AUDIO_FS)
        conv: dict[int, torch.Tensor] = {}
        hooks = [
            model.feature_extractor.conv_layers[layer].conv.register_forward_hook(
                lambda _m, _i, output, layer=layer: conv.__setitem__(layer, output)
            )
            for block, layer in self.block_layers
            if block == "conv"
        ]
        chunk = int(self.chunk_duration * _W2V_AUDIO_FS)
        per_layer: list[list[np.ndarray]] = [[] for _ in self.block_layers]
        try:
            for begin in range(0, len(audio), chunk):
                piece = audio[begin : begin + chunk]
                inputs = processor(
                    piece, sampling_rate=_W2V_AUDIO_FS, return_tensors="pt"
                )
                with torch.no_grad():
                    out = model(
                        inputs["input_values"].to(device), output_hidden_states=True
                    )
                for k, (block, layer) in enumerate(self.block_layers):
                    act = conv[layer][0] if block == "conv" else out.hidden_states[layer][0].T
                    per_layer[k].append(act.float().cpu().numpy())
        finally:
            for hook in hooks:
                hook.remove()
        # The early conv blocks are ~4 GB per chapter: free each layer's chunks
        # once concatenated.
        acts = []
        for p in per_layer:
            a = np.concatenate(p, axis=1)
            p.clear()
            acts.append(a.T.astype(np.float32, copy=False))
        return acts, len(audio) / _W2V_AUDIO_FS

    def _compute_book(self, book: list[str]) -> dict[str, TimedArray]:
        model, processor, device = self._load()
        out: dict[str, TimedArray] = {}
        if self.n_components is None:
            for filepath in tqdm(book, desc="wav2vec chapters"):
                acts, _ = self._chapter_activations(filepath, model, processor, device)
                layers = [
                    _resample(a, _w2v_native_rate(b, l), float(self.frequency))
                    for a, (b, l) in zip(acts, self.block_layers)
                ]
                n = min(len(a) for a in layers)
                x = np.concatenate([a[:n] for a in layers], axis=1)
                out[filepath] = TimedArray(
                    data=np.ascontiguousarray(x.T, dtype=np.float32),
                    frequency=float(self.frequency),
                    start=ns.base._UNSET_START,
                    duration=n / float(self.frequency),
                )
            return out

        ((block, layer),) = self.block_layers
        native_rate = _w2v_native_rate(block, layer)
        # Native-rate activations of the whole book do not fit in memory for
        # the early conv blocks (conv_2: ~9 GB), so they go to disk.
        tmp_root = self.infra.folder
        if tmp_root is not None:
            Path(tmp_root).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="w2v-", dir=tmp_root) as tmp:
            scaler = StandardScaler()
            chapters: list[tuple[str, np.memmap, float]] = []
            for i, filepath in enumerate(tqdm(book, desc="wav2vec chapters")):
                (chapter_acts,), seconds = self._chapter_activations(
                    filepath, model, processor, device
                )
                mm = np.lib.format.open_memmap(
                    f"{tmp}/{i}.npy", mode="w+", dtype=np.float32, shape=chapter_acts.shape
                )
                mm[:] = chapter_acts
                scaler.partial_fit(chapter_acts)
                chapters.append((filepath, mm, seconds))
                del chapter_acts
            for _, mm, _ in chapters:
                mm[:] = scaler.transform(mm)
            ipca = _fit_incremental_pca([mm for _, mm, _ in chapters], self.n_components)
            for filepath, mm, seconds in chapters:
                x = ipca.transform(mm).astype(np.float32)
                if self.frequency == "native":
                    freq = len(x) / seconds
                else:
                    freq = float(self.frequency)
                    x = _resample(x, native_rate, freq)
                out[filepath] = TimedArray(
                    data=np.ascontiguousarray(x.T, dtype=np.float32),
                    frequency=freq,
                    start=ns.base._UNSET_START,
                    duration=len(x) / freq,
                )
        return out
