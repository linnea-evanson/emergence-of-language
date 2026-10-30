"""Event transforms of the study chain (text cleaning, POS, constituency parse)."""

import logging
import os
import re
import string
import typing as tp
import warnings

import benepar
import exca
import nltk
import numpy as np
import pandas as pd
import spacy
import spacy.cli
import spacy.util
import torch
from neuralset import utils as nsutils
from neuralset.events import etypes as ev
from neuralset.events.transforms import EventsTransform
from num2words import num2words
from tqdm import tqdm

logger = logging.getLogger(__name__)


class CleanTextNewlines(EventsTransform):
    """Collapse newlines and repeated whitespace in Text events.

    Applied before ``AddSentenceToWords``, as formatting newlines interfere with
    sentence boundary detection.
    """

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        events = events.copy(deep=True)
        text_mask = events.type == "Text"
        for idx in events[text_mask].index:
            events.at[idx, "text"] = re.sub(r"\s+", " ", events.at[idx, "text"]).strip()
        return events


class RemoveMissingSentenceChar(EventsTransform):
    """Drop Word events with missing ``sentence_char`` (unmatched alignment)."""

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        word_mask = events["type"] == "Word"
        n_words = int(word_mask.sum())
        if n_words == 0:
            return events

        missing = events.loc[word_mask, "sentence_char"].fillna("").eq("")
        n_dropped = int(missing.sum())
        logger.info(
            "RemoveMissingSentenceChar: dropped %d/%d Word events", n_dropped, n_words
        )
        keep = np.logical_or(~word_mask, ~missing.reindex(events.index, fill_value=False))
        return events.loc[keep].reset_index(drop=True)


class NormalizeNumbers(EventsTransform):
    """Replace digit sequences in ``Text`` event text with spelled-out word forms.

    Also normalises hyphenated compound number words in ``Word.text`` (e.g.
    ``'six-cents-douze'`` → ``'six cent douze'``) so that ``TextWordMatcher``
    in ``AddSentenceToWords`` can align them against the normalised Text.

    Run this *before* ``AddSentenceToWords``.

    ``language`` should be an ISO 639-1 code (``'fr'``, ``'en'``, …) or a
    full name (``'french'``, ``'english'``, …); full names are mapped
    automatically.
    """

    _LANG_MAP: tp.ClassVar[dict[str, str]] = {
        "french": "fr", "english": "en", "german": "de",
        "spanish": "es", "italian": "it", "dutch": "nl",
    }

    # French number-component words (singular + plural forms)
    _FR_NUMBER_WORDS: tp.ClassVar[frozenset[str]] = frozenset([
        "zéro", "un", "une", "deux", "trois", "quatre", "cinq", "six", "sept",
        "huit", "neuf", "dix", "onze", "douze", "treize", "quatorze", "quinze",
        "seize", "vingt", "vingts", "trente", "quarante", "cinquante",
        "soixante", "cent", "cents", "mille", "million", "millions",
        "milliard", "milliards",
    ])

    language: str = "fr"

    def _looks_like_compound_number(self, word: str) -> bool:
        """Return True if the word is a hyphenated compound of number-component words."""
        if "-" not in word:
            return False
        parts = [p.lower().rstrip("s") for p in word.split("-")]
        return all(p in self._FR_NUMBER_WORDS or p + "s" in self._FR_NUMBER_WORDS for p in parts if p)

    @staticmethod
    def _normalise_compound(word: str) -> str:
        """Replace hyphens with spaces and strip the plural 's' of cent/vingt/million/milliard.

        ``'six-cents-douze'`` → ``'six cent douze'``, matching num2words, which
        omits the 's' in compounds.
        """
        parts = word.split("-")
        normalised = []
        for part in parts:
            p = part.lower()
            if p.endswith("s") and p[:-1] in ("cent", "vingt", "million", "milliard"):
                p = p[:-1]
            normalised.append(p)
        return " ".join(normalised)

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        events = events.copy()

        # ── 1. Normalise digit sequences in Text events ───────────────────────
        text_mask = events["type"] == "Text"
        if text_mask.any():
            def _replace_digits(text: str, lang: str) -> str:
                return re.sub(
                    r"\b\d+\b",
                    lambda m: num2words(int(m.group()), lang=lang),
                    text,
                )

            text_cache: dict[tuple[str, str], str] = {}
            for _, row in events.loc[text_mask].iterrows():
                raw = str(row["text"] or "")
                lang_raw = str(row.get("language") or self.language)
                lang = self._LANG_MAP.get(lang_raw.lower(), lang_raw)
                key = (raw, lang)
                if key not in text_cache:
                    text_cache[key] = _replace_digits(raw, lang)

            def _lookup_text(row: pd.Series) -> str:
                raw = str(row["text"] or "")
                lang_raw = str(row.get("language") or self.language)
                lang = self._LANG_MAP.get(lang_raw.lower(), lang_raw)
                return text_cache.get((raw, lang), raw)

            events.loc[text_mask, "text"] = events.loc[text_mask].apply(_lookup_text, axis=1).values

        # ── 2. Normalise hyphenated compound number words in Word events ──────
        word_mask = events["type"] == "Word"
        if word_mask.any():
            words = events.loc[word_mask, "text"]
            new_texts = words.map(
                lambda w: self._normalise_compound(str(w))
                if self._looks_like_compound_number(str(w)) else w
            )
            events.loc[word_mask, "text"] = new_texts.values

        return events


# The 8 part-of-speech categories of the lexical features (``pd.get_dummies`` after collapse).
POS_TAGS: list[str] = [
    "ADJ",
    "ADP",
    "ADV",
    "CONJ",
    "DET",
    "NOUN",
    "PRON",
    "VERB",
]


POS_COLLAPSE: dict[str, str] = {
    "PROPN": "NOUN",
    "NUM": "ADJ",
    "AUX": "VERB",
    "INTJ": "X",
    "SCONJ": "CONJ",
    "CCONJ": "CONJ",
    "PUNCT": "X",
}


def collapse_pos(tag: str | None) -> str:
    """Map a spaCy UPOS tag to the 8 collapsed tags."""
    if tag is None or (isinstance(tag, float) and pd.isna(tag)):
        return "X"
    tag = str(tag)
    return POS_COLLAPSE.get(tag, tag)


def pos_onehot(tag: str | None) -> np.ndarray:
    """Binary vector over ``POS_TAGS`` (``pd.get_dummies`` row)."""
    collapsed = collapse_pos(tag)
    return np.array(
        [1.0 if collapsed == name else 0.0 for name in POS_TAGS],
        dtype=np.float32,
    )


def assign_pos(words: pd.DataFrame, text: str, *, language: str = "french") -> pd.Series:
    """Assign collapsed POS tags to Word rows using full-text alignment.

    1. Character-align concatenated word texts to the chapter ``text``.
    2. Run spaCy on the full text and propagate token POS to each Word.
    3. Collapse UPOS tags and set ambiguous assignments to ``"X"``.

    Parameters
    ----------
    words
        Word events for one timeline, with a ``text`` column.
    text
        Full chapter text.

    Returns
    -------
    pd.Series
        Collapsed POS tag per row, indexed like ``words``.
    """
    if words.empty:
        return pd.Series(dtype=object)

    words = words.sort_values("start")
    word_txt = "".join(words["text"].astype(str))
    i, j = nsutils.match_list(text, word_txt)

    nlp = nsutils.get_spacy_model(language=language)
    doc = nlp(text)

    char_pos = [
        token.pos_
        for token in doc
        for _ in token.text_with_ws
    ]
    txt_pos = pd.DataFrame({"pos": char_pos})
    if len(txt_pos) != len(text):
        raise RuntimeError(
            f"spaCy character expansion length {len(txt_pos)} != text length {len(text)}"
        )

    word_idx = np.array(
        [idx for idx, word in zip(words.index, words["text"].astype(str)) for _ in word]
    )
    txt_pos.loc[i, "word_id"] = word_idx[j]

    tags = pd.Series(index=words.index, dtype=object)
    for word_id, group in txt_pos.groupby("word_id"):
        unique = group["pos"].unique()
        if len(unique) > 1:
            pos = "X"
        else:
            pos = collapse_pos(str(unique[0]))
        tags.loc[int(word_id)] = pos
    return tags


class AddPOS(EventsTransform):
    """Append collapsed ``pos_tag`` tags via full-text character alignment.

    Requires ``Text`` events on each timeline (run after ``CleanTextNewlines`` /
    ``NormalizeNumbers``).
    """

    language: str = "french"
    column: str = "pos_tag"

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        events = events.copy()
        if "timeline" in events.columns and events["timeline"].nunique() > 1:
            parts = [
                self._run_timeline(sub)
                for _, sub in events.groupby("timeline", sort=False)
            ]
            return pd.concat(parts, ignore_index=True)
        return self._run_timeline(events)

    def _run_timeline(self, events: pd.DataFrame) -> pd.DataFrame:
        word_mask = events["type"] == "Word"
        if not word_mask.any():
            return events

        texts = events.loc[events["type"] == "Text"].sort_values("start")
        if texts.empty:
            raise RuntimeError("AddPOS requires Text events")
        chapter_text = "".join(texts["text"].astype(str))

        words = events.loc[word_mask]
        events.loc[word_mask, self.column] = assign_pos(
            words, chapter_text, language=self.language
        )
        return events


class _TreeNode:
    def __init__(self, label: str | None, parent: "_TreeNode | None" = None):
        self.label = label
        self.parent = parent
        self.children: list[_TreeNode] = []
        self.word: str | None = None

    def get_leftmost_leaf(self) -> "_TreeNode | None":
        if self.word:
            return self
        if not self.children:
            return None
        return self.children[0].get_leftmost_leaf()


class _SyntaxParser:
    """Per-word node counts from a bracketed constituency parse string."""

    def parse(self, parse_string: str) -> pd.DataFrame:
        root, terminals = self._build_tree_and_get_terminals(parse_string)
        if not root:
            return pd.DataFrame()
        return pd.DataFrame(self._extract_features_from_tree(terminals))

    def _build_tree_and_get_terminals(self, s: str) -> tuple:
        s = s.replace("(", " ( ").replace(")", " ) ")
        tokens = s.split()
        root, current_node, stack, terminals = None, None, [], []

        for token in tokens:
            if token == "(":
                new_node = _TreeNode(label=None, parent=current_node)
                if current_node:
                    current_node.children.append(new_node)
                stack.append(new_node)
                current_node = new_node
                if root is None:
                    root = new_node
            elif token == ")":
                if stack:
                    stack.pop()
                current_node = stack[-1] if stack else None
            else:
                if current_node and current_node.label is None:
                    current_node.label = token
                elif current_node:
                    leaf = _TreeNode(label=current_node.label, parent=current_node)
                    leaf.word = token
                    current_node.children.append(leaf)
                    terminals.append(leaf)
        return root, terminals

    def _extract_features_from_tree(self, terminals: list[_TreeNode]) -> list[dict]:
        extractors_list = []
        pre_terminals = [t.parent for t in terminals]
        max_depth = max(self._get_depth(pt) for pt in pre_terminals) if pre_terminals else 0

        for i, node in enumerate(terminals):
            depth = self._get_depth(node.parent)
            extractors_list.append(
                {
                    "pos": node.parent.label if node.parent else "ROOT",
                    "word_id": i,
                    "word": node.word,
                    "n_closing": self._get_n_closing(node),
                    "n_opening": self._get_n_opening(node),
                    "depth": depth,
                    "height": max_depth - depth,
                    "sequence_id": 0,
                    "is_last_word": i == len(terminals) - 1,
                }
            )
        return extractors_list

    def _get_depth(self, node: _TreeNode | None) -> int:
        if not node:
            return 0
        depth, current = 0, node.parent
        while current:
            depth += 1
            current = current.parent
        return depth

    def _get_n_opening(self, node: _TreeNode) -> int:
        count, current = 0, node.parent
        while current:
            if current.get_leftmost_leaf() is node:
                count += 1
            else:
                break
            current = current.parent
        return count

    def _get_n_closing(self, node: _TreeNode) -> int:
        if not node.parent:
            return 1
        count = 1
        current = node.parent
        while (
            current.parent
            and current.parent.parent
            and current is current.parent.children[-1]
        ):
            count += 1
            current = current.parent
        return count


class _BeneparParser:
    def __init__(self, model: str, device: tp.Literal["auto", "cpu", "cuda"] = "cpu"):
        self.model = model
        self.device = device
        self._nlp: tp.Any = None

    def _initialize_parser(self) -> None:
        if self._nlp is not None:
            return

        warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")
        warnings.filterwarnings("ignore", message=".*T5TokenizerFast.*")
        warnings.filterwarnings("ignore", message=".*TreeCRF.*")
        warnings.filterwarnings("ignore", message=".*arg_constraints.*")
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"

        use_cuda = self.device == "cuda" or (
            self.device == "auto" and torch.cuda.is_available()
        )
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("Benepar requested CUDA, but CUDA is not available; using CPU")
            use_cuda = False

        try:
            nltk.data.find(f"models/{self.model}")
        except LookupError:
            benepar.download(self.model)

        spacy_model = "fr_core_news_lg" if self.model.startswith("benepar_fr") else "en_core_web_sm"
        if not spacy.util.is_package(spacy_model):
            spacy.cli.download(spacy_model)
        self._nlp = spacy.load(spacy_model)

        if use_cuda:
            self._nlp.add_pipe("benepar", config={"model": self.model})
        else:
            original_cuda_available = torch.cuda.is_available
            try:
                torch.cuda.is_available = lambda: False
                self._nlp.add_pipe("benepar", config={"model": self.model})
            finally:
                torch.cuda.is_available = original_cuda_available

    def parse_sentence(self, sentence: str) -> str | None:
        self._initialize_parser()
        try:
            sentence = sentence.strip()
            if not sentence:
                return None
            doc = self._nlp(sentence)
            sents = list(doc.sents)
            if len(sents) > 0:
                parse_result = sents[0]._.parse_string
                if parse_result and parse_result.strip():
                    return parse_result
            return None
        except Exception as e:
            logger.debug("Parse failed for %r: %s", sentence[:30], e)
            return None


class SyntaxEnhancer(EventsTransform):
    """Add constituency-parse node counts (``n_opening``, ``n_closing``, …) to Words.

    Each sentence is parsed whole with benepar and its terminals are aligned to
    the sentence's Word events. Words that cannot be aligned get failure values
    (``n_closing == -1``) and are dropped when ``filter_failed``.
    """

    sentence_field: str = "sentence"
    benepar_model: str = "benepar_fr2"
    benepar_device: tp.Literal["auto", "cpu", "cuda"] = "cpu"
    filter_failed: bool = True
    timeline_infra: exca.MapInfra = exca.MapInfra(version="1")

    def __init__(self, **kwargs: tp.Any) -> None:
        super().__init__(**kwargs)
        self._parser = _BeneparParser(model=self.benepar_model, device=self.benepar_device)

    def model_post_init(self, log__: object) -> None:
        super().model_post_init(log__)
        if self.infra is not None:
            if self.infra.folder is not None and self.timeline_infra.folder is None:
                self.timeline_infra.folder = self.infra.folder

    @timeline_infra.apply(item_uid=lambda df: str(df["timeline"].iloc[0]))
    def _process_timelines(
        self, timeline_dfs: list[pd.DataFrame]
    ) -> tp.Iterator[pd.DataFrame]:
        for df in timeline_dfs:
            yield self._process_single_timeline(df)

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        groups = [g for _, g in events.groupby("timeline", sort=False)]
        processed_events = pd.concat(list(self._process_timelines(groups)), ignore_index=True)

        if self.filter_failed:
            word_mask = processed_events.type == "Word"
            failed = (processed_events["n_closing"] == -1) | (processed_events["pos"] == "UNK")
            processed_events = processed_events[~(word_mask & failed)].copy()

        if "index" in processed_events.columns:
            processed_events = processed_events.drop(columns=["index"])
        return processed_events

    def _process_single_timeline(self, events: pd.DataFrame) -> pd.DataFrame:
        events = events.copy(deep=True)
        syntax_cols = ["pos", "n_opening", "n_closing", "depth", "height", "is_last_word"]

        wtypes = ev.EventTypesHelper("Word")
        word_events = events[events.type.isin(wtypes.names)].copy()
        if len(word_events) == 0:
            return events

        failure_values = {
            "pos": "UNK",
            "n_opening": -1,
            "n_closing": -1,
            "depth": -1,
            "height": -1,
            "is_last_word": False,
        }

        word_mask = events.type.isin(wtypes.names)
        for col in syntax_cols:
            if col not in events.columns:
                if isinstance(failure_values[col], bool):
                    events[col] = pd.NA
                    events[col] = events[col].astype("boolean")
                elif isinstance(failure_values[col], int):
                    events[col] = pd.NA
                    events[col] = events[col].astype("Int64")
                else:
                    events[col] = None
            events.loc[word_mask, col] = failure_values[col]

        return self._process_by_sentences(events, word_events, syntax_cols)

    def _process_by_sentences(
        self, events: pd.DataFrame, word_events: pd.DataFrame, syntax_cols: list
    ) -> pd.DataFrame:
        syntax_parser = _SyntaxParser()
        word_events["_sentence_group"] = word_events[self.sentence_field].astype(str)

        for _, sentence_words in tqdm(
            word_events.groupby("_sentence_group", sort=False), desc="Processing sentences"
        ):
            sentence_text = sentence_words[self.sentence_field].iloc[0]
            if not sentence_text or str(sentence_text).strip() == "":
                continue

            sentence_words = sentence_words.sort_values("start").copy()
            if len(sentence_words) == 0:
                continue

            try:
                parse_string = self._parser.parse_sentence(str(sentence_text))
                if not parse_string:
                    continue
                syntax_df = syntax_parser.parse(parse_string)
                if len(syntax_df) == 0:
                    continue
                self._smart_align_words(events, sentence_words, syntax_df, syntax_cols)
            except Exception as e:
                logger.warning("Error processing %r: %s", str(sentence_text)[:30], e)

        return events

    def _smart_align_words(
        self,
        events: pd.DataFrame,
        sentence_words: pd.DataFrame,
        syntax_df: pd.DataFrame,
        syntax_cols: list,
    ) -> int:
        word_texts = [str(w).strip() for w in sentence_words["text"].values]
        syntax_words = [str(w).strip() for w in syntax_df["word"].values]

        content_syntax = []
        content_words = []
        for i, syntax_word in enumerate(syntax_words):
            if not self._is_punctuation_only(syntax_word):
                content_words.append(syntax_word)
                content_syntax.append(syntax_df.iloc[i])

        if len(word_texts) == len(content_words):
            if all(self._words_match(w, s) for w, s in zip(word_texts, content_words)):
                for i, word_idx in enumerate(sentence_words.index):
                    self._assign_syntax_features(
                        events, word_idx, content_syntax[i], syntax_cols
                    )
                return len(word_texts)

        matched = self._fuzzy_align(
            events, sentence_words, content_words, content_syntax, syntax_cols
        )
        if matched > 0:
            return matched

        if len(content_words) > 0:
            return self._position_based_align(
                events, sentence_words, content_syntax, syntax_cols
            )
        return 0

    def _fuzzy_align(
        self,
        events: pd.DataFrame,
        sentence_words: pd.DataFrame,
        content_words: list,
        content_syntax: list,
        syntax_cols: list,
    ) -> int:
        word_texts = [self._normalize_word(str(w)) for w in sentence_words["text"].values]
        norm_syntax = [self._normalize_word(w) for w in content_words]

        matched = 0
        syntax_idx = 0
        word_idx = 0

        while word_idx < len(word_texts) and syntax_idx < len(norm_syntax):
            word_text = word_texts[word_idx]

            if word_text == norm_syntax[syntax_idx]:
                word_event_idx = sentence_words.iloc[word_idx].name
                self._assign_syntax_features(
                    events, word_event_idx, content_syntax[syntax_idx], syntax_cols
                )
                matched += 1
                syntax_idx += 1
                word_idx += 1
                continue

            if word_idx + 1 < len(word_texts) and word_texts[word_idx + 1] == "t":
                if (
                    syntax_idx + 1 < len(norm_syntax)
                    and norm_syntax[syntax_idx + 1] in ["nt", "n't"]
                    and (
                        word_text == norm_syntax[syntax_idx] + "n"
                        or word_text + "t"
                        == norm_syntax[syntax_idx] + norm_syntax[syntax_idx + 1]
                    )
                ):
                    word_event_idx = sentence_words.iloc[word_idx].name
                    self._assign_syntax_features(
                        events, word_event_idx, content_syntax[syntax_idx], syntax_cols
                    )
                    matched += 1
                    word_idx += 2
                    syntax_idx += 2
                    continue

            matched_contraction = False
            for lookahead in range(2, min(4, len(norm_syntax) - syntax_idx + 1)):
                combined_tokens = "".join(norm_syntax[syntax_idx : syntax_idx + lookahead])
                combined_with_apostrophe = norm_syntax[syntax_idx]
                for i in range(1, lookahead):
                    combined_with_apostrophe += "'" + norm_syntax[syntax_idx + i]

                if word_text in [combined_tokens, combined_with_apostrophe]:
                    word_event_idx = sentence_words.iloc[word_idx].name
                    self._assign_syntax_features(
                        events, word_event_idx, content_syntax[syntax_idx], syntax_cols
                    )
                    matched += 1
                    syntax_idx += lookahead
                    word_idx += 1
                    matched_contraction = True
                    break

            if matched_contraction:
                continue

            if len(word_text) > 3:
                for offset in range(min(3, len(norm_syntax) - syntax_idx)):
                    if word_text.startswith(norm_syntax[syntax_idx + offset]) and len(
                        norm_syntax[syntax_idx + offset]
                    ) > 2:
                        word_event_idx = sentence_words.iloc[word_idx].name
                        self._assign_syntax_features(
                            events,
                            word_event_idx,
                            content_syntax[syntax_idx + offset],
                            syntax_cols,
                        )
                        matched += 1
                        syntax_idx += offset + 1
                        word_idx += 1
                        matched_contraction = True
                        break

            if matched_contraction:
                continue

            if matched > word_idx * 0.5:
                syntax_idx += 1
                if syntax_idx < len(norm_syntax) and word_text == norm_syntax[syntax_idx]:
                    word_event_idx = sentence_words.iloc[word_idx].name
                    self._assign_syntax_features(
                        events, word_event_idx, content_syntax[syntax_idx], syntax_cols
                    )
                    matched += 1
                    syntax_idx += 1
                    word_idx += 1
                    continue

            word_idx += 1

        return matched

    def _position_based_align(
        self,
        events: pd.DataFrame,
        sentence_words: pd.DataFrame,
        content_syntax: list,
        syntax_cols: list,
    ) -> int:
        matched = 0
        max_align = min(len(sentence_words), len(content_syntax))
        for i in range(max_align):
            word_event_idx = sentence_words.iloc[i].name
            self._assign_syntax_features(events, word_event_idx, content_syntax[i], syntax_cols)
            matched += 1
        return matched

    @staticmethod
    def _normalize_word(word: str) -> str:
        return word.lower().strip().replace("'", "").replace('"', "").replace("'", "")

    def _words_match(self, word1: str, word2: str) -> bool:
        return self._normalize_word(word1) == self._normalize_word(word2)

    @staticmethod
    def _is_punctuation_only(word: str) -> bool:
        cleaned = word.strip()
        return len(cleaned) > 0 and all(c in string.punctuation for c in cleaned)

    @staticmethod
    def _assign_syntax_features(
        events: pd.DataFrame, word_idx: int, syntax_row: pd.Series, syntax_cols: list
    ) -> None:
        for col in syntax_cols:
            if col in syntax_row.index and syntax_row[col] is not None:
                events.loc[word_idx, col] = syntax_row[col]
