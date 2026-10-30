"""Le Petit Prince sEEG study (Evanson2023) and its stimulus-only variant."""

import re
import typing as tp
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import yaml
from neuralset.events import study


def _load_chapter_events(
    path: Path, chapter: int, start: float, stop: float
) -> pd.DataFrame:
    """Load Word/Phoneme/Audio/Text events for one chapter.

    Args:
        path: study root directory
        chapter: chapter number
        start: onset (seconds) to add to all event times; also used as the
               Audio event start and the Text event start.
        stop: end of the chapter window; events with onset > stop are dropped.
    """
    filename = path / f"events/MAU_events_chapter_{chapter}.csv"
    annot = pd.read_csv(filename, index_col=0)

    annot["onset"] += start
    annot = annot.loc[annot.onset <= stop]
    annot = annot.rename(columns={"onset": "start", "kind": "type"})

    words = annot.query("type=='word'").dropna(axis="columns").reset_index(drop=True)
    words["text"] = words.word

    phonemes = (
        annot.query("type=='phoneme'").dropna(axis="columns").reset_index(drop=True)
    )
    phonemes["text"] = phonemes.phoneme

    sound_path = path / "audiobook" / f"Le_Petit_Prince_{chapter}.wav"
    sounds = pd.DataFrame(
        dict(filepath=str(sound_path), start=start, type="Audio"), index=[0]
    )

    text_path = path / "text" / f"run{chapter}.txt"
    text_content = Path(text_path).read_text("utf8")
    texts = pd.DataFrame(
        dict(
            filepath=str(text_path),
            start=start,
            type="text",
            text=text_content,
            stop=stop,
            duration=stop - start,
        ),
        index=[0],
    )

    events = pd.concat([words, phonemes, sounds, texts]).reset_index(drop=True)
    events["language"] = "french"
    events["type"] = events.type.str.capitalize()

    # sequence_id: groups of 10 events
    base_vector = np.arange(int(len(events) / 10) + 1)
    seq_id = np.repeat(base_vector, 10)
    events["sequence_id"] = seq_id[: len(events)]

    # word_id
    words_df = events.query('type == "Word"')
    events["word_id"] = None
    events.loc[words_df.index, "word_id"] = np.arange(len(words_df))

    # phoneme → parent word linkage
    events["stop"] = events.start + events.duration.fillna(0)
    words_df = events.query('type == "Word"')
    for _, w in words_df.iterrows():
        sub = events.query("start >= @w.start and start < @w.stop - 1e-8")
        ph = sub.query('type == "Phoneme"')
        events.loc[ph.index, "word"] = w.text
        events.loc[ph.index, "word_id"] = w.word_id

    return events


def _iter_patient_timelines(path: Path) -> tp.Iterator[dict[str, tp.Any]]:
    subjects = pd.read_csv(path / "participants.tsv", sep="\t").participant_id.tolist()
    assert subjects, f"no participant in {path / 'participants.tsv'}"
    for subject in subjects:
        with (path / "annotations" / f"{subject}.yaml").open() as stream:
            patient_info = yaml.safe_load(stream)
        for chapter in sorted(patient_info.keys()):
            yield dict(subject=subject, chapter=chapter)


class Evanson2023(study.Study):
    """sEEG of patients listening to Le Petit Prince (French audiobook)."""

    url: tp.ClassVar[str] = ""
    licence: tp.ClassVar[str] = ""
    description: tp.ClassVar[str] = (
        "Patients listen to Le Petit Prince with sEEG recording"
    )
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = ()

    def _download(self) -> None:
        raise NotImplementedError("The raw recordings are not distributed with this code.")

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        yield from _iter_patient_timelines(self.path)

    def _load_raw(self, timeline: dict[str, tp.Any]) -> mne.io.BaseRaw:
        subject = timeline["subject"]
        chapter = timeline["chapter"]

        path = self.path
        with (path / "annotations" / f"{subject}.yaml").open() as stream:
            patient_info = yaml.safe_load(stream)

        raw_fname = path / "neuro" / subject / patient_info[chapter]["filename"]

        try:
            raw = mne.io.read_raw(raw_fname)
        except UnicodeDecodeError:
            raw = mne.io.read_raw(raw_fname, encoding="latin1")

        ch_types = dict()
        for ch in raw.ch_names:
            if "+" in ch.lower() or "lpp" in ch.lower():
                ch_type = "misc"
            elif "ecg" in ch.lower():
                ch_type = "ecg"
            else:
                ch_type = "seeg"
            ch_types[ch] = ch_type
        raw.set_channel_types(ch_types)

        # A few recordings spell a channel differently from the electrode table,
        # which leaves those channels with no location. Normalise towards it:
        #   "EEG XY1,"  -> "EEG XY1"    (trailing comma)
        #   "EEG XY 10" -> "EEG XY10"   (space before the contact)
        #   "EEG X1.5"  -> "EEG X1'5"   (dot instead of prime)
        # A rename is skipped when the target already exists or is already
        # taken by another rename, so these can never collapse two distinct
        # channels onto one name.
        renames: dict[str, str] = {}
        for ch in raw.ch_names:
            fixed = ch.rstrip(",")
            fixed = re.sub(r"^(EEG [A-Za-z]+) (\d+)$", r"\1\2", fixed)
            fixed = re.sub(r"^(EEG [A-Za-z]+\d*)\.(\d+)$", r"\1'\2", fixed)
            if fixed != ch and fixed not in raw.ch_names and fixed not in renames.values():
                renames[ch] = fixed
        if renames:
            raw.rename_channels(renames)

        # Set channel names to be patient dependent
        raw.rename_channels({ch: f"{ch}_{subject}" for ch in raw.ch_names})
        return raw

    def _overflow_chapter(
        self, timeline: dict[str, tp.Any], patient_info: dict
    ) -> tuple[float, float]:
        chapter = timeline["chapter"]
        # Seconds of the chapter already listened to in the previous file
        prev_chap = patient_info["overflow_from"]
        timeadd = (
            patient_info[prev_chap]["endtime"] - patient_info[prev_chap]["starttime"]
        )
        start = (patient_info[chapter]["starttime"] + timeadd) / 1000
        stop = (patient_info[chapter]["endtime"] + timeadd) / 1000
        return start, stop

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        subject = timeline["subject"]
        chapter = timeline["chapter"]

        path = self.path
        with (path / "annotations" / f"{subject}.yaml").open() as stream:
            patient_info = yaml.safe_load(stream)

        filename = path / f"events/MAU_events_chapter_{int(chapter)}.csv"
        annot = pd.read_csv(filename, index_col=0)

        if "overflow_from" in patient_info.keys():  # chapter started in another file
            start, stop = self._overflow_chapter(timeline, patient_info)
        else:
            start = patient_info[chapter]["starttime"] / 1000  # ms to s
            stop = patient_info[chapter]["endtime"] / 1000

        annot["onset"] += start
        # remove events for the part of the story they didn't listen to
        annot = annot.loc[(annot.onset <= stop)]
        annot = annot.rename(columns={"onset": "start", "kind": "type"})

        words = annot.query("type=='word'").dropna(axis="columns").reset_index(drop=True)
        words["text"] = words.word

        phonemes = (
            annot.query("type=='phoneme'").dropna(axis="columns").reset_index(drop=True)
        )
        phonemes["text"] = phonemes.phoneme

        sound_path = path / "audiobook" / f"Le_Petit_Prince_{int(chapter)}.wav"
        sounds = pd.DataFrame(
            dict(filepath=sound_path, start=start, type="Audio"), index=[0]
        )

        text_path = Path(path / "text") / f"run{int(chapter)}.txt"
        text = Path(text_path).read_text("utf8")
        texts = pd.DataFrame(
            dict(
                filepath=text_path,
                start=start,
                type="text",
                text=text,
                stop=stop,
                duration=stop - start,
            ),
            index=[0],
        )

        events = pd.concat([words, phonemes, sounds, texts]).reset_index(drop=True)
        events["language"] = "french"
        events["type"] = events.type.str.capitalize()  # Word and Phoneme

        info = study.SpecialLoader(method=self._load_raw, timeline=timeline)
        seeg = {"type": "Ieeg", "filepath": info.to_json(), "start": 0}
        events = pd.concat([pd.DataFrame([seeg]), events]).reset_index(drop=True)

        # sequences of 10 events
        base_vector = np.arange(int(len(events) / 10) + 1)
        seq_id = np.repeat(base_vector, 10)
        events["sequence_id"] = seq_id[: len(events)]

        words = events.query('type == "Word"')
        events["word_id"] = None
        events.loc[words.index, "word_id"] = np.arange(len(words))

        # word id for phonemes
        events["stop"] = events.start + events.duration
        words = events.query('type == "Word"')
        for _, w in words.iterrows():
            sub = events.query("start >= @w.start and start < @w.stop-1e-8")
            phonemes = sub.query('type == "Phoneme"')
            events.loc[phonemes.index, "word"] = w.text
            events.loc[phonemes.index, "word_id"] = w.word_id

        return events


class EvansonStimSingle(study.Study):
    """Le Petit Prince stimulus events for one synthetic subject, all chapters.

    Same event files as Evanson2023, without neural recordings; event times are
    offsets within each chapter's audio file.
    """

    url: tp.ClassVar[str] = ""
    licence: tp.ClassVar[str] = ""
    description: tp.ClassVar[str] = (
        "Single-subject stimulus-only version of Evanson2023 (no neural data)."
    )
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = ()

    def _download(self) -> None:
        raise NotImplementedError("The stimulus is on OpenNeuro (ds003643), see the README.")

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        chapter_files = sorted((self.path / "events").glob("MAU_events_chapter_*.csv"))
        chapters = sorted(int(f.stem.split("_")[-1]) for f in chapter_files)
        assert chapters, f"No MAU_events_chapter_*.csv under {self.path / 'events'}"
        for chapter in chapters:
            yield dict(subject="Subject0", chapter=chapter)

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        chapter = int(timeline["chapter"])
        annot = pd.read_csv(
            self.path / f"events/MAU_events_chapter_{chapter}.csv", index_col=0
        )
        stop = (annot["onset"] + annot["duration"].fillna(0)).max()
        return _load_chapter_events(self.path, chapter, 0.0, stop)
