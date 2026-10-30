"""Run the paper's analyses and save their score tables.

    childbrain-run decoding --set linguistic                  # Fig. 3, 4
    childbrain-run decoding --set linguistic --roi stc        # one ROI panel
    childbrain-run decoding --set hierarchy --neuro llama_pca1  # stimulus side
    childbrain-run encoding --set linguistic --splits         # Fig. 2, S attention
    childbrain-run encoding --set linguistic --participant P01 --predictions
    childbrain-run evoked --event word                        # Fig. 1
    childbrain-run predictor-corr                             # Fig. S predictor corr.

Each subject is one job (``CHILDBRAIN_CLUSTER=slurm``) and is cached: running
the same command again collects the scores. See :mod:`.configs` for the paths.
"""

import argparse
import logging
import typing as tp
from pathlib import Path

import mne
import neuralset as ns
import numpy as np
import pandas as pd

from . import configs
from . import extractors as _extractors  # noqa: F401  registers the extractors
from . import study as _studies  # noqa: F401  registers the studies
from . import transforms as _transforms  # noqa: F401  registers the transforms
from .decoding import EpochData, SlidingWindow, scores_to_df
from .encoding import TRFData, TRFEncoding, trf_scores_to_df
from .evoked import EvokedResponses
from .predictor_corr import compute as compute_predictor_corr
from .rois import FIGURE_ROIS, roi_covered_subjects

logger = logging.getLogger(__name__)


def _study(neuro: str) -> dict:
    return configs.chain("Evanson2023" if neuro in configs.IEEG_NEUROS else "EvansonStimSingle")


def _neuro(neuro: str, rois: list[str]) -> dict:
    cfg = configs.neuros[neuro]
    if rois:
        cfg = {**cfg, "rois": tuple(rois), "roi_table": str(configs.ELECTRODES)}
    return cfg


def _infra(folder: Path) -> dict:
    folder.parent.mkdir(parents=True, exist_ok=True)
    return {**configs.infra_exp, "folder": str(folder)}


def _roi_tag(rois: list[str]) -> str:
    return f"_roi_{'-'.join(sorted(rois))}_tbl_{configs.ELECTRODES.stem}" if rois else ""


def _features_tag(keys: list[str]) -> str:
    return f"many_features_{keys[0]}_{keys[-1]}" if len(keys) > 4 else "_".join(keys)


def select_subjects(neuro: str, rois: list[str], participant: str | None) -> list[str]:
    """Subjects with a chapter of at least ``MIN_WORDS`` words (and ROI coverage)."""
    chain = _study(neuro)
    events = ns.Chain(steps=chain["steps"], infra=chain["infra"]).run()
    n_words = events.loc[events.type == "Word"].groupby("timeline").text.count()
    timelines = n_words.index[n_words >= configs.MIN_WORDS]
    subjects = sorted(events.loc[events.timeline.isin(timelines), "subject"].unique())
    if rois:
        covered = roi_covered_subjects(rois, configs.ELECTRODES, configs.DATA)
        subjects = [s for s in subjects if s.rpartition("/")[2] in covered]
    if participant is not None:
        subjects = [s for s in subjects if s == f"Evanson2023/{participant}"]
    print(f"{len(subjects)} subject(s)")
    return subjects


def run_decoding(args: argparse.Namespace) -> None:
    features = configs.DECODING_SETS[args.set]
    rois = args.roi or []
    if rois and args.neuro not in configs.IEEG_NEUROS:
        raise SystemExit("--roi needs an iEEG --neuro")
    subjects = select_subjects(args.neuro, rois, args.participant)
    out = []
    for key, cfg in features.items():
        print(f"── {key} ──")
        folder = configs.CACHE / "decoding" / args.neuro / f"{key}{_roi_tag(rois)}"
        sw = SlidingWindow(
            data=EpochData(
                study=_study(args.neuro),
                event_type=cfg["event_type"],
                neuro=_neuro(args.neuro, rois),
                extractor=cfg["extractor"],
                start=configs.EPOCH_START,
                stop=configs.EPOCH_STOP,
            ),
            model=configs.ridge,
            cv={"n_splits": configs.N_SPLITS},
            neuro_time_radius=configs.NEURO_TIME_RADIUS,
            infra=_infra(folder),
        )
        df = scores_to_df(sw.get_scores(subjects), key, dim_names=cfg.get("dim_names"))
        out.append(
            df.assign(
                score_by="subject",
                neuro=args.neuro,
                stacking=True,
                roi="-".join(sorted(rois)) if rois else "all",
                roi_table=str(configs.ELECTRODES) if rois else "default",
            )
        )
    prefix = args.participant or ""
    name = (
        f"{prefix}scores_stacking_{args.neuro}_{_features_tag(list(features))}"
        f"_scoreby_subject{_roi_tag(rois)}.csv"
    )
    _save(pd.concat(out, ignore_index=True), args.out / name)


def _trf(key: str, n_permutations: int) -> TRFEncoding:
    cfg = configs.trf_features[key]
    pca = cfg.get("chapter_pca", {})
    return TRFEncoding(
        data=TRFData(
            study=_study("ieeg"),
            event_type="Audio",
            neuro={**configs.ieeg, "allow_missing": True},
            extractor=cfg["extractor"],
            duration=configs.TRF_SEGMENT_DURATION,
            stride=configs.TRF_SEGMENT_DURATION,
        ),
        tmin=configs.TRF_TMIN,
        tmax=configs.TRF_TMAX,
        alphas=configs.ridge["alphas"],
        cv={"n_splits": configs.N_SPLITS},
        alpha_holdout=configs.TRF_ALPHA_HOLDOUT,
        n_permutations=n_permutations,
        chapter_pca_components=pca.get("n_components"),
        chapter_pca_incremental=pca.get("incremental", False),
        infra=_infra(configs.CACHE / "trf" / key),
    )


def run_encoding(args: argparse.Namespace) -> None:
    keys = configs.TRF_SETS[args.set]
    prefix = f"{args.participant}_" if args.participant else ""
    stem = args.out / f"{prefix}trf_encod_ieeg_{_features_tag(keys)}_alphaholdout"
    if args.predictions:
        if args.participant is None:
            raise SystemExit("--predictions needs --participant")
        # the figure shows the phonemes + words model
        key = "phoneme_word" if "phoneme_word" in keys else keys[0]
        arrays = _trf(key, 0).held_out_predictions(f"Evanson2023/{args.participant}")
        stem.parent.mkdir(parents=True, exist_ok=True)
        prefixed = {f"{key}/{k}": v for k, v in arrays.items()}
        np.savez_compressed(f"{stem}_predictions.npz", **prefixed)  # type: ignore[arg-type]
        print(f"saved {stem}_predictions.npz")
        return
    subjects = select_subjects("ieeg", [], args.participant)
    out, splits = [], []
    for key in keys:
        print(f"── {key} ──")
        scores = _trf(key, configs.TRF_N_PERMUTATIONS).get_scores(subjects)
        out.append(trf_scores_to_df(scores, key))
        cols = ["subject", "split", "dim", "timeline", "score", "channel"]
        splits.append(scores.dropna(subset=["score"])[cols].assign(feature=key))
    df = pd.concat(out, ignore_index=True).assign(
        neuro="ieeg", tmin=configs.TRF_TMIN, tmax=configs.TRF_TMAX, roi="all"
    )
    _save(df, Path(f"{stem}.csv"))
    if args.splits:
        _save(pd.concat(splits, ignore_index=True), Path(f"{stem}_splits.csv"))


def run_evoked(args: argparse.Namespace) -> None:
    cfg = configs.linguistic_features[configs.EVOKED_EVENTS[args.event]]
    evoked = EvokedResponses(
        data=EpochData(
            study=_study("ieeg_evoked"),
            event_type=cfg["event_type"],
            neuro=configs.ieeg_evoked,
            extractor=cfg["extractor"],
            start=configs.EVOKED_TMIN,
            stop=configs.EVOKED_TMAX + 1.0 / configs.FREQUENCY,
        ),
        infra=_infra(configs.CACHE / "evoked" / args.event),
    )
    df = evoked.get_scores(select_subjects("ieeg_evoked", [], args.participant))
    prefix = f"{args.participant}_" if args.participant else ""
    _save(df.assign(neuro="ieeg_evoked"), args.out / f"{prefix}evoked_{args.event}_broadband.csv")


def run_predictor_corr(args: argparse.Namespace) -> None:
    path = args.out / "predictor_corr.npz"
    compute_predictor_corr(path)
    print(f"saved {path}")


def _save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"saved {path} ({len(df):,} rows)")


def main(argv: tp.Sequence[str] | None = None) -> None:
    mne.set_log_level("WARNING")
    logging.basicConfig(format="[%(asctime)s %(name)s %(levelname)s] %(message)s", level=logging.WARNING)
    logging.getLogger("exca").setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=Path("scores"), help="score folder")
    sub = parser.add_subparsers(dest="analysis", required=True)

    dec = sub.add_parser("decoding", help="time-resolved decoding of word features")
    dec.add_argument("--set", choices=sorted(configs.DECODING_SETS), required=True)
    dec.add_argument("--neuro", choices=sorted(configs.neuros), default="ieeg")
    dec.add_argument("--roi", nargs="*", choices=sorted(FIGURE_ROIS), help="union of ROIs")
    dec.set_defaults(run=run_decoding)

    enc = sub.add_parser("encoding", help="TRF encoding of every channel")
    enc.add_argument("--set", choices=sorted(configs.TRF_SETS), required=True)
    enc.add_argument("--splits", action="store_true", help="also save every (chapter, fold) score")
    enc.add_argument("--predictions", action="store_true", help="refit --participant, save its predictions")
    enc.set_defaults(run=run_encoding)

    evo = sub.add_parser("evoked", help="event-locked broadband responses")
    evo.add_argument("--event", choices=sorted(configs.EVOKED_EVENTS), default="word")
    evo.set_defaults(run=run_evoked)

    pc = sub.add_parser("predictor-corr", help="correlation between the TRF predictors")
    pc.set_defaults(run=run_predictor_corr)

    for p in (dec, enc, evo):
        p.add_argument("--participant", help="one participant, e.g. P01")
    args = parser.parse_args(argv)
    configs.CACHE.mkdir(parents=True, exist_ok=True)
    args.run(args)


if __name__ == "__main__":
    main()
