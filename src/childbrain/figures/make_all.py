"""Make every figure and table of the paper from the released data.

    childbrain-figures --release path/to/release --out figures/
    childbrain-figures --release path/to/release --only fig2 tables

The figures and statistics of exact age (Fig. S5, Fig. S7 C, age correlations
quoted in the text) and of the seizure onset zone need the full participants
table (``--participants``); without it they are skipped.

Each item is a module of :mod:`childbrain.figures` with a ``make(out: Path)`` function;
outputs are described in :mod:`childbrain.figures.output`.
"""

from __future__ import annotations

import argparse
import importlib
import time
from pathlib import Path

import matplotlib

#: Module of each figure / table group, in paper order.
ITEMS = [
    "fig1",          # Fig. 1, Fig. S1 (glassbrain), Fig. S2 (fig1_phon)
    "fig2",          # Fig. 2
    "fig3",          # Fig. 3
    "fig4",          # Fig. 4
    "fig5",          # Fig. 5
    "fig6",          # Fig. 6
    "figS_encoding",  # Fig. S3, S4, S9
    "figS_age_trends",  # Fig. S5
    "figS_layers",   # Fig. S6, S7
    "figS_soz_coverage",  # Fig. S8
    "figS_stimulus",  # Fig. S10
    "tables",        # Tables S1-S3
]


def main(argv: list[str] | None = None) -> None:
    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release", type=Path, default=None,
                        help="released data folder (default: $CHILDBRAIN_RELEASE or ./release)")
    parser.add_argument("--participants", type=Path, default=None,
                        help="full participants table with exact ages "
                             "(default: $CHILDBRAIN_PARTICIPANTS; without it, exact-age items are skipped)")
    parser.add_argument("--out", type=Path, default=Path("figures"))
    parser.add_argument("--only", nargs="*", choices=ITEMS, default=None)
    args = parser.parse_args(argv)

    from . import data

    if args.release is not None:
        data.set_release(args.release)
    if args.participants is not None:
        data.set_participants(args.participants)
    data.release_dir()  # fail early without data
    for name in args.only or ITEMS:
        start = time.time()
        module = importlib.import_module(f"childbrain.figures.{name}")
        module.make(args.out)
        print(f"{name}: done in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
