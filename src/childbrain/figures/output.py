"""Where a figure script writes: the figure, its source data, its statistics.

    out/
      fig2.pdf (.png)                 the figure
      source_data/fig2/B_bars.csv     one table per panel (Science Advances source data)
      stats/fig2.json                 every statistic shown or quoted in the text
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import style


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


class Output:
    """Outputs of one figure (or table) ``name`` under ``root``."""

    def __init__(self, root: str | Path, name: str) -> None:
        self.root = Path(root)
        self.name = name
        self._stats: dict = {}

    def figure(self, fig, png: bool = False, suffix: str = "") -> list[Path]:
        return style.save(fig, self.root / f"{self.name}{suffix}", png=png)

    def source(self, panel: str, df: pd.DataFrame) -> Path:
        """Source data of one panel, e.g. ``source("B_bars", df)``."""
        path = self.root / "source_data" / self.name / f"{panel}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, float_format="%.6g")
        return path

    def stat(self, key: str, value) -> None:
        """Record a statistic (nested dicts allowed); written by :meth:`write_stats`."""
        self._stats[key] = value

    def write_stats(self) -> Path | None:
        if not self._stats:
            return None
        path = self.root / "stats" / f"{self.name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_jsonable(self._stats), indent=1, ensure_ascii=False) + "\n")
        return path

    def text(self, filename: str, content: str) -> Path:
        """A text output next to the figures (e.g. a LaTeX table)."""
        path = self.root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path
