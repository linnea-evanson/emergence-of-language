# Emergence of Language in the Developing Brain

Code and data for Evanson et al., *Emergence of Language in the Developing Brain*,
Science Advances (2026). DOI: TBD.

The Python package is imported as `childbrain`.

- `childbrain.figures`: every figure, table and statistic of the paper, recomputed from
  the released data (`release/`, on Zenodo: https://doi.org/10.5281/zenodo.23812322, released
  under the CC BY-NC 4.0 license).
- `childbrain.pipeline`: the original analyses (decoding, encoding, evoked responses).

## Reproduce the figures

Python ≥ 3.11.

```bash
git clone https://github.com/linnea-evanson/emergence-of-language.git
cd emergence-of-language
pip install .
# released data: download from Zenodo (https://doi.org/10.5281/zenodo.23812322) and unpack to ./release
childbrain-figures --release release --out figures            # everything, ~6 min
childbrain-figures --release release --out figures --only fig2 tables
```

The fsaverage brain used by the brain plots is downloaded once by MNE
(`mne.datasets.fetch_fsaverage`, ~200 MB, to `~/mne_data`).

For each figure `<name>`, `figures/` holds:

- `<name>.pdf` (and `.png` for figures with brains, included as PNG in the paper);
- `source_data/<name>/<panel>.csv`: the values plotted in each panel;
- `stats/<name>.json`: every statistic drawn on the figure or quoted in the text.

Cluster-permutation tests and bootstraps are seeded, so the outputs are deterministic.

## Figures, scripts and data

All modules are in `src/childbrain/figures/`; tables in `release/` (described in
`release/README.md`).

| Paper | Output | Module | Released data |
|---|---|---|---|
| Fig. 1 | `fig1` | `fig1.py` | `participants.tsv`, `electrodes.tsv`, `evoked/word_onset.csv.gz` |
| Fig. 2 | `fig2` | `fig2.py` | `encoding/trf_linguistic.csv.gz`, `electrodes.tsv`, `evoked/word_onset.csv.gz` (text) |
| Fig. 3 | `fig3` | `fig3.py` | `decoding/brain_linguistic.csv.gz`, `encoding/trf_linguistic.csv.gz` |
| Fig. 4 | `fig4` | `fig4.py` | `decoding/brain_linguistic.csv.gz` |
| Fig. 5 | `fig5_1` | `fig5.py` | `encoding/trf_embeddings.csv.gz`, `decoding/brain_embeddings.csv.gz` |
| Fig. 6 | `fig6` | `fig6.py` | `decoding/brain_embeddings.csv.gz`, `encoding/trf_embeddings.csv.gz` |
| Fig. S1 | `glassbrain` | `fig1.py` | `evoked/phoneme_onset.csv.gz`, `electrodes.tsv` |
| Fig. S2 | `fig1_phon` | `fig1.py` | `evoked/phoneme_onset.csv.gz`, `electrodes.tsv` |
| Fig. S3 | `figS_prop_sig_electrodes` | `figS_encoding.py` | `encoding/trf_linguistic.csv.gz`, `electrodes.tsv` |
| Fig. S4 | `figS_single_subject_predictions` | `figS_encoding.py` | `encoding/trf_example_predictions.csv`, `encoding/trf_linguistic.csv.gz` |
| Fig. S5 | `figS_age_trends` | `figS_age_trends.py` | `decoding/brain_linguistic.csv.gz`, `decoding/brain_linguistic_pvalues.csv.gz`; exact ages |
| Fig. S6 | `figS_llm_layers_decode_features` | `figS_layers.py` | `decoding/stimulus_feature_to_feature.csv.gz` |
| Fig. S7 | `figS_layerwise_gain_by_age` | `figS_layers.py` | `decoding/brain_layerwise.csv.gz`; C: exact ages |
| Fig. S8 | `figS_soz_coverage` | `figS_soz_coverage.py` | `participants.tsv`, `electrodes.tsv` |
| Fig. S9 | `figS_attention` | `figS_encoding.py` | `encoding/trf_phonetic_splits.csv.gz` |
| Fig. S10 | `figS_predictor_corr`, `figS_feature_to_feature_matrix`, `figS_feature_to_feature_curves` | `figS_stimulus.py` | `stimulus/predictor_correlation.csv`, `decoding/stimulus_feature_to_feature.csv.gz` |
| Table S1 | `roi_coverage.tex` | `tables.py` | `participants.tsv`, `electrodes.tsv`, `encoding/trf_linguistic.csv.gz` |
| Table S2 | `ci_table.tex` | `tables.py` | `encoding/trf_linguistic.csv.gz` |
| Table S3 | `decoding_effects_table.tex` | `tables.py` | `decoding/brain_linguistic.csv.gz` |

Statistics quoted in the text that are not shown on a figure are in the stats file of the
figure they relate to (e.g. label-permutation tests in `stats/fig2.json`, decoding age
effects in `stats/fig4.json`, cohort counts in `stats/tables.json`).

## Analysis pipeline

```bash
pip install ".[pipeline]"
export CHILDBRAIN_DATA=/path/to/raw   # participants.tsv, electrodes.tsv, annotations/, neuro/, stimulus
export CHILDBRAIN_CACHE=/path/to/cache
childbrain-run --help
```

The released tables (`release/README.md`) are derived from the outputs of these
analyses (score tables per participant, feature and electrode or time sample).

The stimulus is the French *Le Petit Prince* audiobook, from the Le Petit Prince
multilingual naturalistic fMRI corpus on OpenNeuro (dataset ds003643, version 2.0.7:
https://openneuro.org/datasets/ds003643/versions/2.0.7; Li et al., *Sci. Data* 9, 530
(2022), https://doi.org/10.1038/s41597-022-01625-7).

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

Code and released data: Creative Commons Attribution-NonCommercial 4.0 International
(CC BY-NC 4.0); see [`LICENSE`](LICENSE).
