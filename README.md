# Fragrance shelf-life stability; root cause analysis and prediction

Business case for the Symrise Graduate Program in Data Science (final round).

## Objectives
1. **Root cause analysis**: why has the shelf-life (`stability_months`) of a fragrance compound dropped?
2. **Predictive model**: estimate `stability_months` from sensor readings at production time, with calibrated uncertainty and explanations.

## Key results
- **Root cause**: from 8 May 2023, three reactors of Plant-04 (R03, R07, R08) lose ≈ 3.2 months of shelf-life (21.9 → 18.6). On the same day their mixing torque fell from 12.0 to 10.8 N·m, still inside the documented normal range (9–12.5 N·m), so no control flagged it.
- **Model**: gradient boosting on the 7 process sensors, validated on future data (temporal split): MAE 0.73 months (vs 1.63 for the per-reactor mean), R² 0.80, and a 90% prediction interval with 88.9% coverage. SHAP explanations agree with the root cause.

## Slides
[`reports/SymrisePPTX.pdf`](reports/SymrisePPTX.pdf) (also as `presentation.pptx`, with speaker notes). They present the results first, then the work phase by phase.

## Method: CRISP-DM
| Phase | Where |
|---|---|
| 1 Business understanding | Slides |
| 2 Data understanding | [`notebooks/02_data_understanding.ipynb`](notebooks/02_data_understanding.ipynb) |
| 3 Data preparation | [`notebooks/03_data_preparation.ipynb`](notebooks/03_data_preparation.ipynb) · [`src/data.py`](src/data.py) |
| 4 Modelling: root cause | [`notebooks/04_root_cause.ipynb`](notebooks/04_root_cause.ipynb) |
| 4 Modelling: prediction · 5 Evaluation | [`notebooks/05_model.ipynb`](notebooks/05_model.ipynb) |
| 6 Deployment | Slides |

The notebooks are saved with their outputs, so they can be read on GitHub without running anything.

## Structure
```
configs/config.yaml   Paths, domain ranges, leakage columns, random seed
data/raw/             Original dataset (not versioned)
data/processed/       Cleaned dataset, written by notebook 03 (not versioned)
notebooks/            One notebook per CRISP-DM phase
src/data.py           load_raw() and clean(): shared loading and cleaning, with assert checks
reports/              Slides
```

## Reproducing the results
1. Place the original dataset (not included) at
   `data/raw/fragrance_stability_dataset_corrupted_20260601_161558.csv`
   (SHA-256 `870204af5b3e6ab0f320f3699b40624dd7ad20401eff70668723e91e762f2d36`).
2. Install the pinned dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows (source .venv/bin/activate on Linux/macOS)
   pip install -r requirements.txt
   ```
3. Run the notebooks in order (02 → 05). Notebook 03 writes the cleaned parquet used by 04 and 05.
