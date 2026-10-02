# Spotter freight rate assessment

Predict total load rates for 12,000 validation loads and the 31 fixed December
shipments. The solution uses CatBoost models selected using chronological
development folds and assessed on a frozen October holdout.

## Run on this Mac

The project already has a working `.venv` and trained models. From the Spotter
folder in VS Code's terminal, run:

```sh
.venv/bin/python run_all.py
```

The runner checks the data, reuses matching saved models, fills both prediction
files, runs the supplied scorer's checks, regenerates the December chart and
opens `outputs/results.html` in your browser. The page includes model metrics,
the chart, CSV downloads and a searchable view of all 12,000 predictions.

You can also double-click **RunSpotter.command** in Finder. If its executable
permission was lost after downloading the repository, restore it once:

```sh
chmod +x RunSpotter.command
```

Run without opening a browser:

```sh
.venv/bin/python run_all.py --no-open
```

No notebooks, browser server or report generation are needed for the quick run.

## Setup on another computer

Use Python 3.10 or 3.11. The tested local environment is Python 3.10.20.
Install the exact versions in `requirements.txt`:

```sh
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run_all.py
```

If you installed Python 3.11, replace `python3.10` with `python3.11` when creating
the environment. On Windows use `.venv\Scripts\python.exe` for the Python
commands; the Finder launcher is for macOS.

Before running, place the four supplied input CSVs in the project root or in
`data/`. The loader accepts the hyphen and underscore versions of their names:

| Input | Purpose |
| --- | --- |
| `train-test.csv` / `train_test.csv` | 48,000 labeled development loads |
| `validation.csv` | 12,000 unlabeled final loads |
| `validation-predictions-template.csv` / `validation_predictions_template.csv` | Required submission IDs and row order |
| `december-chart-inputs.csv` / `december_chart_inputs.csv` | 31 fixed December scenario inputs |

If saved models are absent, the runner trains the frozen model choices in
`configs/selected_configuration.json`. It evaluates October once when no
holdout history exists, then fits final models on all development rows.
`configs/model_comparison.csv` and its manifest archive the earlier notebook
comparison; the quick run does not recompute those folds.

## Outputs

| File | What it contains |
| --- | --- |
| `outputs/results.html` | Local results page; open in a browser |
| `outputs/validation_predictions.csv` | Exactly `load_id,predicted_rate` for all 12,000 loads |
| `outputs/december_predictions.csv` | The original seven columns with all 31 predicted rates filled |
| `outputs/candidate_december.png` | Fixed December chart from `score.py` |
| `outputs/training/holdout_metrics.json` | Frozen October evaluation results |
| `outputs/training/holdout_predictions.csv` | Individual October predictions and errors |
| `models/main_model.joblib` | Final submission model |
| `models/december_model.joblib` | Final December-compatible model |

All rates are total dollars, rounded to cents in the prediction CSVs. Source
CSV files are unchanged. Template IDs determine the submission order.

## Validation and models

Notebook 02 compares seven candidates on expanding July, August and September
folds: the equipment baseline, Ridge and CatBoost, including rate-per-mile and
total-rate alternatives. Model selection uses pooled MAE with RMSE as the tie
breaker. October is reserved for assessment of the frozen choices.

The main model is `catboost_full_per_mile`; the scenario model is
`catboost_scenario_per_mile`. Rate-per-mile predictions are multiplied by each
shipment's distance to return total dollars. The main model uses market and
quote signals; their availability at the time of quoting remains an assumption
that needs confirmation. The December model excludes market, quote and
geographic inputs, matching the fixed scenario's available features.

Nonpositive weights become missing values. CatBoost handles numeric missing
values natively. Ridge preprocessing is fitted on each training split only.
Load IDs, target values and prediction columns are excluded from model inputs.
No target-based rows are removed.

Original October results:

| Model purpose | MAE ($) | RMSE ($) |
| --- | ---: | ---: |
| Main submission | 137.94 | 647.45 |
| December-compatible | 145.86 | 647.63 |

These are local October metrics. `score.py` validates file structure and creates
the chart; hidden validation accuracy is calculated by Spotter after submission.

## Notebooks and code walkthrough

1. `notebooks/01_data_exploration.ipynb`: data sizes, distributions and quality.
2. `notebooks/02_model_experiments.ipynb`: earlier chronological folds and model
   selection. Keep `RUN_FINAL_HOLDOUT=False`; `src.train` handles the frozen
   October assessment.
3. `notebooks/03_results_review.ipynb`: deeper review of saved comparisons,
   October errors, submission checks and the December chart.

In VS Code select the `.venv` notebook kernel. A fresh checkout can use the quick
runner directly with the frozen configuration. To reproduce the complete
selection-fold analysis and create the out-of-fold records required by notebook
03, run notebook 02 first, then the runner, then notebook 03.

| Code | Responsibility |
| --- | --- |
| `src/data.py` | Load, clean and validate inputs; split development by date |
| `src/features.py` | Deterministic features and unfitted preprocessing |
| `src/train.py` | Frozen October assessment and final fitting |
| `src/predict.py` | Match the 12,000 predictions to template IDs |
| `src/predict_december.py` | Predict the fixed scenario with the separate model |
| `score.py` | Supplied output checks and required chart |
| `src/results.py` | Independently check saved metrics and build the local viewer |
| `run_all.py` | Run the steps in the right order |

## Re-running and troubleshooting

Normal runs reuse models only when the data, code, model settings, library
versions and saved artifact hashes match. To refit the final models with the
same frozen choices:

```sh
.venv/bin/python run_all.py --retrain
```

Matching October metrics are reused. If data, training code, versions or model
choices differ from the saved October evaluation, training stops with an
explanation. Keep that history; changing choices after inspecting October makes
it model-selection data rather than an untouched holdout.

- **Missing packages:** run `.venv/bin/python -m pip install -r requirements.txt`.
- **Missing input files:** put the four supplied CSVs in the root or `data/`.
- **Browser did not open:** open `outputs/results.html` manually.
- **Wrong kernel in VS Code:** select the interpreter inside `.venv`.
- **Missing comparison files in notebook 03:** run notebook 02 to generate its
  complete experiment outputs. The quick run only needs the frozen choices.

## Submission checklist

- Upload the solution code, notebooks, `configs/`, `requirements.txt`, README
  and scorer to an accessible GitHub repository. No GitHub repository has been
  published by the runner. `.gitignore` excludes the environment, generated
  files, models and supplied raw data; reviewers add the assessment CSVs locally.
- Submit `outputs/validation_predictions.csv` separately.
- Prepare the required PDF/DOCX report with the split approach and December
  chart later; this runner does not create a report.
- Record a 2-3 minute Loom: show exploration findings and cleaning, explain the
  chronological split and model choice, walk through the data/training/prediction
  functions, then show the results page and December chart.
