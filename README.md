# Spotter freight rate assessment

Predict total load rates for 12,000 validation loads and the 31 fixed December
shipments. The solution uses CatBoost models selected using chronological
development folds and assessed on a frozen October holdout.

## Requirements

- Windows, macOS or Linux.
- Python **3.10 or 3.11** installed on the machine. The reference results were
  produced with Python 3.10.20 on macOS; Windows and Linux were not tested locally.
- Git to clone the repository, or download and extract its ZIP instead.
- The four input CSVs supplied with the assessment.

Use the pinned package versions in `requirements.txt`. Create a separate `.venv`
on each computer; virtual environments and generated models are not included
in the repository.

## Setup

### 1. Download the project

Run these commands in a terminal, PowerShell, Command Prompt or VS Code's
integrated terminal:

```sh
git clone https://github.com/Nuwanga-Wijamuni/SpotterML.git
cd SpotterML
```

If you downloaded a ZIP, open a terminal in the extracted project folder.
If you already have the project, use your existing folder and skip cloning.
Run the remaining commands from the folder containing `run_all.py`.

### 2. Add the assessment data

Place the four supplied CSVs in the project root or in `data/`. These raw inputs
are excluded from Git. The loader accepts hyphen and underscore naming styles:

| Input | Purpose |
| --- | --- |
| `train-test.csv` / `train_test.csv` | 48,000 labeled development loads |
| `validation.csv` | 12,000 unlabeled final loads |
| `validation-predictions-template.csv` / `validation_predictions_template.csv` | Required submission IDs and row order |
| `december-chart-inputs.csv` / `december_chart_inputs.csv` | 31 fixed December scenario inputs |

### 3. Create the Python environment and install packages

**Windows (PowerShell or Command Prompt)**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Use `py -3.10` instead if Python 3.10 is installed. If the `py` launcher is
unavailable, check `python --version`; when it reports 3.10 or 3.11, use
`python -m venv .venv` to create the environment.

**macOS or Linux (bash/zsh)**

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Use `python3.10` instead if that is your installed interpreter. If `python3`
reports version 3.10 or 3.11, `python3 -m venv .venv` also works. On Linux, install
the `venv` package for your chosen Python version through your distribution's
package manager if the module is unavailable.

These commands use the environment's Python directly, so activation is optional.
See the [Python virtual-environment documentation](https://docs.python.org/3.11/library/venv.html)
for platform-specific activation commands.

### 4. Run the solution

| Operating system | Command |
| --- | --- |
| Windows (PowerShell or Command Prompt) | `.\.venv\Scripts\python.exe run_all.py` |
| macOS or Linux | `.venv/bin/python run_all.py` |
| Any of these systems with `.venv` already activated | `python run_all.py` |

The runner checks the data, prepares the models, fills both prediction files,
runs the supplied scorer's checks, creates the December chart and opens
`outputs/results.html` in your browser. The page includes model metrics, the
chart, CSV downloads and a searchable view of all 12,000 predictions.

If saved models are absent, the runner trains the frozen model choices in
`configs/selected_configuration.json`. It evaluates October once when no
holdout history exists, then fits final models on all development rows.
Later runs reuse matching models and the saved October assessment.
`configs/model_comparison.csv` and its manifest archive the earlier notebook
comparison; the quick run does not recompute those folds.

No notebooks, browser server or report generation are needed for the quick run.

### Run without opening a browser

| Operating system | Command |
| --- | --- |
| Windows | `.\.venv\Scripts\python.exe run_all.py --no-open` |
| macOS or Linux | `.venv/bin/python run_all.py --no-open` |

Open `outputs/results.html` manually to view the page afterward. This option is
also suitable for a Linux machine without a desktop browser.

### Optional macOS launcher

After setting up `.venv` and adding the CSVs, macOS users can double-click
**RunSpotter.command** in Finder. If its executable permission was lost after
downloading the repository, restore it once:

```sh
chmod +x RunSpotter.command
```

Windows and Linux users should use the Python run commands above.

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

On any supported system, install VS Code's Python and Jupyter extensions and
select the `.venv` notebook kernel. Its interpreter is `.venv\Scripts\python.exe`
on Windows and `.venv/bin/python` on macOS/Linux. A fresh checkout can use the quick
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

| Operating system | Command |
| --- | --- |
| Windows | `.\.venv\Scripts\python.exe run_all.py --retrain` |
| macOS or Linux | `.venv/bin/python run_all.py --retrain` |
| Any of these systems with `.venv` activated | `python run_all.py --retrain` |

Matching October metrics are reused. If data, training code, versions or model
choices differ from the saved October evaluation, training stops with an
explanation. Keep that history; changing choices after inspecting October makes
it model-selection data rather than an untouched holdout.

- **Missing packages:** repeat the package-install command for your operating
  system in setup step 3.
- **Python command not found:** install Python 3.10 or 3.11 and use the matching
  launcher or executable shown in setup step 3.
- **PowerShell blocks activation:** use `.\.venv\Scripts\python.exe` directly;
  activation is not required.
- **Missing input files:** put the four supplied CSVs in the root or `data/`.
- **Browser did not open:** open `outputs/results.html` manually.
- **Wrong kernel in VS Code:** select the interpreter inside `.venv`.
- **Missing comparison files in notebook 03:** run notebook 02 to generate its
  complete experiment outputs. The quick run only needs the frozen choices.

## Submission checklist

- Provide access to the GitHub repository containing the solution code,
  notebooks, `configs/`, `requirements.txt`, README and scorer. `.gitignore`
  excludes the environment, generated files, models and supplied raw data;
  reviewers add the assessment CSVs locally.
- Submit `outputs/validation_predictions.csv` separately.
- Submit the required PDF/DOCX report with the split approach and December
  chart; this runner does not create a report.
- Record a 2-3 minute Loom: show exploration findings and cleaning, explain the
  chronological split and model choice, walk through the data/training/prediction
  functions, then show the results page and December chart.
