"""Run the freight assessment and open its local results page.

    .venv/bin/python run_all.py
    .venv/bin/python run_all.py --no-open
    .venv/bin/python run_all.py --retrain

Normal runs reuse matching saved models. A new checkout trains the frozen
choices automatically. Existing October evaluation history is preserved.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import webbrowser


ROOT = Path(__file__).resolve().parent


def models_match(project: Path, selection: dict) -> bool:
    """Reuse models only when data, code, settings, versions and hashes match."""
    from src.data import find_file
    from src.train import MODEL_FILENAMES, ROLES, package_versions, sha256_file

    try:
        report = json.loads((project / 'outputs/training/training_report.json').read_text())
        holdout = json.loads((project / 'outputs/training/holdout_metrics.json').read_text())
        signature = {
            'models': {role: selection[role] for role in ROLES},
            'hyperparameters': selection['hyperparameters'],
            'holdout_start': selection['holdout_start'],
            'development_sha256': sha256_file(find_file(project, 'development')),
            'data_code_sha256': sha256_file(project / 'src/data.py'),
            'features_code_sha256': sha256_file(project / 'src/features.py'),
            'training_code_sha256': sha256_file(project / 'src/train.py'),
            'versions': package_versions(),
        }
        fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        return (report['fingerprint'] == holdout['fingerprint'] == fingerprint
                and (project / 'outputs/training/holdout_predictions.csv').is_file()
                and all(sha256_file(project / 'models' / MODEL_FILENAMES[role])
                        == report['artifacts'][role]['sha256'] for role in ROLES))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def run(project: Path, *, retrain: bool = False, open_results: bool = True) -> Path:
    from src.data import load_project_data
    from src.train import load_selection, run_training
    from src.predict import run_prediction
    from src.predict_december import run_december_prediction
    from src.results import build_results_page
    from score import save_december_chart, validate_december, validate_predictions
    import pandas as pd

    print('\nSpotter freight assessment', flush=True)
    print(f'Project: {project}', flush=True)
    print('\n[1/4] Checking the input files...', flush=True)
    data = load_project_data(project)
    print(f'{len(data.development):,} labeled loads; {len(data.validation):,} submission loads; '
          f'{len(data.december)} December days.', flush=True)

    selection_path = project / 'outputs/model_experiments/selected_configuration.json'
    if not selection_path.is_file():
        selection_path = project / 'configs/selected_configuration.json'
    selection = load_selection(selection_path)
    print('\n[2/4] Preparing the saved models...', flush=True)
    if not retrain and models_match(project, selection):
        print('Matching saved models found. Training is already complete.', flush=True)
    else:
        run_training(project, selection_path, project / 'models', project / 'outputs/training')

    print('\n[3/4] Generating and checking all predictions...', flush=True)
    output = project / 'outputs'
    submission = run_prediction(project, project / 'models/main_model.joblib',
                                output / 'validation_predictions.csv')
    december = run_december_prediction(project, project / 'models/december_model.joblib',
                                      output / 'december_predictions.csv')
    validate_predictions(submission)
    completed_december = validate_december(december)
    save_december_chart(completed_december, output / 'candidate_december.png')
    # Check the saved files, not just the in-memory values.
    validate_predictions(pd.read_csv(output / 'validation_predictions.csv'))
    validate_december(pd.read_csv(output / 'december_predictions.csv'))

    print('\n[4/4] Preparing the results page...', flush=True)
    data.quality_report.to_csv(output / 'training/data_quality_report.csv', index=False)
    page = build_results_page(project)
    print('\nComplete. Open:', page, flush=True)
    print('Submission CSV:', output / 'validation_predictions.csv', flush=True)
    print('December chart:', output / 'candidate_december.png', flush=True)
    print('Hidden validation accuracy is calculated by Spotter after submission.', flush=True)
    if open_results:
        try:
            opened = webbrowser.open(page.resolve().as_uri())
        except (OSError, webbrowser.Error):
            opened = False
        if not opened:
            print('Open outputs/results.html in your browser to see the results.', flush=True)
    return page


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retrain', action='store_true', help='Refit the frozen models; reuse matching October results')
    parser.add_argument('--no-open', action='store_true', help='Generate outputs without opening a browser')
    args = parser.parse_args()
    if sys.version_info < (3, 10):
        print('Use Python 3.10 or 3.11 with the project .venv.', file=sys.stderr)
        return 2
    missing = [module for module in ['numpy', 'pandas', 'sklearn', 'scipy', 'catboost', 'joblib', 'matplotlib']
               if importlib.util.find_spec(module) is None]
    if missing:
        print('Some project packages are missing: ' + ', '.join(missing), file=sys.stderr)
        print('Run .venv/bin/python -m pip install -r requirements.txt, then try again.', file=sys.stderr)
        return 2
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'outputs/.matplotlib'))
    try:
        run(ROOT, retrain=args.retrain, open_results=not args.no_open)
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f'\nCould not finish: {exc}', file=sys.stderr)
        print('See README.md for setup and troubleshooting. Existing October history is kept.', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
