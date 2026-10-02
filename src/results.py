"""Create a standalone local results viewer from checked assessment outputs."""
from __future__ import annotations

import base64
from html import escape
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DataValidationError, find_file
from .predict import validate_submission
from .train import sha256_file


def table_html(frame: pd.DataFrame) -> str:
    return frame.to_html(index=False, border=0, classes='data-table', escape=True)


def checked_results(project: Path) -> dict:
    """Independently verify saved October metrics and submission contracts."""
    from score import validate_december

    output = project / 'outputs'
    training = json.loads((output / 'training/training_report.json').read_text())
    holdout = json.loads((output / 'training/holdout_metrics.json').read_text())
    if training['fingerprint'] != holdout['fingerprint']:
        raise DataValidationError('Training and October reports do not match.')
    signature = holdout['signature']
    checks = [(find_file(project, 'development'), 'development_sha256'),
              (project / 'src/data.py', 'data_code_sha256'),
              (project / 'src/features.py', 'features_code_sha256'),
              (project / 'src/train.py', 'training_code_sha256')]
    for path, key in checks:
        if sha256_file(path) != signature[key]:
            raise DataValidationError(f'{path.name} changed after the saved evaluation.')
    for role, filename in [('main', 'main_model.joblib'), ('december_scenario', 'december_model.joblib')]:
        if sha256_file(project / 'models' / filename) != training['artifacts'][role]['sha256']:
            raise DataValidationError(f'{role} model does not match the training report.')

    predictions = pd.read_csv(output / 'training/holdout_predictions.csv', parse_dates=['date'])
    if set(predictions.role) != {'main', 'december_scenario'}:
        raise DataValidationError('Missing a saved October model evaluation.')
    summaries = []
    for role, group in predictions.groupby('role'):
        truth = group.actual_rate.to_numpy(dtype=float)
        predicted = group.predicted_rate.to_numpy(dtype=float)
        if (not group.load_id.is_unique or not np.isfinite(truth).all()
                or not np.isfinite(predicted).all() or (truth <= 0).any() or (predicted <= 0).any()):
            raise DataValidationError(f'{role}: invalid October prediction rows.')
        if (len(group) != holdout['holdout_rows']
                or not group.experiment.eq(training['selection'][role]['name']).all()
                or group.date.min() != pd.Timestamp(holdout['holdout_first_date'])
                or group.date.max() != pd.Timestamp(holdout['holdout_last_date'])):
            raise DataValidationError(f'{role}: October rows or model selection differ.')
        error = predicted - truth
        measured = {'rows': len(group), 'mae': float(np.abs(error).mean()),
                    'rmse': float(np.sqrt(np.square(error).mean())),
                    'bias': float(error.mean()), 'median_absolute_error': float(np.median(np.abs(error)))}
        for key, value in measured.items():
            if not np.isclose(value, holdout['metrics'][role][key]):
                raise DataValidationError(f'{role}: saved {key} does not match October predictions.')
        summaries.append({'role': role, **measured})

    submission = pd.read_csv(output / 'validation_predictions.csv', dtype={'load_id': 'string'})
    validate_submission(submission)
    template = pd.read_csv(find_file(project, 'template'), dtype={'load_id': 'string'})
    if submission.load_id.tolist() != template.load_id.str.strip().tolist():
        raise DataValidationError('Submission row order differs from the template.')
    december = validate_december(pd.read_csv(output / 'december_predictions.csv'))
    quality = pd.read_csv(output / 'training/data_quality_report.csv')
    chart_path = output / 'candidate_december.png'
    if not chart_path.is_file():
        raise FileNotFoundError('Run score.py to create the December chart.')

    comparison_path = output / 'model_experiments/model_comparison.csv'
    if not comparison_path.is_file():
        comparison_path = project / 'configs/model_comparison.csv'
    comparison = None
    if comparison_path.is_file():
        # The quick-run archive is a saved comparison, not a fresh CV run.
        provenance_path = project / 'configs/experiment_manifest.json'
        if provenance_path.is_file():
            manifest = json.loads(provenance_path.read_text())
            matches = (manifest['development_sha256'] == signature['development_sha256']
                       and manifest['features_code_sha256'] == signature['features_code_sha256']
                       and manifest['data_code_sha256'] == signature['data_code_sha256']
                       and manifest['comparison_sha256'] == sha256_file(comparison_path)
                       and manifest['versions'] == training['selection']['versions']
                       and all(manifest['models'][role] == training['selection'][role]
                               for role in ['main', 'december_scenario'])
                       and manifest['hyperparameters'] == training['selection']['hyperparameters'])
            if matches:
                comparison = pd.read_csv(comparison_path)
        else:
            comparison = pd.read_csv(comparison_path)

    main_rows = predictions.loc[predictions.role.eq('main')]
    squared = np.square(main_rows.predicted_rate-main_rows.actual_rate)
    tail_rows = max(1, int(np.ceil(len(main_rows)*0.01)))
    tail_share = 100*float(squared.nlargest(tail_rows).sum()/squared.sum()) if squared.sum() else 0.0
    return {'training': training, 'holdout': holdout, 'metrics': pd.DataFrame(summaries).set_index('role'),
            'submission': submission, 'december': december, 'quality': quality,
            'comparison': comparison, 'chart': chart_path,
            'tail_rows': tail_rows, 'tail_share': tail_share}


def build_results_page(project: Path) -> Path:
    results = checked_results(project)
    main = results['metrics'].loc['main']
    december = results['december']
    training, holdout = results['training'], results['holdout']
    metrics = results['metrics'].reset_index().rename(columns={
        'role': 'Model purpose', 'rows': 'Loads', 'mae': 'MAE ($)', 'rmse': 'RMSE ($)',
        'bias': 'Average bias ($)', 'median_absolute_error': 'Median absolute error ($)'})
    metrics['Model purpose'] = metrics['Model purpose'].replace(
        {'main': 'Main submission model', 'december_scenario': 'December-compatible model'})
    metrics['Loads'] = metrics['Loads'].map(lambda value: f'{int(value):,}')
    for column in metrics.columns[2:]:
        metrics[column] = metrics[column].map(lambda value: f'{value:,.2f}')
    comparison = results['comparison']
    if comparison is not None:
        comparison_display = comparison[['experiment', 'evaluation_rows', 'pooled_mae', 'pooled_rmse']].copy()
        comparison_display.columns = ['Candidate', 'Evaluation loads', 'Selection MAE ($)', 'Selection RMSE ($)']
        comparison_display['Candidate'] = comparison_display['Candidate'].replace({
            'catboost_full_per_mile': 'CatBoost with market and quote signals (per mile)',
            'catboost_without_quote': 'CatBoost without the quote signal',
            'catboost_scenario_per_mile': 'CatBoost with December-available inputs',
            'catboost_full_total': 'CatBoost predicting the total rate directly',
            'ridge_full_per_mile': 'Ridge with market and quote signals',
            'ridge_scenario_per_mile': 'Ridge with December-available inputs',
            'equipment_baseline': 'Equipment-based median baseline',
        })
        comparison_display['Evaluation loads'] = comparison_display['Evaluation loads'].map(
            lambda value: f'{int(value):,}')
        for column in comparison_display.columns[2:]:
            comparison_display[column] = comparison_display[column].map(lambda value: f'{value:,.2f}')
        comparison_html = table_html(comparison_display)
    else:
        comparison_html = '<p>The earlier experiment archive does not match this run. Review notebook 02 for its comparison.</p>'
    preview_data = json.dumps(results['submission'].values.tolist(), separators=(',', ':'), allow_nan=False)
    preview_data = preview_data.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    chart = base64.b64encode(results['chart'].read_bytes()).decode('ascii')
    quality_display = results['quality'].rename(columns={
        'dataset': 'Data', 'column': 'Field', 'issue': 'Cleaning action', 'rows': 'Loads'}).copy()
    quality_display['Data'] = quality_display['Data'].replace({
        'development': 'Development', 'validation': 'Validation'})
    quality_display['Field'] = quality_display['Field'].replace({
        'weight': 'Shipment weight', 'market_index': 'Market index'})
    quality_display['Cleaning action'] = quality_display['Cleaning action'].replace({
        'nonpositive weight converted to missing': 'Zero or negative weight treated as missing',
        'missing after cleaning': 'Missing after cleaning',
    })
    quality_display['Loads'] = quality_display['Loads'].map(lambda value: f'{int(value):,}')
    replacements = {
        '@@MAE@@': f'{main.mae:,.2f}', '@@RMSE@@': f'{main.rmse:,.2f}',
        '@@ROWS@@': f"{len(results['submission']):,}", '@@OCTOBER_ROWS@@': f"{holdout['holdout_rows']:,}",
        '@@LABELED@@': f"{training['final_training_rows']:,}",
        '@@EARLIER@@': f"{holdout['training_rows']:,}",
        '@@MAIN_MODEL@@': escape(training['selection']['main']['name']),
        '@@SCENARIO_MODEL@@': escape(training['selection']['december_scenario']['name']),
        '@@TAIL_SHARE@@': f"{results['tail_share']:.1f}", '@@TAIL_ROWS@@': str(results['tail_rows']),
        '@@METRICS@@': table_html(metrics), '@@COMPARISON@@': comparison_html,
        '@@QUALITY@@': table_html(quality_display),
        '@@CHART@@': chart, '@@MINIMUM@@': f'{december.predicted_rate.min():,.2f}',
        '@@MAXIMUM@@': f'{december.predicted_rate.max():,.2f}', '@@PREDICTIONS@@': preview_data,
    }
    html = TEMPLATE
    for token, value in replacements.items():
        html = html.replace(token, value)
    page = project / 'outputs/results.html'
    temporary = page.with_suffix('.html.tmp')
    temporary.write_text(html, encoding='utf-8')
    temporary.replace(page)
    return page


TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Spotter | Freight rate results</title>
<style>
:root{--ink:#13343d;--muted:#5d7177;--line:#dce6e8;--accent:#075466;--paper:#fff;--bg:#f2f6f7}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1160px;margin:auto;padding:46px 24px 64px}header{display:flex;justify-content:space-between;gap:24px;align-items:start;margin-bottom:28px}
.eyebrow{text-transform:uppercase;letter-spacing:.16em;color:var(--accent);font-size:12px;font-weight:750;margin:0 0 12px}
h1{font-size:clamp(30px,5vw,44px);line-height:1.15;letter-spacing:-.03em;margin:0 0 14px}h2{font-size:23px;letter-spacing:-.02em;margin:0 0 8px}
p{margin:8px 0 16px}.muted{color:var(--muted)}.status{white-space:nowrap;border:1px solid #bad5cf;background:#e3f2ed;color:#286651;padding:7px 13px;border-radius:22px;font-size:13px;font-weight:650}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:22px}.metric,.panel{background:var(--paper);border:1px solid var(--line);border-radius:15px}
.metric{padding:20px 22px}.metric small{display:block;color:var(--muted);font-size:13px}.number{display:block;font-size:31px;font-weight:700;letter-spacing:-.04em;margin:7px 0 3px}
.panel{padding:25px 28px;margin-bottom:22px}.actions{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0 0}.button,button{display:inline-block;background:var(--accent);color:#fff;padding:11px 16px;border:1px solid var(--accent);border-radius:8px;text-decoration:none;font:inherit;font-weight:600;cursor:pointer}
.secondary,button.secondary{background:#fff;color:var(--accent)}button:disabled{opacity:.4;cursor:default}a{color:var(--accent)}.note{font-size:14px;color:var(--muted)}
.table-wrap{overflow:auto;margin-top:17px}.data-table{border-collapse:collapse;width:100%;font-size:14px;white-space:nowrap}.data-table th{text-align:left;background:#eef4f5;color:#49656d;font-size:12px;font-weight:650}
.data-table th,.data-table td{padding:12px 14px;border-bottom:1px solid var(--line)}.data-table tbody tr:hover{background:#f7fafb}.chart{display:block;width:100%;height:auto;margin:20px 0 4px}
.scenario{display:flex;flex-wrap:wrap;gap:8px;margin:16px 0}.scenario span{border:1px solid var(--line);border-radius:6px;padding:5px 9px;font-size:13px;color:var(--muted)}
.toolbar{display:flex;gap:15px;align-items:center;flex-wrap:wrap;margin:20px 0 0}label{font-size:14px;font-weight:600}input{width:min(300px,100%);border:1px solid #b9ccd1;border-radius:7px;padding:11px 12px;color:var(--ink);font:inherit}
.pagination{display:flex;gap:12px;align-items:center;justify-content:space-between;margin-top:18px}.pagination button{font-size:13px;padding:8px 13px}code{background:#eef4f5;padding:2px 6px;border-radius:4px;font-size:13px}
details summary{cursor:pointer;font-size:18px;font-weight:650}details p{margin-top:15px}
footer{font-size:13px;color:var(--muted);text-align:center;padding-top:12px}input:focus,button:focus-visible,a:focus-visible{outline:3px solid #80c4d3;outline-offset:3px}
@media(max-width:720px){main{padding:28px 16px}header{display:block}.status{display:inline-block;margin-top:12px}.metrics{grid-template-columns:repeat(2,1fr);gap:10px}.metric{padding:16px}.number{font-size:27px}.panel{padding:20px 17px}.pagination{flex-wrap:wrap}}
</style></head><body><main>
<header><div><p class="eyebrow">Spotter / Freight rate predictions</p><h1>Your prediction results</h1><p class="muted">Explore @@ROWS@@ load predictions and the daily December forecast.</p></div><span class="status">Predictions checked</span></header>
<section class="metrics" aria-label="Result summary">
<div class="metric"><small>Average error in October</small><span class="number">$@@MAE@@</span><small>Average absolute error (MAE)</small></div>
<div class="metric"><small>October error (RMSE)</small><span class="number">$@@RMSE@@</span><small>Weights larger misses more heavily</small></div>
<div class="metric"><small>Loads with predicted rates</small><span class="number">@@ROWS@@</span><small>Every required load ID included</small></div>
<div class="metric"><small>December forecast</small><span class="number">31 days</span><small>Same shipment; only the date changes</small></div>
</section>
<section class="panel"><h2>Download your predictions</h2><p class="muted">Your files are ready. Each rate is the predicted total price for a load, in dollars and rounded to cents.</p>
<div class="actions"><a class="button" href="validation_predictions.csv" download>Download submission CSV</a><a class="button secondary" href="december_predictions.csv" download>Download December CSV</a><a class="button secondary" href="candidate_december.png" target="_blank" rel="noopener">Open December chart</a></div>
<p class="note" style="margin-top:18px">The submission file includes all required load IDs in the template's order. Spotter measures final validation accuracy after submission.</p></section>
<section class="panel"><h2>How the models performed in October</h2><p class="muted">The models learned from @@EARLIER@@ January-September loads, then predicted @@OCTOBER_ROWS@@ October loads they had not trained on. The final models used for your predictions were then trained on all @@LABELED@@ labeled loads.</p>
<div class="table-wrap">@@METRICS@@</div><p class="note" style="margin-top:17px">A few large misses explain the gap between MAE and RMSE: @@TAIL_ROWS@@ loads account for @@TAIL_SHARE@@% of squared error. These loads remain included. Positive bias means prices were too high on average.</p></section>
<section class="panel"><h2>December daily rate forecast</h2><p class="muted">The same shipment is priced for each day in December. Predicted total rates range from $@@MINIMUM@@ to $@@MAXIMUM@@.</p>
<div class="scenario"><span>Lexington to Fort Wayne</span><span>360 miles</span><span>Dry Van</span><span>32,000 lb</span><span>December 2025</span></div>
<img class="chart" src="data:image/png;base64,@@CHART@@" alt="Predicted total load rates for every day in December 2025, generated by the supplied scorer">
<p class="note">This forecast uses a separate model trained with the information available for the fixed shipment. The chart shows predicted prices; actual prices for this December shipment are unavailable.</p></section>
<section class="panel"><h2>Look up a load</h2><p class="muted">Enter a load ID to find its predicted price, or browse all @@ROWS@@ loads.</p>
<div class="toolbar"><label for="search">Load ID</label><input id="search" type="search" placeholder="For example: TE-000001" autocomplete="off"><span id="count" class="note" aria-live="polite"></span></div>
<div class="table-wrap"><table class="data-table"><thead><tr><th scope="col">Load ID</th><th scope="col">Predicted total rate ($)</th></tr></thead><tbody id="prediction-rows"></tbody></table></div>
<div class="pagination"><span id="page-label" class="note" aria-live="polite"></span><div><button id="previous" class="secondary" type="button">Previous</button> <button id="next" class="secondary" type="button">Next</button></div></div>
</section>
<details class="panel"><summary>Compare modeling approaches</summary><p class="muted">The approaches were compared on July, August and September loads using only earlier loads for training. These saved results guided model choice before the separate October evaluation.</p><div class="table-wrap">@@COMPARISON@@</div>
<p><strong>Main:</strong> <code>@@MAIN_MODEL@@</code><br><strong>December:</strong> <code>@@SCENARIO_MODEL@@</code></p><p class="note">The main model uses market and quote signals. Their availability at the time of quoting remains an assumption that needs confirmation.</p></details>
<details class="panel"><summary>Data checks and cleaning</summary><p class="muted">Zero or negative weights were treated as missing. The chosen models can work with missing numeric values. Counts overlap: weights changed to missing are also included in the final missing count.</p><div class="table-wrap">@@QUALITY@@</div><p class="note">The original data files remain unchanged. The models use shipment information and calendar features; load IDs, actual target rates and prediction columns are excluded from inputs.</p></details>
<footer>Local prediction preview · All required load IDs and December shipment inputs checked.</footer>
</main><script id="prediction-data" type="application/json">@@PREDICTIONS@@</script><script>
const allRows = JSON.parse(document.getElementById('prediction-data').textContent);
const perPage = 25;
let filtered = allRows, page = 0;
const search = document.getElementById('search');
const body = document.getElementById('prediction-rows');
const previous = document.getElementById('previous'), next = document.getElementById('next');
function render() {
  body.replaceChildren();
  const start = page * perPage;
  for (const [id, rate] of filtered.slice(start, start + perPage)) {
    const row = document.createElement('tr'), idCell = document.createElement('td'), rateCell = document.createElement('td');
    idCell.textContent = id;
    rateCell.textContent = Number(rate).toLocaleString('en-US', {minimumFractionDigits:2, maximumFractionDigits:2});
    row.append(idCell, rateCell); body.append(row);
  }
  if (!filtered.length) {
    const row = document.createElement('tr'), cell = document.createElement('td');
    cell.colSpan = 2; cell.textContent = 'No matching load IDs.'; row.append(cell); body.append(row);
  }
  document.getElementById('count').textContent = filtered.length.toLocaleString() + ' matching loads';
  document.getElementById('page-label').textContent = filtered.length ? `Rows ${start+1}-${Math.min(start+perPage,filtered.length)} of ${filtered.length.toLocaleString()}` : '0 rows';
  previous.disabled = page === 0; next.disabled = start + perPage >= filtered.length;
}
search.addEventListener('input', () => { const query = search.value.trim().toUpperCase(); filtered = allRows.filter(row => row[0].toUpperCase().includes(query)); page = 0; render(); });
previous.addEventListener('click', () => { if (page > 0) { page--; render(); } });
next.addEventListener('click', () => { if ((page+1)*perPage < filtered.length) { page++; render(); } });
render();
</script></body></html>'''
