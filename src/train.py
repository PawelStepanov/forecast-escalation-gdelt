from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from .config import load_config
from .data import load_and_merge, save_processed
from .diagnostics import escalation_diagnostics
from .evaluate import choose_models, evaluation_weeks, metric_table, walk_forward_predictions
from .features import CALENDAR_FEATURES, TARGET_HISTORY_FEATURES, build_origin_features, feature_sets, make_supervised
from .models import (
    ets_prediction,
    make_ridge_log,
    last_value_prediction,
    rolling_mean_prediction,
    seasonal_naive_prediction,
)


def save_figures(df, predictions, selected, cv_metrics, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. История с известными инцидентами, связанными с качеством данных,
    # и окончательный контрольный образец.
    _, holdout = evaluation_weeks(df, 52, 12)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df['week_start'], df['conflict_count'])
    ax.axvspan(pd.Timestamp('2025-06-15'), pd.Timestamp('2025-07-01'), alpha=0.18, label='GDELT coverage incident')
    ax.axvspan(holdout.min(), holdout.max(), alpha=0.12, label='final holdout')
    ax.set_title('Weekly GDELT conflict activity — Israel')
    ax.set_xlabel('Week')
    ax.set_ylabel('GDELT conflict event records')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / 'target_history.png', dpi=160)
    plt.close(fig)

    # 2. Окончательная контрольная выборка для модели, отобранной с помощью
    # перекрестной проверки, отдельно для h=1..4.
    for horizon in sorted(selected['horizon'].astype(int).unique()):
        model = selected.loc[selected['horizon'] == horizon, 'model'].iloc[0]
        g = predictions.loc[
            (predictions['split'] == 'holdout')
            & (predictions['horizon'] == horizon)
            & (predictions['model'] == model)
        ].sort_values('target_week')
        if g.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(g['target_week'], g['actual'], marker='o', label='actual')
        ax.plot(g['target_week'], g['prediction'], marker='o', label=model)
        ax.set_title(f'Final holdout — horizon {horizon} week' + ('' if horizon == 1 else 's'))
        ax.set_xlabel('Target week')
        ax.set_ylabel('GDELT conflict event records')
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f'holdout_h{horizon}.png', dpi=160)
        plt.close(fig)

    # 3. Фактические значения CV по сравнению с выбранной моделью, отдельно для h=1..4.
    # Эти графики показывают режимы сбоев в точке перегиба/сглаживания,
    # которые могут быть скрыты с помощью суммирования MASE/RMSE.
    for horizon in sorted(selected['horizon'].astype(int).unique()):
        model = selected.loc[selected['horizon'] == horizon, 'model'].iloc[0]
        g = predictions.loc[
            (predictions['split'] == 'cv')
            & (predictions['horizon'] == horizon)
            & (predictions['model'] == model)
        ].sort_values('target_week')
        if g.empty:
            continue

        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(g['target_week'], g['actual'], marker='o', label='actual')
        ax.plot(g['target_week'], g['prediction'], marker='o', label=model)
        ax.set_title(f'Walk-forward CV — selected model, horizon {horizon} week' + ('' if horizon == 1 else 's'))
        ax.set_xlabel('Target week')
        ax.set_ylabel('GDELT conflict event records')
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f'cv_selected_h{horizon}.png', dpi=160)
        plt.close(fig)

    # 4. CV MASE по модели и горизонту. Это основной показатель выбора модели.
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for model_name, grp in cv_metrics.groupby('model'):
        grp = grp.sort_values('horizon')
        ax.plot(grp['horizon'], grp['MASE'], marker='o', label=model_name)
    ax.set_title('Walk-forward CV MASE by horizon')
    ax.set_xlabel('Forecast horizon, weeks')
    ax.set_ylabel('MASE')
    ax.set_xticks([1, 2, 3, 4])
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / 'cv_mase_by_horizon.png', dpi=160)
    plt.close(fig)

    # 5. СV RMSE в зависимости от модели и горизонта прогнозирования как
    # дополнительный показатель абсолютной ошибки.
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for model_name, grp in cv_metrics.groupby('model'):
        grp = grp.sort_values('horizon')
        ax.plot(grp['horizon'], grp['RMSE'], marker='o', label=model_name)
    ax.set_title('Walk-forward CV RMSE by horizon')
    ax.set_xlabel('Forecast horizon, weeks')
    ax.set_ylabel('RMSE')
    ax.set_xticks([1, 2, 3, 4])
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / 'cv_rmse_by_horizon.png', dpi=160)
    plt.close(fig)


def future_forecast(df, predictions, selected, cfg):
    origin = pd.Timestamp(df['week_start'].max())
    lookup = df.set_index('week_start')['conflict_count'].astype(float)
    bad_weeks = pd.to_datetime(cfg['quality']['corrupted_week_starts'])
    fsets = feature_sets()
    rows = []

    for _, chosen in selected.iterrows():
        h = int(chosen['horizon'])
        model_name = str(chosen['model'])
        target_week = origin + pd.Timedelta(weeks=h)

        if model_name == 'last_value':
            pred = last_value_prediction(lookup, origin)
        elif model_name == 'rolling_mean_4w':
            pred = rolling_mean_prediction(df, origin, 4)
        elif model_name == 'seasonal_naive_52w':
            pred = seasonal_naive_prediction(lookup, target_week, 52)
        elif model_name == 'ets':
            pred = ets_prediction(df, origin, h)
        elif model_name == 'ridge_log_history':
            sup = make_supervised(df, h, bad_weeks)
            cols = TARGET_HISTORY_FEATURES + CALENDAR_FEATURES
            train = sup.loc[
                (sup['target_week'] <= origin)
                & (~sup['target_is_bad'])
                & (sup['origin_valid_full_features'])
            ].dropna(subset=cols + ['target'])

            row = df.loc[df['week_start'] == origin].copy()
            row['origin_week'] = origin
            row['target_week'] = target_week
            iso_week = float(target_week.isocalendar().week)
            row['target_week_sin'] = np.sin(2 * np.pi * iso_week / 52.18)
            row['target_week_cos'] = np.cos(2 * np.pi * iso_week / 52.18)
            row['target_month'] = float(target_week.month)

            model = make_ridge_log(float(cfg['ridge']['alpha']))
            model.fit(
                train[cols].to_numpy(float),
                np.log1p(train['target'].to_numpy(float)),
            )
            pred = max(0.0, float(np.expm1(model.predict(row[cols].to_numpy(float))[0])))
        elif model_name in fsets:
            sup = make_supervised(df, h, bad_weeks)
            cols = fsets[model_name]
            train = sup.loc[
                (sup['target_week'] <= origin)
                & (~sup['target_is_bad'])
                & (sup['origin_valid_full_features'])
            ].dropna(subset=cols + ['target'])

            row = df.loc[df['week_start'] == origin].copy()
            row['origin_week'] = origin
            row['target_week'] = target_week
            iso_week = float(target_week.isocalendar().week)
            row['target_week_sin'] = np.sin(2 * np.pi * iso_week / 52.18)
            row['target_week_cos'] = np.cos(2 * np.pi * iso_week / 52.18)
            row['target_month'] = float(target_week.month)

            params = dict(cfg['xgboost'])
            params['random_state'] = int(cfg['project']['random_state'])
            model = XGBRegressor(**params)
            model.fit(train[cols].to_numpy(float), train['target'].to_numpy(float))
            pred = max(0.0, float(model.predict(row[cols].to_numpy(float))[0]))
        else:
            raise ValueError(model_name)

        alpha = float(cfg['forecast']['prediction_interval_alpha'])
        residuals = predictions.loc[
            (predictions['split'] == 'cv')
            & (predictions['horizon'] == h)
            & (predictions['model'] == model_name),
            'residual',
        ].dropna()
        lo = max(0.0, pred + float(residuals.quantile(alpha / 2)))
        hi = max(lo, pred + float(residuals.quantile(1 - alpha / 2)))

        rows.append({
            'country': cfg['project']['country_name'],
            'forecast_week': f'{target_week.isocalendar().year}-W{int(target_week.isocalendar().week):02d}',
            'horizon': h,
            'target_type': 'count',
            'predicted_value': pred,
            'predicted_prob': np.nan,
            'ci_lower': lo,
            'ci_upper': hi,
            'model': model_name,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw = load_and_merge(cfg)
    df = build_origin_features(raw)
    save_processed(df, cfg)

    predictions = walk_forward_predictions(df, cfg)
    cv_metrics = metric_table(predictions, df, 'cv')
    holdout_metrics = metric_table(predictions, df, 'holdout')
    selected = choose_models(cv_metrics)
    forecast = future_forecast(df, predictions, selected, cfg)

    out = Path('outputs')
    out.mkdir(exist_ok=True)
    predictions.to_csv(out / 'walk_forward_predictions.csv', index=False)
    cv_metrics.to_csv(out / 'cv_metrics.csv', index=False)
    holdout_metrics.to_csv(out / 'holdout_metrics.csv', index=False)
    selected.to_csv(out / 'selected_models_from_cv.csv', index=False)
    forecast.to_csv(out / 'forecast.csv', index=False)

    # Абляция XGBoost для эксперимента с внешним источником.
    xnames = ['xgb_target_history', 'xgb_gdelt_features', 'xgb_gdelt_world_bank']
    ablation = cv_metrics.loc[cv_metrics['model'].isin(xnames), ['horizon', 'model', 'RMSE']]
    pivot = ablation.pivot(index='horizon', columns='model', values='RMSE').reset_index()
    pivot['gdelt_vs_history_improvement_pct'] = (
        (pivot['xgb_target_history'] - pivot['xgb_gdelt_features']) / pivot['xgb_target_history'] * 100
    )
    pivot['world_bank_vs_gdelt_improvement_pct'] = (
        (pivot['xgb_gdelt_features'] - pivot['xgb_gdelt_world_bank']) / pivot['xgb_gdelt_features'] * 100
    )
    pivot.to_csv(out / 'world_bank_ablation.csv', index=False)

    selected_holdout = holdout_metrics.merge(selected[['horizon', 'model']], on=['horizon', 'model'])
    selected_holdout.to_csv(out / 'selected_policy_holdout.csv', index=False)

    # Вторичная диагностика, ориентированная на бизнес. Она НЕ используется для выбора модели.
    # Рост на 20% — это лишь прозрачная эвристика чувствительности для проверки того,
    # сглаживает ли модель с низкой погрешностью амплитуду эскалации.
    origin_lookup = df.set_index('week_start')['conflict_count'].astype(float)
    diagnostics = escalation_diagnostics(predictions, origin_lookup, large_rise_fraction=0.20)
    diagnostics.to_csv(out / 'escalation_diagnostics.csv', index=False)

    metadata = {
        'country': cfg['project']['country_name'],
        'target': "weekly count of GDELT EventRootCode in {'18','19'}",
        'forecast_origin_last_week': str(pd.Timestamp(df['week_start'].max()).date()),
        'selected_models': {str(int(r.horizon)): str(r.model) for _, r in selected.iterrows()},
        'corrupted_weeks': cfg['quality']['corrupted_week_starts'],
        'secondary_diagnostic_large_rise_fraction': 0.20,
    }
    (out / 'metrics.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')

    save_figures(df, predictions, selected, cv_metrics, out / 'figures')

    print('\nSelected models from CV:')
    print(selected.to_string(index=False))
    print('\nSelected-policy holdout:')
    print(selected_holdout[['horizon','model','MASE','sMAPE','RMSE']].to_string(index=False))
    print('\nFuture forecast:')
    print(forecast.to_string(index=False))
    print('\nSelected-model escalation diagnostics on CV (secondary, not selection metrics):')
    selected_diag = diagnostics.merge(selected[['horizon', 'model']], on=['horizon', 'model'])
    selected_diag = selected_diag.loc[selected_diag['split'] == 'cv']
    print(selected_diag[[
        'horizon', 'model', 'direction_accuracy',
        'n_actual_large_rises', 'large_rise_direction_recall',
        'large_rise_magnitude_recall', 'mean_error_on_large_rises'
    ]].to_string(index=False))


if __name__ == '__main__':
    main()
