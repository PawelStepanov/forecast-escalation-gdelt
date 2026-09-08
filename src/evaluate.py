from __future__ import annotations

import numpy as np
import pandas as pd

from .features import CALENDAR_FEATURES, TARGET_HISTORY_FEATURES, feature_sets, make_supervised
from .metrics import mase, rmse, smape
from .models import (
    ets_prediction,
    fit_predict_ridge_log,
    fit_predict_xgb,
    last_value_prediction,
    rolling_mean_prediction,
    seasonal_naive_prediction,
)


def evaluation_weeks(df: pd.DataFrame, cv_weeks: int, holdout_weeks: int):
    weeks = pd.DatetimeIndex(df['week_start'].sort_values().unique())
    holdout = weeks[-holdout_weeks:]
    cv = weeks[-(holdout_weeks + cv_weeks):-holdout_weeks]
    return cv, holdout


def mase_scale(df: pd.DataFrame, cutoff: pd.Timestamp) -> float:
    hist = df.loc[
        (df['week_start'] < cutoff) & (~df['data_quality_gap_week']),
        ['week_start', 'conflict_count'],
    ]
    lookup = hist.set_index('week_start')['conflict_count'].astype(float)
    diffs = []
    for t, value in lookup.items():
        prev = t - pd.Timedelta(weeks=1)
        if prev in lookup.index:
            diffs.append(abs(value - lookup.loc[prev]))
    return float(np.mean(diffs))


def walk_forward_predictions(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    cv_weeks, holdout_weeks = evaluation_weeks(
        df,
        int(cfg['forecast']['cv_weeks']),
        int(cfg['forecast']['holdout_weeks']),
    )
    lookup = df.set_index('week_start')['conflict_count'].astype(float)
    bad_weeks = pd.to_datetime(cfg['quality']['corrupted_week_starts'])
    fsets = feature_sets()

    rows = []
    for horizon in cfg['forecast']['horizons']:
        horizon = int(horizon)
        sup = make_supervised(df, horizon, bad_weeks)

        for split_name, target_weeks in [('cv', cv_weeks), ('holdout', holdout_weeks)]:
            for target_week in target_weeks:
                target_week = pd.Timestamp(target_week)
                if target_week in bad_weeks:
                    continue

                origin = target_week - pd.Timedelta(weeks=horizon)
                if origin not in lookup.index:
                    continue

                origin_row = df.loc[df['week_start'] == origin]
                if origin_row.empty or not bool(origin_row['origin_valid_full_features'].iloc[0]):
                    continue

                actual = float(lookup.loc[target_week])

                candidates = {
                    'last_value': last_value_prediction(lookup, origin),
                    'rolling_mean_4w': rolling_mean_prediction(df, origin, 4),
                    'seasonal_naive_52w': seasonal_naive_prediction(lookup, target_week, 52),
                    'ets': ets_prediction(df, origin, horizon),
                }

                ridge_cols = TARGET_HISTORY_FEATURES + CALENDAR_FEATURES
                candidates['ridge_log_history'] = fit_predict_ridge_log(
                    sup,
                    origin,
                    target_week,
                    ridge_cols,
                    float(cfg['ridge']['alpha']),
                )

                for model_name, cols in fsets.items():
                    candidates[model_name] = fit_predict_xgb(
                        sup,
                        origin,
                        target_week,
                        cols,
                        cfg['xgboost'],
                        int(cfg['project']['random_state']),
                    )

                for model_name, prediction in candidates.items():
                    if not np.isfinite(prediction):
                        continue
                    rows.append({
                        'split': split_name,
                        'horizon': horizon,
                        'model': model_name,
                        'origin_week': origin,
                        'target_week': target_week,
                        'actual': actual,
                        'prediction': float(prediction),
                        'residual': actual - float(prediction),
                    })

    return pd.DataFrame(rows)


def metric_table(predictions: pd.DataFrame, df: pd.DataFrame, split: str) -> pd.DataFrame:
    part = predictions.loc[predictions['split'] == split].copy()
    scale = mase_scale(df, part['target_week'].min())

    rows = []
    for (horizon, model), grp in part.groupby(['horizon', 'model']):
        rows.append({
            'horizon': int(horizon),
            'model': model,
            'n': int(len(grp)),
            'MASE': mase(grp['actual'], grp['prediction'], scale),
            'sMAPE': smape(grp['actual'], grp['prediction']),
            'RMSE': rmse(grp['actual'], grp['prediction']),
        })

    out = pd.DataFrame(rows)
    base = out.loc[out['model'] == 'last_value', ['horizon', 'RMSE']].rename(
        columns={'RMSE': 'last_value_RMSE'}
    )
    out = out.merge(base, on='horizon', how='left')
    out['vs_last_value_RMSE_pct'] = (
        (out['last_value_RMSE'] - out['RMSE']) / out['last_value_RMSE'] * 100.0
    )
    return out.sort_values(['horizon', 'RMSE']).reset_index(drop=True)


def choose_models(cv_metrics: pd.DataFrame) -> pd.DataFrame:
    # Отбор производится только на основе резюме,
    # никогда не на основе контрольной выборки.
    return (
        cv_metrics
        .sort_values(['horizon', 'MASE', 'RMSE'])
        .groupby('horizon', as_index=False)
        .first()[['horizon', 'model', 'MASE', 'sMAPE', 'RMSE']]
    )
