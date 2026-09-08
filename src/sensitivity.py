from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import load_config
from .data import load_and_merge
from .evaluate import evaluation_weeks, mase_scale
from .features import (
    CALENDAR_FEATURES,
    TARGET_HISTORY_FEATURES,
    build_origin_features,
    make_supervised,
)
from .metrics import mase, rmse, smape


CANDIDATES = {
    'ridge_raw_alpha1': ('ridge_raw', {'alpha': 1.0}),
    'ridge_log_alpha1': ('ridge_log', {'alpha': 1.0}),
    'ridge_delta_alpha1': ('ridge_delta', {'alpha': 1.0}),
    'ridge_logratio_alpha1': ('ridge_logratio', {'alpha': 1.0}),
    'huber_raw': ('huber_raw', {'epsilon': 1.35, 'alpha': 1e-4}),
    'huber_log': ('huber_log', {'epsilon': 1.35, 'alpha': 1e-4}),
    'elasticnet_log_a001': ('elasticnet_log', {'alpha': 0.01, 'l1_ratio': 0.2}),
    'elasticnet_log_a01': ('elasticnet_log', {'alpha': 0.1, 'l1_ratio': 0.2}),
    'elasticnet_log_a1': ('elasticnet_log', {'alpha': 1.0, 'l1_ratio': 0.2}),
}


def _fit_predict(kind, params, train, row, cols, origin_value):
    X = train[cols].to_numpy(float)
    Xrow = row[cols].to_numpy(float)
    y = train['target'].to_numpy(float)
    y_origin_train = train['conflict_count_t'].to_numpy(float)

    if kind == 'ridge_raw':
        model = make_pipeline(StandardScaler(), Ridge(alpha=params['alpha']))
        model.fit(X, y)
        pred = model.predict(Xrow)[0]
    elif kind == 'ridge_log':
        model = make_pipeline(StandardScaler(), Ridge(alpha=params['alpha']))
        model.fit(X, np.log1p(y))
        pred = np.expm1(model.predict(Xrow)[0])
    elif kind == 'ridge_delta':
        model = make_pipeline(StandardScaler(), Ridge(alpha=params['alpha']))
        model.fit(X, y - y_origin_train)
        pred = origin_value + model.predict(Xrow)[0]
    elif kind == 'ridge_logratio':
        model = make_pipeline(StandardScaler(), Ridge(alpha=params['alpha']))
        model.fit(X, np.log1p(y) - np.log1p(y_origin_train))
        pred = np.expm1(np.log1p(origin_value) + model.predict(Xrow)[0])
    elif kind == 'huber_raw':
        model = make_pipeline(
            StandardScaler(),
            HuberRegressor(
                epsilon=params['epsilon'], alpha=params['alpha'], max_iter=2000
            ),
        )
        model.fit(X, y)
        pred = model.predict(Xrow)[0]
    elif kind == 'huber_log':
        model = make_pipeline(
            StandardScaler(),
            HuberRegressor(
                epsilon=params['epsilon'], alpha=params['alpha'], max_iter=2000
            ),
        )
        model.fit(X, np.log1p(y))
        pred = np.expm1(model.predict(Xrow)[0])
    elif kind == 'elasticnet_log':
        model = make_pipeline(
            StandardScaler(),
            ElasticNet(
                alpha=params['alpha'],
                l1_ratio=params['l1_ratio'],
                max_iter=10000,
                random_state=42,
            ),
        )
        model.fit(X, np.log1p(y))
        pred = np.expm1(model.predict(Xrow)[0])
    else:
        raise ValueError(kind)

    return max(0.0, float(pred))


def sensitivity_predictions(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    cv_weeks, _ = evaluation_weeks(
        df,
        int(cfg['forecast']['cv_weeks']),
        int(cfg['forecast']['holdout_weeks']),
    )
    lookup = df.set_index('week_start')['conflict_count'].astype(float)
    bad_weeks = pd.to_datetime(cfg['quality']['corrupted_week_starts'])
    cols = TARGET_HISTORY_FEATURES + CALENDAR_FEATURES

    rows = []
    for h in cfg['forecast']['horizons']:
        h = int(h)
        sup = make_supervised(df, h, bad_weeks)
        for target_week in cv_weeks:
            target_week = pd.Timestamp(target_week)
            if target_week in bad_weeks:
                continue
            origin = target_week - pd.Timedelta(weeks=h)
            if origin not in lookup.index:
                continue

            row = sup.loc[
                (sup['origin_week'] == origin) & (sup['target_week'] == target_week)
            ].copy()
            if row.empty or not bool(row['origin_valid_full_features'].iloc[0]):
                continue
            row = row.dropna(subset=cols)
            if row.empty:
                continue

            train = sup.loc[
                (sup['target_week'] <= origin)
                & (~sup['target_is_bad'])
                & (sup['origin_valid_full_features'])
            ].dropna(subset=cols + ['target'])
            if len(train) < 80:
                continue

            for name, (kind, params) in CANDIDATES.items():
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    pred = _fit_predict(
                        kind, params, train, row, cols, float(lookup.loc[origin])
                    )
                rows.append({
                    'horizon': h,
                    'model': name,
                    'origin_week': origin,
                    'target_week': target_week,
                    'actual': float(lookup.loc[target_week]),
                    'prediction': pred,
                })

    return pd.DataFrame(rows)


def summarize(predictions: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    scale = mase_scale(df, predictions['target_week'].min())
    rows = []
    for (h, model), g in predictions.groupby(['horizon', 'model']):
        origin_lookup = df.set_index('week_start')['conflict_count'].astype(float)
        origin_actual = g['origin_week'].map(origin_lookup)
        actual_change = g['actual'].to_numpy(float) - origin_actual.to_numpy(float)
        pred_change = g['prediction'].to_numpy(float) - origin_actual.to_numpy(float)
        large = g['actual'].to_numpy(float) >= 1.2 * origin_actual.to_numpy(float)

        rows.append({
            'horizon': int(h),
            'model': model,
            'n': int(len(g)),
            'MASE': mase(g['actual'], g['prediction'], scale),
            'sMAPE': smape(g['actual'], g['prediction']),
            'RMSE': rmse(g['actual'], g['prediction']),
            'direction_accuracy': float((np.sign(actual_change) == np.sign(pred_change)).mean()),
            'n_actual_large_rises_20pct': int(large.sum()),
            'large_rise_direction_recall': (
                float((pred_change[large] > 0).mean()) if large.any() else np.nan
            ),
            'large_rise_20pct_recall': (
                float((g['prediction'].to_numpy(float)[large] >= 1.2 * origin_actual.to_numpy(float)[large]).mean())
                if large.any() else np.nan
            ),
        })

    return pd.DataFrame(rows).sort_values(['horizon', 'MASE']).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--output', default='outputs/model_sensitivity_cv.csv')
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw = load_and_merge(cfg)
    df = build_origin_features(raw)
    preds = sensitivity_predictions(df, cfg)
    result = summarize(preds, df)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    print(result.to_string(index=False))


if __name__ == '__main__':
    main()
