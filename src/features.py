from __future__ import annotations

import numpy as np
import pandas as pd


TARGET_HISTORY_FEATURES = [
    'conflict_count_t',
    'conflict_count_lag1',
    'conflict_count_lag2',
    'conflict_count_lag4',
    'conflict_count_lag12',
    'rolling_mean_4w',
    'rolling_std_4w',
    'rolling_mean_8w',
    'rolling_std_12w',
    'rolling_max_8w',
    'cumulative_conflict_12w',
]

GDELT_FEATURES = [
    'avg_goldstein',
    'goldstein_std',
    'avg_tone',
    'total_events_country',
    'news_volume_change_wow',
    'total_mentions_country',
    'conflict_share',
    'mentions_per_conflict',
    'goldstein_volatility_4w',
]

WORLD_BANK_FEATURES = [
    'inflation_pct',
    'unemployment_pct',
    'gdp_growth_pct',
    'wb_feature_age_years',
]

CALENDAR_FEATURES = ['target_week_sin', 'target_week_cos', 'target_month']


def build_origin_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values('week_start').reset_index(drop=True)
    y = out['conflict_count'].astype(float)

    out['conflict_count_t'] = y
    for lag in [1, 2, 4, 12]:
        out[f'conflict_count_lag{lag}'] = y.shift(lag)

    # Прогноз составляется после завершения недели t, поэтому значения за неделю t
    # известны и могут быть использованы для формирования значений y(t+h).
    out['rolling_mean_4w'] = y.rolling(4, min_periods=4).mean()
    out['rolling_std_4w'] = y.rolling(4, min_periods=4).std()
    out['rolling_mean_8w'] = y.rolling(8, min_periods=8).mean()
    out['rolling_std_12w'] = y.rolling(12, min_periods=12).std()
    out['rolling_max_8w'] = y.rolling(8, min_periods=8).max()
    out['cumulative_conflict_12w'] = y.rolling(12, min_periods=12).sum()

    out['goldstein_volatility_4w'] = (
        out['avg_goldstein'].rolling(4, min_periods=4).std()
    )
    out['news_volume_change_wow'] = (
        out['total_events_country']
        .pct_change(fill_method=None)
        .replace([np.inf, -np.inf], np.nan)
    )
    out['conflict_share'] = (
        out['conflict_count'] / out['total_events_country'].replace(0, np.nan)
    )
    out['mentions_per_conflict'] = (
        out['conflict_mentions'] / out['conflict_count'].replace(0, np.nan)
    )

    return out


def make_supervised(origin_features: pd.DataFrame, horizon: int, bad_weeks) -> pd.DataFrame:
    out = origin_features.copy()
    out['origin_week'] = out['week_start']
    out['target_week'] = out['week_start'] + pd.to_timedelta(7 * horizon, unit='D')
    out['target'] = out['conflict_count'].shift(-horizon)
    out['target_is_bad'] = out['target_week'].isin(pd.to_datetime(bad_weeks))

    iso_week = out['target_week'].dt.isocalendar().week.astype(float)
    out['target_week_sin'] = np.sin(2 * np.pi * iso_week / 52.18)
    out['target_week_cos'] = np.cos(2 * np.pi * iso_week / 52.18)
    out['target_month'] = out['target_week'].dt.month.astype(float)
    return out


def feature_sets():
    return {
        'xgb_target_history': TARGET_HISTORY_FEATURES + CALENDAR_FEATURES,
        'xgb_gdelt_features': TARGET_HISTORY_FEATURES + GDELT_FEATURES + CALENDAR_FEATURES,
        'xgb_gdelt_world_bank': TARGET_HISTORY_FEATURES + GDELT_FEATURES + WORLD_BANK_FEATURES + CALENDAR_FEATURES,
    }
