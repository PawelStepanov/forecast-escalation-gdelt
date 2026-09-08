from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_and_merge(cfg: dict) -> pd.DataFrame:
    gdelt = pd.read_csv(cfg['data']['gdelt_csv'], parse_dates=['week_start'])
    wb = pd.read_csv(cfg['data']['world_bank_csv'], parse_dates=['available_from'])

    gdelt = gdelt.sort_values('week_start').reset_index(drop=True)
    wb = wb.sort_values('available_from').reset_index(drop=True)

    macro_cols = ['inflation_pct', 'unemployment_pct', 'gdp_growth_pct']

    merged = pd.merge_asof(
        gdelt,
        wb[['available_from', 'year'] + macro_cols],
        left_on='week_start',
        right_on='available_from',
        direction='backward',
    ).rename(columns={'year': 'wb_observation_year'})

    merged['wb_feature_age_years'] = (
        merged['week_start'].dt.year - merged['wb_observation_year']
    )

    bad_weeks = pd.to_datetime(cfg['quality']['corrupted_week_starts'])
    merged['data_quality_gap_week'] = merged['week_start'].isin(bad_weeks)

    lookback = int(cfg['quality']['max_feature_lookback_weeks'])

    def origin_affected(t: pd.Timestamp) -> bool:
        start = t - pd.Timedelta(weeks=lookback - 1)
        return bool(((bad_weeks >= start) & (bad_weeks <= t)).any())

    merged['origin_valid_full_features'] = ~merged['week_start'].map(origin_affected)

    # Базовые проверки
    if merged['week_start'].duplicated().any():
        raise ValueError('Duplicate weekly timestamps found.')

    diffs = merged['week_start'].diff().dropna()
    if not diffs.eq(pd.Timedelta(days=7)).all():
        raise ValueError('Weekly time index is not continuous.')

    return merged


def save_processed(df: pd.DataFrame, cfg: dict) -> None:
    out = Path(cfg['data']['processed_csv'])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
