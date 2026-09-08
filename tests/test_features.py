import pandas as pd
from src.features import build_origin_features, make_supervised


def test_target_shift():
    weeks = pd.date_range('2024-01-01', periods=20, freq='W-MON')
    df = pd.DataFrame({
        'week_start': weeks,
        'conflict_count': range(20),
        'total_events_country': [100] * 20,
        'avg_goldstein': [-5.0] * 20,
        'goldstein_std': [1.0] * 20,
        'avg_tone': [-2.0] * 20,
        'conflict_mentions': [10] * 20,
        'total_mentions_country': [1000] * 20,
        'inflation_pct': [2.0] * 20,
        'unemployment_pct': [4.0] * 20,
        'gdp_growth_pct': [3.0] * 20,
        'wb_feature_age_years': [2.0] * 20,
        'data_quality_gap_week': [False] * 20,
        'origin_valid_full_features': [True] * 20,
    })
    feat = build_origin_features(df)
    sup = make_supervised(feat, 4, [])
    assert sup.loc[0, 'target'] == 4
    assert sup.loc[0, 'target_week'] == weeks[4]
