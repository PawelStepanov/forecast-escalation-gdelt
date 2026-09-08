import pandas as pd

from src.diagnostics import escalation_diagnostics


def test_escalation_diagnostics_direction_and_large_rise():
    weeks = pd.to_datetime(['2025-01-06', '2025-01-13', '2025-01-20'])
    lookup = pd.Series([100.0, 100.0, 130.0], index=weeks)
    predictions = pd.DataFrame({
        'split': ['cv', 'cv'],
        'horizon': [1, 1],
        'model': ['m', 'm'],
        'origin_week': [weeks[0], weeks[1]],
        'target_week': [weeks[1], weeks[2]],
        'actual': [100.0, 130.0],
        'prediction': [100.0, 125.0],
    })

    out = escalation_diagnostics(predictions, lookup, large_rise_fraction=0.20)
    row = out.iloc[0]
    assert row['n_actual_large_rises'] == 1
    assert row['large_rise_direction_recall'] == 1.0
    assert row['large_rise_magnitude_recall'] == 1.0
