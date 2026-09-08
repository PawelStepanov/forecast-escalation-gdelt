from __future__ import annotations

import numpy as np
import pandas as pd


def escalation_diagnostics(
    predictions: pd.DataFrame,
    origin_lookup: pd.Series,
    large_rise_fraction: float = 0.20,
) -> pd.DataFrame:
    """Вторичная диагностика поведения в точке перелома/эскалации.
    
    Эти метрики намеренно *не* используются для выбора модели. Они помогают
    объяснить режим отказа, который могут скрывать агрегированные метрики
    регрессии: модель может иметь хорошие показатели MASE/RMSE, при этом
    сильно сглаживая большие восходящие движения.
    
    large_rise_fraction=0.20 означает, что наблюдаемое целевое значение
    как минимум на 20% выше значения из точки прогноза. Порог является
    диагностической эвристикой, а не официальным определением целевого
    значения задачи.
    """
    p = predictions.copy()
    p['origin_actual'] = p['origin_week'].map(origin_lookup)
    p = p.dropna(subset=['origin_actual', 'actual', 'prediction'])

    p['actual_change'] = p['actual'] - p['origin_actual']
    p['predicted_change'] = p['prediction'] - p['origin_actual']
    p['direction_correct'] = (
        np.sign(p['actual_change']) == np.sign(p['predicted_change'])
    )
    p['actual_large_rise'] = (
        p['actual'] >= (1.0 + large_rise_fraction) * p['origin_actual']
    )
    p['predicted_large_rise'] = (
        p['prediction'] >= (1.0 + large_rise_fraction) * p['origin_actual']
    )

    rows = []
    for (split, horizon, model), g in p.groupby(['split', 'horizon', 'model']):
        large = g['actual_large_rise']
        n_large = int(large.sum())

        rows.append({
            'split': split,
            'horizon': int(horizon),
            'model': model,
            'n': int(len(g)),
            'direction_accuracy': float(g['direction_correct'].mean()),
            'large_rise_fraction': float(large_rise_fraction),
            'n_actual_large_rises': n_large,
            'large_rise_direction_recall': (
                float((g.loc[large, 'predicted_change'] > 0).mean())
                if n_large else np.nan
            ),
            'large_rise_magnitude_recall': (
                float(g.loc[large, 'predicted_large_rise'].mean())
                if n_large else np.nan
            ),
            'mean_error_on_large_rises': (
                float((g.loc[large, 'prediction'] - g.loc[large, 'actual']).mean())
                if n_large else np.nan
            ),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(['split', 'horizon', 'model'])
        .reset_index(drop=True)
    )
