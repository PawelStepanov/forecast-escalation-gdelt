from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


def last_value_prediction(lookup: pd.Series, origin_week: pd.Timestamp) -> float:
    return max(0.0, float(lookup.loc[origin_week]))


def rolling_mean_prediction(df: pd.DataFrame, origin_week: pd.Timestamp, window: int = 4) -> float:
    history = df.loc[df['week_start'] <= origin_week, 'conflict_count'].tail(window)
    return max(0.0, float(history.mean()))


def seasonal_naive_prediction(lookup: pd.Series, target_week: pd.Timestamp, period: int = 52) -> float:
    ref_week = target_week - pd.Timedelta(weeks=period)
    if ref_week not in lookup.index:
        return float('nan')
    return max(0.0, float(lookup.loc[ref_week]))


def ets_prediction(df: pd.DataFrame, origin_week: pd.Timestamp, horizon: int, alpha: float = 0.30) -> float:
    """Быстрый статистический baseline ETS(A,N,N) / простое экспоненциальное сглаживание.
    
    Параметр alpha фиксируется до начала оценки, чтобы результат был полностью
    воспроизводимым и чтобы избежать подбора параметров по финальному holdout.
    Для ETS(A,N,N) прогноз на h шагов вперёд равен последнему сглаженному уровню ряда.
    """
    history = df.loc[
        (df['week_start'] <= origin_week) & (~df['data_quality_gap_week']),
        'conflict_count',
    ].astype(float).to_numpy()

    if len(history) == 0:
        return float('nan')

    level = float(history[0])
    for value in history[1:]:
        level = alpha * float(value) + (1.0 - alpha) * level
    return max(0.0, level)


def make_xgb(params: dict, random_state: int) -> XGBRegressor:
    p = dict(params)
    p['random_state'] = random_state
    return XGBRegressor(**p)


def fit_predict_xgb(
    supervised: pd.DataFrame,
    origin_week: pd.Timestamp,
    target_week: pd.Timestamp,
    feature_cols: list[str],
    params: dict,
    random_state: int,
) -> float:
    row = supervised.loc[
        (supervised['origin_week'] == origin_week)
        & (supervised['target_week'] == target_week)
    ].copy()

    if row.empty or not bool(row['origin_valid_full_features'].iloc[0]):
        return float('nan')

    row = row.dropna(subset=feature_cols)
    if row.empty:
        return float('nan')

    # Критически важное правило предотвращения утечек в многогоризонтном прогнозе:
    # метка подходит для размещения только в том случае, если целевая неделя
    # не позже текущей точки отсчета прогноза.
    train = supervised.loc[
        (supervised['target_week'] <= origin_week)
        & (~supervised['target_is_bad'])
        & (supervised['origin_valid_full_features'])
    ].dropna(subset=feature_cols + ['target'])

    if len(train) < 80:
        return float('nan')

    model = make_xgb(params, random_state)
    model.fit(
        train[feature_cols].to_numpy(dtype=float),
        train['target'].to_numpy(dtype=float),
    )
    pred = model.predict(row[feature_cols].to_numpy(dtype=float))[0]
    return max(0.0, float(pred))


def make_ridge_log(alpha: float = 1.0):
    """Регуляризованная линейная модель, обучаемая на log1p(target).
    
    StandardScaler обучается только на текущем обучающем окне, чтобы избежать
    утечки данных. Логарифмическое преобразование уменьшает влияние экстремальных
    всплесков conflict_count на обучение модели. После обратного преобразования
    прогноз дополнительно ограничивается снизу нулём.
    """
    return make_pipeline(StandardScaler(), Ridge(alpha=float(alpha)))


def fit_predict_ridge_log(
    supervised: pd.DataFrame,
    origin_week: pd.Timestamp,
    target_week: pd.Timestamp,
    feature_cols: list[str],
    alpha: float = 1.0,
) -> float:
    row = supervised.loc[
        (supervised['origin_week'] == origin_week)
        & (supervised['target_week'] == target_week)
    ].copy()

    if row.empty or not bool(row['origin_valid_full_features'].iloc[0]):
        return float('nan')

    row = row.dropna(subset=feature_cols)
    if row.empty:
        return float('nan')

    # Правило присвоения меток на определенный момент времени такое же, как и для XGBoost:
    # пример может перейти в обучающую выборку только после того, как фактически наступит
    # целевая неделя, на которой он был основан.
    train = supervised.loc[
        (supervised['target_week'] <= origin_week)
        & (~supervised['target_is_bad'])
        & (supervised['origin_valid_full_features'])
    ].dropna(subset=feature_cols + ['target'])

    if len(train) < 80:
        return float('nan')

    model = make_ridge_log(alpha)
    model.fit(
        train[feature_cols].to_numpy(dtype=float),
        np.log1p(train['target'].to_numpy(dtype=float)),
    )
    pred_log = model.predict(row[feature_cols].to_numpy(dtype=float))[0]
    pred = np.expm1(pred_log)
    return max(0.0, float(pred))
