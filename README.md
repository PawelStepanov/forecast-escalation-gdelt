# Forecast DS — прогноз конфликтной активности по GDELT

Тестовое задание. Цель — спрогнозировать недельную конфликтную активность в Израиле на горизонтах **1–4 недели** по историческим данным GDELT и дополнительным открытым источникам.

## Структура репозитория

```text
forecast-escalation/
├── README.md
├── config.yaml
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── run.ps1
├── sql/
│   └── gdelt_weekly.sql
├── src/
│   ├── config.py
│   ├── data.py
│   ├── download_world_bank.py
│   ├── features.py
│   ├── metrics.py
│   ├── models.py
│   ├── evaluate.py
│   └── train.py
├── tests/
├── reports/
├── data/
│   ├── raw/
│   └── processed/
└── outputs/
```

---

## 1. Страна и target

Выбрана **Israel** (`ActionGeo_CountryCode='IS'` в GDELT). Причина выбора: по стране есть длинный непустой недельный ряд, выраженные периоды роста и снижения активности и структурные сдвиги, поэтому задача не сводится к прогнозу почти постоянного ряда.

Используется вариант **A — регрессия**. Для forecast origin недели `t`, когда неделя `t` уже завершена:

```text
y(t+h) = conflict_count на неделе t+h,  h ∈ {1,2,3,4}
```

где

```text
conflict_count = COUNT(
    GDELT events
    WHERE ActionGeo_CountryCode = 'IS'
      AND EventRootCode IN ('18','19')
)
```

`18` и `19` — CAMEO root codes для Assault / Fight. Target — число **GDELT event records**, а не уникальное физическое число боевых инцидентов; поэтому результат зависит и от медиа-покрытия.

Используется **direct forecasting**: для каждого горизонта `h=1..4` формируется отдельная supervised-задача.

---

## 2. Источники данных

| Источник | Что используется | Период | Дата snapshot |
|---|---|---|---|
| GDELT Events | conflict count, Goldstein, tone, event/news volume, mentions | 2022-01-03 — 2026-08-24 | 2026-09-03 |
| World Bank WDI | inflation, unemployment, GDP growth | 2018–2025 | 2026-09-06 |

**GDELT**
- Source: https://www.gdeltproject.org/data.html
- BigQuery dataset: `gdelt-bq.gdeltv2.events_partitioned`
- SQL: [`sql/gdelt_weekly.sql`](sql/gdelt_weekly.sql)
- Submitted snapshot: `data/raw/gdelt_israel_weekly.csv`

**World Bank**
- API: https://api.worldbank.org/v2
- Indicators: `FP.CPI.TOTL.ZG`, `SL.UEM.TOTL.ZS`, `NY.GDP.MKTP.KD.ZG`
- Snapshot: `data/raw/world_bank_israel_annual.csv`

World Bank имеет годовую частоту, а GDELT — недельную. Для каждой недели используется последнее macro observation, которое к этой неделе считается доступным. Чтобы не использовать итоговый годовой показатель слишком рано, применяется консервативное правило: observation за год `Y` становится доступным с `1 January Y+2`, после чего выполняется backward `merge_asof`. Это защитная эвристика против leakage.

---

## 3. Data quality

При EDA обнаружен резкий провал общего GDELT coverage примерно **2025-06-15 — 2025-07-01**. Проблема затронула недели:

```text
2025-06-09
2025-06-16
2025-06-23
2025-06-30
```

Эти значения не интерполируются, чтобы не создавать искусственный ground truth. Повреждённые недели исключаются как target labels. Дополнительно исключаются forecast origins, если их 12-недельное окно признаков пересекает этот период. Календарная сетка при этом не сжимается: lag `1` всегда означает ровно одну календарную неделю назад.

---

## 4. Feature engineering

В задании был задан минимальный набор направлений для feature engineering: лаги target, rolling statistics, tone/Goldstein, объём новостей, календарь, внешний источник и trend-признаки. В решении этот минимум расширен собственными признаками. Ниже явно указано, что было прямо предложено в задании, а что добавлено самостоятельно.

| Feature | Что измеряет | Происхождение идеи |
|---|---|---|
| `conflict_count_t` | Текущий известный уровень conflict activity на завершённой неделе `t` | **Добавлен в решении** как сильный persistence-сигнал |
| `conflict_count_lag1` | Значение target 1 неделю назад | Прямо предложен в задании |
| `conflict_count_lag2` | Значение target 2 недели назад | Прямо предложен в задании |
| `conflict_count_lag4` | Значение target 4 недели назад | Прямо предложен в задании |
| `conflict_count_lag12` | Значение target 12 недель назад | Прямо предложен в задании |
| `rolling_mean_4w` | Локальный средний уровень за последние 4 недели | Прямо предложен в задании |
| `rolling_std_4w` | Краткосрочная нестабильность за последние 4 недели | **Добавлен в решении** |
| `rolling_mean_8w` | Средний уровень на промежуточном 8-недельном окне | **Добавлен в решении** |
| `rolling_std_12w` | Более длинная 12-недельная волатильность target | Прямо предложен в задании |
| `rolling_max_8w` | Максимальный недавний spike за 8 недель | Прямо предложен в задании |
| `cumulative_conflict_12w` | Суммарная conflict activity примерно за квартал | Прямо предложен в задании |
| `avg_goldstein` | Средняя Goldstein severity/cooperation score конфликтных событий недели | Прямо предложен в задании |
| `goldstein_std` | Разброс Goldstein scores внутри текущей недели | **Добавлен в решении** |
| `avg_tone` | Средняя новостная тональность конфликтных событий недели | Прямо предложен в задании |
| `total_events_country` | Общий GDELT event volume по стране, не только conflict events | Прямо предложен в задании |
| `news_volume_change_wow` | Изменение общего event volume относительно предыдущей недели | Прямо предложен в задании |
| `total_mentions_country` | Общая интенсивность упоминаний событий страны в новостях | **Добавлен в решении** |
| `conflict_share` | Доля conflict events среди всех GDELT events страны | **Добавлен в решении** |
| `mentions_per_conflict` | Среднее число mentions на один conflict event | **Добавлен в решении** после EDA вместо почти дублирующего target `conflict_mentions` |
| `goldstein_volatility_4w` | Насколько менялся средний Goldstein между последними 4 неделями | Идея volatility дана в задании, **4-недельное окно выбрано в решении** |
| `target_week_sin`, `target_week_cos` | Циклическое представление недели года, где week 52 и week 1 остаются близкими | **Добавлено в решении** вместо подачи номера недели как обычного числа |
| `target_month` | Месяц прогнозируемой target week | Calendar feature предложен в задании |
| `inflation_pct` | Последний доступный годовой показатель инфляции World Bank | Macro indicator предложен в задании; **конкретный показатель выбран в решении** |
| `unemployment_pct` | Последний доступный годовой показатель безработицы | Macro indicator предложен в задании; **конкретный показатель выбран в решении** |
| `gdp_growth_pct` | Последний доступный годовой GDP growth | Macro indicator предложен в задании; **конкретный показатель выбран в решении** |
| `wb_feature_age_years` | Насколько старое World Bank observation сейчас используется | **Добавлен в решении** для явного учёта stale macro data |

`conflict_mentions` напрямую не включён в основной XGB feature set: EDA показал, что он почти дублирует `conflict_count`. Вместо него используется `mentions_per_conflict`, который описывает не абсолютный объём, а интенсивность медийного освещения на один зарегистрированный conflict event.

Все lag/rolling features используют только данные, доступные к forecast origin. Текущая неделя `t` допустима, потому что прогноз строится после её завершения.

---

## 5. Pipeline и validation

```text
GDELT + World Bank
        ↓
weekly aggregation + point-in-time join
        ↓
data-quality filtering
        ↓
feature engineering
        ↓
direct models h=1..4
        ↓
expanding-window walk-forward CV
        ↓
model selection by MASE
        ↓
12-week final holdout
        ↓
forecast.csv
```

Random split не используется. Последние **12 недель** оставлены как final holdout и не участвуют в выборе модели.

В walk-forward train постепенно расширяется. Для supervised-моделей новая training row становится доступной только когда её target уже известен:

```text
train.target_week <= current_forecast_origin
```

Это особенно важно для `h=2..4`: признаки недавней недели уже известны, но её target через несколько недель ещё может находиться в будущем.

Primary metric для model selection — **MASE**; дополнительно считаются **sMAPE** и **RMSE**.

---

## 6. Модели и результаты CV

Сравниваются:

- Last value;
- Rolling mean 4w;
- Seasonal naive 52w;
- ETS(A,N,N);
- Ridge Regression на `log1p(target)`;
- XGBoost на target-history features;
- XGBoost + GDELT context;
- XGBoost + GDELT + World Bank.

### Walk-forward CV — MASE

| Model | h=1 | h=2 | h=3 | h=4 |
|---|---:|---:|---:|---:|
| Last value | **0.639** | 0.910 | 1.151 | 1.449 |
| Rolling mean 4w | 0.880 | 1.114 | 1.389 | 1.620 |
| Seasonal naive 52w | 2.367 | 2.107 | 1.980 | 1.896 |
| ETS(A,N,N) | 0.855 | 1.096 | 1.307 | 1.522 |
| Ridge log history | 0.772 | **0.853** | **0.933** | **1.090** |
| XGB: target history | 1.013 | 1.519 | 2.058 | 2.330 |
| XGB: + GDELT | 0.935 | 1.505 | 2.225 | 2.248 |
| XGB: + GDELT + World Bank | 0.955 | 1.540 | 2.207 | 2.341 |

### Walk-forward CV — RMSE

| Model | h=1 | h=2 | h=3 | h=4 |
|---|---:|---:|---:|---:|
| Last value | **2347** | **3321** | 3979 | 4662 |
| Rolling mean 4w | 3087 | 3764 | 4361 | 4842 |
| Seasonal naive 52w | 6951 | 5643 | 5201 | 4959 |
| ETS(A,N,N) | 2949 | 3602 | 4115 | 4552 |
| Ridge log history | 2887 | 3353 | **3647** | **3980** |
| XGB: target history | 3314 | 4421 | 5550 | 5802 |
| XGB: + GDELT | 3088 | 4214 | 5870 | 5645 |
| XGB: + GDELT + World Bank | 3128 | 4516 | 5757 | 5811 |

По заранее выбранной primary metric MASE финальная policy:

```text
h=1 → last_value
h=2 → ridge_log_history
h=3 → ridge_log_history
h=4 → ridge_log_history
```

World Bank не дал устойчивого улучшения XGBoost на горизонте 1–4 недели, что ожидаемо для медленно меняющихся годовых показателей.

---

## 7. Final holdout

После выбора policy по CV она проверена на последних 12 неделях:

| h | Model | MASE | sMAPE | RMSE |
|---:|---|---:|---:|---:|
| 1 | Last value | 0.487 | 15.83% | 1269 |
| 2 | Ridge log history | 0.578 | 18.78% | 1408 |
| 3 | Ridge log history | 0.631 | 20.08% | 1665 |
| 4 | Ridge log history | 0.689 | 21.63% | 1757 |

`outputs/holdout_metrics.csv` содержит диагностические результаты всех моделей на holdout, но не используется для переизбрания модели.

---

## 8. Финальный forecast

`outputs/forecast.csv`:

| Week | h | Prediction | 90% empirical interval | Model |
|---|---:|---:|---:|---|
| 2026-W36 | 1 | 5641 | [2275; 8512] | Last value |
| 2026-W37 | 2 | 5302 | [4403; 11543] | Ridge log history |
| 2026-W38 | 3 | 5587 | [4289; 13937] | Ridge log history |
| 2026-W39 | 4 | 5838 | [3713; 15011] | Ridge log history |

Интервалы построены по 5%/95% квантилям historical CV residuals соответствующей модели и горизонта.
---

## 9. Три главных вывода

1. **На h=1 очень силён persistence baseline.** Краткосрочная динамика ряда инерционна, поэтому Last Value оказался лучшим по CV MASE.
2. **На h=2–h=4 Ridge устойчивее сложного boosting по средней ошибке**, но визуально сглаживает резкие изменения и недооценивает амплитуду некоторых spikes.
3. **Дополнительные GDELT-признаки иногда улучшают сам XGBoost, а World Bank не даёт стабильного прироста** на таком коротком горизонте.

---

## 10. Что улучшить при большем времени / prod-данных

1. **Разделить count forecast и escalation alert.** Сохранить прогноз ожидаемого уровня и отдельно обучить classifier/alert model для резкого роста, чтобы не оптимизировать раннее предупреждение только через среднюю regression error.
2. **Использовать тексты новостей.** Добавить embeddings/topic/novelty features и семантические сигналы, которые могут появляться раньше роста GDELT count.
3. **Использовать LLM как feature extractor, а не как прямой forecast model.** Для новостей, опубликованных до forecast origin, извлекать фиксированные признаки вроде `military_mobilization`, `explicit_threat`, `ceasefire_breakdown`, `new_actor`, `new_region`, `weapon_escalation`, `forward_looking_risk`; затем агрегировать их по неделям и проверять прирост тем же walk-forward backtest.
4. **Добавить независимый источник конфликтов**, например ACLED, чтобы отделять реальное изменение активности от изменения медиа-покрытия.
5. **Лучше учитывать смену режима:** сравнить expanding и rolling training windows, добавить несколько стран.

---

## 12. Запуск

### Docker

```bash
docker compose up --build
```

Контейнер устанавливает pinned dependencies, запускает тесты и затем полный training/backtest pipeline. Результаты сохраняются в `outputs/`.

### Локально

Python 3.11:

```bash
python -m venv .venv
pip install -r requirements.txt
pytest -q
python -m src.train --config config.yaml
```

Windows PowerShell также может использовать `./run.ps1`.

Основные outputs:

```text
outputs/forecast.csv                  # final forecast
outputs/cv_metrics.csv                # CV metrics всех моделей
outputs/selected_models_from_cv.csv   # выбранная модель для h=1..4
outputs/selected_policy_holdout.csv   # final holdout выбранной policy
outputs/figures/                      # основные графики
```

Полные historical predictions и дополнительные diagnostics также сохранены в `outputs/`.

---

## 13. Использование AI

AI использовался как вспомогательный инструмент для review постановки задачи и leakage risks, code review, проверки структуры pipeline/README и анализа возможных failure modes. Фактические данные, SQL-выгрузки, метрики и model-selection результаты получены из приложенных snapshots и воспроизводятся кодом проекта.
