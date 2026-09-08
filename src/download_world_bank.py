from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import requests

API = 'https://api.worldbank.org/v2'
INDICATORS = {
    'inflation_pct': 'FP.CPI.TOTL.ZG',
    'unemployment_pct': 'SL.UEM.TOTL.ZS',
    'gdp_growth_pct': 'NY.GDP.MKTP.KD.ZG',
}


def fetch(country: str, code: str, start_year: int, end_year: int):
    url = f'{API}/country/{country}/indicator/{code}'
    r = requests.get(
        url,
        params={'format': 'json', 'date': f'{start_year}:{end_year}', 'per_page': 1000},
        timeout=60,
    )
    r.raise_for_status()
    payload = r.json()
    if not isinstance(payload, list) or len(payload) < 2:
        raise RuntimeError(f'Unexpected World Bank response for {code}')
    rows = payload[1] or []
    return pd.DataFrame({'year': [int(x['date']) for x in rows], 'value': [x.get('value') for x in rows]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--country', default='ISR')
    parser.add_argument('--start-year', type=int, default=2018)
    parser.add_argument('--end-year', type=int, default=2025)
    parser.add_argument('--out', default='data/raw/world_bank_israel_annual.csv')
    args = parser.parse_args()

    result = None
    for name, code in INDICATORS.items():
        part = fetch(args.country, code, args.start_year, args.end_year).rename(columns={'value': name})
        result = part if result is None else result.merge(part, on='year', how='outer', validate='one_to_one')

    result = result.sort_values('year').reset_index(drop=True)

    # Консервативно учитываем доступность данных на момент прогноза.
    # WDI API возвращает текущие версии исторических значений, поэтому
    # не предполагаем, что показатель за год Y был известен уже в году Y.
    # Значение за год Y разрешаем использовать только с 1 января Y+2.
    result['available_from'] = pd.to_datetime((result['year'] + 2).astype(str) + '-01-01')

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    print(result.to_string(index=False))
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
