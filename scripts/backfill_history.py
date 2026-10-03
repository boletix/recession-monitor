#!/usr/bin/env python3
"""
Reconstruye docs/history.json a partir de los macro_indicators.json que la v3
dejó en el historial de git (21-may-2026 en adelante).

Se ejecuta una vez, a mano:  python scripts/backfill_history.py
Recalcula cada día con la puntuación de la v4: los NaN de la v3 quedan como
dato que falta (la v3 los puntuaba como peligro máximo). Las nóminas son el
dato mensual suelto de la v3, no la media de 3 meses, y el crédito HY es el
proxy LQD−HYG: por eso las filas llevan bf=1.
"""

import json, os, re, subprocess, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from update_data import SCORERS, composite, pillar_scores, rnd  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True,
                          text=True, encoding='utf-8', check=True).stdout


def val(d, *path):
    for p in path:
        if not isinstance(d, dict):
            return None
        d = d.get(p)
    return d


def main():
    commits = git('log', '--format=%h', '--', 'docs/macro_indicators.json').split()
    rows = {}
    for c in reversed(commits):
        try:
            raw = git('show', f'{c}:docs/macro_indicators.json')
        except subprocess.CalledProcessError:
            continue
        m = json.loads(re.sub(r'\bNaN\b', 'null', raw))
        ms, lab, con, yc = (m.get('market_signals', {}), m.get('labor_market', {}),
                            m.get('consumer_signals', {}), m.get('yield_curve', {}))
        v = {
            'vix': val(ms, 'vix', 'value'),
            'sox_bubble': val(ms, 'sox_bubble', 'value'),
            'spy_drawdown': val(ms, 'spy_drawdown', 'value'),
            'hy': val(ms, 'hy_stress', 'value'),
            'bond_30y': val(ms, 'bond_30y', 'value'),
            'dxy': val(ms, 'dxy', 'value'),
            'kre_3m': val(ms, 'kre_3m', 'value'),
            'sahm': val(lab, 'sahm_rule', 'value'),
            'unrate': val(lab, 'unemployment', 'value'),
            'payrolls_3m': val(lab, 'payrolls_mom', 'value'),
            'curve_10y3m': yc.get('spread_10y3m'),
            'oil_rise': val(con, 'oil_rise', 'value'),
            'oil_price': val(con, 'oil_price', 'value'),
            'savings': val(con, 'savings_rate', 'value'),
            'umich': val(con, 'umich_sentiment', 'value'),
        }
        v = {k: rnd(x, 3) for k, x in v.items()}
        sc = {k: SCORERS['hy_proxy' if k == 'hy' else k](x) for k, x in v.items()
              if k in SCORERS or k == 'hy'}
        comp, cov = composite(sc)
        # Los días en que la v3 perdió los ETF (NaN) no son comparables: fuera
        if comp is None or cov < 0.85:
            continue
        # La ejecución de las 00:xx UTC trae el cierre del día hábil anterior
        ts = m['timestamp'][:19]
        date = ts[:10]
        if ts[11:13] < '06':
            from datetime import date as D, timedelta
            date = str(D.fromisoformat(date) - timedelta(days=1))
        rows[date] = {'date': date, 'score': comp, 'coverage': cov,
                      **{f'p_{k}': x for k, x in pillar_scores(sc).items()},
                      **{k: v.get(k) for k in ['vix', 'sox_bubble', 'spy_drawdown', 'hy',
                                               'bond_30y', 'curve_10y3m', 'oil_price',
                                               'dxy', 'sahm', 'unrate', 'umich']},
                      'hy_kind': 'proxy', 'bf': 1}

    path = os.path.join(ROOT, 'docs', 'history.json')
    try:
        current = {h['date']: h for h in json.load(open(path, encoding='utf-8'))}
    except Exception:
        current = {}
    rows.update(current)                       # lo ya calculado por la v4 manda
    out = sorted(rows.values(), key=lambda h: h['date'])
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    print(f'{len(out)} días en history.json ({out[0]["date"]} → {out[-1]["date"]})')


if __name__ == '__main__':
    main()
