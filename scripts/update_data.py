#!/usr/bin/env python3
"""
Recession Monitor — Data Fetcher v4
Fuentes: yfinance (precios), Treasury.gov XML (curva oficial), FRED (macro y
crédito, si hay FRED_API_KEY), BLS (respaldo de empleo).

Salidas en docs/:
  dashboard.json  — estado actual: score compuesto, indicadores, series 1 año
  history.json    — una fila por día con el score y los valores clave

Reglas que corrigen la v3:
  * Un dato que falta es None, nunca NaN, y NO puntúa: el compuesto se
    renormaliza sobre los indicadores disponibles y publica su cobertura.
    (La v3 puntuaba NaN como peligro máximo y None como 50.)
  * Las series se limpian de filas vacías: después de las 00:00 UTC yfinance
    añade una fila del día nuevo sin cierre para los ETF.
  * El JSON se escribe con allow_nan=False: si algo se cuela, falla el
    script y el workflow no publica datos rotos.
"""

import os, json, math, datetime as dt, statistics, time, warnings
import xml.etree.ElementTree as ET
import urllib.request

warnings.filterwarnings('ignore')

FRED_API_KEY = os.environ.get('FRED_API_KEY', '')
NOW = dt.datetime.now(dt.timezone.utc)
TS = NOW.strftime('%Y-%m-%dT%H:%M:%SZ')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'docs')


# ══════════════════════════════════════════════════════════════════════════════
# UTILIDADES
# ══════════════════════════════════════════════════════════════════════════════

def num(x):
    """float finito o None."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def rnd(x, n=2):
    x = num(x)
    return None if x is None else round(x, n)


def clean(obj):
    """Sustituye NaN/inf por None en cualquier estructura."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def http_get(url, data=None, headers=None, timeout=20):
    h = {'User-Agent': 'Mozilla/5.0 recession-monitor'}
    h.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


# ══════════════════════════════════════════════════════════════════════════════
# FUENTES
# ══════════════════════════════════════════════════════════════════════════════

def yf_close(ticker, period='2y', retries=3):
    """Serie de cierres sin filas vacías, o None."""
    import yfinance as yf
    for i in range(retries):
        try:
            d = yf.download(ticker, period=period, progress=False,
                            auto_adjust=True, threads=False)
            if d is None or d.empty:
                raise ValueError('vacío')
            s = d['Close']
            if hasattr(s, 'columns'):          # MultiIndex de yfinance >= 0.2.48
                s = s.iloc[:, 0]
            s = s.astype(float).dropna()
            if len(s) < 5:
                raise ValueError(f'solo {len(s)} filas válidas')
            return s
        except Exception as e:
            if i == retries - 1:
                print(f'  ✗ yfinance {ticker}: {e}')
                return None
            time.sleep(2 * (i + 1))


def fred(sid, start=None):
    """Lista [(fecha_iso, valor)] de más antigua a más reciente, o []."""
    if not FRED_API_KEY:
        return []
    start = start or (NOW - dt.timedelta(days=800)).strftime('%Y-%m-%d')
    url = ('https://api.stlouisfed.org/fred/series/observations'
           f'?series_id={sid}&api_key={FRED_API_KEY}&file_type=json'
           f'&observation_start={start}')
    for i in range(3):
        try:
            d = json.loads(http_get(url))
            out = []
            for o in d.get('observations', []):
                v = num(o.get('value'))
                if v is not None:
                    out.append((o['date'], v))
            return out
        except Exception as e:
            if i == 2:
                print(f'  ✗ FRED {sid}: {e}')
                return []
            time.sleep(2)


def bls(sid):
    """Serie mensual BLS [(YYYY-MM-01, valor)] de más antigua a más reciente."""
    y = NOW.year
    payload = json.dumps({'seriesid': [sid], 'startyear': str(y - 3),
                          'endyear': str(y)}).encode()
    try:
        d = json.loads(http_get('https://api.bls.gov/publicAPI/v2/timeseries/data/',
                                data=payload,
                                headers={'Content-Type': 'application/json'}))
        rows = []
        for x in d['Results']['series'][0]['data']:
            if not x['period'].startswith('M') or x['period'] == 'M13':
                continue
            v = num(x['value'])
            if v is not None:
                rows.append((f"{x['year']}-{x['period'][1:]}-01", v))
        return sorted(rows)
    except Exception as e:
        print(f'  ✗ BLS {sid}: {e}')
        return []


def treasury_curve():
    """Última curva oficial de Treasury.gov. Prueba el mes actual y el anterior
    (los primeros días hábiles del mes el XML del mes nuevo viene vacío)."""
    ns = {'d': 'http://schemas.microsoft.com/ado/2007/08/dataservices',
          'm': 'http://schemas.microsoft.com/ado/2007/08/dataservices/metadata',
          'a': 'http://www.w3.org/2005/Atom'}
    tags = {'1M': 'BC_1MONTH', '3M': 'BC_3MONTH', '6M': 'BC_6MONTH',
            '1Y': 'BC_1YEAR', '2Y': 'BC_2YEAR', '3Y': 'BC_3YEAR',
            '5Y': 'BC_5YEAR', '7Y': 'BC_7YEAR', '10Y': 'BC_10YEAR',
            '20Y': 'BC_20YEAR', '30Y': 'BC_30YEAR'}
    first = NOW.date().replace(day=1)
    for month in (first, (first - dt.timedelta(days=1)).replace(day=1)):
        url = ('https://home.treasury.gov/resource-center/data-chart-center/'
               'interest-rates/pages/xml?data=daily_treasury_yield_curve'
               f'&field_tdr_date_value_month={month:%Y%m}')
        try:
            root = ET.fromstring(http_get(url).decode('utf-8'))
            entries = root.findall('.//a:entry', ns)
            if not entries:
                continue
            last = entries[-1]
            def gv(tag):
                el = last.find(f'.//d:{tag}', ns)
                return num(el.text) if el is not None and el.text else None
            date_el = last.find('.//d:NEW_DATE', ns)
            date = date_el.text[:10] if date_el is not None else None
            curve = {k: gv(t) for k, t in tags.items()}
            if curve.get('10Y') is not None:
                return {'date': date, 'curve': curve}
        except Exception as e:
            print(f'  ✗ Treasury {month:%Y%m}: {e}')
    return {'date': None, 'curve': {}}


# ══════════════════════════════════════════════════════════════════════════════
# CÁLCULOS SOBRE SERIES
# ══════════════════════════════════════════════════════════════════════════════

def last(s):
    return None if s is None or len(s) == 0 else num(s.iloc[-1])


def last_date(s):
    return None if s is None or len(s) == 0 else str(s.index[-1].date())


def pct_chg(s, n):
    if s is None or len(s) <= n:
        return None
    return num((s.iloc[-1] / s.iloc[-1 - n] - 1) * 100)


def series_points(s, every=1, n=260, digits=2):
    """Últimos n puntos de una serie pandas → [[fecha, valor], ...]."""
    if s is None:
        return []
    s = s.dropna().tail(n).iloc[::every]
    return [[str(i.date()), rnd(v, digits)] for i, v in s.items() if num(v) is not None]


def sahm_from_unrate(rows):
    """Regla de Sahm real: media móvil 3M del paro menos el mínimo de esa media
    en los 12 meses anteriores. rows de más antigua a más reciente."""
    vals = [v for _, v in rows]
    if len(vals) < 15:
        return None
    ma3 = [statistics.mean(vals[i - 2:i + 1]) for i in range(2, len(vals))]
    return round(ma3[-1] - min(ma3[-13:-1]), 2)


# ══════════════════════════════════════════════════════════════════════════════
# PUNTUACIÓN (0 = sin riesgo, 100 = máximo riesgo)
# ══════════════════════════════════════════════════════════════════════════════

def band(value, breakpoints, scores):
    """Tramos ascendentes. Devuelve None si no hay dato: un dato que falta
    no puede puntuar ni como calma ni como peligro."""
    value = num(value)
    if value is None:
        return None
    for bp, sc in zip(breakpoints, scores):
        if value <= bp:
            return sc
    return scores[-1]


SCORERS = {
    'vix':          lambda v: band(v, [15, 20, 25, 30, 40], [5, 15, 30, 50, 75, 95]),
    'sox_bubble':   lambda v: band(v, [10, 20, 30, 50, 70], [5, 15, 30, 60, 85, 97]),
    'hy_oas':       lambda v: band(v, [3.0, 3.5, 4.5, 6.0, 8.0], [5, 15, 35, 60, 85, 95]),
    'hy_proxy':     lambda v: band(v, [-3, -1, 0, 1, 3], [5, 10, 25, 50, 75, 90]),
    'spy_drawdown': lambda v: band(v, [-30, -20, -15, -10, -5], [95, 80, 60, 35, 15, 5]),
    'bond_30y':     lambda v: band(v, [3.5, 4.0, 4.5, 5.0, 5.5], [5, 10, 25, 60, 80, 90]),
    'dxy':          lambda v: band(v, [95, 100, 103, 106, 110], [5, 10, 25, 50, 70, 85]),
    'sahm':         lambda v: band(v, [0.0, 0.2, 0.35, 0.5, 0.75], [5, 15, 35, 70, 88, 97]),
    'curve_10y3m':  lambda v: band(v, [-1.5, -0.5, 0, 0.3, 1.0], [95, 80, 55, 25, 10, 5]),
    'curve_10y2y':  lambda v: band(v, [-1.5, -0.5, 0, 0.5, 1.5], [95, 75, 50, 25, 10, 5]),
    'payrolls_3m':  lambda v: band(v, [-100, 0, 50, 100, 200], [95, 75, 50, 25, 10, 5]),
    'oil_rise':     lambda v: band(v, [20, 40, 60, 90, 120], [5, 10, 20, 55, 80, 95]),
    'savings':      lambda v: band(v, [2, 3, 4, 6, 8], [95, 75, 50, 20, 8, 3]),
    'umich':        lambda v: band(v, [50, 58, 65, 75, 85], [95, 75, 55, 30, 12, 5]),
    'kre_3m':       lambda v: band(v, [-30, -20, -10, -5, 0], [95, 80, 60, 35, 15, 5]),
    'unrate':       lambda v: band(v, [3.5, 4.0, 4.5, 5.0, 6.0], [5, 10, 25, 50, 75, 90]),
    'claims_4w':    lambda v: band(v, [220e3, 250e3, 280e3, 320e3], [10, 25, 50, 75, 90]),
    'bkln_3m':      lambda v: band(v, [-10, -5, -2, 0, 2], [95, 80, 55, 30, 10, 5]),
    'nfci':         lambda v: band(v, [-0.5, -0.25, 0, 0.25, 0.5], [5, 15, 35, 60, 80, 95]),
    'quits_rate':   lambda v: band(v, [1.8, 2.0, 2.2, 2.5], [85, 60, 35, 15, 5]),
    'oil_price':    lambda v: band(v, [70, 85, 100, 120], [5, 15, 35, 65, 85]),
}

# Peso en el compuesto. Lo que no está aquí se muestra pero no puntúa.
WEIGHTS = {
    # Mercados
    'vix': 0.08, 'sox_bubble': 0.09, 'hy': 0.08, 'spy_drawdown': 0.05,
    'bond_30y': 0.08, 'dxy': 0.04,
    # Economía
    'sahm': 0.12, 'curve_10y3m': 0.08, 'payrolls_3m': 0.08, 'oil_rise': 0.05,
    # Consumidor y bancos
    'savings': 0.07, 'umich': 0.08, 'kre_3m': 0.03,
}
PILLARS = {
    'Mercados':   ['vix', 'sox_bubble', 'hy', 'spy_drawdown', 'bond_30y', 'dxy'],
    'Economía':   ['sahm', 'curve_10y3m', 'payrolls_3m', 'oil_rise'],
    'Consumidor': ['savings', 'umich', 'kre_3m'],
}


def status_of(sc):
    if sc is None:
        return 'SIN DATO'
    return 'OK' if sc < 25 else 'VIGILAR' if sc < 50 else 'ALERTA' if sc < 75 else 'PELIGRO'


def label_of(sc):
    if sc is None: return 'SIN DATOS'
    if sc < 20: return 'RIESGO BAJO'
    if sc < 40: return 'MODERADO'
    if sc < 55: return 'ELEVADO'
    if sc < 70: return 'ALTO RIESGO'
    if sc < 85: return 'CRISIS INMINENTE'
    return 'CRISIS ACTIVA'


def composite(scores):
    """Media ponderada sobre los indicadores con dato. Devuelve también la
    cobertura (peso con dato / peso total) para que se vea cuánto falta."""
    tot_w = sum(WEIGHTS.values())
    got_w = sum(w for k, w in WEIGHTS.items() if scores.get(k) is not None)
    if got_w == 0:
        return None, 0.0
    sc = sum(scores[k] * w for k, w in WEIGHTS.items() if scores.get(k) is not None) / got_w
    return round(sc, 1), round(got_w / tot_w, 3)


def pillar_scores(scores):
    out = {}
    for name, keys in PILLARS.items():
        ws = [(scores[k], WEIGHTS[k]) for k in keys if scores.get(k) is not None]
        out[name] = round(sum(s * w for s, w in ws) / sum(w for _, w in ws), 1) if ws else None
    return out


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print(f'[{TS}] Recession Monitor v4')

    # ── Precios ──────────────────────────────────────────────────────────────
    print('Precios (yfinance)…')
    tick = {'vix': '^VIX', 'spy': 'SPY', 'soxx': 'SOXX', 'hyg': 'HYG',
            'lqd': 'LQD', 'kre': 'KRE', 'xlf': 'XLF', 'bkln': 'BKLN',
            'oil': 'CL=F', 'dxy': 'DX-Y.NYB', 'tnx': '^TNX', 'irx': '^IRX',
            'tyx': '^TYX'}
    px = {k: yf_close(t) for k, t in tick.items()}
    sectors = {'Tecnología': 'XLK', 'Financieras': 'XLF', 'Energía': 'XLE',
               'Materiales': 'XLB', 'Salud': 'XLV', 'Utilities': 'XLU',
               'Industriales': 'XLI', 'Consumo disc.': 'XLY',
               'Consumo básico': 'XLP', 'Comunicación': 'XLC',
               'Inmobiliario': 'XLRE'}
    sector_px = {n: yf_close(t, '1y') for n, t in sectors.items()}

    # ── Curva oficial ────────────────────────────────────────────────────────
    print('Curva (Treasury.gov)…')
    tsy = treasury_curve()
    curve = tsy['curve']

    # ── Macro ────────────────────────────────────────────────────────────────
    print('Macro (FRED / BLS)…' + ('' if FRED_API_KEY else '  [sin FRED_API_KEY]'))
    f = {sid: fred(sid) for sid in
         ['SAHMREALTIME', 'UNRATE', 'PAYEMS', 'PSAVERT', 'UMCSENT', 'ICSA',
          'BAMLH0A0HYM2', 'NFCI', 'JTSQUR', 'INDPRO', 'TEMPHELPS',
          'T10Y3M', 'T10Y2Y', 'DGS30']}
    unrate = f['UNRATE'] or bls('LNS14000000')
    payems = f['PAYEMS'] or bls('CES0000000001')

    # ══════════════════════════════════════════════════════════════════════════
    # VALORES
    # ══════════════════════════════════════════════════════════════════════════
    v, asof, src = {}, {}, {}

    def put(key, value, date, source):
        v[key] = num(value)
        asof[key] = date if v[key] is not None else None
        src[key] = source

    put('vix', last(px['vix']), last_date(px['vix']), 'CBOE ^VIX')

    s = px['soxx']
    if s is not None and len(s) >= 200:
        ma200 = s.rolling(200).mean()
        put('sox_bubble', (s.iloc[-1] / ma200.iloc[-1] - 1) * 100, last_date(s), 'SOXX vs MM200')
        sox_series = ((s / ma200 - 1) * 100).dropna()
    else:
        put('sox_bubble', None, None, 'SOXX vs MM200'); sox_series = None

    s = px['spy']
    if s is not None:
        dd = (s / s.rolling(252, min_periods=20).max() - 1) * 100
        put('spy_drawdown', dd.iloc[-1], last_date(s), 'SPY vs máx. 52 sem.')
        put('spy', s.iloc[-1], last_date(s), 'SPY')
        spy_dd_series = dd.dropna()
    else:
        put('spy_drawdown', None, None, 'SPY'); put('spy', None, None, 'SPY')
        spy_dd_series = None

    # Crédito HY: el diferencial real (ICE BofA, FRED) manda; el proxy LQD-HYG
    # solo entra si no hay FRED.
    if f['BAMLH0A0HYM2']:
        d, val = f['BAMLH0A0HYM2'][-1]
        put('hy', val, d, 'ICE BofA HY OAS (FRED)')
        hy_kind = 'oas'
    else:
        h3, l3 = pct_chg(px['hyg'], 63), pct_chg(px['lqd'], 63)
        put('hy', (l3 - h3) if h3 is not None and l3 is not None else None,
            last_date(px['hyg']), 'Proxy LQD−HYG 3M')
        hy_kind = 'proxy'

    y30 = curve.get('30Y')
    if y30 is None and f['DGS30']:
        put('bond_30y', f['DGS30'][-1][1], f['DGS30'][-1][0], 'FRED DGS30')
    else:
        put('bond_30y', y30, tsy['date'], 'Treasury.gov')

    put('dxy', last(px['dxy']), last_date(px['dxy']), 'ICE DXY')

    c10, c3m, c2 = curve.get('10Y'), curve.get('3M'), curve.get('2Y')
    if c10 is not None and c3m is not None:
        put('curve_10y3m', c10 - c3m, tsy['date'], 'Treasury.gov')
    elif f['T10Y3M']:
        put('curve_10y3m', f['T10Y3M'][-1][1], f['T10Y3M'][-1][0], 'FRED T10Y3M')
    else:
        put('curve_10y3m', None, None, 'Treasury.gov')
    if c10 is not None and c2 is not None:
        put('curve_10y2y', c10 - c2, tsy['date'], 'Treasury.gov')
    elif f['T10Y2Y']:
        put('curve_10y2y', f['T10Y2Y'][-1][1], f['T10Y2Y'][-1][0], 'FRED T10Y2Y')
    else:
        put('curve_10y2y', None, None, 'Treasury.gov')

    if f['SAHMREALTIME']:
        put('sahm', f['SAHMREALTIME'][-1][1], f['SAHMREALTIME'][-1][0], 'FRED SAHMREALTIME')
    else:
        put('sahm', sahm_from_unrate(unrate), unrate[-1][0] if unrate else None,
            'Calculada con paro BLS')

    put('unrate', unrate[-1][1] if unrate else None, unrate[-1][0] if unrate else None,
        'BLS / FRED UNRATE')

    # Nóminas: media de 3 meses (un mes suelto es ruido y se revisa mucho)
    if len(payems) >= 4:
        chg = [payems[i][1] - payems[i - 1][1] for i in range(len(payems) - 3, len(payems))]
        put('payrolls_3m', statistics.mean(chg), payems[-1][0], 'BLS / FRED PAYEMS')
        put('payrolls_1m', chg[-1], payems[-1][0], 'BLS / FRED PAYEMS')
    else:
        put('payrolls_3m', None, None, 'PAYEMS'); put('payrolls_1m', None, None, 'PAYEMS')

    s = px['oil']
    if s is not None:
        low = s.tail(252).min()
        put('oil_price', s.iloc[-1], last_date(s), 'WTI CL=F')
        put('oil_rise', (s.iloc[-1] / low - 1) * 100, last_date(s), 'WTI vs mín. 52 sem.')
        v['oil_low52'] = rnd(low, 1)
    else:
        put('oil_price', None, None, 'WTI'); put('oil_rise', None, None, 'WTI')

    for key, sid, name in [('savings', 'PSAVERT', 'BEA vía FRED'),
                           ('umich', 'UMCSENT', 'U. Michigan vía FRED'),
                           ('nfci', 'NFCI', 'Chicago Fed NFCI'),
                           ('quits_rate', 'JTSQUR', 'JOLTS vía FRED')]:
        obs = f[sid]
        put(key, obs[-1][1] if obs else None, obs[-1][0] if obs else None, name)

    if len(f['ICSA']) >= 4:
        put('claims_4w', statistics.mean(x for _, x in f['ICSA'][-4:]), f['ICSA'][-1][0], 'FRED ICSA (media 4 sem.)')
    else:
        put('claims_4w', None, None, 'FRED ICSA')

    put('kre_3m', pct_chg(px['kre'], 63), last_date(px['kre']), 'KRE 3 meses')
    put('xlf_3m', pct_chg(px['xlf'], 63), last_date(px['xlf']), 'XLF 3 meses')
    put('bkln_3m', pct_chg(px['bkln'], 63), last_date(px['bkln']), 'BKLN 3 meses')

    # ══════════════════════════════════════════════════════════════════════════
    # PUNTUACIONES
    # ══════════════════════════════════════════════════════════════════════════
    sc = {}
    for key in v:
        if key == 'hy':
            sc[key] = SCORERS['hy_oas' if hy_kind == 'oas' else 'hy_proxy'](v[key])
        elif key in SCORERS:
            sc[key] = SCORERS[key](v[key])
    comp, coverage = composite(sc)
    pillars = pillar_scores(sc)
    missing = [k for k in WEIGHTS if sc.get(k) is None]
    print(f'  Compuesto {comp} ({label_of(comp)}), cobertura {coverage:.0%}'
          + (f', faltan: {", ".join(missing)}' if missing else ''))

    # ══════════════════════════════════════════════════════════════════════════
    # UMBRALES DE DOLOR (Hartnett / Carpatos) — capitulación, no recesión
    # ══════════════════════════════════════════════════════════════════════════
    def pain(key, label, value, threshold, op, unit=''):
        value = num(value)
        active = None if value is None else (value > threshold if op == '>' else
                                             value >= threshold if op == '>=' else
                                             value < threshold)
        return {'key': key, 'label': label, 'value': rnd(value, 2),
                'threshold': threshold, 'op': op, 'unit': unit, 'active': active}

    pains = [
        pain('vix', 'VIX en pánico', v['vix'], 35, '>'),
        pain('spy_drawdown', 'S&P cae más de un 15%', v['spy_drawdown'], -15, '<', '%'),
        pain('curve_10y3m', 'Curva 10A−3M invertida', v['curve_10y3m'], 0, '<', ' pp'),
        pain('bond_30y', 'Bono 30A por encima del 5%', v['bond_30y'], 5.0, '>=', '%'),
        pain('sahm', 'Regla de Sahm activada', v['sahm'], 0.5, '>='),
        pain('hy', 'Crédito HY en estrés',
             v['hy'], 5.0 if hy_kind == 'oas' else 1.5, '>', '%' if hy_kind == 'oas' else ' pp'),
        pain('oil_price', 'Petróleo por encima de 100 $', v['oil_price'], 100, '>=', ' $'),
        pain('oil_rise', 'Petróleo +90% desde mínimos', v['oil_rise'], 90, '>=', '%'),
    ]
    n_active = sum(1 for p in pains if p['active'])
    n_known = sum(1 for p in pains if p['active'] is not None)
    action = ('DESPLEGAR capital — varias señales de suelo' if n_active >= 5 else
              'ACUMULAR por tramos — las condiciones mejoran' if n_active >= 3 else
              'MANTENER liquidez — sin capitulación')

    # ══════════════════════════════════════════════════════════════════════════
    # INDICADORES PARA LA TABLA
    # ══════════════════════════════════════════════════════════════════════════
    META = [
        # key, etiqueta, grupo, unidad, decimales, adelanto, umbral/nota, cadencia (días)
        ('vix', 'VIX', 'Mercados', '', 1, '0–3M', '>35 = pánico', 5),
        ('sox_bubble', 'Semis (SOXX) sobre su MM200', 'Mercados', '%', 1, '3–12M', '>30% = burbuja (Hartnett)', 5),
        ('spy_drawdown', 'S&P 500: caída desde máximo 52 sem.', 'Mercados', '%', 1, '0–6M', '<−15% = corrección seria', 5),
        ('bond_30y', 'Bono del Tesoro a 30 años', 'Mercados', '%', 2, '3–12M', '≥5% = «bubble killer»', 5),
        ('dxy', 'Dólar (DXY)', 'Mercados', '', 1, '3–9M', '>105 = estrés en emergentes', 5),
        ('hy', 'Diferencial high yield (OAS)' if hy_kind == 'oas' else 'Estrés HY (LQD−HYG, 3M)', 'Crédito',
         '%' if hy_kind == 'oas' else ' pp', 2, '0–6M', '>5% = estrés; >8% = crisis' if hy_kind == 'oas' else '>1,5 pp = estrés', 5),
        ('nfci', 'Condiciones financieras (NFCI)', 'Crédito', '', 2, '0–6M', '>0 = más tensas que la media', 10),
        ('kre_3m', 'Bancos regionales (KRE), 3 meses', 'Crédito', '%', 1, '0–6M', '<−10% = alerta', 5),
        ('bkln_3m', 'Préstamos apalancados (BKLN), 3 meses', 'Crédito', '%', 1, '0–6M', '<−2% = debilidad', 5),
        ('curve_10y3m', 'Curva 10A − 3M', 'Economía', ' pp', 2, '6–18M', '<0 = invertida', 5),
        ('curve_10y2y', 'Curva 10A − 2A', 'Economía', ' pp', 2, '6–24M', '<0 = invertida', 5),
        ('sahm', 'Regla de Sahm', 'Empleo', '', 2, '0–3M', '≥0,50 = recesión en curso', 75),
        ('unrate', 'Tasa de paro', 'Empleo', '%', 1, '0–6M', '', 75),
        ('payrolls_3m', 'Nóminas no agrícolas (media 3M)', 'Empleo', 'K', 0, '0–6M', '<0 = destrucción de empleo', 75),
        ('claims_4w', 'Peticiones de paro (media 4 sem.)', 'Empleo', '', 0, '3–6M', '>300K = deterioro', 14),
        ('quits_rate', 'Tasa de abandonos (JOLTS)', 'Empleo', '%', 1, '3–12M', '<2% = trabajadores sin confianza', 100),
        ('savings', 'Tasa de ahorro personal', 'Consumidor', '%', 1, '3–9M', '<3% = colchón agotado', 100),
        ('umich', 'Confianza del consumidor (UMich)', 'Consumidor', '', 1, '3–9M', '<58 = zona de recesión', 75),
        ('oil_price', 'Petróleo WTI', 'Consumidor', ' $', 1, '3–9M', '>100 $ = freno al consumo', 5),
        ('oil_rise', 'Petróleo vs mínimo 52 sem.', 'Consumidor', '%', 0, '6–12M', '>90% = antesala de recesión', 5),
    ]
    today = NOW.date()
    indicators = []
    for key, label, group, unit, dec, lead, note, cadence in META:
        a = asof.get(key)
        age = (today - dt.date.fromisoformat(a)).days if a else None
        indicators.append({
            'key': key, 'label': label, 'group': group, 'unit': unit,
            'decimals': dec, 'value': rnd(v.get(key), 4), 'score': sc.get(key),
            'status': status_of(sc.get(key)), 'weight': WEIGHTS.get(key, 0),
            'lead': lead, 'note': note, 'source': src.get(key),
            'as_of': a, 'stale': age is not None and age > cadence,
        })

    # ══════════════════════════════════════════════════════════════════════════
    # SERIES PARA GRÁFICOS (1 año, diarias o semanales)
    # ══════════════════════════════════════════════════════════════════════════
    def fred_pts(sid, n=260):
        return [[d, rnd(x, 3)] for d, x in f[sid][-n:]]

    # La serie oficial (FRED T10Y3M) manda; ^TNX−^IRX es solo respaldo y se
    # desvía unas décimas porque ^IRX es tipo de descuento, no rentabilidad.
    curve_series = fred_pts('T10Y3M')
    if not curve_series and px['tnx'] is not None and px['irx'] is not None:
        curve_series = series_points((px['tnx'] - px['irx']).dropna())

    if f['SAHMREALTIME']:
        sahm_series = [[d, rnd(x, 2)] for d, x in f['SAHMREALTIME'][-36:]]
    else:
        sahm_series = [[unrate[i][0], sahm_from_unrate(unrate[:i + 1])]
                       for i in range(max(15, len(unrate) - 24), len(unrate))]

    charts = {
        'spy_drawdown': series_points(spy_dd_series, digits=1),
        'sox_bubble':   series_points(sox_series, digits=1),
        'vix':          series_points(px['vix'], digits=1),
        'bond_30y':     series_points(px['tyx'], digits=2) or fred_pts('DGS30'),
        'curve_10y3m':  curve_series,
        'hy':           fred_pts('BAMLH0A0HYM2') if hy_kind == 'oas' else [],
        'oil_price':    series_points(px['oil'], digits=1),
        'unrate':       [[d, x] for d, x in unrate[-36:]],
        'sahm':         sahm_series,
        'claims_4w':    fred_pts('ICSA', 104),
    }

    sector_rows = []
    for n, s in sector_px.items():
        sector_rows.append({'name': n, 'ticker': sectors[n],
                            '1M': rnd(pct_chg(s, 21), 1), '3M': rnd(pct_chg(s, 63), 1),
                            '12M': rnd(pct_chg(s, len(s) - 1) if s is not None else None, 1)})

    dashboard = {
        'timestamp': TS,
        'version': 4,
        'fred_enabled': bool(FRED_API_KEY),
        'composite': {'score': comp, 'label': label_of(comp), 'coverage': coverage,
                      'missing': missing, 'pillars': pillars,
                      'weights': WEIGHTS, 'pillar_keys': PILLARS},
        'indicators': indicators,
        'pain': {'levels': pains, 'active': n_active, 'known': n_known, 'action': action},
        'curve': {'date': tsy['date'], 'points': curve},
        'sectors': sector_rows,
        'charts': charts,
        'extra': {'spy': rnd(v.get('spy'), 2), 'oil_low52': v.get('oil_low52'),
                  'payrolls_1m': rnd(v.get('payrolls_1m'), 0), 'hy_kind': hy_kind,
                  'xlf_3m': rnd(v.get('xlf_3m'), 1)},
    }

    # ── Histórico: una fila por día (la última ejecución del día manda) ─────
    hist_path = os.path.join(OUT, 'history.json')
    try:
        with open(hist_path, encoding='utf-8') as fh:
            history = json.load(fh)
    except Exception:
        history = []
    # Fecha del último cierre de mercado, no la del reloj: la ejecución de las
    # 00:xx UTC o la de un sábado describen el último día hábil.
    mkt_date = asof.get('spy') or asof.get('vix') or str(today)
    row = {'date': mkt_date, 'score': comp, 'coverage': coverage,
           **{f'p_{k}': x for k, x in pillars.items()},
           **{k: rnd(v.get(k), 3) for k in ['vix', 'sox_bubble', 'spy_drawdown', 'hy',
                                             'bond_30y', 'curve_10y3m', 'oil_price',
                                             'dxy', 'sahm', 'unrate', 'umich']},
           'hy_kind': hy_kind}
    # Una corrida sin datos de mercado no debe pisar un día bueno
    if comp is not None and coverage >= 0.6:
        history = [h for h in history if h.get('date') != row['date']] + [row]
        history.sort(key=lambda h: h['date'])

    os.makedirs(OUT, exist_ok=True)
    for name, payload in [('dashboard.json', dashboard), ('history.json', history)]:
        path = os.path.join(OUT, name)
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(clean(payload), fh, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':') if name == 'history.json' else None,
                      indent=None if name == 'history.json' else 1)
        print(f'  ✓ {os.path.normpath(path)}')

    if coverage < 0.6:
        raise SystemExit(f'Cobertura {coverage:.0%} < 60%: no se publica.')


if __name__ == '__main__':
    main()
