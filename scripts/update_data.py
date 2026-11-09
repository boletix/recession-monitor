#!/usr/bin/env python3
"""
Recession Monitor — Data Fetcher v5 (indicadores adelantados)

Qué cambia respecto a la v4
  * El score ya no mide el humor del mercado: mide la ECONOMÍA. Los indicadores
    se ordenan por lo que la literatura dice que avisa antes de una recesión
    (curva, prima de riesgo de los bonos, crédito bancario, vivienda, pedidos,
    paro) y lo que la confirma cuando ya ha empezado (Sahm, Michaillat-Saez,
    paro asegurado). El mercado entra con poco peso: anticipa, pero se equivoca
    a menudo.
  * Tres scores: EE. UU. (el principal), zona euro y global.
  * Backtest mensual desde 1988 con el mismo cálculo y las recesiones del NBER:
    docs/backtest.json. Usa datos revisados, no los que se conocían en cada
    momento, así que es optimista; sirve para ver cuándo avisó y cuándo falló.

Fuentes: FRED (con FRED_API_KEY usa la API; sin ella, fredgraph.csv), Fed
(prima de riesgo de los bonos, EBP), Eurostat, BCE, Treasury.gov y yfinance.

Salidas en docs/:
  dashboard.json  — estado actual: scores, indicadores, modelos, series
  history.json    — una fila por día con los scores v5
  backtest.json   — score mensual de EE. UU. desde 1988 y estadísticas

Reglas heredadas de la v4
  * Un dato que falta es None y NO puntúa: el compuesto se renormaliza sobre
    los pesos disponibles y publica su cobertura.
  * El JSON se escribe con allow_nan=False: si algo se cuela, falla el script
    y el workflow no publica datos rotos.
"""

import os, io, csv, json, math, datetime as dt, time, warnings
import xml.etree.ElementTree as ET
import urllib.request

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

FRED_API_KEY = os.environ.get('FRED_API_KEY', '')
NOW = dt.datetime.now(dt.timezone.utc)
TS = NOW.strftime('%Y-%m-%dT%H:%M:%SZ')
TODAY = NOW.date()
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'docs')
BT_START = '1988-01-01'   # dos años antes de la recesión de 1990


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
    if isinstance(obj, (float, np.floating)):
        obj = float(obj)
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def http_get(url, data=None, headers=None, timeout=40):
    h = {'User-Agent': 'Mozilla/5.0 recession-monitor'}
    h.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def retry(fn, what, tries=3):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            if i == tries - 1:
                print(f'  ✗ {what}: {e}')
                return None
            time.sleep(2 * (i + 1))


def series(dates, values):
    s = pd.Series([num(v) for v in values], index=pd.to_datetime(dates), dtype=float)
    return s.dropna().sort_index()


def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# ══════════════════════════════════════════════════════════════════════════════
# FUENTES
# ══════════════════════════════════════════════════════════════════════════════

def fred(sid, start='1960-01-01'):
    """Serie completa de FRED como pd.Series, o None."""
    def get():
        if FRED_API_KEY:
            url = ('https://api.stlouisfed.org/fred/series/observations'
                   f'?series_id={sid}&api_key={FRED_API_KEY}&file_type=json'
                   f'&observation_start={start}')
            obs = json.loads(http_get(url)).get('observations', [])
            s = series([o['date'] for o in obs], [o.get('value') for o in obs])
        else:
            url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd={start}'
            rows = list(csv.reader(io.StringIO(http_get(url).decode('utf-8'))))[1:]
            s = series([r[0] for r in rows], [r[1] for r in rows])
        if s.empty:
            raise ValueError('vacía')
        return s
    return retry(get, f'FRED {sid}')


def yf_close(ticker, period='2y'):
    """Serie de cierres sin filas vacías ni zona horaria, o None."""
    def tidy(s):
        if hasattr(s, 'columns'):          # MultiIndex de yfinance >= 0.2.48
            s = s.iloc[:, 0]
        s = s.astype(float).dropna()
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        if len(s) < 5:
            raise ValueError(f'solo {len(s)} filas válidas')
        return s

    def get():
        import yfinance as yf
        d = yf.download(ticker, period=period, progress=False, auto_adjust=True, threads=False)
        if d is None or d.empty:
            raise ValueError('vacío')
        return tidy(d['Close'])

    def get_hist():
        import yfinance as yf
        return tidy(yf.Ticker(ticker).history(period='2y', auto_adjust=True)['Close'])

    s = retry(get, f'yfinance {ticker}')
    # Desde GitHub Actions, Yahoo a veces devuelve una sola fila de un futuro
    # (TTF=F, domingo 1-nov-2026), aunque desde casa funcione: se prueba con
    # un plazo fijo y luego con Ticker.history.
    if s is None and period == 'max':
        period = '10y'
        s = retry(get, f'yfinance {ticker} (10y)')
    if s is None:
        s = retry(get_hist, f'yfinance {ticker} (history 2y)', tries=2)
    return s


def previous_indicator(key):
    """Último valor publicado de un indicador (dashboard.json anterior), para no
    perderlo si una fuente falla un día. Devuelve (valor, fecha) o (None, None)."""
    try:
        with open(os.path.join(OUT, 'dashboard.json'), encoding='utf-8') as fh:
            for i in json.load(fh).get('indicators', []):
                if i.get('key') == key and i.get('value') is not None and i.get('as_of'):
                    return i['value'], i['as_of']
    except Exception:
        pass
    return None, None


def eurostat(dataset, since='1990-01', **params):
    """Serie mensual de Eurostat (todas las dimensiones fijadas salvo el tiempo)."""
    def get():
        q = '&'.join(f'{k}={v}' for k, v in params.items())
        url = ('https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/'
               f'{dataset}?format=JSON&sinceTimePeriod={since}&{q}')
        d = json.loads(http_get(url))
        idx, vals = d['dimension']['time']['category']['index'], d['value']
        rows = sorted((p, vals.get(str(i))) for p, i in idx.items())
        s = series([p + '-01' for p, _ in rows], [v for _, v in rows])
        if s.empty:
            raise ValueError('vacía')
        return s
    return retry(get, f'Eurostat {dataset}')


def ecb(flow, key, start='1999-01-01'):
    """Serie del portal de datos del BCE (diaria, mensual o trimestral)."""
    def get():
        url = (f'https://data-api.ecb.europa.eu/service/data/{flow}/{key}'
               f'?format=csvdata&startPeriod={start}')
        rows = list(csv.DictReader(io.StringIO(http_get(url).decode('utf-8'))))
        def to_date(p):
            if '-Q' in p:
                y, q = p.split('-Q')
                return f'{y}-{(int(q) - 1) * 3 + 1:02d}-01'
            return p if len(p) == 10 else p + '-01'
        s = series([to_date(r['TIME_PERIOD']) for r in rows], [r['OBS_VALUE'] for r in rows])
        if s.empty:
            raise ValueError('vacía')
        return s
    return retry(get, f'BCE {flow}/{key}')


def fed_ebp():
    """Prima de riesgo de los bonos (Gilchrist-Zakrajšek) y la probabilidad de
    recesión a 12 meses que publica la Fed con ella."""
    def get():
        url = 'https://www.federalreserve.gov/econres/notes/feds-notes/ebp_csv.csv'
        rows = list(csv.DictReader(io.StringIO(http_get(url).decode('utf-8'))))
        d = [r['date'] for r in rows]
        return {'ebp': series(d, [r['ebp'] for r in rows]),
                'prob': series(d, [num(r['est_prob']) * 100 if num(r['est_prob']) is not None else None
                                   for r in rows])}
    return retry(get, 'Fed EBP') or {'ebp': None, 'prob': None}


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
    first = TODAY.replace(day=1)
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


def debt_limit():
    """Deuda sujeta al techo legal y el techo, del Daily Treasury Statement (API
    Fiscal Data del Tesoro, sin clave). Miles de millones de dólares. Incluye el
    ritmo de los últimos 90 días y la fecha en la que, a ese ritmo, se llegaría
    al techo (estimación lineal: los cobros de impuestos la mueven)."""
    def get():
        since = (TODAY - dt.timedelta(days=200)).isoformat()
        url = ('https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/dts/'
               f'debt_subject_to_limit?filter=record_date:gte:{since}&page[size]=5000')
        by = {}
        for r in json.loads(http_get(url))['data']:
            d = by.setdefault(r['record_date'], {'subject': 0.0, 'limit': None})
            v, c = num(r.get('close_today_bal')), r.get('debt_catg')
            if v is None:
                continue
            if c in ('Debt Held by the Public', 'Intragovernmental Holdings', 'Other Debt Subject to Limit'):
                d['subject'] += v
            elif c == 'Debt Not Subject to Limit':
                d['subject'] -= v
            elif c == 'Statutory Debt Limit':
                d['limit'] = v
        s = pd.Series({pd.Timestamp(k): v['subject'] for k, v in by.items()}).sort_index() / 1e3
        if s.empty:
            raise ValueError('vacía')
        last = s.index[-1]
        limit = num(by[str(last.date())]['limit'])
        limit = limit / 1e3 if limit is not None else None
        past = s[s.index <= last - pd.Timedelta(days=90)]
        pace = num((s.iloc[-1] - past.iloc[-1]) / (last - past.index[-1]).days) if len(past) else None
        head = limit - s.iloc[-1] if limit is not None else None
        proj = (str((last + pd.Timedelta(days=head / pace)).date())
                if head is not None and pace and pace > 0 and head > 0 else None)
        return {'date': str(last.date()), 'subject_bn': rnd(s.iloc[-1], 0), 'limit_bn': rnd(limit, 0),
                'headroom_bn': rnd(head, 0), 'pace_bn_day': rnd(pace, 2), 'projected': proj}
    return retry(get, 'Tesoro: techo de deuda')


# ══════════════════════════════════════════════════════════════════════════════
# CÁLCULOS SOBRE SERIES
# ══════════════════════════════════════════════════════════════════════════════

def mon(s, how='mean'):
    """Diaria/semanal → mensual con índice al día 1. El mes en curso usa lo que
    haya hasta hoy."""
    if s is None or s.empty:
        return None
    r = s.resample('MS')
    return (r.mean() if how == 'mean' else r.last()).dropna()


def qtr_to_month(s, lag=0):
    """Trimestral → mensual repitiendo el valor; lag = meses que tarda en publicarse."""
    if s is None or s.empty:
        return None
    idx = pd.date_range(s.index[0], TODAY.replace(day=1), freq='MS')
    m = s.reindex(idx, method='ffill')
    return m.shift(lag).dropna() if lag else m


def ann_chg(s, n=6):
    """Variación anualizada a n meses, en %."""
    return ((s / s.shift(n)) ** (12 / n) - 1) * 100


def yoy(s, n=12):
    return (s / s.shift(n) - 1) * 100


def sahm_like(u):
    """Media móvil de 3 meses menos el mínimo de esa media en los 12 meses anteriores."""
    ma3 = u.rolling(3).mean()
    return (ma3 - ma3.shift(1).rolling(12).min()).dropna()


def last_valid(s):
    if s is None:
        return None, None
    s = s.dropna()
    if s.empty:
        return None, None
    return num(s.iloc[-1]), s.index[-1]


def pts(s, n=180, digits=2):
    """Últimos n puntos → [[fecha, valor], ...]."""
    if s is None:
        return []
    s = s.dropna().tail(n)
    return [[str(i.date()), rnd(v, digits)] for i, v in s.items() if num(v) is not None]


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


def up(*b):    # más alto = más riesgo
    return lambda v: band(v, b, [5, 20, 40, 60, 80, 95])


def down(*b):  # más bajo = más riesgo
    return lambda v: band(v, b, [95, 80, 60, 40, 20, 5])


SCORERS = {
    # ── EE. UU., crédito y tipos
    'curve_prob':  up(10, 20, 30, 40, 55),          # % del probit de la Fed de NY
    'ntfs':        down(-0.50, -0.25, 0.0, 0.25, 0.50),
    'ebp_prob':    up(15, 25, 35, 50, 65),          # % del modelo de la Fed
    'sloos':       up(0, 10, 20, 35, 50),           # % neto de bancos que endurecen
    'anfci':       up(-0.50, -0.25, 0.0, 0.25, 0.50),
    # ── EE. UU., economía real adelantada
    'claims_rise': up(8, 12, 16, 22, 30),           # % sobre el mínimo de 52 semanas
    'permits':     down(-25, -18, -12, -6, 0),      # % interanual, media 3M
    'neworders':   down(-12, -7, -3, 0, 4),         # % anualizado a 6M, media 3M
    'temphelp':    down(-10, -7, -4, -2, 0),        # % interanual
    'cli_us':      down(98.5, 99.0, 99.4, 99.8, 100.2),
    'umich_chg':   down(-20, -15, -10, -5, 0),      # puntos en 12 meses, media 3M
    'oil_yoy':     up(10, 25, 45, 70, 100),         # % interanual del Brent
    'profits':     down(-10, -5, 0, 3, 6),          # % interanual
    # ── EE. UU., confirmación
    'sahm':        up(0.10, 0.20, 0.35, 0.50, 0.80),
    'ms':          up(0.10, 0.20, 0.29, 0.50, 0.81),
    'sos':         up(0.05, 0.10, 0.15, 0.20, 0.30),
    'payrolls_3m': down(-50, 0, 25, 75, 150),       # miles al mes
    'income6':     down(-2, -1, 0, 1, 2),
    'ip6':         down(-5, -3, -1, 0, 2),
    # ── EE. UU., mercados
    'spx_dd':      down(-30, -20, -15, -10, -5),
    'hy':          up(3.5, 4.5, 5.5, 7.0, 8.5),
    'vix':         up(15, 20, 25, 30, 40),
    # ── Zona euro
    'esi':         down(90, 94, 97, 100, 103),
    'ea_curve':    down(-0.75, -0.25, 0.0, 0.25, 0.75),
    'bls':         up(0, 10, 20, 30, 45),
    'ea_sahm':     up(0.10, 0.20, 0.30, 0.50, 0.80),
    'cli_ea':      down(98.5, 99.0, 99.4, 99.8, 100.2),
    'ttf':         up(30, 45, 60, 80, 110),
    # ── Global
    'cli_china':   down(98.5, 99.0, 99.4, 99.8, 100.2),
    'cli_g7':      down(98.5, 99.0, 99.4, 99.8, 100.2),
    'copper6':     down(-25, -15, -8, -2, 5),
    'cu_au6':      down(-25, -15, -8, -2, 5),
    'oil_yoy_g':   up(10, 25, 45, 70, 100),
    'dxy':         up(95, 100, 104, 108, 112),
}

# Pesos (suman 1 en cada score). Lo que no está aquí se muestra pero no puntúa.
US_WEIGHTS = {
    'curve_prob': 0.08, 'ntfs': 0.06, 'ebp_prob': 0.08, 'sloos': 0.05, 'anfci': 0.03,
    'claims_rise': 0.07, 'permits': 0.05, 'neworders': 0.04, 'temphelp': 0.03,
    'cli_us': 0.04, 'umich_chg': 0.03, 'oil_yoy': 0.06, 'profits': 0.03,
    'sahm': 0.06, 'ms': 0.06, 'sos': 0.04, 'payrolls_3m': 0.04, 'income6': 0.025, 'ip6': 0.025,
    'spx_dd': 0.04, 'hy': 0.04, 'vix': 0.02,
}
US_PILLARS = {
    'Crédito y tipos':    ['curve_prob', 'ntfs', 'ebp_prob', 'sloos', 'anfci'],
    'Economía adelantada': ['claims_rise', 'permits', 'neworders', 'temphelp', 'cli_us',
                            'umich_chg', 'oil_yoy', 'profits'],
    'Confirmación':       ['sahm', 'ms', 'sos', 'payrolls_3m', 'income6', 'ip6'],
    'Mercados':           ['spx_dd', 'hy', 'vix'],
}
EA_WEIGHTS = {'esi': 0.25, 'ea_curve': 0.20, 'bls': 0.20, 'ea_sahm': 0.15, 'cli_ea': 0.10, 'ttf': 0.10}
GL_WEIGHTS = {'cli_china': 0.25, 'cli_g7': 0.20, 'copper6': 0.15, 'cu_au6': 0.10,
              'oil_yoy_g': 0.20, 'dxy': 0.10}


def status_of(sc):
    if sc is None:
        return 'SIN DATO'
    return 'OK' if sc < 25 else 'VIGILAR' if sc < 50 else 'ALERTA' if sc < 75 else 'PELIGRO'


def label_of(sc):
    if sc is None: return 'SIN DATOS'
    if sc < 20: return 'RIESGO BAJO'
    if sc < 35: return 'MODERADO'
    if sc < 50: return 'ELEVADO'
    if sc < 65: return 'ALTO RIESGO'
    return 'RECESIÓN PROBABLE'


def composite(scores, weights):
    """Media ponderada sobre los indicadores con dato y cobertura (peso con dato / total)."""
    tot = sum(weights.values())
    got = sum(w for k, w in weights.items() if scores.get(k) is not None)
    if got == 0:
        return None, 0.0
    sc = sum(scores[k] * w for k, w in weights.items() if scores.get(k) is not None) / got
    return round(sc, 1), round(got / tot, 3)


def pillar_scores(scores, weights, pillars):
    out = {}
    for name, keys in pillars.items():
        ws = [(scores[k], weights[k]) for k in keys if scores.get(k) is not None]
        out[name] = round(sum(s * w for s, w in ws) / sum(w for _, w in ws), 1) if ws else None
    return out


# ══════════════════════════════════════════════════════════════════════════════
# FICHA DE CADA INDICADOR
# key: (etiqueta, grupo, unidad, decimales, adelanto, umbral de alerta, por qué, fuente, días máximos
#       desde la fecha del dato antes de marcarlo atrasado: los mensuales van fechados el día 1 y salen
#       semanas después, los trimestrales hasta 5 meses después)
# ══════════════════════════════════════════════════════════════════════════════

META = {
    # ── Crédito y tipos
    'curve_prob': ('Curva 10A−3M → probabilidad de recesión', 'Crédito y tipos', '%', 0, '12M',
                   '>30% = alerta', 'Probit de Estrella y Mishkin (Fed de NY): el mejor predictor a 4-6 trimestres. Falló en 2022-24.',
                   'FRED T10Y3M, media del mes', 5),
    'ntfs': ('Diferencial a plazo cercano (aprox.)', 'Crédito y tipos', ' pp', 2, '6–12M',
             '<0 = el mercado espera recortes', 'Engstrom y Sharpe (Fed, 2019): el tipo a 3M esperado dentro de 1-2 años menos el actual. Domina a la curva 10A-2A.',
             'FRED DGS1, DGS2, DGS3MO', 5),
    'ebp_prob': ('Prima de riesgo de los bonos → probabilidad', 'Crédito y tipos', '%', 0, '12M',
                 '>35% = alerta', 'Gilchrist y Zakrajšek (2012): la parte del diferencial de crédito que no es riesgo de impago. La Fed publica la probabilidad.',
                 'Fed, EBP mensual', 75),
    'sloos': ('Bancos que endurecen el crédito a empresas', 'Crédito y tipos', '%', 1, '6–12M',
              '>20% = alerta', 'Lown, Morgan y Rohatgi (Fed de NY, 2000): predice préstamos y producción. Dio falsa alarma en 2023.',
              'Fed SLOOS (FRED DRTSCILM)', 140),
    'anfci': ('Condiciones financieras (ANFCI)', 'Crédito y tipos', '', 2, '0–6M',
              '>0 = más tensas que la media', 'Fed de Chicago, ajustado por el ciclo. Las crisis de crédito llegan por aquí.',
              'Fed de Chicago (FRED ANFCI)', 14),
    # ── Economía adelantada
    'claims_rise': ('Peticiones de paro sobre su mínimo de 52 sem.', 'Economía adelantada', '%', 1, '3–9M',
                    '>16% = alerta', 'Antes de cada recesión, la media de 4 semanas sube un 15-20% sobre su mínimo. Dato semanal y sin revisiones grandes.',
                    'FRED ICSA, media 4 sem.', 14),
    'permits': ('Permisos de construcción (interanual)', 'Economía adelantada', '%', 1, '6–18M',
                '<−12% = alerta', 'Leamer (2007), «Housing IS the business cycle». Cayó más de un 25% en 2022-23 sin recesión.',
                'Census (FRED PERMIT), media 3M', 75),
    'neworders': ('Pedidos de bienes de equipo (6M anualizado)', 'Economía adelantada', '%', 1, '3–9M',
                  '<−3% = alerta', 'Bienes de equipo sin defensa ni aviones: lo que las empresas planean invertir. Sustituye al ISM, que es de pago.',
                  'Census (FRED NEWORDER), media 3M', 90),
    'temphelp': ('Empleo temporal (interanual)', 'Economía adelantada', '%', 1, '6–12M',
                 '<−4% = alerta', 'Lo primero que recortan las empresas. Lleva cayendo desde 2022 sin recesión: peso bajo.',
                 'BLS (FRED TEMPHELPS)', 75),
    'cli_us': ('Indicador adelantado de la OCDE (EE. UU.)', 'Economía adelantada', '', 2, '6–9M',
               '<99,4 = alerta', 'Índice compuesto de la OCDE, 100 = tendencia. En pruebas ROC rinde peor que los diferenciales y el crédito (Camacho y Palmieri, 2021).',
               'OCDE (FRED USALOLITOAASTSAM)', 75),
    'umich_chg': ('Confianza del consumidor (cambio en 12M)', 'Economía adelantada', ' pts', 1, '3–9M',
                  '<−10 pts = alerta', 'Se usa el cambio, no el nivel: desde 2022 el nivel está en mínimos sin recesión.',
                  'U. Michigan (FRED UMCSENT), media 3M', 75),
    'oil_yoy': ('Petróleo Brent (interanual)', 'Economía adelantada', '%', 0, '3–12M',
                '>45% = choque', 'Hamilton: nueve de las diez últimas recesiones vinieron tras una subida fuerte del petróleo. Empuja a una economía ya frágil.',
                'FRED DCOILBRENTEU + BZ=F', 5),
    'profits': ('Beneficios empresariales (interanual)', 'Economía adelantada', '%', 1, '3–12M',
                '<0% = alerta', 'Cuando caen los beneficios, las empresas recortan empleo e inversión. Llega con 3-5 meses de retraso.',
                'BEA (FRED A446RC1Q027SBEA)', 250),
    # ── Confirmación
    'sahm': ('Regla de Sahm', 'Confirmación', ' pp', 2, '0–3M',
             '≥0,50 = recesión en curso', 'Sahm (2019). Saltó en 2024 por la inmigración, no por despidos: un falso positivo.',
             'FRED UNRATE (calculada)', 75),
    'ms': ('Regla de Michaillat-Saez (paro y vacantes)', 'Confirmación', ' pp', 2, '0–3M',
           '≥0,29 = recesión probable', 'Michaillat y Saez (2025): el mínimo de la señal de paro y la de vacantes. Detecta las 15 recesiones desde 1929 sin falsos positivos.',
           'FRED UNRATE y JTSJOR', 100),     # JOLTS sale ~5 semanas después del mes
    'sos': ('Paro asegurado (indicador SOS)', 'Confirmación', ' pp', 2, '0–3M',
            '≥0,20 = recesión', 'Fed de Richmond: la regla de Sahm con datos administrativos semanales. Avisa de las 7 recesiones desde los 70.',
            'FRED IURSA, media 26 sem.', 21),
    'payrolls_3m': ('Nóminas no agrícolas (media 3M)', 'Confirmación', 'K', 0, '0–3M',
                    '<25K = alerta', 'Con menos inmigración, el empleo «neutral» es más bajo que antes (~25-50K al mes).',
                    'BLS (FRED PAYEMS)', 75),
    'income6': ('Renta real sin transferencias (6M anualizado)', 'Confirmación', '%', 1, '0–3M',
                '<0% = alerta', 'Uno de los indicadores con los que el NBER fecha las recesiones.',
                'BEA (FRED W875RX1)', 90),
    'ip6': ('Producción industrial (6M anualizado)', 'Confirmación', '%', 1, '0–3M',
            '<−1% = alerta', 'Coincidente. Pesa menos que antes: la industria es ~10% de la economía.',
            'Fed (FRED INDPRO)', 75),
    # ── Mercados
    'spx_dd': ('S&P 500: caída desde máximo 52 sem.', 'Mercados', '%', 1, '0–6M',
               '<−15% = corrección seria', 'La bolsa anticipa, pero «ha predicho nueve de las cinco últimas recesiones» (Samuelson).',
               'Yahoo ^GSPC', 5),
    'hy': ('Diferencial high yield (OAS)', 'Mercados', '%', 2, '0–6M',
           '>5,5% = estrés', 'El crédito arriesgado suele moverse antes que la volatilidad.',
           'ICE BofA (FRED BAMLH0A0HYM2)', 5),
    'vix': ('VIX', 'Mercados', '', 1, '0–3M', '>30 = pánico', 'Volatilidad implícita del S&P 500.',
            'CBOE ^VIX', 5),
    # ── Zona euro
    'esi': ('Sentimiento económico (ESI)', 'Zona euro', '', 1, '0–6M',
            '<97 = alerta', 'Encuesta de la Comisión Europea; 100 = media histórica. Sigue bien al PIB trimestral.',
            'Eurostat ei_bssi_m_r2, EA21', 75),
    'ea_curve': ('Curva euro 10A−3M (AAA)', 'Zona euro', ' pp', 2, '6–18M',
                 '<0 = invertida', 'El BCE (2023) confirma su poder predictivo; un tramo negativo pesa mucho más que uno bajo y positivo.',
                 'BCE, curva AAA', 5),
    'bls': ('Bancos que endurecen el crédito a empresas', 'Zona euro', '%', 1, '6–12M',
            '>20% = alerta', 'Encuesta de préstamos bancarios del BCE: el equivalente europeo del SLOOS.',
            'BCE BLS (WFNET)', 140),
    'ea_sahm': ('Paro de la zona euro: regla tipo Sahm', 'Zona euro', ' pp', 2, '0–3M',
                '≥0,30 = alerta', 'La misma lógica de Sahm aplicada al paro europeo, que se mueve más despacio.',
                'Eurostat une_rt_m, EA21', 90),
    'cli_ea': ('Indicador adelantado de la OCDE (4 grandes del euro)', 'Zona euro', '', 2, '6–9M',
               '<99,4 = alerta', 'Media ponderada por PIB de Alemania, Francia, Italia y España.',
               'OCDE vía FRED', 75),
    'ttf': ('Gas natural TTF', 'Zona euro', ' €', 1, '3–12M',
            '>60 € = choque', 'El choque de energía es el canal por el que Europa entró en estancamiento en 2022-23.',
            'Yahoo TTF=F, media del mes', 5),
    # ── Global
    'cli_china': ('Indicador adelantado de la OCDE (China)', 'Global', '', 2, '6–9M',
                  '<99,4 = alerta', 'China es la mayor fuente de demanda de materias primas y de bienes intermedios.',
                  'OCDE (FRED CHNLOLITOAASTSAM)', 75),
    'cli_g7': ('Indicador adelantado de la OCDE (G7)', 'Global', '', 2, '6–9M',
               '<99,4 = alerta', 'Las recesiones globales (1975, 1982, 1991, 2009, 2020) son sincronizadas.',
               'OCDE (FRED G7LOLITOAASTSAM)', 75),
    'copper6': ('Cobre (cambio a 6 meses)', 'Global', '%', 1, '3–6M',
                '<−8% = alerta', '«Doctor Cobre»: demanda industrial y de construcción en tiempo real. Tiene ruido.',
                'Yahoo HG=F', 5),
    'cu_au6': ('Ratio cobre/oro (cambio a 6 meses)', 'Global', '%', 1, '3–6M',
               '<−8% = alerta', 'Crecimiento frente a refugio: cae cuando el mercado teme por la economía mundial.',
               'Yahoo HG=F / GC=F', 5),
    'oil_yoy_g': ('Petróleo Brent (interanual)', 'Global', '%', 0, '3–12M',
                  '>45% = choque', 'El mismo dato que en EE. UU.: un choque de oferta golpea a todos los importadores.',
                  'FRED DCOILBRENTEU + BZ=F', 5),
    'dxy': ('Dólar (DXY)', 'Global', '', 1, '3–9M', '>104 = estrés en emergentes',
            'Un dólar fuerte encarece la deuda en dólares de medio mundo.', 'ICE DXY', 5),
}

# Informativos (no puntúan): los usan las pestañas de mercado y la tesis de Hartnett
INFO_META = {
    'curve_10y3m': ('Curva 10A − 3M', 'Mercados', ' pp', 2),
    'curve_10y2y': ('Curva 10A − 2A', 'Mercados', ' pp', 2),
    'bond_30y': ('Bono del Tesoro a 30 años', 'Mercados', '%', 2),
    'sox_bubble': ('Semis (SOXX) sobre su MM200', 'Mercados', '%', 1),
    'oil_price': ('Petróleo WTI', 'Mercados', ' $', 1),
    'oil_rise': ('Petróleo vs mín. 52 sem.', 'Mercados', '%', 0),
    'brent': ('Petróleo Brent', 'Mercados', ' $', 1),
    'unrate': ('Tasa de paro', 'Confirmación', '%', 1),
    'claims_4w': ('Peticiones de paro (media 4 sem.)', 'Confirmación', '', 0),
    'ea_unrate': ('Tasa de paro zona euro', 'Zona euro', '%', 1),
}

# Regla de amplitud global de BofA (Hartnett): compra contraria cuando el 88% de
# las bolsas del MSCI ACWI cotiza a la vez por debajo de su media de 50 y de 200
# sesiones. Se aproxima con los ETF de países MSCI, en dólares como el índice.
# Portugal (PGAL) ya no cotiza en Yahoo.
BREADTH = {
    'EWA': 'Australia', 'EWO': 'Austria', 'EWK': 'Bélgica', 'EWC': 'Canadá', 'EDEN': 'Dinamarca',
    'EFNL': 'Finlandia', 'EWQ': 'Francia', 'EWG': 'Alemania', 'EWH': 'Hong Kong', 'EIRL': 'Irlanda',
    'EIS': 'Israel', 'EWI': 'Italia', 'EWJ': 'Japón', 'EWN': 'Países Bajos', 'ENZL': 'Nueva Zelanda',
    'ENOR': 'Noruega', 'EWS': 'Singapur', 'EWP': 'España', 'EWD': 'Suecia', 'EWL': 'Suiza',
    'EWU': 'Reino Unido', 'SPY': 'EE. UU.',
    'EWZ': 'Brasil', 'ECH': 'Chile', 'MCHI': 'China', 'GREK': 'Grecia', 'INDA': 'India',
    'EIDO': 'Indonesia', 'KWT': 'Kuwait', 'EWY': 'Corea', 'EWM': 'Malasia', 'EWW': 'México',
    'EPU': 'Perú', 'EPHE': 'Filipinas', 'EPOL': 'Polonia', 'QAT': 'Catar', 'KSA': 'Arabia Saudí',
    'EZA': 'Sudáfrica', 'EWT': 'Taiwán', 'THD': 'Tailandia', 'TUR': 'Turquía', 'UAE': 'Emiratos',
}

# Recesiones del NBER, por si FRED (USREC) falla el día de la ejecución
NBER_SPANS = [['1990-07-01', '1991-03-01'], ['2001-03-01', '2001-11-01'],
              ['2007-12-01', '2009-06-01'], ['2020-02-01', '2020-04-01']]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print(f'[{TS}] Recession Monitor v5' + ('' if FRED_API_KEY else '  [sin FRED_API_KEY: fredgraph.csv]'))

    # ── Descargas ────────────────────────────────────────────────────────────
    print('FRED…')
    F = {sid: fred(sid) for sid in [
        'T10Y3M', 'T10Y2Y', 'DGS1', 'DGS2', 'DGS3MO', 'DGS30', 'DRTSCILM', 'ANFCI',
        'ICSA', 'PERMIT', 'NEWORDER', 'TEMPHELPS', 'USALOLITOAASTSAM', 'UMCSENT',
        'DCOILBRENTEU', 'A446RC1Q027SBEA', 'UNRATE', 'JTSJOR', 'IURSA', 'PAYEMS',
        'W875RX1', 'INDPRO', 'BAMLH0A0HYM2', 'USREC', 'RECPROUSM156N',
        'CHNLOLITOAASTSAM', 'G7LOLITOAASTSAM', 'DEULOLITOAASTSAM', 'FRALOLITOAASTSAM',
        'ITALOLITOAASTSAM', 'ESPLOLITOAASTSAM', 'DFEDTARU', 'DGS10']}
    print('Fed, Eurostat, BCE…')
    EBP = fed_ebp()
    ESI = eurostat('ei_bssi_m_r2', geo='EA21', indic='BS-ESI-I', s_adj='SA')
    EAU = eurostat('une_rt_m', since='1999-01', geo='EA21', s_adj='SA', age='TOTAL', sex='T', unit='PC_ACT')
    EA10 = ecb('YC', 'B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y', '2004-01-01')
    EA3M = ecb('YC', 'B.U2.EUR.4F.G_N_A.SV_C_YM.SR_3M', '2004-01-01')
    BLS = ecb('BLS', 'Q.U2.ALL.O.E.Z.B3.ST.S.WFNET', '2003-01-01')
    print('Precios (yfinance)…')
    P = {k: yf_close(t, per) for k, (t, per) in {
        'spx': ('^GSPC', 'max'), 'vix': ('^VIX', 'max'), 'copper': ('HG=F', 'max'),
        'gold': ('GC=F', 'max'), 'ttf': ('TTF=F', 'max'), 'brent': ('BZ=F', '2y'),
        'wti': ('CL=F', '2y'), 'dxy': ('DX-Y.NYB', 'max'), 'soxx': ('SOXX', '2y'),
        'tyx': ('^TYX', '2y')}.items()}
    sectors = {'Tecnología': 'XLK', 'Financieras': 'XLF', 'Energía': 'XLE',
               'Materiales': 'XLB', 'Salud': 'XLV', 'Utilities': 'XLU',
               'Industriales': 'XLI', 'Consumo disc.': 'XLY',
               'Consumo básico': 'XLP', 'Comunicación': 'XLC',
               'Inmobiliario': 'XLRE'}
    sector_px = {n: yf_close(t, '1y') for n, t in sectors.items()}
    # Lo que vigilan Hartnett y Cárpatos: las Siete Magníficas, semis, Nasdaq,
    # equiponderado y empresas con caja libre (pestaña «Hartnett y Cárpatos»)
    PULSE = {'mags': 'MAGS', 'smh': 'SMH', 'qqq': 'QQQ', 'rsp': 'RSP', 'spy': 'SPY', 'vflo': 'VFLO',
             'aapl': 'AAPL', 'msft': 'MSFT', 'googl': 'GOOGL', 'amzn': 'AMZN', 'meta': 'META',
             'nvda': 'NVDA', 'tsla': 'TSLA',
             # canarios y termómetros de Hartnett; futuros para los niveles de los CTA (Cárpatos)
             'tsm': 'TSM', 'ixg': 'IXG', 'mdy': 'MDY', 'jnk': 'JNK', 'es': 'ES=F', 'nq': 'NQ=F'}
    Q = {k: yf_close(t, '2y') for k, t in PULSE.items()}
    print('Amplitud global (ETF de países MSCI), semis desde 1994 y techo de deuda…')
    BR = {t: yf_close(t, 'max') for t in BREADTH}
    acwi = yf_close('ACWI', 'max')
    sox_long = yf_close('^SOX', 'max')
    debt = debt_limit()
    print('Curva (Treasury.gov)…')
    tsy = treasury_curve()
    curve = tsy['curve']
    try:
        with open(os.path.join(OUT, 'manual.json'), encoding='utf-8') as fh:
            manual = json.load(fh)
    except Exception:
        manual = {}

    # ══════════════════════════════════════════════════════════════════════════
    # SERIES MENSUALES DE CADA INDICADOR (las mismas para hoy y para el backtest)
    # ══════════════════════════════════════════════════════════════════════════
    M, asof = {}, {}

    def put(key, s, raw=None):
        """Guarda la serie mensual y la fecha del último dato real que la alimenta."""
        M[key] = s.dropna() if s is not None else None
        src = raw if raw is not None else s
        asof[key] = str(src.dropna().index[-1].date()) if src is not None and not src.dropna().empty else None

    sp = mon(F['T10Y3M'])
    put('curve_prob', sp.apply(lambda x: 100 * phi(-0.5333 - 0.6330 * x)) if sp is not None else None, F['T10Y3M'])
    if all(F[k] is not None for k in ('DGS1', 'DGS2', 'DGS3MO')):
        y1, y2, y3m = mon(F['DGS1']), mon(F['DGS2']), mon(F['DGS3MO'])
        fwd = ((1 + y2 / 100) ** 2 / (1 + y1 / 100) - 1) * 100
        put('ntfs', fwd - y3m, F['DGS3MO'])
    else:
        put('ntfs', None)
    put('ebp_prob', EBP['prob'])
    put('sloos', qtr_to_month(F['DRTSCILM'], lag=1), F['DRTSCILM'])
    put('anfci', mon(F['ANFCI'], 'last'), F['ANFCI'])
    if F['ICSA'] is not None:
        ma4 = F['ICSA'].rolling(4).mean()
        put('claims_rise', mon((ma4 / ma4.rolling(52).min() - 1) * 100, 'last'), F['ICSA'])
    else:
        put('claims_rise', None)
    put('permits', yoy(F['PERMIT'].rolling(3).mean()) if F['PERMIT'] is not None else None, F['PERMIT'])
    put('neworders', ann_chg(F['NEWORDER'].rolling(3).mean()) if F['NEWORDER'] is not None else None, F['NEWORDER'])
    put('temphelp', yoy(F['TEMPHELPS']) if F['TEMPHELPS'] is not None else None, F['TEMPHELPS'])
    put('cli_us', F['USALOLITOAASTSAM'])
    put('umich_chg', (F['UMCSENT'].rolling(3).mean().diff(12)) if F['UMCSENT'] is not None else None, F['UMCSENT'])

    # Brent: FRED (histórico, 2-3 días de retraso) + Yahoo para los últimos días
    brent = F['DCOILBRENTEU']
    if brent is not None and P['brent'] is not None:
        brent = pd.concat([brent, P['brent'][P['brent'].index > brent.index[-1]]])
    elif brent is None:
        brent = P['brent']
    bm = mon(brent)
    put('oil_yoy', yoy(bm) if bm is not None else None, brent)
    put('oil_yoy_g', M['oil_yoy'], brent)

    prof = F['A446RC1Q027SBEA']
    put('profits', qtr_to_month(yoy(prof, 4), lag=5) if prof is not None else None, prof)

    u = F['UNRATE']
    put('sahm', sahm_like(u) if u is not None else None, u)
    if u is not None and F['JTSJOR'] is not None:
        v = F['JTSJOR']
        vma = v.rolling(3).mean()
        vind = (vma.shift(1).rolling(12).max() - vma).dropna()
        uind = sahm_like(u)
        idx = uind.index.union(vind.index)
        vind = vind.reindex(idx).ffill(limit=1)    # JOLTS llega un mes después
        put('ms', pd.concat([uind.reindex(idx), vind], axis=1).min(axis=1, skipna=False).dropna(), v)
    else:
        put('ms', None)
    if F['IURSA'] is not None:
        ma26 = F['IURSA'].rolling(26).mean()
        put('sos', mon(ma26 - ma26.shift(1).rolling(52).min(), 'last'), F['IURSA'])
    else:
        put('sos', None)
    put('payrolls_3m', F['PAYEMS'].diff().rolling(3).mean() if F['PAYEMS'] is not None else None, F['PAYEMS'])
    put('income6', ann_chg(F['W875RX1']) if F['W875RX1'] is not None else None, F['W875RX1'])
    put('ip6', ann_chg(F['INDPRO']) if F['INDPRO'] is not None else None, F['INDPRO'])

    if P['spx'] is not None:
        dd = (P['spx'] / P['spx'].rolling(252, min_periods=20).max() - 1) * 100
        put('spx_dd', mon(dd, 'last'), P['spx'])
    else:
        put('spx_dd', None)
    put('hy', mon(F['BAMLH0A0HYM2']), F['BAMLH0A0HYM2'])
    put('vix', mon(P['vix']), P['vix'])

    # ── Zona euro
    put('esi', ESI)
    put('ea_curve', (mon(EA10) - mon(EA3M)) if EA10 is not None and EA3M is not None else None, EA10)
    put('bls', qtr_to_month(BLS, lag=1), BLS)
    put('ea_sahm', sahm_like(EAU) if EAU is not None else None, EAU)
    cli4 = {'DEULOLITOAASTSAM': 0.36, 'FRALOLITOAASTSAM': 0.27, 'ITALOLITOAASTSAM': 0.21, 'ESPLOLITOAASTSAM': 0.16}
    if all(F[k] is not None for k in cli4):
        df = pd.concat({k: F[k] for k in cli4}, axis=1).dropna()
        put('cli_ea', sum(df[k] * w for k, w in cli4.items()), df.iloc[:, 0])
    else:
        put('cli_ea', None)
    put('ttf', mon(P['ttf']), P['ttf'])
    if M['ttf'] is None:
        # Si Yahoo falla, se mantiene el último dato publicado con su fecha: el
        # panel lo marca como atrasado a los 5 días en lugar de perder cobertura.
        pv, pd_ = previous_indicator('ttf')
        if pv is not None:
            s_ = pd.Series([pv], index=pd.to_datetime([pd_]))
            put('ttf', s_.resample('MS').last(), s_)
            print(f'  ↺ TTF: se mantiene el último dato publicado ({pv} del {pd_})')

    # ── Global
    put('cli_china', F['CHNLOLITOAASTSAM'])
    put('cli_g7', F['G7LOLITOAASTSAM'])
    cu = mon(P['copper'])
    put('copper6', yoy(cu, 6) if cu is not None else None, P['copper'])
    if P['copper'] is not None and P['gold'] is not None:
        r = mon((P['copper'] / P['gold']).dropna())
        put('cu_au6', yoy(r, 6), P['copper'])
    else:
        put('cu_au6', None)
    put('dxy', mon(P['dxy']), P['dxy'])

    # ══════════════════════════════════════════════════════════════════════════
    # VALORES DE HOY Y PUNTUACIONES
    # ══════════════════════════════════════════════════════════════════════════
    val, sc = {}, {}
    for k in META:
        val[k], _ = last_valid(M.get(k))
        sc[k] = SCORERS[k](val[k])

    us, us_cov = composite(sc, US_WEIGHTS)
    ea, ea_cov = composite(sc, EA_WEIGHTS)
    gl, gl_cov = composite(sc, GL_WEIGHTS)
    us_p = pillar_scores(sc, US_WEIGHTS, US_PILLARS)
    print(f'  EE. UU. {us} ({label_of(us)}), cobertura {us_cov:.0%} · zona euro {ea} ({ea_cov:.0%}) · global {gl} ({gl_cov:.0%})')
    print('  pilares: ' + ', '.join(f'{k} {v}' for k, v in us_p.items()))

    # Informativos
    info = {}
    c10, c3m, c2, c30 = curve.get('10Y'), curve.get('3M'), curve.get('2Y'), curve.get('30Y')
    info['curve_10y3m'] = (c10 - c3m if c10 is not None and c3m is not None else last_valid(F['T10Y3M'])[0], tsy['date'])
    info['curve_10y2y'] = (c10 - c2 if c10 is not None and c2 is not None else last_valid(F['T10Y2Y'])[0], tsy['date'])
    info['bond_30y'] = (c30 if c30 is not None else last_valid(F['DGS30'])[0], tsy['date'])
    s = P['soxx']
    if s is not None and len(s) >= 200:
        ma200 = s.rolling(200).mean()
        sox_series = ((s / ma200 - 1) * 100).dropna()
        info['sox_bubble'] = (num(sox_series.iloc[-1]), str(s.index[-1].date()))
    else:
        sox_series, info['sox_bubble'] = None, (None, None)
    w = P['wti']
    if w is not None:
        low = w.tail(252).min()
        info['oil_price'] = (num(w.iloc[-1]), str(w.index[-1].date()))
        info['oil_rise'] = (num((w.iloc[-1] / low - 1) * 100), str(w.index[-1].date()))
        oil_low52 = rnd(low, 1)
    else:
        info['oil_price'] = info['oil_rise'] = (None, None); oil_low52 = None
    info['brent'] = (num(brent.iloc[-1]) if brent is not None else None, str(brent.index[-1].date()) if brent is not None else None)
    info['unrate'] = (num(u.iloc[-1]) if u is not None else None, str(u.index[-1].date()) if u is not None else None)
    if F['ICSA'] is not None:
        info['claims_4w'] = (num(F['ICSA'].tail(4).mean()), str(F['ICSA'].index[-1].date()))
    else:
        info['claims_4w'] = (None, None)
    info['ea_unrate'] = (num(EAU.iloc[-1]) if EAU is not None else None, str(EAU.index[-1].date()) if EAU is not None else None)

    # ══════════════════════════════════════════════════════════════════════════
    # TABLA DE INDICADORES
    # ══════════════════════════════════════════════════════════════════════════
    weight_of = {**US_WEIGHTS, **EA_WEIGHTS, **GL_WEIGHTS}
    indicators = []
    for k, (label, group, unit, dec, lead, thr, why, source, cadence) in META.items():
        a = asof.get(k)
        age = (TODAY - dt.date.fromisoformat(a)).days if a else None
        indicators.append({
            'key': k, 'label': label, 'group': group, 'unit': unit, 'decimals': dec,
            'value': rnd(val[k], 4), 'score': sc[k], 'status': status_of(sc[k]),
            'weight': weight_of.get(k, 0), 'lead': lead, 'note': thr, 'why': why,
            'source': source, 'as_of': a, 'stale': age is not None and age > cadence,
        })
    for k, (label, group, unit, dec) in INFO_META.items():
        v_, a = info.get(k, (None, None))
        indicators.append({'key': k, 'label': label, 'group': group, 'unit': unit, 'decimals': dec,
                           'value': rnd(v_, 4), 'score': None, 'status': 'SIN DATO', 'weight': 0,
                           'lead': '', 'note': '', 'why': '', 'source': '', 'as_of': a,
                           'stale': False, 'info': True})

    # ══════════════════════════════════════════════════════════════════════════
    # LO QUE DICEN LOS MODELOS PUBLICADOS
    # ══════════════════════════════════════════════════════════════════════════
    ms_v = val.get('ms')
    cp, cp_d = last_valid(F['RECPROUSM156N'])
    models = [
        {'key': 'nyfed', 'label': 'Curva de tipos (probit de la Fed de NY)', 'prob': rnd(val.get('curve_prob'), 1),
         'horizon': 'recesión en los próximos 12 meses', 'as_of': asof.get('curve_prob'),
         'note': 'Estrella y Mishkin; P = Φ(−0,5333 − 0,6330 × diferencial 10A−3M).'},
        {'key': 'ebp', 'label': 'Prima de riesgo de los bonos (Fed)', 'prob': rnd(val.get('ebp_prob'), 1),
         'horizon': 'recesión en los próximos 12 meses', 'as_of': asof.get('ebp_prob'),
         'note': 'Favara, Gilchrist, Lewis y Zakrajšek (2016); la Fed lo publica cada mes.'},
        {'key': 'ms', 'label': 'Regla de Michaillat-Saez', 'prob': rnd(min(100, max(0, (ms_v - 0.29) / (0.81 - 0.29) * 100)) if ms_v is not None else None, 1),
         'horizon': 'que la recesión ya haya empezado', 'as_of': asof.get('ms'),
         'note': 'Interpolación lineal entre 0,29 pp (probable) y 0,81 pp (segura), como en el artículo.'},
        {'key': 'cp', 'label': 'Chauvet-Piger (coincidente)', 'prob': rnd(cp, 1),
         'horizon': 'estar ya en recesión', 'as_of': str(cp_d.date()) if cp_d is not None else None,
         'note': 'Modelo de cambio de régimen con 4 indicadores coincidentes; llega con 2 meses de retraso.'},
    ]

    # ══════════════════════════════════════════════════════════════════════════
    # BACKTEST MENSUAL DE EE. UU. DESDE 1988
    # ══════════════════════════════════════════════════════════════════════════
    print('Backtest…')
    idx = pd.date_range(BT_START, TODAY.replace(day=1), freq='MS')
    frame = pd.DataFrame({k: (M[k].reindex(idx) if M.get(k) is not None else np.nan) for k in US_WEIGHTS}, index=idx)
    # Un dato mensual vale hasta que llega el siguiente (como en tiempo real)
    frame = frame.ffill(limit=4)
    bt_scores, bt_cov, bt_pill = [], [], {p: [] for p in US_PILLARS}
    for _, row in frame.iterrows():
        s_ = {k: SCORERS[k](row[k]) for k in US_WEIGHTS}
        c, cov = composite(s_, US_WEIGHTS)
        bt_scores.append(c); bt_cov.append(cov)
        for p, x in pillar_scores(s_, US_WEIGHTS, US_PILLARS).items():
            bt_pill[p].append(x)
    usrec = F['USREC'].reindex(idx).ffill(limit=2).fillna(0) if F['USREC'] is not None else pd.Series(0, index=idx)
    bt = pd.Series(bt_scores, index=idx, dtype=float)

    # Recesiones: inicio = primer mes con USREC = 1
    rec_starts = [idx[i] for i in range(1, len(idx)) if usrec.iloc[i] == 1 and usrec.iloc[i - 1] == 0]
    rec_spans, start = [], None
    for i, d in enumerate(idx):
        if usrec.iloc[i] == 1 and start is None:
            start = d
        if usrec.iloc[i] == 0 and start is not None:
            rec_spans.append([str(start.date()), str(idx[i - 1].date())]); start = None
    if start is not None:
        rec_spans.append([str(start.date()), str(idx[-1].date())])

    def at(d, k):
        t = d - pd.DateOffset(months=k)
        return rnd(bt.get(t), 1)
    rec_stats = [{'start': str(d.date()), 'm12': at(d, 12), 'm6': at(d, 6), 'm3': at(d, 3), 'm0': at(d, 0),
                  'max12': rnd(bt[(bt.index >= d - pd.DateOffset(months=12)) & (bt.index < d)].max(), 1)}
                 for d in rec_starts]
    # Falsas alarmas: tramos de ≥3 meses con score ≥ 35 sin recesión en los 18 meses siguientes
    warn = (bt >= 35)
    false_eps, run = [], []
    for d, flag in warn.items():
        if flag:
            run.append(d)
        if (not flag or d == bt.index[-1]) and run:
            if len(run) >= 3:
                end = run[-1]
                near = any(r >= run[0] - pd.DateOffset(months=1) and r <= end + pd.DateOffset(months=18) for r in rec_starts) \
                    or usrec.loc[run[0]:end].max() == 1
                open_ = end >= bt.index[-1] - pd.DateOffset(months=18)
                if not near:
                    false_eps.append({'from': str(run[0].date()), 'to': str(end.date()),
                                      'max': rnd(bt.loc[run[0]:end].max(), 1), 'open': bool(open_)})
            run = []
    backtest = {
        'dates': [str(d.date()) for d in idx], 'score': [rnd(x, 1) for x in bt_scores],
        'coverage': bt_cov, 'pillars': {p: [rnd(x, 1) for x in v] for p, v in bt_pill.items()},
        'recessions': rec_spans, 'rec_stats': rec_stats, 'false_alarms': false_eps,
        'threshold': 35, 'generated': TS,
        'note': 'Datos revisados (no los que se conocían en cada momento): el backtest es optimista.',
    }

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

    vix_now = last_valid(P['vix'])[0]
    dd_now = num(((P['spx'] / P['spx'].rolling(252, min_periods=20).max() - 1) * 100).iloc[-1]) if P['spx'] is not None else None
    hy_now = last_valid(F['BAMLH0A0HYM2'])[0]
    pains = [
        pain('vix', 'VIX en pánico', vix_now, 35, '>'),
        pain('spx_dd', 'S&P cae más de un 15%', dd_now, -15, '<', '%'),
        pain('curve_10y3m', 'Curva 10A−3M invertida', info['curve_10y3m'][0], 0, '<', ' pp'),
        pain('bond_30y', 'Bono 30A por encima del 5%', info['bond_30y'][0], 5.0, '>=', '%'),
        pain('sahm', 'Regla de Sahm activada', val.get('sahm'), 0.5, '>='),
        pain('hy', 'Crédito HY en estrés', hy_now, 5.0, '>', '%'),
        pain('oil_price', 'Petróleo por encima de 100 $', info['oil_price'][0], 100, '>=', ' $'),
        pain('oil_rise', 'Petróleo +90% desde mínimos', info['oil_rise'][0], 90, '>=', '%'),
    ]
    n_active = sum(1 for p in pains if p['active'])
    n_known = sum(1 for p in pains if p['active'] is not None)
    action = ('DESPLEGAR capital — varias señales de suelo' if n_active >= 5 else
              'ACUMULAR por tramos — las condiciones mejoran' if n_active >= 3 else
              'MANTENER liquidez — sin capitulación')

    # ══════════════════════════════════════════════════════════════════════════
    # SERIES PARA GRÁFICOS: 15 años mensuales, con las recesiones sombreadas
    # ══════════════════════════════════════════════════════════════════════════
    charts = {k: pts(M.get(k), 180) for k in META}
    charts['sox_bubble'] = [[str(i.date()), rnd(x, 1)] for i, x in sox_series.tail(260).items()] if sox_series is not None else []

    sector_rows = []
    def pct(s, n):
        return None if s is None or len(s) <= n else num((s.iloc[-1] / s.iloc[-1 - n] - 1) * 100)
    for n, s in sector_px.items():
        sector_rows.append({'name': n, 'ticker': sectors[n], '1M': rnd(pct(s, 21), 1),
                            '3M': rnd(pct(s, 63), 1),
                            '12M': rnd(pct(s, len(s) - 1) if s is not None else None, 1)})

    # ══════════════════════════════════════════════════════════════════════════
    # PULSO DEL MERCADO (Hartnett / Cárpatos): los niveles que vigilan, en vivo.
    # No entra en ningún score: mide el humor del mercado, no la economía.
    # Los umbrales se pueden cambiar en manual.json → "levels".
    # ══════════════════════════════════════════════════════════════════════════
    lv = {'mags': 70, 'smh': 520, 'gold': 4000, 'brent': 80, 'bb_sell': 8, 'bb_buy': 2,
          'tsm': 400, 'nvda': 200, 'ixg': 125, 'mdy': 666, 'jnk': 92, 'hyper_spread': 200,
          'es_cta_mid': 7606, 'es_cta_long': 7450, 'nq_res': 28580, 'spx_breadth': 7323,
          'h_10y': 4.5, 'h_oil': 80, 'h_breadth': 88, 'h_semis': -20, 'h_semis_strong': -25,
          **{k: v for k, v in manual.get('levels', {}).items() if not k.startswith('_')}}
    hk = manual.get('hartnett', {})

    def lastd(s):
        return (num(s.iloc[-1]), str(s.index[-1].date())) if s is not None and len(s) else (None, None)

    def rel(a, b, n=21):
        pa, pb = pct(a, n), pct(b, n)
        return None if pa is None or pb is None else pa - pb

    levels = []

    def ma_gap(s, n=200):
        """(% sobre (+) o bajo (−) la media de n sesiones, la media)."""
        if s is None or len(s) < n:
            return None, None
        m = s.rolling(n).mean().iloc[-1]
        return num((s.iloc[-1] / m - 1) * 100), num(m)

    def level(key, label, value, date, unit, dec, threshold, good, note, source, group='rot', s=None):
        """good = 'above' si estar por encima del umbral es lo bueno para la bolsa,
        'below' si lo es estar por debajo, None si el nivel es solo una referencia.
        group = bloque de la pestaña; s = serie de precios para enseñar su MM200."""
        value = num(value)
        above = None if value is None or threshold is None else value >= threshold
        ok = None if above is None or good is None else (above if good == 'above' else not above)
        g, m = ma_gap(s)
        levels.append({'key': key, 'label': label, 'value': rnd(value, 3), 'as_of': date,
                       'unit': unit, 'decimals': dec, 'threshold': threshold, 'above': above,
                       'ok': ok, 'note': note, 'source': source, 'group': group,
                       'ma200': rnd(m, 2), 'ma_gap': rnd(g, 1)})

    qma = None
    if Q['qqq'] is not None and len(Q['qqq']) >= 200:
        qma = ((Q['qqq'] / Q['qqq'].rolling(200).mean() - 1) * 100).dropna()
    m7 = {'aapl': 'Apple', 'msft': 'Microsoft', 'googl': 'Alphabet', 'amzn': 'Amazon',
          'meta': 'Meta', 'nvda': 'Nvidia', 'tsla': 'Tesla'}
    mag7 = []
    for k, name in m7.items():
        s = Q.get(k)
        if s is None or len(s) < 200:
            continue
        mag7.append({'ticker': PULSE[k], 'name': name,
                     'dd52': rnd((s.iloc[-1] / s.tail(252).max() - 1) * 100, 1),
                     'ma200': rnd((s.iloc[-1] / s.rolling(200).mean().iloc[-1] - 1) * 100, 1),
                     'm1': rnd(pct(s, 21), 1)})
    n_bear = sum(1 for x in mag7 if x['dd52'] is not None and x['dd52'] <= -20) if mag7 else None
    corr = None
    if Q['aapl'] is not None and Q['qqq'] is not None:
        r_ = pd.concat([Q['aapl'], Q['qqq']], axis=1).dropna().pct_change().dropna()
        corr = r_.iloc[:, 0].rolling(21).corr(r_.iloc[:, 1]).dropna()
    sec1m = {r['ticker']: r['1M'] for r in sector_rows}
    defs = [sec1m.get(t) for t in ('XLE', 'XLP', 'XLU')]
    def_tech = (sum(defs) / 3 - sec1m['XLK']
                if all(x is not None for x in defs) and sec1m.get('XLK') is not None else None)
    bb = manual.get('bull_bear', {})
    cbh = manual.get('cb_hikes_6m', {})
    hs = hk.get('hyperscaler_spread_bp', {})
    # ── Canarios: tecnología y semis
    level('mags', 'Siete Magníficas (ETF MAGS)', *lastd(Q['mags']), ' $', 2, lv['mags'], 'above',
          'Línea roja de Hartnett: pesan cerca del 40% del índice', 'Yahoo MAGS', 'tech', Q['mags'])
    level('mags_ma200', 'MAGS sobre su media de 200 sesiones', ma_gap(Q['mags'])[0], lastd(Q['mags'])[1],
          '%', 1, 0, 'above', 'Cárpatos: si la pierde, «se le echa la cruz al mercado» y arrastra al S&P 500',
          'Yahoo MAGS', 'tech')
    level('smh', 'Semiconductores (ETF SMH)', *lastd(Q['smh']), ' $', 2, lv['smh'], 'above',
          'Nivel crítico de Hartnett', 'Yahoo SMH', 'tech', Q['smh'])
    level('tsm', 'Taiwan Semiconductor (TSM)', *lastd(Q['tsm']), ' $', 2, lv['tsm'], 'above',
          'Línea roja de Hartnett; perderla con la MM200 confirma el doble techo', 'Yahoo TSM', 'tech', Q['tsm'])
    level('nvda', 'Nvidia', *lastd(Q['nvda']), ' $', 2, lv['nvda'], 'above',
          'Alarma de Hartnett por debajo de 200 $', 'Yahoo NVDA', 'tech', Q['nvda'])
    level('qqq_ma200', 'Nasdaq 100 (QQQ) sobre su media de 200 sesiones',
          num(qma.iloc[-1]) if qma is not None else None, lastd(Q['qqq'])[1], '%', 1, 0, 'above',
          'Cárpatos: si la pierde, ventas forzadas de los fondos CTA', 'Yahoo QQQ', 'tech')
    level('soxx_ma200', 'Semis (SOXX) sobre su media de 200 sesiones', *info['sox_bubble'], '%', 1, 0, 'above',
          'El otro soporte que vigila Cárpatos', 'Yahoo SOXX', 'tech')
    level('mag7_bear', 'Siete Magníficas en mercado bajista', n_bear, lastd(Q['aapl'])[1], ' de 7', 0, 4, 'below',
          'Más de un 20% por debajo de su máximo de 52 semanas, con cierres', 'Yahoo', 'tech')
    # ── Termómetros de la economía y del crédito
    level('ixg', 'Bancos del mundo (ETF IXG)', *lastd(Q['ixg']), ' $', 2, lv['ixg'], 'above',
          'Nivel crítico de Hartnett para el sector bancario', 'Yahoo IXG', 'credit', Q['ixg'])
    level('mdy', 'Medianas de EE. UU. (ETF MDY)', *lastd(Q['mdy']), ' $', 2, lv['mdy'], 'above',
          'Termómetro de la economía doméstica de Hartnett', 'Yahoo MDY', 'credit', Q['mdy'])
    level('jnk', 'Bonos basura (ETF JNK)', *lastd(Q['jnk']), ' $', 2, lv['jnk'], 'above',
          'Si pierde los 92 $, aviso de contagio de crédito', 'Yahoo JNK', 'credit', Q['jnk'])
    level('hyper_spread', 'Diferencial de los bonos de los hiperescaladores', hs.get('value'), hs.get('date'),
          ' pb', 0, lv['hyper_spread'], 'below',
          'Pánico por encima de 200 pb: miedo a un evento de crédito corporativo (dato manual de BofA)',
          'manual.json', 'credit')
    # ── Fontanería: niveles de los CTA según Cárpatos (futuros)
    spx_s = P['spx'].tail(520) if P['spx'] is not None else None
    level('spx_ma200', 'S&P 500 sobre su media de 200 sesiones', ma_gap(spx_s)[0], lastd(spx_s)[1], '%', 1, 0,
          'above', 'Coincide con la prueba de la MM200 del MAGS', 'Yahoo ^GSPC', 'flows')
    level('spx_breadth', 'S&P 500 frente a 7.323', *lastd(spx_s), '', 0, lv['spx_breadth'], 'above',
          'Hartnett: perderlo (con Europa y Reino Unido un 3-4% más abajo) llevaría la amplitud al 88%',
          'Yahoo ^GSPC', 'flows', spx_s)
    level('es_cta_mid', 'Futuro del S&P 500 frente a 7.606', *lastd(Q['es']), '', 0, lv['es_cta_mid'], 'above',
          'Cárpatos: por debajo entran ventas medias de los CTA', 'Yahoo ES=F', 'flows', Q['es'])
    level('es_cta_long', 'Futuro del S&P 500 frente a 7.450', *lastd(Q['es']), '', 0, lv['es_cta_long'], 'above',
          'Cárpatos: por debajo se activan las ventas de largo plazo de los CTA', 'Yahoo ES=F', 'flows')
    level('nq_res', 'Futuro del Nasdaq 100 frente a 28.580', *lastd(Q['nq']), '', 0, lv['nq_res'], 'above',
          'Cárpatos: resistencia crítica tras perder la MM200', 'Yahoo NQ=F', 'flows', Q['nq'])
    # ── Rotación y refugios
    level('rsp_spy_1m', 'Equiponderado (RSP) frente al S&P 500, 1 mes', rel(Q['rsp'], Q['spy']),
          lastd(Q['rsp'])[1], ' pp', 1, 0, None,
          'Positivo = rotación fuera de los gigantes. Cárpatos: el equiponderado solo ganó 10 de los últimos 22 años',
          'Yahoo RSP y SPY')
    level('vflo_spy_1m', 'Empresas con caja libre (VFLO) frente al S&P 500, 1 mes', rel(Q['vflo'], Q['spy']),
          lastd(Q['vflo'])[1], ' pp', 1, 0, None, 'Hartnett: el refugio frente a la IA que quema caja',
          'Yahoo VFLO y SPY')
    level('def_tech_1m', 'Defensivos frente a tecnología, 1 mes', def_tech, lastd(sector_px.get('Tecnología'))[1],
          ' pp', 1, 0, None, 'Media de energía, consumo básico y utilities menos tecnología',
          'Yahoo XLE, XLP, XLU y XLK')
    level('aapl_corr', 'Correlación Apple–Nasdaq 100 (21 sesiones)',
          num(corr.iloc[-1]) if corr is not None and len(corr) else None, lastd(Q['aapl'])[1], '', 2, 0, None,
          'Negativa = Apple funciona como cobertura «anti-IA»', 'Yahoo AAPL y QQQ')
    # ── Macro, oro y sentimiento
    level('gold', 'Oro', *lastd(P['gold']), ' $', 0, lv['gold'], None,
          'Hartnett: acumular oro antes del pulso por el techo de deuda (ABD, «todo menos el dólar»)',
          'Yahoo GC=F', 'macro', P['gold'])
    level('brent', 'Petróleo Brent', *info['brent'], ' $', 1, lv['brent'], 'below',
          'Hartnett: por debajo de 80 $ es su señal de calma; corto en petróleo para el 1S 2027',
          'FRED + Yahoo BZ=F', 'macro')
    level('bond_30y', 'Bono del Tesoro a 30 años', *info['bond_30y'], '%', 2, 5.0, 'below',
          '«Bubble killer» por encima del 5%', 'Treasury.gov', 'macro')
    level('bull_bear', 'Bull & Bear de BofA', bb.get('value'), bb.get('date'), '', 1, lv['bb_sell'], 'below',
          f"Venta en {lv['bb_sell']} o más, compra en {lv['bb_buy']} o menos (dato manual)", 'BofA, manual.json',
          'macro')
    level('cb_hikes', 'Subidas de tipos de bancos centrales (6 meses)', cbh.get('value'), cbh.get('date'), '', 0,
          None, None, 'Hartnett: «doloroso pero no mortal» mientras no se destruya empleo ni haya un evento de '
          'crédito (dato manual)', 'BofA, manual.json', 'macro')

    # ══════════════════════════════════════════════════════════════════════════
    # SEÑALES DE COMPRA DE HARTNETT: tres condiciones financieras que tienen que
    # destensarse y tres señales de capitulación (compra contraria).
    # ══════════════════════════════════════════════════════════════════════════
    nber = rec_spans or NBER_SPANS

    def in_recession(d, before=6, after=12):
        """¿Hubo recesión del NBER entre `before` meses antes y `after` después?"""
        a, b = d - pd.DateOffset(months=before), d + pd.DateOffset(months=after)
        return any(pd.Timestamp(s) <= b and pd.Timestamp(e) >= a for s, e in nber)

    def fwd(s, d, months):
        later = s[s.index >= d + pd.DateOffset(months=months)]
        return rnd((later.iloc[0] / s[s.index >= d].iloc[0] - 1) * 100, 1) if s is not None and len(later) else None

    # 1) Regla de amplitud global (aproximada con los ETF de países)
    breadth = None
    cols = {t: s for t, s in BR.items() if s is not None and len(s) > 260}
    if len(cols) >= 30:
        px = pd.concat(cols, axis=1).sort_index().ffill(limit=3)
        m50, m200 = px.rolling(50).mean(), px.rolling(200).mean()
        valid = px.notna() & m200.notna()
        below = (px < m50) & (px < m200) & valid
        nval = valid.sum(axis=1)
        bpct = (below.sum(axis=1) / nval * 100)[nval >= 30]
        last_i = bpct.index[-1]
        n_now, b_now = int(nval[last_i]), int(below.loc[last_i].sum())
        need_n = max(0, math.ceil(lv['h_breadth'] / 100 * n_now) - b_now)
        closest = sorted(
            [[BREADTH[t], rnd((min(m50[t].loc[last_i], m200[t].loc[last_i]) / px[t].loc[last_i] - 1) * 100, 1)]
             for t in px.columns if valid[t].loc[last_i] and not below[t].loc[last_i]],
            key=lambda x: -x[1] if x[1] is not None else 99)
        sigs, last_hit = [], None
        for i, (d, v) in enumerate(bpct.items()):
            if v >= lv['h_breadth']:
                if last_hit is None or i - last_hit > 60:
                    sigs.append({'date': str(d.date()), 'acwi_3m': fwd(acwi, d, 3), 'acwi_12m': fwd(acwi, d, 12),
                                 'peak': rnd(bpct.iloc[i:i + 60].max(), 0), 'recession': in_recession(d)})
                last_hit = i
        breadth = {'value': rnd(bpct.iloc[-1], 1), 'date': str(last_i.date()), 'n': n_now, 'below': b_now,
                   'need': need_n, 'closest': closest,
                   'below_names': sorted(BREADTH[t] for t in px.columns if below[t].loc[last_i]),
                   'since': str(bpct.index[0].date()),
                   'signals': sigs, 'chart': pts(bpct.resample('W-FRI').last(), 900, 0)}
        print(f"  Amplitud global: {breadth['value']}% ({b_now} de {n_now}); faltan {need_n} países para el 88%")

    # 2) Semiconductores: distancia a su media de 200 sesiones y señales desde 1996
    semis = None
    if sox_long is not None and len(sox_long) > 400:
        sg = ((sox_long / sox_long.rolling(200).mean() - 1) * 100).dropna()
        ma_now = num(sox_long.rolling(200).mean().iloc[-1])
        sigs, last_hit = [], None
        for i, (d, v) in enumerate(sg.items()):
            if v <= lv['h_semis']:
                if last_hit is None or i - last_hit > 126:
                    sigs.append({'date': str(d.date()), 'gap': rnd(v, 1), 'sox_12m': fwd(sox_long, d, 12),
                                 'recession': in_recession(d)})
                last_hit = i
        done = [s_ for s_ in sigs if s_['sox_12m'] is not None]
        semis = {'value': rnd(sg.iloc[-1], 1), 'date': str(sg.index[-1].date()), 'price': rnd(sox_long.iloc[-1], 0),
                 'ma200': rnd(ma_now, 0), 'min_1y': rnd(sg.tail(252).min(), 1),
                 'min_1y_date': str(sg.tail(252).idxmin().date()),
                 'need_pct': rnd(((1 + lv['h_semis'] / 100) * ma_now / sox_long.iloc[-1] - 1) * 100, 1),
                 'signals': sigs,
                 'stats': {'n': len(done), 'up': sum(1 for s_ in done if s_['sox_12m'] > 0),
                           'n_rec': sum(1 for s_ in done if s_['recession']),
                           'up_rec': sum(1 for s_ in done if s_['recession'] and s_['sox_12m'] > 0),
                           'med_norec': rnd(np.median([s_['sox_12m'] for s_ in done if not s_['recession']]), 0)
                           if any(not s_['recession'] for s_ in done) else None,
                           'med_rec': rnd(np.median([s_['sox_12m'] for s_ in done if s_['recession']]), 0)
                           if any(s_['recession'] for s_ in done) else None},
                 'chart': pts(sg, 520, 1)}

    # 3) Fed: último movimiento del tipo oficial (para la condición «fin de las subidas»)
    fed = F.get('DFEDTARU')
    fed_last = None
    if fed is not None and len(fed) > 1:
        chg = fed[fed.diff().fillna(0) != 0]
        if len(chg):
            d_ = chg.index[-1]
            fed_last = {'date': str(d_.date()), 'from': num(fed[fed.index < d_].iloc[-1]), 'to': num(chg.iloc[-1])}
        fed_last = {**(fed_last or {}), 'level': num(fed.iloc[-1]), 'as_of': str(fed.index[-1].date())}

    def trig(key, label, value, date, unit, dec, threshold, op, note, source, met=None):
        value = num(value)
        if met is None and value is not None and threshold is not None:
            met = value < threshold if op == '<' else value <= threshold if op == '<=' else value >= threshold
        return {'key': key, 'label': label, 'value': rnd(value, 3), 'as_of': date, 'unit': unit,
                'decimals': dec, 'threshold': threshold, 'op': op, 'met': met, 'note': note, 'source': source}

    t10 = (curve.get('10Y'), tsy['date']) if curve.get('10Y') is not None else lastd(F.get('DGS10'))
    wti_now = info['oil_price'][0]
    macro_t = [
        trig('t10y', 'Bono del Tesoro a 10 años por debajo del 4,5%', *t10, '%', 2, lv['h_10y'], '<',
             'Que se destensen las condiciones financieras', 'Treasury.gov'),
        trig('oil80', 'Petróleo por debajo de 80 $', *info['brent'], ' $', 1, lv['h_oil'], '<',
             f"Brent; el WTI está en {wti_now:.1f} $. La pieza clave: alivia la inflación y frena las subidas"
             .replace('.', ',', 1)
             if wti_now is not None else 'Brent. La pieza clave: alivia la inflación y frena las subidas',
             'FRED + Yahoo BZ=F'),
        trig('rate_cycle', 'Fin del ciclo de subidas de tipos', cbh.get('value'), hk.get('rate_cycle_date') or cbh.get('date'),
             ' subidas', 0, None, None,
             hk.get('rate_cycle_note') or 'Lo decide Roger con el Flow Show (manual.json → hartnett.rate_cycle_over)',
             'BofA (manual) + FRED DFEDTARU', met=hk.get('rate_cycle_over')),
    ]
    cap_t = [
        trig('breadth', 'Regla de amplitud global en el 88%', breadth['value'] if breadth else None,
             breadth['date'] if breadth else None, '%', 0, lv['h_breadth'], '>=',
             (f"{breadth['below']} de {breadth['n']} bolsas por debajo de su MM50 y su MM200" if breadth else
              'Sin datos de los ETF de países'), 'Yahoo, ETF de países MSCI'),
        trig('semis', 'Semiconductores un 20% o más bajo su MM200', semis['value'] if semis else None,
             semis['date'] if semis else None, '%', 1, lv['h_semis'], '<=',
             'Desde 1996 las grandes compras en chips llegaron entre −20% y −25%', 'Yahoo ^SOX'),
        trig('bb_buy', 'Bull & Bear de BofA en 2 o menos', bb.get('value'), bb.get('date'), '', 1, lv['bb_buy'],
             '<=', 'Sentimiento extremo: compra contraria (dato manual)', 'BofA, manual.json'),
    ]
    n_macro = sum(1 for t in macro_t if t['met'])
    cap_on = [t['key'] for t in cap_t if t['met']]
    if 'breadth' in cap_on or 'bb_buy' in cap_on:
        verdict = ['buy', 'COMPRA CONTRARIA ACTIVA', 'La regla de amplitud o el Bull & Bear marcan capitulación.']
    elif n_macro == 3:
        verdict = ['buy', 'CONDICIONES DESTENSADAS', 'Se cumplen las tres: Hartnett dejaría la postura defensiva.']
    elif n_macro == 2:
        verdict = ['watch', 'CERCA', 'Se cumplen dos de las tres condiciones financieras.']
    elif 'semis' in cap_on:
        verdict = ['watch', 'SOLO COMPRA TÁCTICA EN SEMIS', 'Los chips están un 20% o más bajo su MM200; '
                   'las condiciones financieras siguen sin destensarse.']
    else:
        verdict = ['wait', 'SIN SEÑAL DE COMPRA', 'Hartnett sigue defensivo: no se cumple ninguna de las seis señales.'
                   if n_macro == 0 else f'Hartnett sigue defensivo: {n_macro} de 3 condiciones y sin capitulación.']
    gold_s = P['gold']
    hartnett = {
        'macro': macro_t, 'capitulation': cap_t, 'n_macro': n_macro, 'cap_on': cap_on, 'verdict': verdict,
        'breadth': breadth, 'semis': semis, 'fed': fed_last,
        'debt': {**(debt or {}), 'deadline': hk.get('debt_limit_date'), 'congress': hk.get('congress_date'),
                 'gold': rnd(lastd(gold_s)[0], 0), 'gold_3m': rnd(pct(gold_s, 63), 1)},
        'spx_to_7323': rnd((lv['spx_breadth'] / lastd(spx_s)[0] - 1) * 100, 1) if lastd(spx_s)[0] else None,
    }

    rsp_spy = (Q['rsp'] / Q['spy']).dropna().tail(260) if Q['rsp'] is not None and Q['spy'] is not None else None
    pulse = {
        'levels': levels, 'mag7': mag7, 'mag7_bear': n_bear, 'hartnett': hartnett,
        'charts': {'mags': pts(Q['mags'], 260), 'smh': pts(Q['smh'], 260), 'qqq_ma200': pts(qma, 260, 1),
                   'gold': pts(P['gold'], 260, 0),
                   'rsp_spy': pts(rsp_spy / rsp_spy.iloc[0] * 100, 260, 1)
                              if rsp_spy is not None and len(rsp_spy) else [],
                   'aapl_corr': pts(corr, 260, 2),
                   'es': pts(Q['es'], 260, 0),
                   'breadth': breadth['chart'] if breadth else [],
                   'sox_ma200': semis['chart'] if semis else []},
    }
    if breadth:
        del breadth['chart']
    if semis:
        del semis['chart']
    print(f"  Pulso: MAGS {lastd(Q['mags'])[0]} · SMH {lastd(Q['smh'])[0]} · Mag7 en bajista {n_bear}")
    print(f"  Hartnett: {verdict[1]} · condiciones {n_macro}/3 · capitulación {cap_on or 'ninguna'}")

    gas = manual.get('eu_gas_storage', {})
    dashboard = {
        'timestamp': TS,
        'version': 5,
        'fred_enabled': bool(FRED_API_KEY),
        'composite': {'score': us, 'label': label_of(us), 'coverage': us_cov,
                      'missing': [k for k in US_WEIGHTS if sc.get(k) is None],
                      'pillars': us_p, 'weights': US_WEIGHTS, 'pillar_keys': US_PILLARS},
        'euro': {'score': ea, 'label': label_of(ea), 'coverage': ea_cov, 'weights': EA_WEIGHTS,
                 'missing': [k for k in EA_WEIGHTS if sc.get(k) is None]},
        'global': {'score': gl, 'label': label_of(gl), 'coverage': gl_cov, 'weights': GL_WEIGHTS,
                   'missing': [k for k in GL_WEIGHTS if sc.get(k) is None]},
        'models': models,
        'indicators': indicators,
        'pain': {'levels': pains, 'active': n_active, 'known': n_known, 'action': action},
        'curve': {'date': tsy['date'], 'points': curve},
        'sectors': sector_rows,
        'charts': charts,
        'recessions': [r for r in rec_spans if r[1] >= str((TODAY - dt.timedelta(days=16 * 365)))],
        'extra': {'spy': rnd(last_valid(P['spx'])[0], 2), 'oil_low52': oil_low52, 'hy_kind': 'oas',
                  'gas_storage': gas},
        'pulse': pulse,
    }

    # ── Histórico: una fila por día (la última ejecución del día manda) ─────
    hist_path = os.path.join(OUT, 'history.json')
    try:
        with open(hist_path, encoding='utf-8') as fh:
            history = json.load(fh)
        history = [h for h in history if h.get('v') == 5]
    except Exception:
        history = []
    mkt_date = str(P['spx'].index[-1].date()) if P['spx'] is not None else str(TODAY)
    row = {'date': mkt_date, 'v': 5, 'score': us, 'coverage': us_cov, 'euro': ea, 'global': gl,
           **{f'p_{k}': x for k, x in us_p.items()},
           **{k: rnd(val.get(k), 3) for k in ['curve_prob', 'ebp_prob', 'claims_rise', 'sahm', 'ms', 'oil_yoy', 'esi']}}
    if us is not None and us_cov >= 0.6:
        history = [h for h in history if h.get('date') != row['date']] + [row]
        history.sort(key=lambda h: h['date'])

    os.makedirs(OUT, exist_ok=True)
    for name, payload in [('dashboard.json', dashboard), ('history.json', history), ('backtest.json', backtest)]:
        path = os.path.join(OUT, name)
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(clean(payload), fh, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':') if name != 'dashboard.json' else None,
                      indent=1 if name == 'dashboard.json' else None)
        print(f'  ✓ {os.path.normpath(path)}')

    if us_cov < 0.6:
        raise SystemExit(f'Cobertura {us_cov:.0%} < 60%: no se publica.')


if __name__ == '__main__':
    main()
