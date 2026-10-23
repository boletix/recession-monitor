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
    def get():
        import yfinance as yf
        d = yf.download(ticker, period=period, progress=False, auto_adjust=True, threads=False)
        if d is None or d.empty:
            raise ValueError('vacío')
        s = d['Close']
        if hasattr(s, 'columns'):          # MultiIndex de yfinance >= 0.2.48
            s = s.iloc[:, 0]
        s = s.astype(float).dropna()
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        if len(s) < 5:
            raise ValueError(f'solo {len(s)} filas válidas')
        return s
    return retry(get, f'yfinance {ticker}')


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
           'FRED UNRATE y JTSJOR', 90),
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
        'ITALOLITOAASTSAM', 'ESPLOLITOAASTSAM']}
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
