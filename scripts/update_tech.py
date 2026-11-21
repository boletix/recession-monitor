#!/usr/bin/env python3
"""
Recession Monitor — Big Tech, semis y caja (pestaña «Big Tech y semis»)

Sale del notebook «Check Market» de Roger (21-nov-2026), corregido y llevado al
monitor. No entra en ningún score: mide la concentración del mercado y la caja
de las empresas que tiran de él, y cómo eso puede llegar a la economía.

Qué calcula
  1. FAAMG (Meta, Apple, Amazon, Microsoft, Alphabet) y semiconductores frente
     al S&P 500: rentabilidad, fuerza relativa, peso en el índice y el «S&P sin
     ellos». Pesos del día: cartera del SPY (State Street); hacia atrás se
     aproximan con precios y acciones diluidas de EDGAR.
  2. Amplitud de los ~500 valores del S&P 500 (lo que hacía el notebook, con
     los errores arreglados): % sobre la MM50 y la MM200, máximos y mínimos de
     52 semanas, oscilador McClellan ajustado y % bajo las dos medias.
  3. Estado de flujos de caja de FAAMG y de 11 semis de EE. UU. (SEC EDGAR,
     10-K y 10-Q, nunca yfinance): caja operativa, capex, arrendamientos
     financieros, FCF, SBC, recompras, dividendos y el resto de la financiación.
     Los 10-Q dan acumulados del año: los trimestres se sacan por diferencia.
  4. Enlace con la economía (FRED): inversión en equipos informáticos y
     software (% del PIB y aportación al crecimiento) y vacantes por parado.

Reglas
  * FCF = caja operativa − capex − amortización de arrendamientos financieros
    (la regla de los hiperescaladores del sistema); «FCF tras SBC» resta además
    la remuneración en acciones.
  * Si una fuente falla, se conserva el bloque de la ejecución anterior con su
    fecha y la marca «stale». El JSON se escribe con allow_nan=False.
  * EDGAR pide un contacto en el User-Agent: variable SEC_CONTACT_EMAIL (en
    GitHub, un secret). Sin ella, el bloque de caja no se refresca.

Salida: docs/tech.json
"""

import os, io, csv, json, math, gzip, time, shutil, subprocess, warnings
import datetime as dt
import urllib.request
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get('RM_OUT') or os.path.join(HERE, '..', 'docs')
PREV = os.path.join(HERE, '..', 'docs', 'tech.json')
FRED_API_KEY = os.environ.get('FRED_API_KEY', '')
SEC_EMAIL = os.environ.get('SEC_CONTACT_EMAIL', '').strip()
NOW = dt.datetime.now(dt.timezone.utc)
TS = NOW.strftime('%Y-%m-%dT%H:%M:%SZ')

# FAAMG y los semis de EE. UU. que presentan 10-Q en EDGAR (TSMC y ASML presentan
# 20-F/6-K sin trimestres en XBRL: entran en precios, no en caja).
COS = {
    'AAPL': (320193, 'Apple', 'faamg'), 'MSFT': (789019, 'Microsoft', 'faamg'),
    'GOOGL': (1652044, 'Alphabet', 'faamg'), 'AMZN': (1018724, 'Amazon', 'faamg'),
    'META': (1326801, 'Meta', 'faamg'),
    'NVDA': (1045810, 'Nvidia', 'semis'), 'AVGO': (1730168, 'Broadcom', 'semis'),
    'AMD': (2488, 'AMD', 'semis'), 'QCOM': (804328, 'Qualcomm', 'semis'),
    'TXN': (97476, 'Texas Instruments', 'semis'), 'MU': (723125, 'Micron', 'semis'),
    'INTC': (50863, 'Intel', 'semis'), 'ADI': (6281, 'Analog Devices', 'semis'),
    'AMAT': (6951, 'Applied Materials', 'semis'), 'LRCX': (707549, 'Lam Research', 'semis'),
    'KLAC': (319201, 'KLA', 'semis'),
}
FAAMG = [t for t, v in COS.items() if v[2] == 'faamg']
SEMIS = [t for t, v in COS.items() if v[2] == 'semis']
HYPER = ['MSFT', 'GOOGL', 'AMZN', 'META']
EXTRA_PX = {'TSM': 'TSMC', 'ASML': 'ASML'}
# Semis dentro del S&P 500 (para su peso en el índice); solo cuentan los que estén en la cartera del SPY
SP_SEMIS = ['NVDA', 'AVGO', 'AMD', 'QCOM', 'TXN', 'MU', 'INTC', 'ADI', 'AMAT', 'LRCX', 'KLAC',
            'MCHP', 'NXPI', 'ON', 'MPWR', 'TER', 'SWKS', 'QRVO', 'MRVL']

# Etiquetas XBRL por orden de preferencia (comprobadas el 21-nov-2026 en las 16 empresas)
TAGS = {
    'rev': ['RevenueFromContractWithCustomerExcludingAssessedTax', 'Revenues', 'SalesRevenueNet',
            'RevenueFromContractWithCustomerIncludingAssessedTax'],
    'cogs': ['CostOfRevenue', 'CostOfGoodsAndServicesSold', 'CostOfGoodsSold'],
    'ni': ['NetIncomeLoss', 'ProfitLoss'],
    'ocf': ['NetCashProvidedByUsedInOperatingActivities',
            'NetCashProvidedByUsedInOperatingActivitiesContinuingOperations'],
    'capex': ['PaymentsToAcquirePropertyPlantAndEquipment', 'PaymentsToAcquireProductiveAssets',
              'PaymentsToAcquirePropertyPlantAndEquipmentAndIntangibleAssets'],
    'fl': ['FinanceLeasePrincipalPayments', 'FinanceLeaseAndFinancingObligationPrincipalPayments'],
    'sbc': ['ShareBasedCompensation', 'AllocatedShareBasedCompensationExpense'],
    'bb': ['PaymentsForRepurchaseOfCommonStock'],
    'div': ['PaymentsOfDividendsCommonStock', 'PaymentsOfDividends', 'PaymentsOfOrdinaryDividends'],
    'fin': ['NetCashProvidedByUsedInFinancingActivities'],
    'inv_cf': ['NetCashProvidedByUsedInInvestingActivities'],
    'da': ['DepreciationDepletionAndAmortization', 'DepreciationAndAmortization',
           'DepreciationAmortizationAndAccretionNet', 'Depreciation'],
}
FLOWS = list(TAGS)
# Splits por si Yahoo no responde (fecha efectiva, factor)
SPLITS = {'AAPL': [('2014-06-09', 7), ('2020-08-31', 4)], 'GOOGL': [('2022-07-18', 20)],
          'AMZN': [('2022-06-06', 20)], 'NVDA': [('2021-07-20', 4), ('2024-06-10', 10)],
          'AVGO': [('2024-07-15', 10)], 'LRCX': [('2024-10-03', 10)]}


# ══════════════════════════════════════════════════════════════════════════════
# UTILIDADES
# ══════════════════════════════════════════════════════════════════════════════

def num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def rnd(x, n=2):
    x = num(x)
    return None if x is None else round(x, n)


def clean(obj):
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        obj = float(obj)
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def pts(s, n=None, digits=2):
    """Serie → [[fecha, valor], ...] (los últimos n puntos)."""
    if s is None:
        return []
    s = s.dropna()
    if n:
        s = s.tail(n)
    return [[str(pd.Timestamp(i).date()), rnd(v, digits)] for i, v in s.items() if num(v) is not None]


def http_get(url, headers=None, timeout=40):
    """urllib y, si falla, curl (en este PC el TLS de Python se cuelga con FRED)."""
    h = {'User-Agent': 'Mozilla/5.0 recession-monitor'}
    h.update(headers or {})
    err = None
    if not os.environ.get('RM_CURL'):
        try:
            req = urllib.request.Request(url, headers=h)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
                if r.headers.get('Content-Encoding') == 'gzip':
                    data = gzip.decompress(data)
                return data
        except Exception as e:
            err = e
    if not shutil.which('curl'):
        raise err or RuntimeError('sin curl')
    # --compressed solo si se pide: FRED corta la conexión con compresión + User-Agent «Mozilla»
    cmd = ['curl', '-sSLf', '-g', '--max-time', str(timeout + 30)]
    if any(k.lower() == 'accept-encoding' for k in h):
        cmd.append('--compressed')
    for k, v in h.items():
        if k.lower() != 'accept-encoding':
            cmd += ['-H', f'{k}: {v}']
    out = subprocess.run(cmd + [url], capture_output=True, timeout=timeout + 40)
    if out.returncode != 0:
        raise RuntimeError(f'{err}; curl {out.returncode}: {out.stderr.decode(errors="ignore")[:160]}')
    return out.stdout


def retry(fn, what, tries=3):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            if i == tries - 1:
                print(f'  ✗ {what}: {e}')
                return None
            time.sleep(2 * (i + 1))


def fred(sid, start='1985-01-01'):
    def get():
        if FRED_API_KEY:
            url = ('https://api.stlouisfed.org/fred/series/observations'
                   f'?series_id={sid}&api_key={FRED_API_KEY}&file_type=json&observation_start={start}')
            obs = json.loads(http_get(url)).get('observations', [])
            d, v = [o['date'] for o in obs], [o.get('value') for o in obs]
        else:
            rows = list(csv.reader(io.StringIO(http_get(
                f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd={start}').decode('utf-8'))))[1:]
            d, v = [r[0] for r in rows], [r[1] for r in rows]
        s = pd.Series([num(x) for x in v], index=pd.to_datetime(d), dtype=float).dropna().sort_index()
        if s.empty:
            raise ValueError('vacía')
        return s
    return retry(get, f'FRED {sid}')


def yf_batch(tickers, period, chunk=80):
    """Descarga por lotes. Devuelve (Close, Adj Close) como DataFrames fecha × ticker.
    Close de Yahoo ya viene ajustado por splits; Adj Close además por dividendos."""
    import yfinance as yf
    closes, adjs = [], []
    tickers = list(dict.fromkeys(tickers))
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i + chunk]

        def get():
            d = yf.download(part, period=period, auto_adjust=False, actions=False, threads=True,
                            progress=False, group_by='column')
            if d is None or d.empty:
                raise ValueError('vacío')
            return d
        d = retry(get, f'yfinance lote {i // chunk + 1} ({len(part)})')
        if d is None:
            continue
        c, a = d['Close'], d['Adj Close'] if 'Adj Close' in d.columns.get_level_values(0) else d['Close']
        if isinstance(c, pd.Series):
            c, a = c.to_frame(part[0]), a.to_frame(part[0])
        closes.append(c)
        adjs.append(a)
    if not closes:
        return None, None
    C, A = pd.concat(closes, axis=1), pd.concat(adjs, axis=1)
    for X in (C, A):
        if X.index.tz is not None:
            X.index = X.index.tz_localize(None)
    C, A = C.sort_index(), A.sort_index()
    C = C.loc[:, C.notna().sum() > 5]
    return C.astype(float), A.reindex(columns=C.columns).astype(float)


def load_prev():
    try:
        with open(PREV, encoding='utf-8') as fh:
            return json.load(fh)
    except Exception:
        return {}


def pct(s, n):
    s = s.dropna() if s is not None else None
    return None if s is None or len(s) <= n else num((s.iloc[-1] / s.iloc[-1 - n] - 1) * 100)


# ══════════════════════════════════════════════════════════════════════════════
# 1) CARTERA DEL SPY (pesos del S&P 500 hoy)
# ══════════════════════════════════════════════════════════════════════════════

def spy_holdings():
    url = 'https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx'

    def get():
        raw = pd.read_excel(io.BytesIO(http_get(url, timeout=60)), header=None)
        as_of = None
        for r in range(6):
            cell = str(raw.iloc[r, 1])
            if cell.startswith('As of'):
                as_of = str(pd.Timestamp(cell.replace('As of', '').strip()).date())
        hdr = raw.index[raw.iloc[:, 1].astype(str).str.strip() == 'Ticker'][0]
        df = raw.iloc[hdr + 1:, :].copy()
        df.columns = [str(c).strip() for c in raw.iloc[hdr]]
        df = df[pd.to_numeric(df['Weight'], errors='coerce').notna()]
        df['Weight'] = df['Weight'].astype(float)
        df = df[df['Ticker'].astype(str).str.fullmatch(r'[A-Z]{1,5}(\.[A-Z])?') & (df['Weight'] > 0)]
        if len(df) < 450:
            raise ValueError(f'solo {len(df)} valores')
        rows = [{'t': str(t).replace('.', '-'), 'name': str(n).title(), 'w': round(float(w), 4)}
                for t, n, w in zip(df['Ticker'], df['Name'], df['Weight'])]
        return {'as_of': as_of, 'rows': rows}
    return retry(get, 'cartera del SPY (State Street)')


# ══════════════════════════════════════════════════════════════════════════════
# 2) EDGAR: estado de flujos de caja trimestral y LTM
# ══════════════════════════════════════════════════════════════════════════════

def edgar_facts(cik):
    url = f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json'
    ua = f'recession-monitor/1.0 ({SEC_EMAIL})'
    return retry(lambda: json.loads(http_get(url, {'User-Agent': ua, 'Accept-Encoding': 'gzip'}, 90)),
                 f'EDGAR CIK {cik}')


def durations(g, tags, unit='USD'):
    """{(inicio, fin): (valor, presentado)} de hechos con duración; manda la etiqueta con
    datos más recientes (Amazon pasó en 2016 de PaymentsToAcquirePropertyPlantAndEquipment,
    neto, a PaymentsToAcquireProductiveAssets, bruto) y las demás solo rellenan huecos.
    Dentro de cada etiqueta, la presentación más reciente (reexpresiones)."""
    def last_end(tg):
        return max((v['end'] for v in g.get(tg, {}).get('units', {}).get(unit, []) if 'start' in v), default='')
    tags = sorted(tags, key=last_end, reverse=True)    # sorted es estable: a igualdad, el orden de la lista
    out = {}
    for tg in tags:
        best = {}
        for v in g.get(tg, {}).get('units', {}).get(unit, []):
            if 'start' not in v or not str(v.get('form', '')).startswith('10-'):
                continue
            k = (v['start'], v['end'])
            if k not in best or v['filed'] > best[k][1]:
                best[k] = (v['val'], v['filed'])
        for k, v in best.items():
            out.setdefault(k, v)
    return out


def instants(g, tags, unit='USD'):
    out = {}
    for tg in tags:
        best = {}
        for v in g.get(tg, {}).get('units', {}).get(unit, []):
            if 'start' in v or not str(v.get('form', '')).startswith('10-'):
                continue
            if v['end'] not in best or v['filed'] > best[v['end']][1]:
                best[v['end']] = (v['val'], v['filed'])
        for k, v in best.items():
            out.setdefault(k, v)
    return {pd.Timestamp(k): v[0] for k, v in out.items()}


def quarterly(dur):
    """Trimestres discretos {fin: valor}. Los 10-Q dan acumulados del año (3, 6 y 9
    meses) y el 10-K el año: cada trimestre es la diferencia entre acumulados con el
    mismo inicio. Un hecho de 3 meses directo manda sobre la diferencia."""
    items = [(pd.Timestamp(s), pd.Timestamp(e), v[0]) for (s, e), v in dur.items()]
    q = {}
    for s, e, v in items:
        if 80 <= (e - s).days <= 100:
            q[e] = v
    by_start = defaultdict(list)
    for s, e, v in items:
        if 80 <= (e - s).days <= 380:
            by_start[s].append((e, v))
    for s, lst in by_start.items():
        lst.sort()
        for (e0, v0), (e1, v1) in zip(lst, lst[1:]):
            if 80 <= (e1 - e0).days <= 100 and e1 not in q:
                q[e1] = v1 - v0
    return dict(sorted(q.items()))


def ltm(q, dur):
    """Suma de 4 trimestres consecutivos en cada fin de trimestre; si hay un hecho
    anual que acaba ese día, manda el anual."""
    fy = {pd.Timestamp(e): v[0] for (s, e), v in dur.items()
          if 350 <= (pd.Timestamp(e) - pd.Timestamp(s)).days <= 380}
    ends = list(q)
    out = {}
    for i in range(3, len(ends)):
        w = ends[i - 3:i + 1]
        gaps = [(b - a).days for a, b in zip(w, w[1:])]
        if all(80 <= g <= 100 for g in gaps):
            out[ends[i]] = sum(q[e] for e in w)
    for e, v in fy.items():
        if e in out or e in q:
            out[e] = v
    return dict(sorted(out.items()))


def cal_q(d):
    """Fin de trimestre fiscal → fin de trimestre natural más cercano (NVDA 26-jul → 30-jun)."""
    d = pd.Timestamp(d)
    cands = [pd.Timestamp(d.year - 1, 12, 31)] + [pd.Timestamp(d.year, m, 1) + pd.offsets.MonthEnd(0)
                                                  for m in (3, 6, 9, 12)]
    return min(cands, key=lambda c: abs((c - d).days))


def yahoo_splits(t):
    try:
        import yfinance as yf
        s = yf.Ticker(t).splits
        if s is not None and len(s):
            if s.index.tz is not None:
                s.index = s.index.tz_localize(None)
            return [(str(i.date()), float(v)) for i, v in s.items() if v and v != 1]
    except Exception:
        pass
    return SPLITS.get(t, [])


def diluted_shares(g, splits):
    """Acciones diluidas medias por trimestre, en la base de acciones de hoy: cada dato se
    multiplica por los splits posteriores a la fecha en que se presentó."""
    dur = durations(g, ['WeightedAverageNumberOfDilutedSharesOutstanding'], 'shares')
    out = {}
    for (s, e), (v, filed) in dur.items():
        d = (pd.Timestamp(e) - pd.Timestamp(s)).days
        if not (80 <= d <= 100 or 350 <= d <= 380):
            continue
        f = 1.0
        for sd, r in splits:
            if sd > filed:
                f *= r
        e = pd.Timestamp(e)
        if 80 <= d <= 100 or e not in out:
            out[e] = v * f
    s = pd.Series(out, dtype=float).sort_index()
    # Los primeros XBRL (2009-10) traen a veces la escala mal (Nvidia: 22 en vez de 22.000 millones)
    med = s.rolling(5, center=True, min_periods=1).median()
    return s[(s / med).between(0.5, 2)]


def company_cash(t, g, splits):
    rows = {}
    for k in FLOWS:
        dur = durations(g, TAGS[k])
        rows[k] = pd.Series(ltm(quarterly(dur), dur), dtype=float)
    df = pd.DataFrame(rows).sort_index()
    # sin caja operativa, capex o ventas el trimestre no vale (Amazon no da el acumulado del 2T de 2017)
    df = df[df['ocf'].notna() & df['capex'].notna() & df['rev'].notna()]
    for k in ['fl', 'bb', 'div', 'sbc']:
        df[k] = df[k].fillna(0.0)        # si no hay dato, la empresa no lo tiene (Amazon no recompra)
    df['fcf'] = df['ocf'] - df['capex'] - df['fl']
    df['fcf_sbc'] = df['fcf'] - df['sbc']
    df['ret'] = df['bb'] + df['div']
    df['fin_rest'] = df['fin'] + df['bb'] + df['div'] + df['fl']   # financiación sin recompras, dividendos ni arrendamientos
    df['inv_rest'] = df['inv_cf'] + df['capex']                      # inversión sin capex (compras, valores...)
    inv = instants(g, ['InventoryNet'])
    df['inventory'] = [inv.get(e) for e in df.index]
    sh = diluted_shares(g, splits)
    # Alphabet no da acciones diluidas en XBRL antes de 2022: se congela el primer dato hacia atrás
    df['shares'] = sh.reindex(df.index.union(sh.index)).ffill().bfill().reindex(df.index) if len(sh) else None
    df.index = pd.DatetimeIndex(df.index)
    return df, sh


def cash_block(prev):
    if not SEC_EMAIL:
        print('  · sin SEC_CONTACT_EMAIL: se conserva la caja de la ejecución anterior')
        return None, None
    per, shares = {}, {}
    for t, (cik, name, grp) in COS.items():
        js = edgar_facts(cik)
        time.sleep(0.15)
        if not js:
            continue
        g = js.get('facts', {}).get('us-gaap', {})
        try:
            df, sh = company_cash(t, g, yahoo_splits(t))
        except Exception as e:
            print(f'  ✗ caja {t}: {e}')
            continue
        if df.empty:
            continue
        per[t], shares[t] = df, sh
    if len(per) < 12:
        print(f'  ✗ EDGAR: solo {len(per)} empresas; se conserva lo anterior')
        return None, None
    return per, shares


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

B = 1e9


def main():
    print(f'[{TS}] Big Tech, semis y caja')
    prev = load_prev()
    out = {'timestamp': TS, 'version': 1}

    # ── Descargas ────────────────────────────────────────────────────────────
    # RM_CACHE (solo pruebas en local): guarda las descargas y las reutiliza
    cache = os.environ.get('RM_CACHE')
    if cache and os.path.exists(cache):
        import pickle
        with open(cache, 'rb') as fh:
            hold, per, shares, LC, LA, BC, F = pickle.load(fh)
        for s in [k for k, v in F.items() if v is None]:
            F[s] = fred(s)
        print('  (descargas de la caché local; FRED que faltaba, de nuevo)')
        out.update(build(prev, hold, per, shares, LC, LA, BC, F))
        return write(out)
    print('Cartera del SPY…')
    hold = spy_holdings()
    if hold is None and prev.get('index'):
        hold = {'as_of': prev['index'].get('as_of'), 'rows': prev['index'].get('rows', []), 'stale': True}
    print('EDGAR (estado de flujos de 16 empresas)…')
    per, shares = cash_block(prev)
    print('Precios largos (índices, FAAMG, semis)…')
    LONG = ['^GSPC', '^SOX', 'RSP', 'SPY', 'SMH', '^TNX'] + FAAMG + SEMIS + list(EXTRA_PX)
    LC, LA = yf_batch(LONG, 'max', chunk=40)
    print('Precios de los valores del S&P 500 (amplitud)…')
    members = [r['t'] for r in (hold or {}).get('rows', [])]
    BC = yf_batch(members, '3y')[0] if members else None
    print('FRED…')
    F = {s: fred(s) for s in ['A679RC1Q027SBEA', 'GDP', 'JTSJOL', 'UNEMPLOY', 'DGS10', 'USREC']}

    if cache:
        import pickle
        with open(cache, 'wb') as fh:
            pickle.dump((hold, per, shares, LC, LA, BC, F), fh)
    out.update(build(prev, hold, per, shares, LC, LA, BC, F))
    write(out)


def write(out):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, 'tech.json')
    tmp = path + '.tmp'                     # se escribe aparte: un fallo no deja el JSON a medias
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(clean(out), fh, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    os.replace(tmp, path)
    print(f'  ✓ {os.path.normpath(path)} ({os.path.getsize(path) / 1024:.0f} KB)')


def at(s, d):
    """Último valor de la serie en o antes de la fecha d."""
    if s is None:
        return None
    s = s.dropna()
    s = s[s.index <= pd.Timestamp(d)]
    return num(s.iloc[-1]) if len(s) else None


def spans(rec):
    """USREC (0/1 mensual) → [[inicio, fin], ...]."""
    if rec is None:
        return []
    out, start = [], None
    for d, v in rec.items():
        if v >= 1 and start is None:
            start = d
        if v < 1 and start is not None:
            out.append([str(start.date()), str((d - pd.offsets.MonthBegin(1)).date())])
            start = None
    if start is not None:
        out.append([str(start.date()), str(rec.index[-1].date())])
    return out


def build(prev, hold, per, shares, LC, LA, BC, F):
    res = {}
    W = {}
    if hold:
        for r in hold['rows']:
            t = 'GOOGL' if r['t'] == 'GOOG' else r['t']
            W[t] = W.get(t, 0) + r['w']
    tot = sum(W.values()) or 100.0

    # ── 1) Pesos de hoy ───────────────────────────────────────────────────────
    if hold:
        top = sorted(W.items(), key=lambda x: -x[1])
        names = {}
        for r in hold['rows']:
            names.setdefault('GOOGL' if r['t'] == 'GOOG' else r['t'], r['name'].replace(' Cl A', '').replace(' Class A', ''))
        wf = sum(W.get(t, 0) for t in FAAMG)
        ws_all = sum(W.get(t, 0) for t in SP_SEMIS)
        ws = sum(W.get(t, 0) for t in SEMIS)
        res['index'] = {
            'as_of': hold.get('as_of'), 'stale': bool(hold.get('stale')), 'n': len(hold['rows']), 'total': rnd(tot, 2),
            'faamg': rnd(wf, 2), 'semis': rnd(ws_all, 2), 'semis11': rnd(ws, 2),
            'mag7': rnd(wf + W.get('NVDA', 0) + W.get('TSLA', 0), 2), 'top10': rnd(sum(w for _, w in top[:10]), 2),
            'rest': rnd(tot - wf - ws_all, 2),
            'top': [[t, names.get(t, t), rnd(w, 2)] for t, w in top[:15]],
            'rows': hold['rows'],
        }
        print(f"  Pesos ({hold.get('as_of')}): FAAMG {wf:.1f}% · semis {ws_all:.1f}% · top 10 {res['index']['top10']}%")
    elif prev.get('index'):
        res['index'] = prev['index']

    # ── 2) Cestas y «S&P sin ellos» ──────────────────────────────────────────
    spx = LC['^GSPC'].dropna() if LC is not None and '^GSPC' in LC else None
    sh_daily = {}
    if shares:
        for t, s in shares.items():
            if t in LC and len(s):
                # antes del primer dato (Alphabet solo da acciones diluidas desde 2022) se congela el primero
                sh_daily[t] = s.reindex(s.index.union(LC.index)).ffill().bfill().reindex(LC.index)
    perf = None
    if spx is not None and W:
        T = spx.index[-1]
        if hold and hold.get('as_of'):
            T = spx.index[spx.index <= pd.Timestamp(hold['as_of'])][-1]

        def basket(members):
            """Nivel en % del S&P 500 de hoy: Σ peso_i × (precio_i / precio_i(T)) × (acciones_i / acciones_i(T))."""
            parts = []
            for t in members:
                if t not in LC or W.get(t) is None:
                    continue
                p = LC[t].dropna()
                p0 = at(p, T)
                if not p0:
                    continue
                r = p / p0
                if t in sh_daily:
                    s = sh_daily[t]
                    s0 = at(s, T)
                    if s0:
                        r = r * (s.reindex(r.index).ffill() / s0)
                parts.append(W[t] * r)
            return pd.concat(parts, axis=1).dropna().sum(axis=1) if parts else None

        M = (spx / at(spx, T) * tot)
        Fb, Sb = basket(FAAMG), basket(SEMIS)
        idx = M.index.intersection(Fb.index).intersection(Sb.index)
        M_, F_, S_ = M.reindex(idx), Fb.reindex(idx), Sb.reindex(idx)
        E_ = M_ - F_ - S_
        # el peso de los semis se escala a todos los del índice (los 11 de la cesta son ~96%)
        ws11 = sum(W.get(t, 0) for t in SEMIS)
        k_all = sum(W.get(t, 0) for t in SP_SEMIS) / ws11 if ws11 else 1.0
        wF, wS = F_ / M_ * 100, S_ / M_ * 100 * k_all
        start5 = T - pd.DateOffset(years=5)
        cut = lambda s: s[s.index >= start5] if s is not None else None
        d5 = cut(M_).index

        def align(s):
            if s is None:
                return [None] * len(d5)
            s = s.dropna()
            return [rnd(x, 4) for x in s.reindex(s.index.union(d5)).ffill().reindex(d5).values]
        ser = {'spx': M_, 'faamg': F_, 'semis': LC['^SOX'] if '^SOX' in LC else None,
               'rsp': LC['RSP'] if 'RSP' in LC else None, 'ex': E_}
        perf = {'base': str(T.date()), 'dates': [str(d.date()) for d in d5],
                'series': {k: align(v) for k, v in ser.items()}}
        wk = lambda s: s.resample('W-FRI').last().dropna()
        perf['weights'] = {'faamg': pts(wk(wF[wF.index >= '2016-01-01']), digits=2),
                           'semis': pts(wk(wS[wS.index >= '2016-01-01']), digits=2)}
        # Fuerza relativa larga (semanal) con su media de 40 semanas ≈ MM200
        rel = {}
        if '^SOX' in LC:
            r = wk((LC['^SOX'] / spx).dropna())
            r = r / r.iloc[0] * 100
            rel['sox'] = {'v': pts(r, digits=1), 'ma': pts(r.rolling(40).mean(), digits=1)}
        r = wk((F_ / M_).dropna())
        r = r[r.index >= '2013-01-01']
        if len(r):
            r = r / r.iloc[0] * 100
            rel['faamg'] = {'v': pts(r, digits=1), 'ma': pts(r.rolling(40).mean(), digits=1)}
        perf['rel'] = rel

        def gap200(s):
            s = s.dropna()
            return num((s.iloc[-1] / s.rolling(200).mean().iloc[-1] - 1) * 100) if len(s) > 200 else None

        def r_n(s, n):
            return pct(s, n)

        def ytd(s):
            s = s.dropna()
            y = s[s.index < pd.Timestamp(s.index[-1].year, 1, 1)]
            return num((s.iloc[-1] / y.iloc[-1] - 1) * 100) if len(y) else None
        rs = {'spx': M_, 'rsp': LC.get('RSP'), 'faamg': F_, 'semis': LC.get('^SOX'), 'ex': E_}
        perf['stats'] = {k: {'m1': rnd(r_n(s, 21), 1), 'm3': rnd(r_n(s, 63), 1), 'ytd': rnd(ytd(s), 1),
                             'y1': rnd(r_n(s, 252), 1), 'y3': rnd(r_n(s, 756), 1), 'ma200': rnd(gap200(s), 1),
                             'dd52': rnd((s.dropna().iloc[-1] / s.dropna().tail(252).max() - 1) * 100, 1)}
                         for k, s in rs.items() if s is not None and len(s.dropna()) > 260}
        rF, rS = (F_ / M_), (LC['^SOX'] / spx).dropna() if '^SOX' in LC else None
        perf['lead'] = {'faamg_vs_ma200': rnd(gap200(rF), 1), 'semis_vs_ma200': rnd(gap200(rS), 1) if rS is not None else None,
                        'w_faamg_1y': rnd(at(wF, T - pd.DateOffset(years=1)), 2), 'w_semis_1y': rnd(at(wS, T - pd.DateOffset(years=1)), 2),
                        'w_faamg_5y': rnd(at(wF, start5), 2), 'w_semis_5y': rnd(at(wS, start5), 2)}
        print(f"  Cestas desde {perf['dates'][0]}: FAAMG peso hoy {wF.iloc[-1]:.1f}% (hace 1 año {perf['lead']['w_faamg_1y']}%)")

        # Tabla por empresa (precio)
        tab = []
        for t in FAAMG + SEMIS + list(EXTRA_PX):
            if t not in LC:
                continue
            s = LC[t].dropna()
            if len(s) < 260:
                continue
            tab.append({'t': t, 'name': COS.get(t, (0, EXTRA_PX.get(t)))[1], 'g': COS.get(t, (0, '', 'semis'))[2],
                        'w': rnd(W.get(t), 2), 'px': rnd(s.iloc[-1], 2), 'm1': rnd(r_n(s, 21), 1), 'm3': rnd(r_n(s, 63), 1),
                        'ytd': rnd(ytd(s), 1), 'y1': rnd(r_n(s, 252), 1), 'ma200': rnd(gap200(s), 1),
                        'dd52': rnd((s.iloc[-1] / s.tail(252).max() - 1) * 100, 1)})
        perf['table'] = tab
        perf['as_of'] = str(spx.index[-1].date())
    res['perf'] = perf or prev.get('perf')

    # ── 3) Amplitud de los valores del S&P 500 ────────────────────────────────
    br = None
    if BC is not None and BC.shape[1] >= 400:
        px = BC.ffill(limit=3)
        m50 = px.rolling(50, min_periods=50).mean()
        m200 = px.rolling(200, min_periods=200).mean()
        hi = px.rolling(252, min_periods=252).max()
        lo = px.rolling(252, min_periods=252).min()
        v200 = m200.notna() & px.notna()
        v50 = m50.notna() & px.notna()
        v252 = hi.notna() & px.notna()
        share = lambda cond, v: (cond & v).sum(axis=1) / v.sum(axis=1).replace(0, np.nan) * 100
        a200, a50 = share(px > m200, v200), share(px > m50, v50)
        nh, nl = share(px >= hi, v252), share(px <= lo, v252)
        both = share((px < m50) & (px < m200), v200)
        ret = px.pct_change(fill_method=None)
        adv, dec = (ret > 0).sum(axis=1), (ret < 0).sum(axis=1)
        rana = ((adv - dec) / (adv + dec).replace(0, np.nan) * 1000).iloc[1:]
        mcc = rana.ewm(span=19, adjust=False).mean() - rana.ewm(span=39, adjust=False).mean()
        mcc = mcc.iloc[60:]                     # el arranque de las medias exponenciales no vale
        start = a200.dropna().index[0]
        last = px.index[-1]
        n_now = int(v200.loc[last].sum())
        p05 = num(mcc[mcc.index >= start].quantile(0.05))
        spx_dd = num((spx.iloc[-1] / spx.tail(252).max() - 1) * 100) if spx is not None else None
        br = {'as_of': str(last.date()), 'n': int(px.loc[last].notna().sum()), 'n200': n_now, 'since': str(start.date()),
              'above200': rnd(a200.iloc[-1], 1), 'above50': rnd(a50.iloc[-1], 1),
              'above200_1m': rnd(a200.iloc[-22], 1), 'nh': rnd(nh.iloc[-1], 1), 'nl': rnd(nl.iloc[-1], 1),
              'nh5': rnd(nh.tail(5).mean(), 1), 'nl5': rnd(nl.tail(5).mean(), 1),
              'below_both': rnd(both.iloc[-1], 1), 'mcc': rnd(mcc.iloc[-1], 1), 'mcc_min5': rnd(mcc.tail(5).min(), 1),
              'mcc_p05': rnd(p05, 1), 'spx_dd52': rnd(spx_dd, 1),
              'charts': {'above200': pts(a200[a200.index >= start], digits=1), 'above50': pts(a50[a50.index >= start], digits=1),
                         'nhnl': pts((nh - nl)[nh.index >= start].rolling(5).mean(), digits=1),
                         'mcc': pts(mcc[mcc.index >= start], digits=1),
                         'below_both': pts(both[both.index >= start], digits=1)}}
        print(f"  Amplitud S&P 500 ({br['n']} valores): {br['above200']}% sobre MM200 · McClellan {br['mcc']}")
    res['breadth'] = br or prev.get('breadth')

    # ── 4) Estado de flujos de caja ───────────────────────────────────────────
    cash = None
    if per:
        t10 = F.get('DGS10')
        if t10 is None and LC is not None and '^TNX' in LC:
            t10 = LC['^TNX'].dropna()
        cos, cal = {}, {}
        for t, df in per.items():
            d = df.copy()
            p = LC[t].dropna() if LC is not None and t in LC else None
            d['mcap'] = [(at(p, e) or np.nan) * (num(s) or np.nan) for e, s in zip(d.index, d['shares'])] if p is not None else np.nan
            d['cal'] = [cal_q(e) for e in d.index]
            cal[t] = d.drop_duplicates('cal', keep='last').set_index('cal')
            L = d.iloc[-1]
            Y = d[d.index <= d.index[-1] - pd.Timedelta(days=330)]
            Y = Y.iloc[-1] if len(Y) else None
            cur_mcap = (num(p.iloc[-1]) or 0) * (num(L['shares']) or 0) if p is not None else None
            fields = ['rev', 'ocf', 'capex', 'fl', 'fcf', 'sbc', 'fcf_sbc', 'bb', 'div', 'ret', 'fin_rest', 'inv_rest', 'da', 'ni']
            cos[t] = {'name': COS[t][1], 'g': COS[t][2], 'last': str(d.index[-1].date()),
                      'ltm': {k: rnd(L[k] / B, 2) for k in fields},
                      'yago': {k: rnd(Y[k] / B, 2) for k in fields} if Y is not None else {},
                      'mcap': rnd(cur_mcap / B, 1) if cur_mcap else None,
                      'series': {k: pts(cal[t][k][cal[t].index >= '2015-01-01'] / B, digits=2)
                                 for k in ['capex', 'ocf', 'fcf_sbc', 'rev']}}

        def agg(members, need_all=True):
            frames = [cal[t] for t in members if t in cal]
            keys = ['rev', 'ocf', 'capex', 'fl', 'fcf', 'sbc', 'fcf_sbc', 'bb', 'div', 'ret', 'fin_rest', 'da', 'mcap', 'inventory', 'cogs']
            idx = sorted(set().union(*[f.index for f in frames]))
            rows = []
            for q in idx:
                have = [f for f in frames if q in f.index]
                if need_all and len(have) < len(members):
                    continue
                rows.append({'q': q, **{k: sum(num(f.loc[q, k]) or 0 for f in have) for k in keys},
                             'mcap_ok': all(num(f.loc[q, 'mcap']) for f in have),
                             'inv_ok': all(num(f.loc[q, 'inventory']) is not None for f in have)})
            a = pd.DataFrame(rows).set_index('q') if rows else pd.DataFrame()
            if a.empty:
                return None
            a = a[a.index >= '2014-01-01']
            o = {'dates': [str(q.date()) for q in a.index]}
            for k in ['rev', 'ocf', 'capex', 'fcf', 'fcf_sbc', 'sbc', 'ret', 'fin_rest', 'da']:
                o[k] = [rnd(v / B, 1) for v in a[k]]
            o['capex_ocf'] = [rnd(c / x * 100, 1) if x > 0 else None for c, x in zip(a['capex'], a['ocf'])]
            o['ret_fcf'] = [rnd(r / f * 100, 0) if f > 0 else None for r, f in zip(a['ret'], a['fcf'])]
            o['capex_da'] = [rnd(c / x, 2) if x > 0 else None for c, x in zip(a['capex'], a['da'])]
            o['fcf_yield'] = [rnd(f / m * 100, 2) if ok and m > 0 else None for f, m, ok in zip(a['fcf_sbc'], a['mcap'], a['mcap_ok'])]
            o['fcf_yield_raw'] = [rnd(f / m * 100, 2) if ok and m > 0 else None for f, m, ok in zip(a['fcf'], a['mcap'], a['mcap_ok'])]
            o['inv_days'] = [rnd(i / c * 365, 0) if ok and c > 0 else None for i, c, ok in zip(a['inventory'], a['cogs'], a['inv_ok'])]
            for k in ['capex', 'rev', 'ocf', 'fcf_sbc']:
                s = a[k]
                # contra el mismo trimestre del año anterior, buscado por fecha (puede faltar alguno)
                yago = [s.get(q - pd.DateOffset(years=1)) for q in s.index]
                o[k + '_yoy'] = [rnd((v / y - 1) * 100, 1) if y is not None and num(y) and y > 0 else None
                                 for v, y in zip(s, yago)]
            o['t10'] = [rnd(at(t10, q), 2) for q in a.index]
            return o

        # el «ahora» de cada cesta: el último LTM de cada empresa, aunque no sean del mismo trimestre
        def now(members):
            ks = ['rev', 'ocf', 'capex', 'fl', 'fcf', 'sbc', 'fcf_sbc', 'bb', 'div', 'ret', 'fin_rest', 'inv_rest', 'da']
            have = [t for t in members if t in cos]
            o = {k: rnd(sum(cos[t]['ltm'].get(k) or 0 for t in have), 1) for k in ks}
            o['yago'] = {k: rnd(sum(cos[t]['yago'].get(k) or 0 for t in have), 1) for k in ks}
            o['mcap'] = rnd(sum(cos[t]['mcap'] or 0 for t in have), 0)
            o['n'], o['last'] = len(have), max(cos[t]['last'] for t in have) if have else None
            o['oldest'] = min(cos[t]['last'] for t in have) if have else None
            return o
        cash = {'fetched': TS[:10], 'source': 'SEC EDGAR (10-K y 10-Q)', 'companies': cos,
                'baskets': {k: {'series': agg(m), 'now': now(m)} for k, m in
                            [('hyper', HYPER), ('faamg', FAAMG), ('semis', SEMIS)]},
                't10': rnd(at(t10, NOW.date()), 2) if t10 is not None else None}
        h = cash['baskets']['hyper']['now']
        print(f"  Caja hiperescaladores LTM: OCF {h['ocf']} · capex {h['capex']} · FCF tras SBC {h['fcf_sbc']} (miles de M$)")
    elif prev.get('cash'):
        cash = {**prev['cash'], 'stale': True}
    res['cash'] = cash

    # ── 5) Economía: inversión en informática y software, vacantes por parado ─
    mac = {}
    ai, gdp = F.get('A679RC1Q027SBEA'), F.get('GDP')
    if ai is not None and gdp is not None:
        df = pd.concat({'ai': ai, 'gdp': gdp}, axis=1).dropna()
        df['share'] = df['ai'] / df['gdp'] * 100
        df['g_yoy'] = (df['gdp'] / df['gdp'].shift(4) - 1) * 100
        df['contrib'] = (df['ai'] - df['ai'].shift(4)) / df['gdp'].shift(4) * 100
        pk = df.loc[:'2005-12-31', 'share']
        last = df.iloc[-1]
        mac['ai'] = {'as_of': str(df.index[-1].date()), 'level': rnd(last['ai'], 0), 'share': rnd(last['share'], 2),
                     'peak2000': rnd(pk.max(), 2), 'peak2000_date': str(pk.idxmax().date()),
                     'g_yoy': rnd(last['g_yoy'], 2), 'contrib': rnd(last['contrib'], 2),
                     'contrib_share': rnd(last['contrib'] / last['g_yoy'] * 100, 0) if last['g_yoy'] > 0 else None,
                     'yoy': rnd((df['ai'].iloc[-1] / df['ai'].iloc[-5] - 1) * 100, 1),
                     'share_chart': pts(df['share'][df.index >= '1990-01-01'], digits=2),
                     'g_chart': pts(df['g_yoy'][df.index >= '2010-01-01'], digits=2),
                     'c_chart': pts(df['contrib'][df.index >= '2010-01-01'], digits=2)}
    jo, un = F.get('JTSJOL'), F.get('UNEMPLOY')
    if jo is not None and un is not None:
        vu = pd.concat({'jo': jo, 'un': un}, axis=1).dropna()
        vu = vu['jo'] / vu['un']
        mac['vu'] = {'as_of': str(vu.index[-1].date()), 'value': rnd(vu.iloc[-1], 2), 'y1': rnd(vu.iloc[-13], 2) if len(vu) > 13 else None,
                     'max': rnd(vu.max(), 2), 'max_date': str(vu.idxmax().date()), 'chart': pts(vu, digits=2)}
    mac['recessions'] = spans(F.get('USREC'))
    res['macro'] = mac if mac.get('ai') or mac.get('vu') else prev.get('macro', mac)
    return res


if __name__ == '__main__':
    main()
