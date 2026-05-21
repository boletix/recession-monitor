#!/usr/bin/env python3
"""
Recession Monitor — Data Fetcher v3.0
Fuentes: yfinance (mercados), Treasury.gov XML (curvas), BLS API (empleo),
         fredapi (macro macro, si FRED_API_KEY disponible).
Outputs: data.json, harnett_data.json, credit_data.json, macro_indicators.json
"""

import os, json, datetime, warnings, statistics, time
import xml.etree.ElementTree as ET
import urllib.request
import numpy as np
import yfinance as yf

warnings.filterwarnings('ignore')

FRED_API_KEY = os.environ.get('FRED_API_KEY', '')
TS = datetime.datetime.utcnow().isoformat() + 'Z'

# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ══════════════════════════════════════════════════════════════════════════════

def dl(ticker, period='400d', retries=3):
    """Download yfinance ticker, return close as float series."""
    for i in range(retries):
        try:
            d = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            if d.empty: raise ValueError("empty")
            return d['Close'].squeeze()
        except Exception as e:
            if i == retries - 1:
                print(f"yfinance {ticker}: {e}")
                return None
            time.sleep(1)

def treasury_yields():
    """Fetch latest daily Treasury yield curve from Treasury.gov."""
    from datetime import date
    ym = date.today().strftime('%Y%m')
    url = (f'https://home.treasury.gov/resource-center/data-chart-center/'
           f'interest-rates/pages/xml?data=daily_treasury_yield_curve'
           f'&field_tdr_date_value_month={ym}')
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=12) as r:
            content = r.read().decode('utf-8')
        ns = {'d': 'http://schemas.microsoft.com/ado/2007/08/dataservices',
              'a': 'http://www.w3.org/2005/Atom'}
        root = ET.fromstring(content)
        entries = root.findall('.//a:entry', ns)
        if not entries: return {}
        last = entries[-1]
        def gv(tag):
            el = last.find(f'.//d:{tag}', ns)
            return float(el.text) if el is not None and el.text else None
        return {
            'y1m':  gv('BC_1MONTH'),  'y3m':  gv('BC_3MONTH'),
            'y6m':  gv('BC_6MONTH'),  'y1y':  gv('BC_1YEAR'),
            'y2y':  gv('BC_2YEAR'),   'y3y':  gv('BC_3YEAR'),
            'y5y':  gv('BC_5YEAR'),   'y7y':  gv('BC_7YEAR'),
            'y10y': gv('BC_10YEAR'),  'y20y': gv('BC_20YEAR'),
            'y30y': gv('BC_30YEAR'),
        }
    except Exception as e:
        print(f"Treasury API: {e}")
        return {}

def bls_series(sid, start='2023', end='2026'):
    """Fetch a single BLS series. Returns list of (period_str, value) newest first."""
    url = 'https://api.bls.gov/publicAPI/v2/timeseries/data/'
    payload = json.dumps({'seriesid': [sid], 'startyear': start, 'endyear': end}).encode()
    req = urllib.request.Request(url, data=payload,
                                 headers={'Content-Type': 'application/json',
                                          'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            d = json.loads(r.read())
        rows = []
        for x in d.get('Results', {}).get('series', [{}])[0].get('data', []):
            try:
                rows.append((x['year'] + x['period'], float(x['value'])))
            except Exception:
                pass
        return rows
    except Exception as e:
        print(f"BLS {sid}: {e}")
        return []

def fred_series(sid, n=36):
    """Fetch FRED series if API key available. Returns list newest first."""
    if not FRED_API_KEY:
        return []
    try:
        from fredapi import Fred
        s = Fred(api_key=FRED_API_KEY).get_series(sid).dropna()
        return [(str(idx.date()), float(v)) for idx, v in s.iloc[-n:].iloc[::-1].items()]
    except Exception as e:
        print(f"FRED {sid}: {e}")
        return []

# ══════════════════════════════════════════════════════════════════════════════
# INDICATOR HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def latest(series_list, default=None):
    return series_list[0][1] if series_list else default

def pct_chg(s, n=1):
    """Percent change n bars back in a pandas Series."""
    if s is None or len(s) <= n: return None
    return float((s.iloc[-1] / s.iloc[-1-n] - 1) * 100)

def dma(s, n):
    """n-day moving average of latest value."""
    if s is None or len(s) < n: return None
    return float(s.rolling(n).mean().iloc[-1])

def zscore(s, window=252):
    """Z-score of latest value vs rolling window."""
    if s is None or len(s) < window // 2: return None
    recent = s.tail(window)
    return float((s.iloc[-1] - recent.mean()) / recent.std())

def score_val(value, breakpoints, scores):
    """
    Map value to 0-100 score.
    breakpoints: thresholds ascending; scores: score at each threshold + one extra.
    """
    if value is None: return 50
    for bp, sc in zip(breakpoints, scores):
        if value <= bp: return sc
    return scores[-1]

def status_label(sc):
    if sc < 25: return "OK"
    if sc < 50: return "WATCH"
    if sc < 75: return "WARNING"
    return "DANGER"

def color_for(sc):
    return "#00d68f" if sc < 25 else "#ffd93d" if sc < 50 else "#ff8c42" if sc < 75 else "#ff4d4d"

def ind(label, value, sc, unit="", fmt=".1f", note=""):
    """Standard indicator dict."""
    v_str = (f"{value:{fmt}}{unit}" if value is not None else "--")
    return {
        "label": label, "value": value, "value_str": v_str,
        "score": int(sc), "status": status_label(sc),
        "color": color_for(sc), "note": note
    }

# ══════════════════════════════════════════════════════════════════════════════
# SCORE FUNCTIONS PER INDICATOR
# ══════════════════════════════════════════════════════════════════════════════

def score_vix(v):
    return score_val(v, [15, 20, 25, 30, 40], [5, 15, 30, 50, 75, 95])

def score_spread_10y2y(v):   # positive = normal, negative = inverted
    return score_val(v, [-1.5, -0.5, 0, 0.5, 1.5], [95, 75, 50, 25, 10, 5])

def score_spread_10y3m(v):
    return score_val(v, [-1.5, -0.5, 0, 0.3, 1.0], [95, 80, 55, 25, 10, 5])

def score_bond30y(v):        # bubble killer at 5%
    return score_val(v, [3.5, 4.0, 4.5, 5.0, 5.5], [5, 10, 25, 60, 80, 90])

def score_sox_bubble(v):     # pct above 200dma
    return score_val(v, [10, 20, 30, 50, 70], [5, 15, 30, 60, 85, 97])

def score_hy_stress(v):      # HYG 3M vs LQD 3M underperformance (positive = bad)
    return score_val(v, [-3, -1, 0, 1, 3], [5, 10, 25, 50, 75, 90])

def score_oil_rise(v):       # pct above 52W low
    return score_val(v, [20, 40, 60, 90, 120], [5, 10, 20, 55, 80, 95])

def score_dxy(v):            # DXY > 105 = EM stress
    return score_val(v, [95, 100, 103, 106, 110], [5, 10, 25, 50, 70, 85])

def score_spy_drawdown(v):   # negative value
    return score_val(v, [-30, -20, -15, -10, -5], [95, 80, 60, 35, 15, 5])

def score_sahm(v):           # 0.5+ = triggered
    return score_val(v, [0.0, 0.2, 0.35, 0.5, 0.75], [5, 15, 35, 70, 88, 97])

def score_unrate(v):         # unemployment
    return score_val(v, [3.5, 4.0, 4.5, 5.0, 6.0], [5, 10, 25, 50, 75, 90])

def score_payrolls_mom(v):   # in thousands
    return score_val(v, [-300, -100, 0, 100, 200], [95, 80, 60, 25, 10, 5])

def score_savings(v):        # personal savings rate %
    return score_val(v, [2, 3, 4, 6, 8], [95, 75, 50, 20, 8, 3])

def score_umich(v):          # consumer sentiment
    return score_val(v, [50, 58, 65, 75, 85], [95, 75, 55, 30, 12, 5])

def score_kre(v):            # KRE 3M pct change
    return score_val(v, [-30, -20, -10, -5, 0], [95, 80, 60, 35, 15, 5])

def score_bkln(v):           # BKLN 3M pct change
    return score_val(v, [-10, -5, -2, 0, 2], [95, 80, 55, 30, 10, 5])

# ══════════════════════════════════════════════════════════════════════════════
# COMPOSITE RECESSION SCORE (weighted)
# ══════════════════════════════════════════════════════════════════════════════

WEIGHTS = {
    # Market signals (45%)
    'vix':          0.08,
    'sox_bubble':   0.09,
    'hy_stress':    0.08,
    'spy_drawdown': 0.05,
    'bond_30y':     0.08,
    'dxy':          0.04,
    # Economic signals (35%)
    'sahm':         0.12,
    'spread_10y3m': 0.08,
    'payrolls':     0.08,
    'oil_rise':     0.05,
    # Consumer signals (20%)
    'savings':      0.07,
    'umich':        0.08,
    'kre':          0.03,
}

def composite_score(scores_dict):
    """Weighted composite 0-100."""
    total_weight = 0
    total_score = 0
    for key, w in WEIGHTS.items():
        if key in scores_dict and scores_dict[key] is not None:
            total_score += scores_dict[key] * w
            total_weight += w
    if total_weight == 0: return 50
    return round(total_score / total_weight, 1)

def composite_label(sc):
    if sc < 20: return "RIESGO BAJO"
    if sc < 40: return "MODERADO"
    if sc < 55: return "ELEVADO"
    if sc < 70: return "ALTO RIESGO"
    if sc < 85: return "CRISIS INMINENTE"
    return "CRISIS ACTIVA"

# ══════════════════════════════════════════════════════════════════════════════
# MAIN FETCH + COMPUTE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"[{TS}] Starting data fetch...")

    # ── Market data ──────────────────────────────────────────────────────────
    print("Fetching market data...")
    vix_s  = dl('^VIX', '90d')
    spy_s  = dl('SPY', '500d')
    soxx_s = dl('SOXX', '500d')
    hyg_s  = dl('HYG', '500d')
    lqd_s  = dl('LQD', '500d')
    tlt_s  = dl('TLT', '500d')
    kre_s  = dl('KRE', '500d')
    xlf_s  = dl('XLF', '500d')
    bkln_s = dl('BKLN', '500d')
    oil_s  = dl('CL=F', '500d')
    dxy_s  = dl('DX-Y.NYB', '90d')
    gld_s  = dl('GLD', '500d')

    # ── Treasury yields ───────────────────────────────────────────────────────
    print("Fetching Treasury yields...")
    yields = treasury_yields()

    # ── BLS data ──────────────────────────────────────────────────────────────
    print("Fetching BLS data...")
    unrate_bl  = bls_series('LNS14000000')   # Unemployment rate
    payrolls_bl= bls_series('CES0000000001') # Nonfarm payrolls (000s)
    quits_bl   = bls_series('LNS13023705')   # JOLTS quits (000s)
    claims_bl  = bls_series('LASBS000000000000003') # Initial claims proxy

    # ── FRED (if key available) ───────────────────────────────────────────────
    savings_fl  = fred_series('PSAVERT')      # Personal savings rate
    umich_fl    = fred_series('UMCSENT')      # UMich consumer sentiment
    sahm_fl     = fred_series('SAHMREALTIME') # Sahm Rule
    indpro_fl   = fred_series('INDPRO')       # Industrial production
    icsa_fl     = fred_series('ICSA')         # Initial claims weekly
    temphelp_fl = fred_series('TEMPHELPS')    # Temp employment

    # ══════════════════════════════════════════════════════════════════════════
    # DERIVED METRICS
    # ══════════════════════════════════════════════════════════════════════════

    vix_val    = float(vix_s.iloc[-1]) if vix_s is not None else None

    # SOX bubble
    soxx_price = float(soxx_s.iloc[-1]) if soxx_s is not None else None
    soxx_200   = dma(soxx_s, 200)
    sox_bubble = (soxx_price / soxx_200 - 1) * 100 if soxx_price and soxx_200 else None

    # SPY drawdown from 1Y high
    spy_1yh    = float(spy_s.tail(252).max()) if spy_s is not None else None
    spy_now    = float(spy_s.iloc[-1]) if spy_s is not None else None
    spy_dd     = (spy_now / spy_1yh - 1) * 100 if spy_now and spy_1yh else None

    # HY spread stress: LQD 3M - HYG 3M (positive = HY underperforming)
    hyg_3m     = pct_chg(hyg_s, 63)
    lqd_3m     = pct_chg(lqd_s, 63)
    hy_stress  = ((lqd_3m or 0) - (hyg_3m or 0)) if hyg_3m is not None else None

    # Oil vs 52W low (recession trigger if > 90%)
    oil_val    = float(oil_s.iloc[-1]) if oil_s is not None else None
    oil_52l    = float(oil_s.tail(252).min()) if oil_s is not None else None
    oil_rise   = (oil_val / oil_52l - 1) * 100 if oil_val and oil_52l else None

    # Treasury yield curve
    y10    = yields.get('y10y')
    y2     = yields.get('y2y')
    y3m    = yields.get('y3m')
    y30    = yields.get('y30y')
    sp_10y2y = (y10 - y2)  if y10 and y2  else None
    sp_10y3m = (y10 - y3m) if y10 and y3m else None

    # DXY
    dxy_val  = float(dxy_s.iloc[-1]) if dxy_s is not None else None

    # KRE (regional banks) 3M performance
    kre_3m   = pct_chg(kre_s, 63)
    xlf_3m   = pct_chg(xlf_s, 63)
    bkln_3m  = pct_chg(bkln_s, 63)
    bkln_dd  = (float(bkln_s.iloc[-1]) / float(bkln_s.tail(252).max()) - 1) * 100 if bkln_s is not None else None

    # Gold/Copper ratio (risk barometer, higher = risk-off)
    gld_val  = float(gld_s.iloc[-1]) if gld_s is not None else None

    # McClellan proxy: SPY 5d vs 20d momentum spread
    spy_5d   = pct_chg(spy_s, 5)
    spy_20d  = pct_chg(spy_s, 20)
    mcclellan_proxy = ((spy_5d or 0) - (spy_20d or 0)) * 10 if spy_5d and spy_20d else None

    # Sahm Rule (FRED preferred, BLS-computed fallback)
    sahm_val = latest(sahm_fl)
    if sahm_val is None and unrate_bl:
        ur_vals = [v for _, v in unrate_bl]
        if len(ur_vals) >= 12:
            avg3m  = statistics.mean(ur_vals[:3])
            min12m = min(ur_vals[:12])
            sahm_val = round(avg3m - min12m, 2)

    # Unemployment
    unrate_val = latest(unrate_bl)

    # Payrolls MoM (thousands)
    pay_mom = None
    if len(payrolls_bl) >= 2:
        pay_mom = (payrolls_bl[0][1] - payrolls_bl[1][1])  # already in 000s

    # Initial claims (FRED preferred)
    icsa_val    = latest(icsa_fl)
    icsa_trend  = None  # 4W avg change
    if len(icsa_fl) >= 5:
        avg4w  = statistics.mean([v for _, v in icsa_fl[:4]])
        avg4wp = statistics.mean([v for _, v in icsa_fl[4:8]])
        icsa_trend = (avg4w / avg4wp - 1) * 100 if avg4wp else None

    # FRED macro
    savings_val = latest(savings_fl)
    umich_val   = latest(umich_fl)
    indpro_mom  = None
    if len(indpro_fl) >= 2:
        indpro_mom = (indpro_fl[0][1] / indpro_fl[1][1] - 1) * 100
    temphelp_yoy = None
    if len(temphelp_fl) >= 13:
        temphelp_yoy = (temphelp_fl[0][1] / temphelp_fl[12][1] - 1) * 100

    # Quits rate (thousands → rate approximation for display)
    quits_val = latest(quits_bl)

    # ══════════════════════════════════════════════════════════════════════════
    # SCORE EACH INDICATOR
    # ══════════════════════════════════════════════════════════════════════════

    sc = {}
    sc['vix']          = score_vix(vix_val)
    sc['sox_bubble']   = score_sox_bubble(sox_bubble)
    sc['hy_stress']    = score_hy_stress(hy_stress)
    sc['spy_drawdown'] = score_spy_drawdown(spy_dd)
    sc['bond_30y']     = score_bond30y(y30)
    sc['dxy']          = score_dxy(dxy_val)
    sc['sahm']         = score_sahm(sahm_val)
    sc['spread_10y3m'] = score_spread_10y3m(sp_10y3m)
    sc['payrolls']     = score_payrolls_mom(pay_mom)
    sc['oil_rise']     = score_oil_rise(oil_rise)
    sc['savings']      = score_savings(savings_val)
    sc['umich']        = score_umich(umich_val)
    sc['kre']          = score_kre(kre_3m)

    comp = composite_score(sc)
    comp_label = composite_label(comp)
    comp_color = color_for(comp)

    # ══════════════════════════════════════════════════════════════════════════
    # BUILD macro_indicators.json
    # ══════════════════════════════════════════════════════════════════════════

    macro = {
        "timestamp": TS,
        "composite": {
            "score": comp,
            "label": comp_label,
            "color": comp_color,
            "components": {k: {"score": int(v), "weight": WEIGHTS.get(k, 0)} for k, v in sc.items()}
        },
        "hartnett_live": {
            "sox_vs_200dma": round(sox_bubble, 1) if sox_bubble else None,
            "sox_price":     round(soxx_price, 2) if soxx_price else None,
            "sox_200dma":    round(soxx_200, 2) if soxx_200 else None,
            "bond_30y":      y30,
            "bond_30y_triggered": (y30 or 0) >= 5.0,
            "oil_price":     round(oil_val, 1) if oil_val else None,
            "oil_rise_pct":  round(oil_rise, 1) if oil_rise else None,
            "oil_trigger_90": (oil_rise or 0) >= 90,
            "vix":           round(vix_val, 1) if vix_val else None,
            "dxy":           round(dxy_val, 1) if dxy_val else None,
            "bull_bear_last": 7.6,
            "bull_bear_date": "2026-05-16",
            "bull_bear_triggered": False,
        },
        "market_signals": {
            "vix":          ind("VIX", vix_val, sc['vix'], "", ".1f", "Estrés mercado — >35 = pánico"),
            "sox_bubble":   ind("SOX vs MM200 (%)", sox_bubble, sc['sox_bubble'], "%", "+.1f", "Hartnett: >30% = burbuja"),
            "spy_drawdown": ind("S&P drawdown 1Y (%)", spy_dd, sc['spy_drawdown'], "%", ".1f", "Distancia desde máx 52 semanas"),
            "hy_stress":    ind("Estrés HY (LQD-HYG 3M)", hy_stress, sc['hy_stress'], "%", "+.2f", "Positivo = HY underperforming"),
            "bond_30y":     ind("Bono 30Y (%)", y30, sc['bond_30y'], "%", ".2f", "Hartnett: 5%+ = bubble killer"),
            "dxy":          ind("DXY (dólar)", dxy_val, sc['dxy'], "", ".1f", ">105 = estrés emergentes"),
            "bkln_3m":      ind("Bank Loans BKLN 3M (%)", bkln_3m, score_bkln(bkln_3m), "%", ".1f", "Crédito leveraged"),
            "kre_3m":       ind("Bancos regionales KRE 3M (%)", kre_3m, sc['kre'], "%", ".1f", "Canario en la mina"),
        },
        "yield_curve": {
            "y3m":  y3m, "y2y": y2, "y10y": y10, "y30y": y30,
            "spread_10y2y": round(sp_10y2y, 2) if sp_10y2y else None,
            "spread_10y3m": round(sp_10y3m, 2) if sp_10y3m else None,
            "score_10y3m":  sc['spread_10y3m'],
            "score_10y2y":  score_spread_10y2y(sp_10y2y),
            "inverted_10y3m": (sp_10y3m or 0) < 0,
            "inverted_10y2y": (sp_10y2y or 0) < 0,
        },
        "labor_market": {
            "sahm_rule":       ind("Sahm Rule", sahm_val, sc['sahm'], "", ".2f",
                                   "≥0.5 = recesión activa (todas las recesiones desde 1970)"),
            "sahm_triggered":  (sahm_val or 0) >= 0.5,
            "unemployment":    ind("Tasa desempleo (%)", unrate_val, score_unrate(unrate_val), "%", ".1f", "BLS, abril 2026"),
            "payrolls_mom":    ind("Nóminas MoM (miles)", pay_mom, sc['payrolls'], "K", "+.0f", "BLS nonfarm payrolls"),
            "initial_claims":  ind("Peticiones desempleo (semanales)", icsa_val,
                                   score_val(icsa_val, [220000, 250000, 280000, 320000], [10, 25, 50, 75, 90]),
                                   "", ",.0f", "FRED ICSA — >300K = deterioro") if icsa_val else None,
            "quits_raw":       ind("JOLTS renuncias (miles)", quits_val, 
                                   score_val(quits_val, [700, 800, 900, 1000], [80, 50, 25, 10, 5]),
                                   "K", ".0f", "Bajo = trabajadores sin confianza"),
            "temp_employment": ind("Temp employment YoY (%)", temphelp_yoy,
                                   score_val(temphelp_yoy, [-5, -2, 0, 2, 5], [85, 60, 40, 15, 5, 3]),
                                   "%", ".1f", "Lidera nóminas 3-6M") if temphelp_yoy else None,
        },
        "consumer_signals": {
            "savings_rate":    ind("Tasa ahorro personal (%)", savings_val, sc['savings'], "%", ".1f", "<3% = buffer agotado"),
            "umich_sentiment": ind("UMich Consumer Sentiment", umich_val, sc['umich'], "", ".0f", "<58 = recesión inminente"),
            "oil_rise":        ind("Petróleo vs mínimo 52S (%)", oil_rise, sc['oil_rise'], "%", "+.0f",
                                   "Hartnett: >90% = señal recesión"),
            "oil_price":       ind("Petróleo WTI ($)", oil_val, 
                                   score_val(oil_val, [70, 85, 100, 120], [5, 15, 35, 65, 85]),
                                   "$", ".1f", ""),
        },
        "nber_coincident": {
            "payrolls_mom":     ind("Nóminas MoM (miles)", pay_mom, sc['payrolls'], "K", "+.0f", "NBER indicator #1"),
            "indpro_mom":       ind("Producción industrial MoM (%)", indpro_mom,
                                    score_val(indpro_mom, [-1.0, -0.3, 0, 0.3, 1.0], [90, 60, 35, 15, 5, 3]),
                                    "%", ".2f", "NBER indicator #2") if indpro_mom else None,
            "sahm_rule":        ind("Sahm Rule", sahm_val, sc['sahm'], "", ".2f", "NBER proxy más rápido"),
            "unemployment_lvl": ind("Desempleo (%)", unrate_val, score_unrate(unrate_val), "%", ".1f", "NBER indicator"),
            "recession_declared": False,
            "nber_note": "El NBER tarda de media 11 meses en declarar recesión. La Sahm Rule es la señal más rápida disponible en tiempo real."
        },
        "oil_breakdown": {
            "price": round(oil_val, 1) if oil_val else None,
            "52w_low": round(oil_52l, 1) if oil_52l else None,
            "rise_from_low_pct": round(oil_rise, 1) if oil_rise else None,
            "recession_trigger_90pct": (oil_rise or 0) >= 90,
            "above_100": (oil_val or 0) >= 100,
        },
    }

    # ══════════════════════════════════════════════════════════════════════════
    # BUILD data.json (EXISTING — enhanced)
    # ══════════════════════════════════════════════════════════════════════════

    buy_signals = 0
    checklist = {}

    def ck(name, condition):
        nonlocal buy_signals
        active = bool(condition)
        checklist[name] = active
        if active: buy_signals += 1

    ck("VIX > 35 (pánico mercado)",           (vix_val or 0) > 35)
    ck("S&P drawdown > 15%",                   (spy_dd or 0) < -15)
    ck("Curva 10Y-3M invertida",               (sp_10y3m or 0) < 0)
    ck("Bono 30Y > 5% (bubble killer activo)", (y30 or 0) >= 5.0)
    ck("Sahm Rule activada (≥ 0.5)",           (sahm_val or 0) >= 0.5)
    ck("HY spreads en estrés",                 (hy_stress or 0) > 1.5)
    ck("Oil > $100",                           (oil_val or 0) >= 100)
    ck("Petróleo +90% desde mínimos 52S",      (oil_rise or 0) >= 90)

    action_text = (
        "DESPLEGAR capital — múltiples señales de suelo" if buy_signals >= 5
        else "ACUMULAR por tramos — condiciones mejorando" if buy_signals >= 3
        else "MANTENER liquidez — esperar capitulación"
    )

    # Sector performance
    sector_tickers = {
        'Tech': 'XLK', 'Financials': 'XLF', 'Energy': 'XLE',
        'Materials': 'XLB', 'Health': 'XLV', 'Utilities': 'XLU',
        'Industrials': 'XLI', 'Cons.Disc': 'XLY', 'Cons.Stap': 'XLP',
    }
    sector_perf = {}
    for name, tkr in sector_tickers.items():
        s = dl(tkr, '90d')
        if s is not None:
            sector_perf[name] = {
                '1M': round(pct_chg(s, 21) or 0, 2),
                '3M': round(pct_chg(s, 63) or 0, 2),
            }

    data_json = {
        "timestamp": TS,
        "composite_score": comp / 100,
        "risk_label": comp_label,
        "risk_color": comp_color,
        "active_buy_signals": buy_signals,
        "action": action_text,
        "buy_checklist": checklist,
        "metrics": {
            "sp500":           round(spy_now, 0) if spy_now else None,
            "sp500_drawdown":  round(spy_dd, 1) if spy_dd else None,
            "vix":             round(vix_val, 1) if vix_val else None,
            "oil":             round(oil_val, 1) if oil_val else None,
            "bond_30y":        y30,
            "yield_curve":     round(sp_10y2y, 2) if sp_10y2y else None,
            "yield_curve_10y3m": round(sp_10y3m, 2) if sp_10y3m else None,
            "pct_above_200":   None,  # requires breadth data
            "mcclellan":       round(mcclellan_proxy, 1) if mcclellan_proxy else None,
            "dxy":             round(dxy_val, 1) if dxy_val else None,
            "sox_vs_200dma":   round(sox_bubble, 1) if sox_bubble else None,
            "sahm_rule":       round(sahm_val, 2) if sahm_val else None,
            "unemployment":    unrate_val,
            "oil_rise_pct":    round(oil_rise, 1) if oil_rise else None,
        },
        "sector_performance": sector_perf,
        "composite_recession_score": comp,
    }

    # ══════════════════════════════════════════════════════════════════════════
    # BUILD harnett_data.json (EXISTING — enhanced with live data)
    # ══════════════════════════════════════════════════════════════════════════

    def pain(label, value, threshold, lower_is_bad=True):
        if value is None: return {"label": label, "value": None, "threshold": threshold, "active": False}
        active = (value > threshold) if not lower_is_bad else (value < threshold)
        return {"label": label, "value": round(value, 2), "threshold": threshold, "active": active}

    pain_levels = {
        "vix35":      pain("VIX > 35", vix_val, 35, lower_is_bad=False),
        "spy15dd":    pain("S&P -15% drawdown", spy_dd, -15, lower_is_bad=True),
        "curva_inv":  pain("Curva 10Y-3M invertida", sp_10y3m, 0, lower_is_bad=True),
        "bono30y":    pain("30Y Bono > 5%", y30, 5.0, lower_is_bad=False),
        "sahm05":     pain("Sahm Rule ≥ 0.5", sahm_val, 0.5, lower_is_bad=False),
        "oil100":     pain("Petróleo > $100", oil_val, 100, lower_is_bad=False),
    }
    n_pain = sum(1 for p in pain_levels.values() if p['active'])

    harnett_danger = comp / 100
    harnett_label  = comp_label

    # Credit composite (reuse hy_stress and kre)
    credit_stress_components = [
        (hy_stress or 0) > 1.0,     # HY underperforming
        (kre_3m or 0) < -10,        # banks weak
        (bkln_3m or 0) < -3,        # leveraged loans
        (sp_10y3m or 0) < 0,        # curve inverted
        (oil_val or 0) > 100 and (hy_stress or 0) > 0.5,  # 2008 parallel
    ]
    credit_composite = sum(1 for x in credit_stress_components if x) / len(credit_stress_components) * 100

    harnett_json = {
        "timestamp": TS,
        "pain_levels": pain_levels,
        "n_pain_active": n_pain,
        "harnett_danger": harnett_danger,
        "harnett_label": harnett_label,
        "contrarian_buy": n_pain >= 3 and credit_composite > 40,
        "credit_composite": round(credit_composite, 1),
        "credit_label": "STRESS" if credit_composite > 60 else "ELEVATED" if credit_composite > 35 else "OK",
        "metrics": {
            "vix": round(vix_val, 1) if vix_val else None,
            "oil": round(oil_val, 1) if oil_val else None,
            "dxy": round(dxy_val, 1) if dxy_val else None,
            "bond_30y": y30,
            "sox_vs_200dma": round(sox_bubble, 1) if sox_bubble else None,
        },
        "credit": {
            "hy_spread_zscore": round(hy_stress, 2) if hy_stress else None,
            "hy_spread_status": "STRESS" if (hy_stress or 0) > 2 else "ELEVATED" if (hy_stress or 0) > 0.5 else "OK",
            "banks_3m": round(kre_3m, 1) if kre_3m else None,
            "banks_status": "ALERT" if (kre_3m or 0) < -15 else "WEAK" if (kre_3m or 0) < -5 else "OK",
            "bank_loans_1m": round(bkln_3m, 1) if bkln_3m else None,
            "bank_loans_status": "STRESS" if (bkln_3m or 0) < -5 else "WEAK" if (bkln_3m or 0) < -2 else "OK",
            "yield_curve": round(sp_10y3m, 2) if sp_10y3m else None,
            "yield_curve_status": "INVERTED" if (sp_10y3m or 0) < 0 else "FLAT" if (sp_10y3m or 0) < 0.5 else "OK",
        },
    }

    # ══════════════════════════════════════════════════════════════════════════
    # BUILD credit_data.json (EXISTING)
    # ══════════════════════════════════════════════════════════════════════════

    sigs = {
        "hy_divergence": {
            "status": "WARNING" if (hy_stress or 0) > 0.5 else "OK",
            "diverging": (hy_stress or 0) > 0.5,
            "sp500_1m": round(pct_chg(spy_s, 21) or 0, 1),
            "hyg_1m":   round(pct_chg(hyg_s, 21) or 0, 1),
        },
        "cds_proxy": {"status": "WARNING" if (hy_stress or 0) > 1.5 else "OK"},
        "bank_loans": {
            "status": "STRESS" if (bkln_3m or 0) < -5 else "WEAK" if (bkln_3m or 0) < -2 else "OK",
            "bkln_1m": round(pct_chg(bkln_s, 21) or 0, 1),
            "bkln_3m": round(bkln_3m or 0, 1),
            "bkln_drawdown": round(bkln_dd or 0, 1),
        },
        "oil_trigger": {
            "status": "DANGER" if (oil_rise or 0) >= 90 else "WARNING" if (oil_rise or 0) >= 60 else "OK",
            "price": round(oil_val or 0, 1),
            "rise_from_low": round(oil_rise or 0, 1),
            "above_100": (oil_val or 0) >= 100,
        },
        "financials": {
            "status": "ALERT" if (kre_3m or 0) < -15 else "WEAK" if (kre_3m or 0) < -5 else "OK",
            "kre_3m": round(kre_3m or 0, 1),
            "xlf_3m": round(xlf_3m or 0, 1),
            "insurance_1m": 0,
        },
        "tech_debt": {"status": "WARNING"},
    }
    n_danger  = sum(1 for s in sigs.values() if s.get('status') == 'DANGER')
    n_warning = sum(1 for s in sigs.values() if s.get('status') in ('WARNING', 'ALERT', 'WEAK'))
    n_ok      = len(sigs) - n_danger - n_warning
    credit_stress_score = n_danger * 25 + n_warning * 12
    credit_json = {
        "timestamp": TS,
        "credit_stress_score": min(credit_stress_score, 100),
        "credit_stress_label": "DANGER" if credit_stress_score >= 50 else "WARNING" if credit_stress_score >= 25 else "OK",
        "n_danger": n_danger, "n_warning": n_warning, "n_ok": n_ok,
        "parallel_2008": (oil_val or 0) > 90 and (hy_stress or 0) > 0.5,
        "signals": sigs,
    }

    # ══════════════════════════════════════════════════════════════════════════
    # WRITE JSON FILES
    # ══════════════════════════════════════════════════════════════════════════

    OUTPUT_DIR = "docs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outputs = {
        "data.json":             data_json,
        "harnett_data.json":     harnett_json,
        "credit_data.json":      credit_json,
        "macro_indicators.json": macro,
    }
    for fname, payload in outputs.items():
        path = os.path.join(OUTPUT_DIR, fname)
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"  ✓ {path}")

    print(f"\n── COMPOSITE RECESSION SCORE: {comp}/100 — {comp_label} ──")
    print(f"   SOX vs 200dma: {sox_bubble:+.1f}%")
    print(f"   30Y yield: {y30:.2f}% {'⚠ BUBBLE KILLER ACTIVO' if (y30 or 0)>=5 else ''}")
    print(f"   Oil: ${oil_val:.1f} (+{oil_rise:.0f}% desde mínimos)")
    print(f"   Sahm Rule: {sahm_val:.2f}" if sahm_val else "   Sahm Rule: N/A")
    print(f"   VIX: {vix_val:.1f}")
    print(f"   Buy signals: {buy_signals}/8")

if __name__ == '__main__':
    main()
