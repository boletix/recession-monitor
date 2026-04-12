"""
CREDIT STRESS MONITOR v1.0
Private Credit & Systemic Risk Signals
Based on Harnett/Cárpatos framework

Monitors:
1. High-Yield bond behavior (HYG, JNK) vs equities
2. Credit Default Swap proxies (via ETFs)
3. Bank loan stress (BKLN)
4. Oil as inflation/recession trigger
5. Financial sector health (banks, insurers)
6. Refinancing wall indicators
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
import warnings
import json
import os

warnings.filterwarnings("ignore")

FROM_DATE = "2020-01-01"
TODAY = datetime.today().date().strftime("%Y-%m-%d")
OUTPUT_DIR = "docs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def safe_last(series, default=0.0):
    if series is None or len(series) == 0:
        return default
    val = series.iloc[-1]
    if isinstance(val, (pd.Series, pd.DataFrame)):
        val = val.iloc[0]
    val = float(val)
    return default if (np.isnan(val) or np.isinf(val)) else val

def extract(batch, ticker):
    try:
        if isinstance(batch.columns, pd.MultiIndex):
            if ticker in batch['Close'].columns:
                return batch['Close'][ticker].dropna()
        return pd.Series(dtype=float)
    except:
        return pd.Series(dtype=float)

def sanitize(obj):
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize(v) for v in obj]
    elif isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else round(obj, 4)
    elif isinstance(obj, (np.floating, np.integer)):
        val = float(obj)
        return None if (np.isnan(val) or np.isinf(val)) else round(val, 4)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj

print("=" * 60)
print("  CREDIT STRESS MONITOR v1.0")
print("=" * 60)

# ═══════════════════════════════════════
# DOWNLOAD
# ═══════════════════════════════════════
print("\n[1/3] Downloading data...")

tickers = [
    '^GSPC',       # S&P 500
    'HYG',         # iShares HY Bond ETF
    'JNK',         # SPDR HY Bond ETF
    'LQD',         # Investment Grade Bond ETF
    'BKLN',        # Invesco Senior Loan ETF (bank loans / leveraged loans)
    'SRLN',        # SPDR Blackstone Senior Loan ETF
    'EMB',         # EM bonds
    'TLT',         # Long Treasury (flight to safety)
    'SHY',         # Short Treasury
    'CL=F',        # Oil
    '^TNX',        # 10Y yield
    '^TYX',        # 30Y yield
    'XLF',         # Financials
    'KRE',         # Regional Banks
    'KIE',         # Insurance ETF
    'IGV',         # Software ETF (refinancing wall proxy)
    'WCLD',        # Cloud computing (tech debt proxy)
    '^VIX',        # VIX
]

raw = yf.download(tickers, start=FROM_DATE, end=TODAY, progress=False, timeout=30, threads=True)

sp500 = extract(raw, '^GSPC')
hyg = extract(raw, 'HYG')
jnk = extract(raw, 'JNK')
lqd = extract(raw, 'LQD')
bkln = extract(raw, 'BKLN')
srln = extract(raw, 'SRLN')
emb = extract(raw, 'EMB')
tlt = extract(raw, 'TLT')
shy = extract(raw, 'SHY')
oil = extract(raw, 'CL=F')
tnx = extract(raw, '^TNX')
tyx = extract(raw, '^TYX')
xlf = extract(raw, 'XLF')
kre = extract(raw, 'KRE')
kie = extract(raw, 'KIE')
igv = extract(raw, 'IGV')
wcld = extract(raw, 'WCLD')
vix = extract(raw, '^VIX')

print(f"  HYG: {len(hyg)}d | BKLN: {len(bkln)}d | KRE: {len(kre)}d | Oil: {len(oil)}d")

# ═══════════════════════════════════════
# COMPUTE CREDIT SIGNALS
# ═══════════════════════════════════════
print("[2/3] Computing credit stress signals...")

signals = {}

# ── SIGNAL 1: HY Bond divergence from equities ──
# If S&P rallies but HYG/JNK don't follow = credit stress
if len(hyg) > 20 and len(sp500) > 20:
    # Normalize both to 100 from 20 days ago
    sp_norm = sp500 / sp500.iloc[-21] * 100
    hyg_norm = hyg / hyg.iloc[-21] * 100
    divergence_1m = float(sp_norm.iloc[-1] - hyg_norm.iloc[-1])

    # 3-month divergence
    if len(hyg) > 63 and len(sp500) > 63:
        sp_norm3 = sp500 / sp500.iloc[-63] * 100
        hyg_norm3 = hyg / hyg.iloc[-63] * 100
        divergence_3m = float(sp_norm3.iloc[-1] - hyg_norm3.iloc[-1])
    else:
        divergence_3m = 0

    # HYG performance
    hyg_1m = float(hyg.iloc[-1] / hyg.iloc[-21] - 1) * 100
    hyg_3m = float(hyg.iloc[-1] / hyg.iloc[-63] - 1) * 100 if len(hyg) > 63 else 0

    # Is HY rebounding with equities?
    sp_1m = float(sp500.iloc[-1] / sp500.iloc[-21] - 1) * 100
    hy_rebounding = hyg_1m > 0 and sp_1m > 0
    hy_diverging = sp_1m > 1 and hyg_1m < 0  # S&P up but HY down = BAD

    signals['hy_divergence'] = {
        'divergence_1m': round(divergence_1m, 2),
        'divergence_3m': round(divergence_3m, 2),
        'hyg_1m': round(hyg_1m, 2),
        'hyg_3m': round(hyg_3m, 2),
        'sp500_1m': round(sp_1m, 2),
        'rebounding': hy_rebounding,
        'diverging': hy_diverging,
        'status': 'DANGER' if hy_diverging else 'WARNING' if not hy_rebounding and sp_1m > 0 else 'OK',
    }

# ── SIGNAL 2: CDS proxy - HY spread widening ──
# LQD/HYG ratio = IG outperforming HY = credit stress rising
if len(lqd) > 20 and len(hyg) > 20:
    ci = lqd.index.intersection(hyg.index)
    lqd_c = lqd[~lqd.index.duplicated(keep='last')]
    hyg_c = hyg[~hyg.index.duplicated(keep='last')]
    ci = lqd_c.index.intersection(hyg_c.index)
    spread_ratio = lqd_c.loc[ci] / hyg_c.loc[ci]
    spread_ratio = spread_ratio[~spread_ratio.index.duplicated(keep='last')]

    spread_current = safe_last(spread_ratio)
    spread_mean = float(spread_ratio.mean())
    spread_std = float(spread_ratio.std())
    spread_zscore = (spread_current - spread_mean) / spread_std if spread_std > 0 else 0

    # Is spread widening (trending up)?
    spread_20d = float(spread_ratio.iloc[-20:].mean()) if len(spread_ratio) > 20 else spread_current
    spread_60d = float(spread_ratio.iloc[-60:].mean()) if len(spread_ratio) > 60 else spread_current
    widening = spread_20d > spread_60d

    signals['cds_proxy'] = {
        'spread_zscore': round(spread_zscore, 2),
        'widening': widening,
        'status': 'DANGER' if spread_zscore > 2.0 else 'WARNING' if spread_zscore > 1.0 or widening else 'OK',
    }

# ── SIGNAL 3: Bank loan stress (BKLN) ──
# Leveraged loans = canary in credit mine
if len(bkln) > 20:
    bkln_1m = float(bkln.iloc[-1] / bkln.iloc[-21] - 1) * 100
    bkln_3m = float(bkln.iloc[-1] / bkln.iloc[-63] - 1) * 100 if len(bkln) > 63 else 0
    bkln_ath = float(bkln.expanding().max().iloc[-1])
    bkln_dd = float((bkln.iloc[-1] / bkln_ath - 1) * 100)

    signals['bank_loans'] = {
        'bkln_1m': round(bkln_1m, 2),
        'bkln_3m': round(bkln_3m, 2),
        'bkln_drawdown': round(bkln_dd, 2),
        'status': 'DANGER' if bkln_1m < -3 or bkln_dd < -8 else 'WARNING' if bkln_1m < -1 or bkln_dd < -4 else 'OK',
    }

# ── SIGNAL 4: Oil as recession trigger ──
if len(oil) > 20:
    oil_val = safe_last(oil)
    # Calculate how much oil has risen from recent low
    oil_52w_low = float(oil.iloc[-252:].min()) if len(oil) > 252 else float(oil.min())
    oil_rise_pct = (oil_val / oil_52w_low - 1) * 100

    signals['oil_trigger'] = {
        'price': round(oil_val, 1),
        'rise_from_low': round(oil_rise_pct, 1),
        'above_100': oil_val > 100,
        'recession_threshold': oil_rise_pct > 90,  # 90%+ rise historically = recession
        'status': 'DANGER' if oil_rise_pct > 90 else 'WARNING' if oil_val > 100 or oil_rise_pct > 50 else 'OK',
    }

# ── SIGNAL 5: Financial sector health ──
if len(kre) > 20 and len(xlf) > 20:
    kre_1m = float(kre.iloc[-1] / kre.iloc[-21] - 1) * 100
    kre_3m = float(kre.iloc[-1] / kre.iloc[-63] - 1) * 100 if len(kre) > 63 else 0
    xlf_1m = float(xlf.iloc[-1] / xlf.iloc[-21] - 1) * 100
    xlf_3m = float(xlf.iloc[-1] / xlf.iloc[-63] - 1) * 100 if len(xlf) > 63 else 0

    # Insurance exposure
    kie_1m = float(kie.iloc[-1] / kie.iloc[-21] - 1) * 100 if len(kie) > 21 else 0

    # Banks underperforming = credit canary
    bank_weak = kre_3m < -10 or xlf_3m < -8

    signals['financials'] = {
        'kre_1m': round(kre_1m, 2),
        'kre_3m': round(kre_3m, 2),
        'xlf_1m': round(xlf_1m, 2),
        'xlf_3m': round(xlf_3m, 2),
        'insurance_1m': round(kie_1m, 2),
        'banks_weak': bank_weak,
        'status': 'DANGER' if kre_3m < -15 else 'WARNING' if bank_weak else 'OK',
    }

# ── SIGNAL 6: Tech debt / refinancing wall proxy ──
if len(igv) > 20:
    igv_1m = float(igv.iloc[-1] / igv.iloc[-21] - 1) * 100
    igv_3m = float(igv.iloc[-1] / igv.iloc[-63] - 1) * 100 if len(igv) > 63 else 0
    wcld_1m = float(wcld.iloc[-1] / wcld.iloc[-21] - 1) * 100 if len(wcld) > 21 else 0
    wcld_3m = float(wcld.iloc[-1] / wcld.iloc[-63] - 1) * 100 if len(wcld) > 63 else 0

    signals['tech_debt'] = {
        'software_1m': round(igv_1m, 2),
        'software_3m': round(igv_3m, 2),
        'cloud_1m': round(wcld_1m, 2),
        'cloud_3m': round(wcld_3m, 2),
        'status': 'DANGER' if igv_3m < -20 or wcld_3m < -25 else 'WARNING' if igv_3m < -10 else 'OK',
    }

# ── COMPOSITE CREDIT STRESS SCORE ──
stress_points = 0
stress_max = 0

def add_stress(status, weight=1):
    global stress_points, stress_max
    stress_max += weight * 100
    if status == 'DANGER':
        stress_points += weight * 100
    elif status == 'WARNING':
        stress_points += weight * 50

for key, sig in signals.items():
    w = 1.5 if key in ['hy_divergence', 'cds_proxy', 'bank_loans'] else 1.0
    add_stress(sig['status'], w)

credit_stress_score = (stress_points / stress_max * 100) if stress_max > 0 else 0
credit_stress_label = 'LOW' if credit_stress_score < 25 else 'MODERATE' if credit_stress_score < 50 else 'ELEVATED' if credit_stress_score < 75 else 'CRITICAL'

# Count danger/warning signals
n_danger = sum(1 for s in signals.values() if s['status'] == 'DANGER')
n_warning = sum(1 for s in signals.values() if s['status'] == 'WARNING')
n_ok = sum(1 for s in signals.values() if s['status'] == 'OK')

# 2008 parallel check
parallel_2008 = (
    signals.get('hy_divergence', {}).get('diverging', False) and
    signals.get('oil_trigger', {}).get('above_100', False) and
    signals.get('cds_proxy', {}).get('widening', False)
)

print(f"\n  Credit Stress Score: {credit_stress_score:.0f}/100 ({credit_stress_label})")
print(f"  Signals: {n_danger} DANGER / {n_warning} WARNING / {n_ok} OK")
print(f"  2008 Parallel: {'ACTIVE' if parallel_2008 else 'Not triggered'}")

# ═══════════════════════════════════════
# GENERATE CHART
# ═══════════════════════════════════════
print("\n[3/3] Generating credit dashboard...")

BG = '#0a0a0f'
PANEL = '#12121a'
TXT = '#e0e0e0'
GRN, RED, YEL, BLU, PUR, ORA, GRD = '#00d68f', '#ff4d4d', '#ffd93d', '#4da8ff', '#b87aff', '#ff8c42', '#1a1a2e'

def style_ax(ax, title=""):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TXT, labelsize=7)
    ax.set_title(title, color=TXT, fontsize=10, fontweight='bold', pad=8)
    for s in ax.spines.values(): s.set_color(GRD)
    ax.grid(True, color=GRD, alpha=0.3, linewidth=0.5)

fig = plt.figure(figsize=(24, 22), facecolor=BG)
gs = gridspec.GridSpec(4, 2, hspace=0.38, wspace=0.25, left=0.06, right=0.94, top=0.93, bottom=0.04)
fig.suptitle('CREDIT STRESS MONITOR', color=RED, fontsize=18, fontweight='bold', family='monospace', y=0.97)

# P1: HYG vs S&P 500 (divergence)
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, "Signal 1: HY Bonds (HYG) vs S&P 500 — Divergence = Danger")
if len(hyg) > 0 and len(sp500) > 0:
    start = max(hyg.index[0], sp500.index[0])
    h = hyg.loc[start:] / hyg.loc[start:].iloc[0] * 100
    s = sp500.loc[start:] / sp500.loc[start:].iloc[0] * 100
    ax1.plot(s, color=BLU, linewidth=1, label='S&P 500 (norm)')
    ax1.plot(h, color=ORA, linewidth=1, label='HYG (norm)')
    ax1.legend(fontsize=7, facecolor=PANEL, edgecolor=GRD, labelcolor=TXT)

# P2: Credit spread (LQD/HYG)
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, "Signal 2: Credit Spread Proxy (LQD/HYG) — Rising = Stress")
if 'cds_proxy' in signals and len(spread_ratio) > 0:
    ax2.plot(spread_ratio, color=RED, linewidth=1)
    ax2.axhline(spread_mean, color=TXT, linewidth=0.5, linestyle='--', label='Mean')
    ax2.axhline(spread_mean + 1.5 * spread_std, color=RED, linewidth=0.8, linestyle='--', label='+1.5 std')
    ax2.axhline(spread_mean + 2.0 * spread_std, color=RED, linewidth=0.8, linestyle='--', alpha=0.5, label='+2.0 std')
    ax2.fill_between(spread_ratio.index, spread_ratio.values, spread_mean + 1.5 * spread_std,
                      where=spread_ratio > spread_mean + 1.5 * spread_std, color=RED, alpha=0.2)
    ax2.legend(fontsize=7, facecolor=PANEL, edgecolor=GRD, labelcolor=TXT)

# P3: Bank Loans (BKLN)
ax3 = fig.add_subplot(gs[1, 0])
style_ax(ax3, "Signal 3: Bank Loans (BKLN) — Leveraged Loan Stress")
if len(bkln) > 0:
    ax3.plot(bkln, color=PUR, linewidth=1.2)
    bkln_200 = bkln.rolling(200).mean()
    ax3.plot(bkln_200, color=YEL, linewidth=0.6, alpha=0.6, label='200 MA')
    ax3.legend(fontsize=7, facecolor=PANEL, edgecolor=GRD, labelcolor=TXT)

# P4: Oil recession trigger
ax4 = fig.add_subplot(gs[1, 1])
style_ax(ax4, "Signal 4: Oil — 90%+ Rise from Low = Recession Trigger")
if len(oil) > 0:
    ax4.plot(oil, color=ORA, linewidth=1.2)
    ax4.axhline(100, color=RED, linestyle='--', linewidth=1, label='Pain: 100')
    # Show the 52w low level
    if len(oil) > 252:
        low_val = float(oil.iloc[-252:].min())
        recession_level = low_val * 1.9  # 90% rise
        ax4.axhline(recession_level, color=RED, linestyle=':', linewidth=0.8, alpha=0.6, label=f'90% rise: {recession_level:.0f}')
    ax4.fill_between(oil.index, oil.values, 100, where=oil > 100, color=RED, alpha=0.15)
    ax4.legend(fontsize=7, facecolor=PANEL, edgecolor=GRD, labelcolor=TXT)

# P5: Banks (KRE vs XLF)
ax5 = fig.add_subplot(gs[2, 0])
style_ax(ax5, "Signal 5: Banks — KRE Regional vs XLF Financials vs KIE Insurance")
if len(kre) > 0:
    start = kre.index[-min(252, len(kre))]
    kn = kre.loc[start:] / kre.loc[start:].iloc[0] * 100
    xn = xlf.loc[start:] / xlf.loc[start:].iloc[0] * 100 if len(xlf) > 0 else pd.Series(dtype=float)
    kien = kie.loc[start:] / kie.loc[start:].iloc[0] * 100 if len(kie) > 0 else pd.Series(dtype=float)
    ax5.plot(kn, color=RED, linewidth=1.2, label='KRE Regional Banks')
    if len(xn) > 0: ax5.plot(xn, color=BLU, linewidth=1, alpha=0.7, label='XLF Financials')
    if len(kien) > 0: ax5.plot(kien, color=PUR, linewidth=1, alpha=0.7, label='KIE Insurance')
    ax5.axhline(100, color=TXT, linewidth=0.5, linestyle='--')
    ax5.legend(fontsize=7, facecolor=PANEL, edgecolor=GRD, labelcolor=TXT)

# P6: Tech / Software (refinancing wall proxy)
ax6 = fig.add_subplot(gs[2, 1])
style_ax(ax6, "Signal 6: Tech Debt Proxy — Software (IGV) & Cloud (WCLD)")
if len(igv) > 0:
    start = igv.index[-min(252, len(igv))]
    in_ = igv.loc[start:] / igv.loc[start:].iloc[0] * 100
    ax6.plot(in_, color=BLU, linewidth=1.2, label='IGV Software')
    if len(wcld) > 0:
        wn = wcld.loc[start:] / wcld.loc[start:].iloc[0] * 100
        ax6.plot(wn, color=PUR, linewidth=1, alpha=0.7, label='WCLD Cloud')
    ax6.axhline(100, color=TXT, linewidth=0.5, linestyle='--')
    ax6.legend(fontsize=7, facecolor=PANEL, edgecolor=GRD, labelcolor=TXT)

# P7: Summary
ax7 = fig.add_subplot(gs[3, :])
ax7.set_facecolor(PANEL)
for s in ax7.spines.values(): s.set_color(GRD)
ax7.axis('off')

sc_color = GRN if credit_stress_score < 25 else YEL if credit_stress_score < 50 else ORA if credit_stress_score < 75 else RED
summary_text = (
    f"  CREDIT STRESS: {credit_stress_score:.0f}/100 ({credit_stress_label})  |  "
    f"DANGER: {n_danger}  WARNING: {n_warning}  OK: {n_ok}  |  "
    f"2008 PARALLEL: {'ACTIVE' if parallel_2008 else 'No'}  |  "
    f"HYG 1M: {signals.get('hy_divergence',{}).get('hyg_1m',0):.1f}%  |  "
    f"BKLN 1M: {signals.get('bank_loans',{}).get('bkln_1m',0):.1f}%  |  "
    f"Oil: {safe_last(oil):.0f}  Rise: {signals.get('oil_trigger',{}).get('rise_from_low',0):.0f}%"
)
ax7.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=9, color=TXT, family='monospace', transform=ax7.transAxes)

output_png = os.path.join(OUTPUT_DIR, "credit_monitor.png")
fig.savefig(output_png, dpi=150, facecolor=BG, bbox_inches='tight')
plt.close('all')
print(f"  Saved: {output_png}")

# ═══════════════════════════════════════
# EXPORT JSON
# ═══════════════════════════════════════
export = sanitize({
    'timestamp': datetime.now().isoformat(),
    'credit_stress_score': credit_stress_score,
    'credit_stress_label': credit_stress_label,
    'n_danger': n_danger,
    'n_warning': n_warning,
    'n_ok': n_ok,
    'parallel_2008': parallel_2008,
    'signals': signals,
})

json_path = os.path.join(OUTPUT_DIR, "credit_data.json")
with open(json_path, 'w') as f:
    json.dump(export, f, indent=2, default=str)

print(f"  Saved: {json_path}")
print(f"\n{'='*60}")
print(f"  CREDIT STRESS: {credit_stress_score:.0f}/100 ({credit_stress_label})")
print(f"  SIGNALS: {n_danger} DANGER / {n_warning} WARNING / {n_ok} OK")
print(f"  2008 PARALLEL: {'ACTIVE' if parallel_2008 else 'No'}")
print(f"{'='*60}")
