"""
HARNETT PAIN LEVELS MONITOR v1.0
4 Political Pain Thresholds + Credit Stress
Based on Michael Harnett (BofA) framework via Cárpatos

Monitors:
1. Oil > $100 (ACTIVE pain)
2. Dollar Index > 100
3. 30Y Bond yield > 5%
4. S&P 500 < 6000

Plus credit market health (2008 parallel)
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

# ═══════════════════════════════════════
# PAIN LEVEL THRESHOLDS
# ═══════════════════════════════════════
PAIN_LEVELS = {
    'oil_above_100': {'threshold': 100, 'direction': 'above', 'label': 'Oil > 100'},
    'dxy_above_100': {'threshold': 100, 'direction': 'above', 'label': 'DXY > 100'},
    'bond30y_above_5': {'threshold': 5.0, 'direction': 'above', 'label': '30Y > 5%'},
    'sp500_below_6000': {'threshold': 6000, 'direction': 'below', 'label': 'S&P < 6000'},
}

# ═══════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════
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
print("  HARNETT PAIN LEVELS MONITOR v1.0")
print("=" * 60)

# ═══════════════════════════════════════
# DOWNLOAD DATA
# ═══════════════════════════════════════
print("\n[1/3] Downloading data...")

tickers = [
    '^GSPC',       # S&P 500
    '^VIX',        # VIX
    '^TYX',        # 30Y yield
    '^TNX',        # 10Y yield
    '^IRX',        # 3M yield
    'DX-Y.NYB',   # Dollar Index
    'CL=F',        # WTI Oil
    'GC=F',        # Gold
    'HYG',         # High Yield Bond ETF
    'LQD',         # Investment Grade Bond ETF
    'TLT',         # Long Treasury ETF
    'BIL',         # Short Treasury ETF (cash proxy)
    'KRE',         # Regional Banks ETF
    'XLF',         # Financials ETF
    'FXI',         # China Large Cap ETF
    'MCHI',        # MSCI China ETF
    'XLY',         # Consumer Discretionary
    'IWM',         # Russell 2000 Small Caps
    'BKLN',        # Bank Loans ETF (leveraged loans)
    'JNK',         # SPDR HY Bond
]

raw = yf.download(tickers, start=FROM_DATE, end=TODAY, progress=False, timeout=30, threads=True)

sp500 = extract(raw, '^GSPC')
vix = extract(raw, '^VIX')
tyx = extract(raw, '^TYX')
tnx = extract(raw, '^TNX')
irx = extract(raw, '^IRX')
dxy = extract(raw, 'DX-Y.NYB')
oil = extract(raw, 'CL=F')
gold = extract(raw, 'GC=F')
hyg = extract(raw, 'HYG')
lqd = extract(raw, 'LQD')
tlt = extract(raw, 'TLT')
kre = extract(raw, 'KRE')
xlf = extract(raw, 'XLF')
fxi = extract(raw, 'FXI')
mchi = extract(raw, 'MCHI')
xly = extract(raw, 'XLY')
iwm = extract(raw, 'IWM')
bkln = extract(raw, 'BKLN')
jnk = extract(raw, 'JNK')

print(f"  S&P: {len(sp500)}d | Oil: {len(oil)}d | DXY: {len(dxy)}d | 30Y: {len(tyx)}d")

# ═══════════════════════════════════════
# COMPUTE INDICATORS
# ═══════════════════════════════════════
print("[2/3] Computing Harnett indicators...")

# Current values
sp_val = safe_last(sp500)
oil_val = safe_last(oil)
dxy_val = safe_last(dxy)
tyx_val = safe_last(tyx)
vix_val = safe_last(vix)
gold_val = safe_last(gold)

# Pain level status
pain = {
    'oil': {'value': oil_val, 'threshold': 100, 'active': oil_val > 100, 'label': 'Oil > 100'},
    'dxy': {'value': dxy_val, 'threshold': 100, 'active': dxy_val > 100, 'label': 'DXY > 100'},
    'bond30y': {'value': tyx_val, 'threshold': 5.0, 'active': tyx_val > 5.0, 'label': '30Y Yield > 5%'},
    'sp500': {'value': sp_val, 'threshold': 6000, 'active': sp_val < 6000, 'label': 'S&P 500 < 6000'},
}
active_pain = sum(1 for p in pain.values() if p['active'])

# Credit stress indicators (2008 parallel)
credit = {}

# HY spread proxy: LQD/HYG ratio (rising = stress)
if len(hyg) > 0 and len(lqd) > 0:
    ci = hyg.index.intersection(lqd.index)
    hyg_c = hyg[~hyg.index.duplicated(keep='last')]
    lqd_c = lqd[~lqd.index.duplicated(keep='last')]
    ci = hyg_c.index.intersection(lqd_c.index)
    hy_spread = lqd_c.loc[ci] / hyg_c.loc[ci]
    hy_spread = hy_spread[~hy_spread.index.duplicated(keep='last')]
    hy_z = (safe_last(hy_spread) - float(hy_spread.mean())) / float(hy_spread.std())
    credit['hy_spread_zscore'] = round(hy_z, 2)
    credit['hy_spread_status'] = 'STRESS' if hy_z > 1.5 else 'ELEVATED' if hy_z > 0.5 else 'NORMAL'

# Bank stress: KRE (regional banks) performance
if len(kre) > 0:
    kre_1m = float(kre.iloc[-1] / kre.iloc[-21] - 1) * 100 if len(kre) > 21 else 0
    kre_3m = float(kre.iloc[-1] / kre.iloc[-63] - 1) * 100 if len(kre) > 63 else 0
    credit['banks_1m'] = round(kre_1m, 1)
    credit['banks_3m'] = round(kre_3m, 1)
    credit['banks_status'] = 'ALERT' if kre_3m < -15 else 'WEAK' if kre_3m < -5 else 'OK'

# Bank loans (BKLN) - leveraged loan stress
if len(bkln) > 0:
    bkln_1m = float(bkln.iloc[-1] / bkln.iloc[-21] - 1) * 100 if len(bkln) > 21 else 0
    credit['bank_loans_1m'] = round(bkln_1m, 1)
    credit['bank_loans_status'] = 'STRESS' if bkln_1m < -3 else 'WEAK' if bkln_1m < -1 else 'OK'

# Yield curve inversion (recession signal)
if len(tnx) > 0 and len(irx) > 0:
    tnx_c = tnx[~tnx.index.duplicated(keep='last')]
    irx_c = irx[~irx.index.duplicated(keep='last')]
    ci = tnx_c.index.intersection(irx_c.index)
    yc = tnx_c.loc[ci] - irx_c.loc[ci]
    yc = yc[~yc.index.duplicated(keep='last')]
    credit['yield_curve'] = round(safe_last(yc), 2)
    credit['yield_curve_status'] = 'INVERTED' if safe_last(yc) < 0 else 'FLAT' if safe_last(yc) < 0.5 else 'NORMAL'

# Safety net check (Harnett: no safety net this time)
safety_net = {}

# TLT performance (flight to safety)
if len(tlt) > 0:
    tlt_1m = float(tlt.iloc[-1] / tlt.iloc[-21] - 1) * 100 if len(tlt) > 21 else 0
    safety_net['treasuries_1m'] = round(tlt_1m, 1)

# Gold as safe haven
if len(gold) > 0:
    gold_1m = float(gold.iloc[-1] / gold.iloc[-21] - 1) * 100 if len(gold) > 21 else 0
    gold_ytd = 0
    gold_ytd_start = gold.loc[gold.index >= f'{datetime.now().year}-01-01']
    if len(gold_ytd_start) > 0:
        gold_ytd = float(gold.iloc[-1] / gold_ytd_start.iloc[0] - 1) * 100
    safety_net['gold_1m'] = round(gold_1m, 1)
    safety_net['gold_ytd'] = round(gold_ytd, 1)

# Harnett trade recommendations tracking
trades = {}

# China (FXI) - Harnett says BUY
if len(fxi) > 0:
    fxi_1m = float(fxi.iloc[-1] / fxi.iloc[-21] - 1) * 100 if len(fxi) > 21 else 0
    fxi_3m = float(fxi.iloc[-1] / fxi.iloc[-63] - 1) * 100 if len(fxi) > 63 else 0
    trades['china_fxi'] = {'1m': round(fxi_1m, 1), '3m': round(fxi_3m, 1), 'rec': 'BUY'}

# Consumer Discretionary (XLY) - low-end consumer
if len(xly) > 0:
    xly_1m = float(xly.iloc[-1] / xly.iloc[-21] - 1) * 100 if len(xly) > 21 else 0
    trades['consumer_disc'] = {'1m': round(xly_1m, 1), 'rec': 'FAVOR (low-end)'}

# Small caps (IWM) - Harnett favors
if len(iwm) > 0:
    iwm_1m = float(iwm.iloc[-1] / iwm.iloc[-21] - 1) * 100 if len(iwm) > 21 else 0
    iwm_3m = float(iwm.iloc[-1] / iwm.iloc[-63] - 1) * 100 if len(iwm) > 63 else 0
    trades['small_caps'] = {'1m': round(iwm_1m, 1), '3m': round(iwm_3m, 1), 'rec': 'FAVOR'}

# Banks (XLF) - Harnett says AVOID/SELL
if len(xlf) > 0:
    xlf_1m = float(xlf.iloc[-1] / xlf.iloc[-21] - 1) * 100 if len(xlf) > 21 else 0
    trades['financials'] = {'1m': round(xlf_1m, 1), 'rec': 'AVOID (sell if drops)'}

# Composite credit stress score (0-100)
credit_score = 0
credit_n = 0
if 'hy_spread_zscore' in credit:
    credit_score += min(max(credit['hy_spread_zscore'] / 2 * 100, 0), 100)
    credit_n += 1
if 'banks_3m' in credit:
    credit_score += min(max(-credit['banks_3m'] / 20 * 100, 0), 100)
    credit_n += 1
if 'bank_loans_1m' in credit:
    credit_score += min(max(-credit['bank_loans_1m'] / 5 * 100, 0), 100)
    credit_n += 1
if 'yield_curve' in credit:
    credit_score += 80 if credit['yield_curve'] < 0 else 30 if credit['yield_curve'] < 0.5 else 0
    credit_n += 1

credit_composite = credit_score / credit_n if credit_n > 0 else 0
credit_label = 'LOW' if credit_composite < 25 else 'MODERATE' if credit_composite < 50 else 'ELEVATED' if credit_composite < 75 else 'CRITICAL'

# Overall Harnett signal
# When pain levels + credit stress align = maximum danger (2008 parallel)
harnett_danger = (active_pain / 4) * 0.5 + (credit_composite / 100) * 0.5
harnett_label = 'LOW' if harnett_danger < 0.3 else 'MODERATE' if harnett_danger < 0.5 else 'HIGH' if harnett_danger < 0.75 else 'EXTREME (2008 parallel)'

# Contrarian signal: when ALL pain levels active = time to BUY the dip
contrarian_buy = active_pain >= 3 and credit_composite > 50

print(f"\n  Pain levels active: {active_pain}/4")
print(f"  Credit stress: {credit_composite:.0f}/100 ({credit_label})")
print(f"  Harnett danger: {harnett_danger:.0%} ({harnett_label})")
print(f"  Contrarian BUY signal: {'YES' if contrarian_buy else 'NO'}")

# ═══════════════════════════════════════
# GENERATE CHART
# ═══════════════════════════════════════
print("\n[3/3] Generating Harnett dashboard...")

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

fig = plt.figure(figsize=(24, 28), facecolor=BG)
gs = gridspec.GridSpec(5, 4, hspace=0.38, wspace=0.3, left=0.05, right=0.95, top=0.93, bottom=0.03)

# Title
fig.suptitle('HARNETT PAIN LEVELS & CREDIT MONITOR', color=ORA, fontsize=18, fontweight='bold', family='monospace', y=0.97)

# ── Row 0: Four Pain Level Gauges ──
for i, (key, p) in enumerate(pain.items()):
    ax = fig.add_subplot(gs[0, i])
    ax.set_facecolor(PANEL)
    for s in ax.spines.values(): s.set_color(RED if p['active'] else GRD)
    ax.axis('off')

    status_color = RED if p['active'] else GRN
    icon = "!" if p['active'] else "OK"
    status_text = "ACTIVE" if p['active'] else "INACTIVE"

    ax.text(0.5, 0.78, p['label'], ha='center', va='center', fontsize=13, fontweight='bold',
            color=TXT, transform=ax.transAxes)
    ax.text(0.5, 0.52, f"{p['value']:.1f}", ha='center', va='center', fontsize=32,
            fontweight='bold', color=status_color, family='monospace', transform=ax.transAxes)
    ax.text(0.5, 0.30, f"Threshold: {p['threshold']}", ha='center', va='center',
            fontsize=10, color='#6b7280', transform=ax.transAxes)
    ax.text(0.5, 0.12, status_text, ha='center', va='center', fontsize=14,
            fontweight='bold', color=status_color, transform=ax.transAxes)

    # Border glow effect
    if p['active']:
        rect = plt.Rectangle((0.02, 0.02), 0.96, 0.96, transform=ax.transAxes,
                              fill=False, edgecolor=RED, linewidth=2, linestyle='-')
        ax.add_patch(rect)

# ── Row 1: Oil + DXY charts ──
ax_oil = fig.add_subplot(gs[1, 0:2])
style_ax(ax_oil, "WTI Oil - Pain Level: above 100 = SELL (Harnett)")
if len(oil) > 0:
    ax_oil.plot(oil, color=ORA, linewidth=1.2)
    ax_oil.axhline(100, color=RED, linestyle='--', linewidth=1.5, label='PAIN: 100')
    ax_oil.fill_between(oil.index, oil.values, 100, where=oil > 100, color=RED, alpha=0.2)
    ax_oil.fill_between(oil.index, oil.values, 100, where=oil <= 100, color=GRN, alpha=0.05)
    ax_oil.legend(fontsize=8, facecolor=PANEL, edgecolor=GRD, labelcolor=TXT)

ax_dxy = fig.add_subplot(gs[1, 2:4])
style_ax(ax_dxy, "Dollar Index (DXY) - Pain Level: above 100 = SELL (Harnett)")
if len(dxy) > 0:
    ax_dxy.plot(dxy, color=BLU, linewidth=1.2)
    ax_dxy.axhline(100, color=RED, linestyle='--', linewidth=1.5, label='PAIN: 100')
    ax_dxy.fill_between(dxy.index, dxy.values, 100, where=dxy > 100, color=RED, alpha=0.2)
    ax_dxy.legend(fontsize=8, facecolor=PANEL, edgecolor=GRD, labelcolor=TXT)

# ── Row 2: 30Y Bond + S&P 500 ──
ax_bond = fig.add_subplot(gs[2, 0:2])
style_ax(ax_bond, "30Y Treasury Yield - Pain: above 5% = BUY BONDS (Harnett)")
if len(tyx) > 0:
    ax_bond.plot(tyx, color=PUR, linewidth=1.2)
    ax_bond.axhline(5.0, color=GRN, linestyle='--', linewidth=1.5, label='PAIN/BUY: 5%')
    ax_bond.fill_between(tyx.index, tyx.values, 5.0, where=tyx > 5.0, color=GRN, alpha=0.2)
    ax_bond.legend(fontsize=8, facecolor=PANEL, edgecolor=GRD, labelcolor=TXT)

ax_sp = fig.add_subplot(gs[2, 2:4])
style_ax(ax_sp, "S&P 500 - Pain Level: below 6000 = BUY (Harnett)")
if len(sp500) > 0:
    ax_sp.plot(sp500, color=BLU, linewidth=1.2)
    ax_sp.axhline(6000, color=GRN, linestyle='--', linewidth=1.5, label='PAIN/BUY: 6000')
    ax_sp.fill_between(sp500.index, sp500.values, 6000, where=sp500 < 6000, color=GRN, alpha=0.2)
    if len(sp500) > 200:
        ax_sp.plot(sp500.rolling(200).mean(), color=YEL, linewidth=0.6, alpha=0.5, label='200 MA')
    ax_sp.legend(fontsize=8, facecolor=PANEL, edgecolor=GRD, labelcolor=TXT)

# ── Row 3: Credit Health ──
ax_hy = fig.add_subplot(gs[3, 0:2])
style_ax(ax_hy, "Credit Stress: HY Spread Proxy (LQD/HYG) — Rising = Stress")
if len(hyg) > 0 and len(lqd) > 0:
    ax_hy.plot(hy_spread, color=RED, linewidth=1)
    mean = float(hy_spread.mean())
    std = float(hy_spread.std())
    ax_hy.axhline(mean, color=TXT, linewidth=0.5, alpha=0.5, linestyle='--', label='Mean')
    ax_hy.axhline(mean + 1.5 * std, color=RED, linewidth=0.8, linestyle='--', label='+1.5 std (STRESS)')
    ax_hy.fill_between(hy_spread.index, hy_spread.values, mean + 1.5 * std,
                         where=hy_spread > mean + 1.5 * std, color=RED, alpha=0.2)
    ax_hy.legend(fontsize=7, facecolor=PANEL, edgecolor=GRD, labelcolor=TXT)

ax_banks = fig.add_subplot(gs[3, 2:4])
style_ax(ax_banks, "Banks: KRE (Regional) vs XLF (Financials) — Harnett: AVOID")
if len(kre) > 0 and len(xlf) > 0:
    # Normalize to 100
    kre_norm = kre / kre.iloc[0] * 100
    xlf_norm = xlf / xlf.iloc[0] * 100
    ax_banks.plot(kre_norm, color=RED, linewidth=1.2, label='KRE Regional Banks')
    ax_banks.plot(xlf_norm, color=BLU, linewidth=1, alpha=0.7, label='XLF Financials')
    ax_banks.legend(fontsize=7, facecolor=PANEL, edgecolor=GRD, labelcolor=TXT)

# ── Row 4: Summary ──
ax_sum = fig.add_subplot(gs[4, 0:2])
ax_sum.set_facecolor(PANEL)
for s in ax_sum.spines.values(): s.set_color(GRD)
ax_sum.axis('off')
ax_sum.set_title("HARNETT TRADE RECOMMENDATIONS", color=ORA, fontsize=11, fontweight='bold', pad=8)

recs = [
    ("SELL Oil", f"above 100 (now: {oil_val:.0f})", oil_val > 100, RED if oil_val > 100 else GRN),
    ("SELL Dollar", f"above 100 (now: {dxy_val:.1f})", dxy_val > 100, RED if dxy_val > 100 else GRN),
    ("BUY 30Y Bonds", f"yield > 5% (now: {tyx_val:.2f}%)", tyx_val > 5.0, GRN if tyx_val > 5.0 else '#6b7280'),
    ("BUY China (FXI)", "deflation exit play", True, BLU),
    ("BUY Small Caps", "Trump consumer play", True, BLU),
    ("AVOID Banks", "credit canary", True, YEL),
    ("AVOID Mag 7 Tech", "overowned, vulnerable", True, YEL),
    ("AVOID Bank Loans", "leveraged loan risk", True, YEL),
]

for i, (name, detail, active, color) in enumerate(recs):
    icon = ">" if active else " "
    ax_sum.text(0.03, 0.92 - i * 0.115, f"{icon}  {name}", transform=ax_sum.transAxes,
                fontsize=10, color=color, fontweight='bold', family='monospace')
    ax_sum.text(0.45, 0.92 - i * 0.115, detail, transform=ax_sum.transAxes,
                fontsize=9, color='#6b7280', family='monospace')

# Overall status
ax_status = fig.add_subplot(gs[4, 2:4])
ax_status.set_facecolor(PANEL)
for s in ax_status.spines.values(): s.set_color(GRD)
ax_status.axis('off')
ax_status.set_title("COMPOSITE STATUS", color=ORA, fontsize=11, fontweight='bold', pad=8)

danger_color = GRN if harnett_danger < 0.3 else YEL if harnett_danger < 0.5 else ORA if harnett_danger < 0.75 else RED
credit_color = GRN if credit_composite < 25 else YEL if credit_composite < 50 else ORA if credit_composite < 75 else RED

lines = [
    (f"Pain Levels Active:  {active_pain}/4", RED if active_pain >= 3 else YEL if active_pain >= 2 else GRN, 'bold'),
    (f"Credit Stress:       {credit_composite:.0f}/100 ({credit_label})", credit_color, 'bold'),
    (f"Harnett Danger:      {harnett_danger:.0%} ({harnett_label})", danger_color, 'bold'),
    ("", TXT, 'normal'),
    (f"S&P 500:  {sp_val:.0f}     VIX: {vix_val:.1f}", TXT, 'normal'),
    (f"Oil:      {oil_val:.1f}     DXY: {dxy_val:.1f}", TXT, 'normal'),
    (f"30Y:      {tyx_val:.2f}%    Gold: {gold_val:.0f}", TXT, 'normal'),
    ("", TXT, 'normal'),
    (f"2008 PARALLEL:", RED if credit_composite > 50 and oil_val > 100 else YEL, 'bold'),
    (f"Oil masking credit = {'YES - WATCH CLOSELY' if credit_composite > 40 and oil_val > 90 else 'Not yet'}", RED if credit_composite > 40 and oil_val > 90 else GRN, 'normal'),
    ("", TXT, 'normal'),
    (f"CONTRARIAN BUY: {'ACTIVE - Deploy capital' if contrarian_buy else 'Not yet triggered'}", GRN if contrarian_buy else '#6b7280', 'bold'),
]

for i, (txt, c, fw) in enumerate(lines):
    ax_status.text(0.05, 0.94 - i * 0.076, txt, transform=ax_status.transAxes,
                   fontsize=9, color=c, family='monospace', fontweight=fw)

# Save
output_png = os.path.join(OUTPUT_DIR, "harnett_monitor.png")
fig.savefig(output_png, dpi=150, facecolor=BG, bbox_inches='tight')
plt.close('all')
print(f"  Saved: {output_png}")

# ═══════════════════════════════════════
# EXPORT JSON
# ═══════════════════════════════════════
export = sanitize({
    'timestamp': datetime.now().isoformat(),
    'pain_levels': {k: {'value': v['value'], 'threshold': v['threshold'], 'active': v['active'], 'label': v['label']} for k, v in pain.items()},
    'active_pain_count': active_pain,
    'credit': credit,
    'credit_composite': credit_composite,
    'credit_label': credit_label,
    'harnett_danger': harnett_danger,
    'harnett_label': harnett_label,
    'contrarian_buy': contrarian_buy,
    'safety_net': safety_net,
    'trades': trades,
    'metrics': {
        'sp500': sp_val,
        'vix': vix_val,
        'oil': oil_val,
        'dxy': dxy_val,
        'bond_30y': tyx_val,
        'gold': gold_val,
    },
})

json_path = os.path.join(OUTPUT_DIR, "harnett_data.json")
with open(json_path, 'w') as f:
    json.dump(export, f, indent=2, default=str)

print(f"  Saved: {json_path}")
print(f"\n{'='*60}")
print(f"  PAIN LEVELS: {active_pain}/4")
print(f"  CREDIT STRESS: {credit_composite:.0f}/100 ({credit_label})")
print(f"  HARNETT DANGER: {harnett_danger:.0%} ({harnett_label})")
print(f"  CONTRARIAN BUY: {'YES' if contrarian_buy else 'NO'}")
print(f"{'='*60}")
