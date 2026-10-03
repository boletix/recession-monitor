"""
RECESSION & MARKET TIMING MONITOR v2.0
Lightweight version for GitHub Actions (fast downloads)
Roger's Investment Framework
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

# Thresholds
SP500_DANGER_LEVEL = 6475
VIX_PANIC = 35
BOND_30Y_BUY = 5.0
BREADTH_CAPITULATION = 0.12

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("  RECESSION MONITOR v2.0")
print("=" * 60)

# ═══════════════════════════════════════
# HELPER: safe download + scalar extract
# ═══════════════════════════════════════
def download_single(ticker, start=FROM_DATE):
    """Download a single ticker, return Series."""
    try:
        raw = yf.download(ticker, start=start, end=TODAY, progress=False, timeout=15)
        if raw.empty:
            return pd.Series(dtype=float)
        col = raw['Close']
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        return col.dropna()
    except Exception as e:
        print(f"  ✗ {ticker}: {e}")
        return pd.Series(dtype=float)

def safe_last(series, default=0.0):
    """Extract last value as float, never NaN."""
    if series is None or len(series) == 0:
        return default
    val = series.iloc[-1]
    if isinstance(val, (pd.Series, pd.DataFrame)):
        val = val.iloc[0]
    val = float(val)
    return default if (np.isnan(val) or np.isinf(val)) else val

# ═══════════════════════════════════════
# DOWNLOAD DATA (fast: only ETFs + indices)
# ═══════════════════════════════════════
print("\n[1/4] Downloading market data...")

# All tickers in ONE batch call (much faster)
all_tickers = [
    '^GSPC', '^VIX', '^TNX', '^TYX', '^IRX',  # Indices
    'GC=F', 'CL=F',                              # Commodities
    'HYG', 'LQD',                                 # Credit
    'RSP',                                         # Equal-weight S&P (breadth proxy)
    'XLE', 'XLF', 'XLK', 'XLV', 'XLI',          # Sectors
    'XLP', 'XLY', 'XLU', 'XLRE', 'XLB', 'XLC',
]

print("  Downloading batch...")
raw_batch = yf.download(all_tickers, start=FROM_DATE, end=TODAY, progress=False, timeout=30, threads=True)

def extract(ticker):
    """Extract a single ticker from batch download."""
    try:
        if isinstance(raw_batch.columns, pd.MultiIndex):
            if ticker in raw_batch['Close'].columns:
                s = raw_batch['Close'][ticker].dropna()
                return s
        return pd.Series(dtype=float)
    except:
        return pd.Series(dtype=float)

sp500 = extract('^GSPC')
vix = extract('^VIX')
tnx = extract('^TNX')    # 10Y yield
tyx = extract('^TYX')    # 30Y yield
irx = extract('^IRX')    # 3M yield
gold = extract('GC=F')
oil = extract('CL=F')
hyg = extract('HYG')
lqd = extract('LQD')
rsp = extract('RSP')     # Equal-weight S&P 500

print(f"  S&P 500: {len(sp500)} days | VIX: {len(vix)} days | Oil: {len(oil)} days")

# Sector ETFs
sectors_map = {
    'XLE': 'Energy', 'XLF': 'Financials', 'XLK': 'Technology',
    'XLV': 'Healthcare', 'XLI': 'Industrials', 'XLP': 'Staples',
    'XLY': 'Discretionary', 'XLU': 'Utilities', 'XLRE': 'Real Estate',
    'XLB': 'Materials', 'XLC': 'Comms'
}
sector_data = {}
for tk in sectors_map:
    s = extract(tk)
    if len(s) > 0:
        sector_data[tk] = s

# ═══════════════════════════════════════
# BREADTH (approximated from RSP vs SPY)
# ═══════════════════════════════════════
print("[2/4] Computing indicators...")

results = {}

# S&P 500 metrics
if len(sp500) > 0:
    sp500_ath = sp500.expanding().max()
    sp500_drawdown = (sp500 / sp500_ath - 1) * 100
    sp500_200ma = sp500.rolling(200).mean()
    sp500_50ma = sp500.rolling(50).mean()
    results['sp500'] = sp500
    results['sp500_drawdown'] = sp500_drawdown

# Breadth proxy: RSP (equal-weight) vs SPY (cap-weight) ratio
# When RSP underperforms, breadth is narrow (few stocks carrying the index)
if len(rsp) > 0 and len(sp500) > 0:
    ci = rsp.index.intersection(sp500.index)
    breadth_ratio = rsp.loc[ci] / sp500.loc[ci]
    breadth_ratio = breadth_ratio / breadth_ratio.rolling(200).mean()  # Normalize
    # Approximate % above 200MA from breadth ratio (calibrated)
    pct_above_200_approx = (breadth_ratio.rolling(20).mean() - 0.9) / 0.2
    pct_above_200_approx = pct_above_200_approx.clip(0.05, 0.95)
    pct_above_50_approx = (breadth_ratio.rolling(5).mean() - 0.9) / 0.2
    pct_above_50_approx = pct_above_50_approx.clip(0.05, 0.95)
    results['pct_above_200'] = pct_above_200_approx
    results['pct_above_50'] = pct_above_50_approx

# McClellan approximation from sector advance/decline
if len(sector_data) >= 8:
    sector_df = pd.DataFrame(sector_data)
    sector_returns = sector_df.pct_change()
    advances = (sector_returns > 0).sum(axis=1)
    declines = (sector_returns < 0).sum(axis=1)
    ad_diff = advances - declines
    ema_19 = ad_diff.ewm(span=19, adjust=False).mean()
    ema_39 = ad_diff.ewm(span=39, adjust=False).mean()
    mcclellan = ema_19 - ema_39
    results['mcclellan'] = mcclellan

# Yield curve
if len(tnx) > 0 and len(irx) > 0:
    tnx_c = tnx[~tnx.index.duplicated(keep='last')]
    irx_c = irx[~irx.index.duplicated(keep='last')]
    ci = tnx_c.index.intersection(irx_c.index)
    yc = (tnx_c.loc[ci] - irx_c.loc[ci])
    yc = yc[~yc.index.duplicated(keep='last')]
    results['yield_curve'] = yc

# Credit spread proxy
if len(hyg) > 0 and len(lqd) > 0:
    hyg_c = hyg[~hyg.index.duplicated(keep='last')]
    lqd_c = lqd[~lqd.index.duplicated(keep='last')]
    ci = hyg_c.index.intersection(lqd_c.index)
    credit = lqd_c.loc[ci] / hyg_c.loc[ci]
    credit = credit[~credit.index.duplicated(keep='last')]
    results['credit_proxy'] = credit

# ═══════════════════════════════════════
# RECESSION RISK SCORE
# ═══════════════════════════════════════
print("[3/4] Computing recession risk score...")

score = 0
weight_total = 0

def add(value, w):
    global score, weight_total
    score += float(value) * w
    weight_total += w

# 1. Yield Curve (15%)
if 'yield_curve' in results:
    yc_val = safe_last(results['yield_curve'])
    add(1.0 if yc_val < 0 else 0.5 if yc_val < 0.5 else 0.0, 0.15)

# 2. VIX (10%)
if len(vix) > 0:
    v = safe_last(vix)
    add(min(max((v - 15) / 35, 0), 1), 0.10)

# 3. Breadth (15%)
if 'pct_above_200' in results:
    b = safe_last(results['pct_above_200'])
    add(max(1 - b * 1.5, 0), 0.15)

# 4. Drawdown (10%)
if 'sp500_drawdown' in results:
    dd = abs(safe_last(results['sp500_drawdown']))
    add(min(dd / 25, 1), 0.10)

# 5. Oil shock (12%)
if len(oil) > 0:
    o = safe_last(oil)
    add(min(max((o - 80) / 60, 0), 1), 0.12)

# 6. McClellan (10%)
if 'mcclellan' in results:
    mc = safe_last(results['mcclellan'])
    add(1.0 if mc < -3 else 0.6 if mc < -1 else 0.0, 0.10)

# 7. 30Y bond (8%)
if len(tyx) > 0:
    t30 = safe_last(tyx)
    add(min(max((t30 - 4) / 1.5, 0), 1), 0.08)

# 8. Credit stress (12%)
if 'credit_proxy' in results:
    cp = results['credit_proxy']
    cpz = (safe_last(cp) - float(cp.mean())) / float(cp.std())
    add(min(max(cpz, 0), 1), 0.12)

# 9. S&P below danger (8%)
if len(sp500) > 0:
    sp_below = 1.0 if safe_last(sp500) < SP500_DANGER_LEVEL else 0.0
    add(sp_below, 0.08)

composite = score / weight_total if weight_total > 0 else 0
risk_label = "LOW" if composite < 0.3 else "MODERATE" if composite < 0.5 else "ELEVATED" if composite < 0.7 else "HIGH"

# Buy checklist
mc_last = safe_last(results.get('mcclellan', pd.Series(dtype=float)))
mc_min5 = float(results['mcclellan'].iloc[-5:].min()) if 'mcclellan' in results and len(results['mcclellan']) >= 5 else 0
vix_last = safe_last(vix)
b200 = safe_last(results.get('pct_above_200', pd.Series(dtype=float)))
tyx_last = safe_last(tyx)
sp_last = safe_last(sp500, 99999)
oil_last = safe_last(oil)

buy_signals = {
    'McClellan > -1 from below': bool(mc_last > -1 and mc_min5 < -1),
    'VIX > 35 (panic)': bool(vix_last > VIX_PANIC),
    'Breadth < 12% above 200W': bool(b200 < BREADTH_CAPITULATION),
    '30Y yield >= 5%': bool(tyx_last >= BOND_30Y_BUY),
    'S&P below 6475': bool(sp_last < SP500_DANGER_LEVEL),
    'Oil > $120 (stress)': bool(oil_last > 120),
}
active_buy = sum(buy_signals.values())

if active_buy >= 4:
    action = "DEPLOY CAPITAL (Tramo 2-3)"
elif active_buy >= 2:
    action = "SELECTIVE BUYING (Tramo 1-2)"
else:
    action = "WAIT / ACCUMULATE CASH"

# Sector performance
sector_perf = {}
for tk, name in sectors_map.items():
    if tk in sector_data:
        s = sector_data[tk]
        if len(s) > 63:
            try:
                ytd = s.loc[s.index >= f'{datetime.now().year}-01-01']
                sector_perf[name] = {
                    '1M': round(float(s.iloc[-1] / s.iloc[-21] - 1) * 100, 1),
                    '3M': round(float(s.iloc[-1] / s.iloc[-63] - 1) * 100, 1),
                    'YTD': round(float(s.iloc[-1] / ytd.iloc[0] - 1) * 100, 1) if len(ytd) > 0 else 0,
                }
            except:
                pass

# ═══════════════════════════════════════
# GENERATE CHART
# ═══════════════════════════════════════
print("[4/4] Generating dashboard...")

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

fig = plt.figure(figsize=(24, 30), facecolor=BG)
gs = gridspec.GridSpec(5, 3, hspace=0.35, wspace=0.28, left=0.05, right=0.95, top=0.94, bottom=0.03)

score_color = GRN if composite < 0.3 else YEL if composite < 0.6 else RED

# P1: S&P + Drawdown
ax1 = fig.add_subplot(gs[0, 0:2])
style_ax(ax1, "S&P 500 & Drawdown from ATH")
if len(sp500) > 0:
    ax1.plot(sp500, color=BLU, linewidth=1.2, label='S&P 500')
    ax1.axhline(SP500_DANGER_LEVEL, color=RED, linestyle='--', alpha=0.7, linewidth=0.8, label=f'Danger: {SP500_DANGER_LEVEL}')
    if len(sp500) > 200:
        ax1.plot(sp500.rolling(200).mean(), color=YEL, linewidth=0.6, alpha=0.5, label='200 MA')
    ax1t = ax1.twinx()
    ax1t.fill_between(sp500_drawdown.index, sp500_drawdown.values, 0, color=RED, alpha=0.15)
    ax1t.plot(sp500_drawdown, color=RED, alpha=0.5, linewidth=0.7)
    ax1t.tick_params(colors=TXT, labelsize=7)
    ax1t.spines['right'].set_color(GRD)
    ax1.legend(loc='upper left', fontsize=7, facecolor=PANEL, edgecolor=GRD, labelcolor=TXT)

# P2: Risk Score
ax2 = fig.add_subplot(gs[0, 2])
ax2.set_facecolor(PANEL)
for s in ax2.spines.values(): s.set_color(GRD)
ax2.axis('off')
ax2.text(0.5, 0.72, f"{composite:.0%}", ha='center', va='center', fontsize=52, fontweight='bold', color=score_color, family='monospace', transform=ax2.transAxes)
ax2.text(0.5, 0.48, risk_label, ha='center', va='center', fontsize=18, fontweight='bold', color=score_color, transform=ax2.transAxes)
ax2.text(0.5, 0.32, "RECESSION RISK", ha='center', va='center', fontsize=10, color=TXT, alpha=0.6, transform=ax2.transAxes)
ax2.text(0.5, 0.18, f"Buy signals: {active_buy}/{len(buy_signals)}", ha='center', fontsize=12, color=GRN if active_buy >= 3 else YEL, fontweight='bold', transform=ax2.transAxes)
ax2.text(0.5, 0.05, action, ha='center', fontsize=9, color=score_color, transform=ax2.transAxes)

# P3: VIX
ax3 = fig.add_subplot(gs[1, 0])
style_ax(ax3, "VIX")
if len(vix) > 0:
    ax3.plot(vix, color=ORA, linewidth=1)
    ax3.axhline(VIX_PANIC, color=RED, linestyle='--', linewidth=1, label=f'Panic: {VIX_PANIC}')
    ax3.fill_between(vix.index, vix.values, VIX_PANIC, where=vix > VIX_PANIC, color=RED, alpha=0.25)
    ax3.legend(fontsize=7, facecolor=PANEL, edgecolor=GRD, labelcolor=TXT)

# P4: Oil
ax4 = fig.add_subplot(gs[1, 1])
style_ax(ax4, "WTI Oil ($)")
if len(oil) > 0:
    ax4.plot(oil, color=ORA, linewidth=1.2)
    ax4.axhline(100, color=RED, linestyle='--', linewidth=0.8, label='$100')
    ax4.axhline(120, color=RED, linestyle='--', linewidth=0.8, alpha=0.5, label='$120')
    ax4.fill_between(oil.index, oil.values, 100, where=oil > 100, color=RED, alpha=0.15)
    ax4.legend(fontsize=7, facecolor=PANEL, edgecolor=GRD, labelcolor=TXT)

# P5: Yield Curve
ax5 = fig.add_subplot(gs[1, 2])
style_ax(ax5, "Yield Curve 10Y-3M")
if 'yield_curve' in results:
    yc = results['yield_curve']
    ax5.fill_between(yc.index, yc.values, 0, where=yc > 0, color=GRN, alpha=0.25)
    ax5.fill_between(yc.index, yc.values, 0, where=yc < 0, color=RED, alpha=0.25)
    ax5.plot(yc, color=TXT, linewidth=0.8)
    ax5.axhline(0, color=RED, linewidth=1, linestyle='--')

# P6: 30Y Bond
ax6 = fig.add_subplot(gs[2, 0])
style_ax(ax6, "30Y Treasury Yield (%)")
if len(tyx) > 0:
    ax6.plot(tyx, color=BLU, linewidth=1.2)
    ax6.axhline(BOND_30Y_BUY, color=GRN, linestyle='--', linewidth=1, label=f'Buy: {BOND_30Y_BUY}%')
    ax6.fill_between(tyx.index, tyx.values, BOND_30Y_BUY, where=tyx > BOND_30Y_BUY, color=GRN, alpha=0.2)
    ax6.legend(fontsize=7, facecolor=PANEL, edgecolor=GRD, labelcolor=TXT)

# P7: McClellan
ax7 = fig.add_subplot(gs[2, 1:3])
style_ax(ax7, "McClellan Oscillator (Sector-based) — Buy when crosses above -1")
if 'mcclellan' in results:
    mc = results['mcclellan']
    ax7.fill_between(mc.index, mc.values, 0, where=mc > 0, color=GRN, alpha=0.25)
    ax7.fill_between(mc.index, mc.values, 0, where=mc < 0, color=RED, alpha=0.25)
    ax7.plot(mc, color=TXT, linewidth=0.8)
    ax7.axhline(-1, color=YEL, linestyle='--', linewidth=1, label='Buy signal: -1')
    ax7.axhline(0, color=TXT, linewidth=0.5, alpha=0.3)
    ax7.legend(fontsize=7, facecolor=PANEL, edgecolor=GRD, labelcolor=TXT)

# P8: Sector Heatmap
ax8 = fig.add_subplot(gs[3, 0:2])
style_ax(ax8, "Sector Performance (%)")
if sector_perf:
    perf_df = pd.DataFrame(sector_perf).T.sort_values('1M', ascending=True)
    y_pos = np.arange(len(perf_df))
    for j, period in enumerate(['1M', '3M', 'YTD']):
        vals = perf_df[period].values.astype(float)
        bar_colors = [GRN if v > 0 else RED for v in vals]
        ax8.barh(y_pos + j * 0.25, vals, height=0.22, color=bar_colors, alpha=0.5 + j * 0.15, label=period)
    ax8.set_yticks(y_pos + 0.25)
    ax8.set_yticklabels(perf_df.index, fontsize=7, color=TXT)
    ax8.axvline(0, color=TXT, linewidth=0.5)
    ax8.legend(fontsize=7, facecolor=PANEL, edgecolor=GRD, labelcolor=TXT)

# P9: Buy Checklist
ax9 = fig.add_subplot(gs[3, 2])
ax9.set_facecolor(PANEL)
for s in ax9.spines.values(): s.set_color(GRD)
ax9.set_title("BUY CHECKLIST", color=TXT, fontsize=10, fontweight='bold', pad=8)
ax9.axis('off')
for i, (sig, active) in enumerate(buy_signals.items()):
    icon = "●" if active else "○"
    c = GRN if active else TXT
    ax9.text(0.05, 0.88 - i * 0.13, f"{icon}  {sig}", transform=ax9.transAxes, fontsize=8, color=c, family='monospace')
ax9.text(0.05, 0.88 - len(buy_signals) * 0.13 - 0.08, f"ACTIVE: {active_buy}/{len(buy_signals)}",
         transform=ax9.transAxes, fontsize=11, fontweight='bold', color=GRN if active_buy >= 3 else YEL)

# P10: Summary
ax10 = fig.add_subplot(gs[4, :])
ax10.set_facecolor(PANEL)
for s in ax10.spines.values(): s.set_color(GRD)
ax10.axis('off')
summary = (
    f"  DATE: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}  |  "
    f"RISK: {composite:.0%} ({risk_label})  |  "
    f"S&P: {safe_last(sp500):.0f}  |  "
    f"VIX: {safe_last(vix):.1f}  |  "
    f"OIL: {safe_last(oil):.1f}  |  "
    f"30Y: {safe_last(tyx):.2f}%  |  "
    f"GOLD: {safe_last(gold):.0f}  |  "
    f"ACTION: {action}"
)
ax10.text(0.5, 0.5, summary, ha='center', va='center', fontsize=10, color=TXT, family='monospace', transform=ax10.transAxes)

fig.suptitle('RECESSION & MARKET TIMING MONITOR', color=GRN, fontsize=18, fontweight='bold', family='monospace', y=0.97)

output_png = os.path.join(OUTPUT_DIR, "recession_monitor_dashboard.png")
fig.savefig(output_png, dpi=150, facecolor=BG, bbox_inches='tight')
plt.close('all')
print(f"✓ Dashboard saved: {output_png}")

# ═══════════════════════════════════════
# EXPORT JSON
# ═══════════════════════════════════════
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
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj

export = sanitize({
    'timestamp': datetime.now().isoformat(),
    'composite_score': composite,
    'risk_label': risk_label,
    'action': action,
    'buy_checklist': {k: bool(v) for k, v in buy_signals.items()},
    'active_buy_signals': active_buy,
    'metrics': {
        'sp500': safe_last(sp500),
        'sp500_drawdown': safe_last(results.get('sp500_drawdown', pd.Series(dtype=float))),
        'vix': safe_last(vix),
        'oil': safe_last(oil),
        'gold': safe_last(gold),
        'bond_30y': safe_last(tyx),
        'yield_curve': safe_last(results.get('yield_curve', pd.Series(dtype=float))),
        'pct_above_200': safe_last(results.get('pct_above_200', pd.Series(dtype=float))) * 100,
        'pct_above_50': safe_last(results.get('pct_above_50', pd.Series(dtype=float))) * 100,
        'mcclellan': safe_last(results.get('mcclellan', pd.Series(dtype=float))),
    },
    'sector_performance': sector_perf,
})

json_path = os.path.join(OUTPUT_DIR, "data.json")
with open(json_path, 'w') as f:
    json.dump(export, f, indent=2, default=str)

print(f"✓ JSON saved: {json_path}")
print(f"\n{'='*60}")
print(f"  RISK: {composite:.0%} ({risk_label})")
print(f"  BUY SIGNALS: {active_buy}/{len(buy_signals)}")
print(f"  ACTION: {action}")
print(f"{'='*60}")
