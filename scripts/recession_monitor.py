"""
RECESSION & MARKET TIMING MONITOR v1.0
Roger's Investment Framework
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta
import warnings
import json
import os

warnings.filterwarnings("ignore")

FROM_DATE = "2018-01-01"
TODAY = datetime.today().date().strftime("%Y-%m-%d")
OUTPUT_DIR = "docs"

MCCLELLAN_BUY_SIGNAL = -1
VIX_PANIC = 35
BREADTH_CAPITULATION = 0.12
SP500_DANGER_LEVEL = 6475
BOND_30Y_BUY = 5.0

print("=" * 60)
print("  RECESSION MONITOR - Downloading data...")
print("=" * 60)

# S&P 500 top 100 tickers for breadth
tickers = [
    'AAPL','MSFT','AMZN','NVDA','GOOGL','META','BRK-B','LLY','AVGO','JPM',
    'TSLA','UNH','XOM','V','MA','PG','JNJ','COST','HD','ABBV',
    'MRK','WMT','NFLX','KO','BAC','CRM','PEP','CVX','AMD','TMO',
    'LIN','ORCL','ACN','MCD','ABT','PM','CSCO','ADBE','WFC','DHR',
    'TXN','GE','AMGN','CMCSA','NEE','DIS','PFE','RTX','ISRG','INTU',
    'IBM','HON','SPGI','VZ','QCOM','UNP','AMAT','GS','CAT','NOW',
    'BLK','T','LOW','COP','SYK','BKNG','AXP','ELV','MDLZ','DE',
    'PLD','SBUX','ADP','MS','SCHW','LRCX','GILD','VRTX','MMC','CB',
    'C','ADI','AMT','REGN','MO','SO','BMY','ZTS','SLB','CI',
    'PANW','CME','PGR','BDX','DUK','ICE','SNPS','CL','TJX','WM'
]

print("[1/5] Downloading S&P 500 constituents...")
raw = yf.download(tickers, start=FROM_DATE, end=TODAY, progress=False)
# Handle both old and new yfinance column formats
if isinstance(raw.columns, pd.MultiIndex):
    df_close = raw['Close']
else:
    df_close = raw[[c for c in raw.columns if 'Close' in str(c)]]
    df_close.columns = [c.replace('Close_', '').replace('Close', '') for c in df_close.columns]

# Flatten any remaining MultiIndex
if isinstance(df_close.columns, pd.MultiIndex):
    df_close.columns = df_close.columns.get_level_values(-1)

available_tickers = [t for t in tickers if t in df_close.columns]
df_close = df_close[available_tickers].dropna(how='all')
n_stocks = len(available_tickers)
print(f"  Downloaded {n_stocks} stocks")

print("[2/5] Downloading indices & macro...")
assets = {
    'SP500': '^GSPC', 'VIX': '^VIX', 'TNX_10Y': '^TNX',
    'TYX_30Y': '^TYX', 'IRX_3M': '^IRX', 'GOLD': 'GC=F',
    'OIL': 'CL=F', 'HYG': 'HYG', 'LQD': 'LQD',
}

macro_data = {}
for name, ticker in assets.items():
    try:
        raw = yf.download(ticker, start=FROM_DATE, end=TODAY, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            data = raw['Close'].iloc[:, 0] if isinstance(raw['Close'], pd.DataFrame) else raw['Close']
        else:
            data = raw['Close']
        if isinstance(data, pd.DataFrame):
            data = data.iloc[:, 0]
        if len(data) > 0:
            macro_data[name] = data
    except:
        pass

print("[3/5] Downloading sector ETFs...")
sectors = {
    'XLE': 'Energy', 'XLF': 'Financials', 'XLK': 'Technology',
    'XLV': 'Healthcare', 'XLI': 'Industrials', 'XLP': 'Staples',
    'XLY': 'Discretionary', 'XLU': 'Utilities', 'XLRE': 'Real Estate',
    'XLB': 'Materials', 'XLC': 'Comms'
}
sector_data = {}
for ticker in sectors:
    try:
        raw = yf.download(ticker, start=FROM_DATE, end=TODAY, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            data = raw['Close'].iloc[:, 0] if isinstance(raw['Close'], pd.DataFrame) else raw['Close']
        else:
            data = raw['Close']
        if isinstance(data, pd.DataFrame):
            data = data.iloc[:, 0]
        if len(data) > 0:
            sector_data[ticker] = data
    except:
        pass

print("[4/5] Computing breadth indicators...")
results = {}

df_50wma = df_close.rolling(50).mean()
df_200wma = df_close.rolling(200).mean()

pct_above_200 = (df_close > df_200wma).sum(axis=1) / n_stocks
pct_above_50 = (df_close > df_50wma).sum(axis=1) / n_stocks

rolling_max_52w = df_close.rolling(252).max()
rolling_min_52w = df_close.rolling(252).min()
pct_new_highs = (df_close >= rolling_max_52w * 0.99).sum(axis=1) / n_stocks
pct_new_lows = (df_close <= rolling_min_52w * 1.01).sum(axis=1) / n_stocks

daily_returns = df_close.pct_change()
advances = (daily_returns > 0).sum(axis=1)
declines = (daily_returns < 0).sum(axis=1)
ad_diff = advances - declines

ema_19 = ad_diff.ewm(span=19, adjust=False).mean()
ema_39 = ad_diff.ewm(span=39, adjust=False).mean()
mcclellan = ema_19 - ema_39

below_both = ((df_close < df_50wma) & (df_close < df_200wma)).sum(axis=1) / n_stocks

results['pct_above_200'] = pct_above_200
results['pct_above_50'] = pct_above_50
results['pct_new_highs'] = pct_new_highs
results['pct_new_lows'] = pct_new_lows
results['mcclellan'] = mcclellan
results['below_both'] = below_both

sp500 = macro_data.get('SP500', pd.Series(dtype=float))
vix = macro_data.get('VIX', pd.Series(dtype=float))
tnx = macro_data.get('TNX_10Y', pd.Series(dtype=float))
tyx = macro_data.get('TYX_30Y', pd.Series(dtype=float))
irx = macro_data.get('IRX_3M', pd.Series(dtype=float))
oil = macro_data.get('OIL', pd.Series(dtype=float))
gold = macro_data.get('GOLD', pd.Series(dtype=float))
hyg = macro_data.get('HYG', pd.Series(dtype=float))
lqd = macro_data.get('LQD', pd.Series(dtype=float))

# Flatten any DataFrames to Series (yfinance v0.2.x can return DataFrame)
def to_series(x):
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0]
    return x

sp500, vix, tnx, tyx, irx, oil, gold, hyg, lqd = [
    to_series(x) for x in [sp500, vix, tnx, tyx, irx, oil, gold, hyg, lqd]
]

# Safe scalar extraction: always returns a Python float, never NaN for comparisons
def safe_last(series, default=0.0):
    if series is None or len(series) == 0:
        return default
    val = series.iloc[-1]
    if isinstance(val, (pd.Series, pd.DataFrame)):
        val = val.iloc[0]
    val = float(val)
    if np.isnan(val) or np.isinf(val):
        return default
    return val

if len(sp500) > 0:
    sp500_ath = sp500.expanding().max()
    sp500_drawdown = (sp500 / sp500_ath - 1) * 100
    results['sp500'] = sp500
    results['sp500_drawdown'] = sp500_drawdown

if len(tnx) > 0 and len(irx) > 0:
    # Flatten to Series if needed (yfinance sometimes returns DataFrame)
    if isinstance(tnx, pd.DataFrame): tnx = tnx.iloc[:, 0]
    if isinstance(irx, pd.DataFrame): irx = irx.iloc[:, 0]
    # Align on common dates, drop duplicates
    tnx_clean = tnx[~tnx.index.duplicated(keep='last')]
    irx_clean = irx[~irx.index.duplicated(keep='last')]
    ci = tnx_clean.index.intersection(irx_clean.index)
    yc = tnx_clean.loc[ci] - irx_clean.loc[ci]
    yc = yc[~yc.index.duplicated(keep='last')]
    results['yield_curve'] = yc

print("[5/5] Computing recession risk score...")

composite_score = 0
n_signals = 0

def add_signal(value, weight):
    global composite_score, n_signals
    composite_score += float(value) * weight
    n_signals += weight

# Yield curve
if 'yield_curve' in results:
    yc_val = safe_last(results['yield_curve'])
    if not np.isnan(yc_val):
        add_signal(1.0 if yc_val < 0 else 0.5 if yc_val < 0.5 else 0.0, 0.15)

# VIX
if len(vix) > 0:
    v = safe_last(vix)
    add_signal(min(max((v - 15) / 35, 0), 1), 0.10)

# Breadth
b200_last = safe_last(pct_above_200)
add_signal(max(1 - b200_last * 1.5, 0), 0.15)

# Drawdown
if 'sp500_drawdown' in results:
    dd_last = safe_last(sp500_drawdown)
    add_signal(min(abs(dd_last) / 25, 1), 0.10)

# Oil shock
if len(oil) > 0:
    o = safe_last(oil)
    add_signal(min(max((o - 80) / 60, 0), 1), 0.12)

# McClellan
mc_val = safe_last(mcclellan)
add_signal(1.0 if mc_val < -20 else 0.6 if mc_val < -1 else 0.0, 0.10)

# 30Y bond stress
if len(tyx) > 0:
    t30 = safe_last(tyx)
    add_signal(min(max((t30 - 4) / 1.5, 0), 1), 0.08)

# Credit proxy
if len(hyg) > 0 and len(lqd) > 0:
    hyg_clean = hyg[~hyg.index.duplicated(keep='last')]
    lqd_clean = lqd[~lqd.index.duplicated(keep='last')]
    ci2 = hyg_clean.index.intersection(lqd_clean.index)
    cp = lqd_clean.loc[ci2] / hyg_clean.loc[ci2]
    cp = cp[~cp.index.duplicated(keep='last')]
    cpz = float((safe_last(cp) - cp.mean()) / cp.std())
    add_signal(min(max(cpz, 0), 1), 0.12)

# New lows
nl_avg = float(pct_new_lows.iloc[-5:].mean())
add_signal(min(nl_avg * 10, 1), 0.08)

composite_score = composite_score / n_signals if n_signals > 0 else 0
risk_label = "LOW" if composite_score < 0.3 else "MODERATE" if composite_score < 0.5 else "ELEVATED" if composite_score < 0.7 else "HIGH"

# Buy signal checklist - all comparisons use safe_last for scalar values
mc_last = safe_last(mcclellan)
mc_min5 = float(mcclellan.iloc[-5:].min()) if len(mcclellan) >= 5 else 0
vix_last = safe_last(vix) if len(vix) > 0 else 0
b200_last_pct = safe_last(pct_above_200)
tyx_last = safe_last(tyx) if len(tyx) > 0 else 0
sp_last = safe_last(sp500) if len(sp500) > 0 else 99999
nl_last = safe_last(pct_new_lows)

buy_signals = {}
buy_signals['McClellan > -1 from below'] = bool(mc_last > -1 and mc_min5 < -1)
buy_signals['VIX > 35 (panic)'] = bool(vix_last > VIX_PANIC)
buy_signals['Breadth < 12% above 200W'] = bool(b200_last_pct < BREADTH_CAPITULATION)
buy_signals['30Y yield >= 5%'] = bool(tyx_last >= BOND_30Y_BUY)
buy_signals['S&P below danger level'] = bool(sp_last < SP500_DANGER_LEVEL)
buy_signals['New lows > 15%'] = bool(nl_last > 0.15)
active_buy = sum(buy_signals.values())

if active_buy >= 4:
    action = "DEPLOY CAPITAL AGGRESSIVELY (Tramo 2-3)"
elif active_buy >= 2:
    action = "SELECTIVE BUYING (Tramo 1-2)"
else:
    action = "WAIT / ACCUMULATE CASH"

# Sector performance
sector_perf = {}
for ticker, name in sectors.items():
    if ticker in sector_data:
        s = sector_data[ticker]
        if len(s) > 63:
            ytd_start = s.loc[s.index >= f'{datetime.now().year}-01-01']
            try:
                sector_perf[name] = {
                    '1M': round(float(s.iloc[-1] / s.iloc[-21] - 1) * 100, 1),
                    '3M': round(float(s.iloc[-1] / s.iloc[-63] - 1) * 100, 1),
                    'YTD': round(float(s.iloc[-1] / ytd_start.iloc[0] - 1) * 100, 1) if len(ytd_start) > 0 else 0,
                }
            except:
                pass

# ═══════════════════════════════════════════════════
# GENERATE CHARTS
# ═══════════════════════════════════════════════════
print("\nGenerating dashboard charts...")

BG = '#0a0a0f'
PANEL = '#12121a'
TXT = '#e0e0e0'
GRN = '#00d68f'
RED = '#ff4d4d'
YEL = '#ffd93d'
BLU = '#4da8ff'
PUR = '#b87aff'
ORA = '#ff8c42'
GRD = '#1a1a2e'

def style_ax(ax, title=""):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TXT, labelsize=7)
    ax.set_title(title, color=TXT, fontsize=10, fontweight='bold', pad=8)
    for sp in ax.spines.values():
        sp.set_color(GRD)
    ax.grid(True, color=GRD, alpha=0.3, linewidth=0.5)

fig = plt.figure(figsize=(26, 34), facecolor=BG)
gs = gridspec.GridSpec(6, 3, hspace=0.35, wspace=0.28,
                       left=0.05, right=0.95, top=0.94, bottom=0.03)

# Panel 1: S&P 500 + Drawdown
ax1 = fig.add_subplot(gs[0, 0:2])
style_ax(ax1, "S&P 500 & Drawdown from ATH")
sp_plot = sp500.loc['2019':]
ax1.plot(sp_plot, color=BLU, linewidth=1.2, label='S&P 500')
ax1.axhline(SP500_DANGER_LEVEL, color=RED, linestyle='--', alpha=0.7, linewidth=0.8, label=f'Danger: {SP500_DANGER_LEVEL}')
ax1t = ax1.twinx()
dd_plot = sp500_drawdown.loc['2019':]
ax1t.fill_between(dd_plot.index, dd_plot.values, 0, color=RED, alpha=0.15)
ax1t.plot(dd_plot, color=RED, alpha=0.5, linewidth=0.7)
ax1t.tick_params(colors=TXT, labelsize=7)
ax1t.spines['right'].set_color(GRD)
ax1.legend(loc='upper left', fontsize=7, facecolor=PANEL, edgecolor=GRD, labelcolor=TXT)

# Panel 2: Risk Score
ax2 = fig.add_subplot(gs[0, 2])
ax2.set_facecolor(PANEL)
ax2.axis('off')
score_color = GRN if composite_score < 0.3 else YEL if composite_score < 0.6 else RED
ax2.text(0.5, 0.7, f"{composite_score:.0%}", ha='center', va='center',
         fontsize=52, fontweight='bold', color=score_color, family='monospace',
         transform=ax2.transAxes)
ax2.text(0.5, 0.45, risk_label, ha='center', va='center',
         fontsize=18, fontweight='bold', color=score_color, transform=ax2.transAxes)
ax2.text(0.5, 0.30, "RECESSION RISK", ha='center', va='center',
         fontsize=10, color=TXT, transform=ax2.transAxes, alpha=0.6)
ax2.text(0.5, 0.12, f"Buy signals: {active_buy}/{len(buy_signals)}", ha='center', va='center',
         fontsize=12, color=GRN if active_buy >= 3 else YEL if active_buy >= 2 else TXT,
         fontweight='bold', transform=ax2.transAxes)
for sp in ax2.spines.values():
    sp.set_color(GRD)

# Panel 3: Market Breadth
ax3 = fig.add_subplot(gs[1, 0:2])
style_ax(ax3, "Market Breadth: % of S&P 500 Above Moving Averages")
b200 = pct_above_200.loc['2019':]
b50 = pct_above_50.loc['2019':]
ax3.plot(b200, color=BLU, linewidth=1.2, label='Above 200 WMA')
ax3.plot(b50, color=PUR, linewidth=0.8, alpha=0.7, label='Above 50 WMA')
ax3.axhline(0.50, color=YEL, linestyle='--', alpha=0.4, linewidth=0.7)
ax3.axhline(BREADTH_CAPITULATION, color=RED, linestyle='--', alpha=0.7, label=f'Capitulation: {BREADTH_CAPITULATION:.0%}')
ax3.fill_between(b200.index, b200.values, BREADTH_CAPITULATION,
                  where=b200 < BREADTH_CAPITULATION, color=RED, alpha=0.2)
ax3.set_ylim(0, 1)
ax3.legend(loc='lower left', fontsize=7, facecolor=PANEL, edgecolor=GRD, labelcolor=TXT)

# Panel 4: New Highs vs Lows
ax4 = fig.add_subplot(gs[1, 2])
style_ax(ax4, "52W New Highs vs Lows")
nh = pct_new_highs.loc['2023':]
nl = pct_new_lows.loc['2023':]
ax4.bar(nh.index, nh.values, color=GRN, alpha=0.5, width=2, label='Highs')
ax4.bar(nl.index, -nl.values, color=RED, alpha=0.5, width=2, label='Lows')
ax4.axhline(0, color=TXT, linewidth=0.5)
ax4.legend(fontsize=7, facecolor=PANEL, edgecolor=GRD, labelcolor=TXT)

# Panel 5: McClellan
ax5 = fig.add_subplot(gs[2, 0:2])
style_ax(ax5, "McClellan Oscillator (Buy signal: crosses above -1 from below)")
mc_plot = mcclellan.loc['2019':]
ax5.fill_between(mc_plot.index, mc_plot.values, 0, where=mc_plot > 0, color=GRN, alpha=0.25)
ax5.fill_between(mc_plot.index, mc_plot.values, 0, where=mc_plot < 0, color=RED, alpha=0.25)
ax5.plot(mc_plot, color=TXT, linewidth=0.8)
ax5.axhline(MCCLELLAN_BUY_SIGNAL, color=YEL, linestyle='--', linewidth=1, label=f'Buy: {MCCLELLAN_BUY_SIGNAL}')
ax5.axhline(0, color=TXT, linewidth=0.5, alpha=0.3)
ax5.legend(fontsize=7, facecolor=PANEL, edgecolor=GRD, labelcolor=TXT)

# Panel 6: VIX
ax6 = fig.add_subplot(gs[2, 2])
style_ax(ax6, "VIX (Fear Index)")
v_plot = vix.loc['2019':]
ax6.plot(v_plot, color=ORA, linewidth=1)
ax6.axhline(VIX_PANIC, color=RED, linestyle='--', linewidth=1, label=f'Panic: {VIX_PANIC}')
ax6.axhline(20, color=YEL, linestyle='--', linewidth=0.5, alpha=0.4)
ax6.fill_between(v_plot.index, v_plot.values, VIX_PANIC, where=v_plot > VIX_PANIC, color=RED, alpha=0.25)
ax6.legend(fontsize=7, facecolor=PANEL, edgecolor=GRD, labelcolor=TXT)

# Panel 7: Yield Curve
ax7 = fig.add_subplot(gs[3, 0])
style_ax(ax7, "Yield Curve 10Y-3M")
if 'yield_curve' in results:
    yc = results['yield_curve'].loc['2019':]
    ax7.fill_between(yc.index, yc.values, 0, where=yc > 0, color=GRN, alpha=0.25)
    ax7.fill_between(yc.index, yc.values, 0, where=yc < 0, color=RED, alpha=0.25)
    ax7.plot(yc, color=TXT, linewidth=0.8)
    ax7.axhline(0, color=RED, linewidth=1, linestyle='--')

# Panel 8: Oil
ax8 = fig.add_subplot(gs[3, 1])
style_ax(ax8, "WTI Oil ($)")
if len(oil) > 0:
    o_plot = oil.loc['2019':]
    ax8.plot(o_plot, color=ORA, linewidth=1.2)
    ax8.axhline(100, color=RED, linestyle='--', linewidth=0.8, label='$100')
    ax8.fill_between(o_plot.index, o_plot.values, 100, where=o_plot > 100, color=RED, alpha=0.2)
    ax8.legend(fontsize=7, facecolor=PANEL, edgecolor=GRD, labelcolor=TXT)

# Panel 9: 30Y Bond
ax9 = fig.add_subplot(gs[3, 2])
style_ax(ax9, "30Y Treasury Yield (%)")
if len(tyx) > 0:
    b30 = tyx.loc['2019':]
    ax9.plot(b30, color=BLU, linewidth=1.2)
    ax9.axhline(BOND_30Y_BUY, color=GRN, linestyle='--', linewidth=1, label=f'Buy: {BOND_30Y_BUY}%')
    ax9.fill_between(b30.index, b30.values, BOND_30Y_BUY, where=b30 > BOND_30Y_BUY, color=GRN, alpha=0.2)
    ax9.legend(fontsize=7, facecolor=PANEL, edgecolor=GRD, labelcolor=TXT)

# Panel 10: Sector Heatmap
ax10 = fig.add_subplot(gs[4, 0:2])
style_ax(ax10, "Sector Performance (%)")
if sector_perf:
    perf_df = pd.DataFrame(sector_perf).T.sort_values('1M', ascending=True)
    y_pos = np.arange(len(perf_df))
    for j, period in enumerate(['1M', '3M', 'YTD']):
        vals = perf_df[period].values.astype(float)
        bar_colors = [GRN if v > 0 else RED for v in vals]
        ax10.barh(y_pos + j * 0.25, vals, height=0.22, color=bar_colors, alpha=0.5 + j * 0.15, label=period)
    ax10.set_yticks(y_pos + 0.25)
    ax10.set_yticklabels(perf_df.index, fontsize=7, color=TXT)
    ax10.axvline(0, color=TXT, linewidth=0.5)
    ax10.legend(fontsize=7, facecolor=PANEL, edgecolor=GRD, labelcolor=TXT)

# Panel 11: Buy Checklist
ax11 = fig.add_subplot(gs[4, 2])
ax11.set_facecolor(PANEL)
for sp in ax11.spines.values():
    sp.set_color(GRD)
ax11.set_title("BUY CHECKLIST", color=TXT, fontsize=10, fontweight='bold', pad=8)
ax11.axis('off')
for i, (sig, active) in enumerate(buy_signals.items()):
    icon = "●" if active else "○"
    c = GRN if active else TXT
    ax11.text(0.05, 0.88 - i * 0.13, f"{icon}  {sig}", transform=ax11.transAxes,
              fontsize=8, color=c, family='monospace')
ax11.text(0.05, 0.88 - len(buy_signals) * 0.13 - 0.08,
          f"ACTIVE: {active_buy}/{len(buy_signals)}", transform=ax11.transAxes,
          fontsize=11, fontweight='bold', color=GRN if active_buy >= 3 else YEL)

# Panel 12: Below Both MAs (Harnett rule)
ax12 = fig.add_subplot(gs[5, 0:2])
style_ax(ax12, "Harnett Rule: % Below BOTH 50 & 200 WMA (88% = capitulation BUY)")
bb = below_both.loc['2019':]
ax12.plot(bb, color=PUR, linewidth=1)
ax12.axhline(0.88, color=GRN, linestyle='--', linewidth=1.5, label='88% = BUY')
ax12.fill_between(bb.index, bb.values, 0.88, where=bb > 0.88, color=GRN, alpha=0.3)
ax12.set_ylim(0, 1)
ax12.legend(fontsize=7, facecolor=PANEL, edgecolor=GRD, labelcolor=TXT)

# Panel 13: Summary
ax13 = fig.add_subplot(gs[5, 2])
ax13.set_facecolor(PANEL)
for sp in ax13.spines.values():
    sp.set_color(GRD)
ax13.set_title("SUMMARY", color=TXT, fontsize=10, fontweight='bold', pad=8)
ax13.axis('off')

lines = [
    (f"Date: {datetime.now().strftime('%Y-%m-%d')}", TXT, 'normal'),
    ("", TXT, 'normal'),
    (f"Risk Score: {composite_score:.0%} ({risk_label})", score_color, 'bold'),
    (f"S&P 500:    {safe_last(sp500):.0f}", TXT, 'normal'),
    (f"VIX:        {safe_last(vix):.1f}" if len(vix) > 0 else "", TXT, 'normal'),
    (f"Oil:        ${safe_last(oil):.1f}" if len(oil) > 0 else "", TXT, 'normal'),
    (f"30Y:        {safe_last(tyx):.2f}%" if len(tyx) > 0 else "", TXT, 'normal'),
    (f"Above 200W: {safe_last(pct_above_200):.0%}", TXT, 'normal'),
    (f"McClellan:  {safe_last(mcclellan):.1f}", TXT, 'normal'),
    ("", TXT, 'normal'),
    (f"ACTION:", TXT, 'normal'),
    (action, GRN if active_buy >= 3 else YEL if active_buy >= 2 else ORA, 'bold'),
]

for i, (txt, c, fw) in enumerate(lines):
    ax13.text(0.05, 0.95 - i * 0.075, txt, transform=ax13.transAxes,
              fontsize=8, color=c, family='monospace', fontweight=fw)

fig.suptitle('RECESSION & MARKET TIMING MONITOR', color=GRN, fontsize=18,
             fontweight='bold', family='monospace', y=0.97)

os.makedirs(OUTPUT_DIR, exist_ok=True)

output_png = os.path.join(OUTPUT_DIR, "recession_monitor_dashboard.png")
fig.savefig(output_png, dpi=150, facecolor=BG, bbox_inches='tight')
plt.close('all')
print(f"\n✓ Dashboard saved: {output_png}")

# Export JSON for React dashboard
export = {
    'timestamp': datetime.now().isoformat(),
    'composite_score': round(composite_score, 3),
    'risk_label': risk_label,
    'action': action,
    'buy_checklist': {k: bool(v) for k, v in buy_signals.items()},
    'active_buy_signals': active_buy,
    'metrics': {
        'sp500': round(safe_last(sp500), 1) if len(sp500) > 0 else None,
        'sp500_drawdown': round(safe_last(sp500_drawdown), 2) if 'sp500_drawdown' in results else None,
        'vix': round(safe_last(vix), 1) if len(vix) > 0 else None,
        'oil': round(safe_last(oil), 1) if len(oil) > 0 else None,
        'gold': round(safe_last(gold), 0) if len(gold) > 0 else None,
        'bond_30y': round(safe_last(tyx), 2) if len(tyx) > 0 else None,
        'yield_curve': round(safe_last(results['yield_curve']), 2) if 'yield_curve' in results else None,
        'pct_above_200': round(safe_last(pct_above_200) * 100, 1),
        'pct_above_50': round(safe_last(pct_above_50) * 100, 1),
        'mcclellan': round(safe_last(mcclellan), 1),
        'below_both_pct': round(safe_last(below_both) * 100, 1),
    },
    'sector_performance': sector_perf,
    'breadth_history': {
        'dates': [d.strftime('%Y-%m-%d') for d in pct_above_200.loc['2022':].index[-120:]],
        'above_200': [round(float(v)*100, 1) for v in pct_above_200.loc['2022':].values[-120:]],
        'above_50': [round(float(v)*100, 1) for v in pct_above_50.loc['2022':].values[-120:]],
        'mcclellan': [round(float(v), 1) for v in mcclellan.loc['2022':].values[-120:]],
    },
    'sp500_history': {
        'dates': [d.strftime('%Y-%m-%d') for d in sp500.loc['2024':].index[-120:]],
        'values': [round(float(v), 1) for v in sp500.loc['2024':].values[-120:]],
        'drawdown': [round(float(v), 2) for v in sp500_drawdown.loc['2024':].values[-120:]],
    },
    'vix_history': {
        'dates': [d.strftime('%Y-%m-%d') for d in vix.loc['2024':].index[-120:]] if len(vix) > 0 else [],
        'values': [round(float(v), 1) for v in vix.loc['2024':].values[-120:]] if len(vix) > 0 else [],
    },
}

json_path = os.path.join(OUTPUT_DIR, "data.json")

# Sanitize NaN/Inf values — JSON doesn't support them
def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    elif isinstance(obj, (np.floating, np.integer)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    return obj

export = sanitize_for_json(export)

with open(json_path, 'w') as f:
    json.dump(export, f, indent=2, default=str)

print(f"✓ Data exported: {json_path}")
print(f"\n{'='*60}")
print(f"  RECESSION RISK: {composite_score:.0%} ({risk_label})")
print(f"  BUY SIGNALS:    {active_buy}/{len(buy_signals)}")
print(f"  ACTION:         {action}")
print(f"{'='*60}")
