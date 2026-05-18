# leg_analysis_standalone.py
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# افزودن پوشه اصلی پروژه به مسیر جستجو
sys.path.insert(0, str(Path(__file__).parent))

from analysis.leg_detector import detect_legs
from preprocessing.candle_classifier import enrich_candles
from analysis.channels_and_trendlines import find_channels

# وارد کردن توابع امتیازدهی از leg_strength
from analysis.leg_strength import (
    _c1_micro_channel_score,
    _c2_overlap_score,
    _c3_close_position_score,
    _c4_big_bodies_score,
    _c5_consecutive_opposite_score,
    _c6_avg_body_ratio_score,
    _c7_length_bonus_score,      # <--- جایگزین _c7_shadow_score
    _c8_gap_score,
    _c9_increasing_bodies_score,
    _c10_god_candles_score
)

from config import BREAK_FACTOR, MERGE_FACTOR


def get_leg_segments(leg_labels):
    """استخراج بازه‌های پیوسته لگ"""
    segments = []
    i, n = 0, len(leg_labels)
    while i < n:
        if leg_labels[i] != '':
            start = i
            label = leg_labels[i]
            while i < n and leg_labels[i] == label:
                i += 1
            end = i - 1
            segments.append((start, end, label))
        else:
            i += 1
    return segments


def compute_detailed_strength(df, leg_labels, micro_channels):
    """محاسبه قدرت هر لگ با جزئیات کامل ۱۰ معیار."""
    detail_list = [''] * len(df)
    highlights = []

    segments = get_leg_segments(leg_labels)

    for start, end, label in segments:
        leg_df = df.iloc[start:end+1].copy()
        leg_height = leg_df['high'].max() - leg_df['low'].min()
        if leg_height == 0:
            continue

        lookback_start = max(0, start - 100)
        avg_body_100 = (abs(df.iloc[lookback_start:start]['close'] - df.iloc[lookback_start:start]['open'])).mean()
        pre_candle = df.iloc[start - 1] if start > 0 else None
        post_candle = df.iloc[end + 1] if end + 1 < len(df) else None

        sc = {}
        sc['1_micro_channel'] = _c1_micro_channel_score(leg_df, label, micro_channels, start, end, leg_height)
        sc['2_overlap'] = _c2_overlap_score(leg_df, label, pre_candle)
        sc['4_big_bodies'] = _c4_big_bodies_score(leg_df, leg_height, avg_body_100)
        c3_raw = _c3_close_position_score(leg_df, label)
        sc['3_close_pos'] = c3_raw if sc['4_big_bodies'] > 0 else 0
        sc['5_opp_seq'] = _c5_consecutive_opposite_score(leg_df, label, leg_height)
        sc['6_body_ratio'] = _c6_avg_body_ratio_score(leg_df, label)
        sc['7_length_bonus'] = _c7_length_bonus_score(leg_df)
        sc['8_gaps'] = _c8_gap_score(leg_df, label, leg_height, pre_candle, post_candle)
        sc['9_inc_bodies'] = _c9_increasing_bodies_score(leg_df, label, pre_candle, avg_body_100)
        sc['10_god_candles'] = _c10_god_candles_score(leg_df, label, avg_body_100)

        n_candles = len(leg_df)
        vertical_per_candle = leg_height / n_candles if n_candles > 0 else 0.0
        if avg_body_100 > 0:
            efficiency_ratio = vertical_per_candle / avg_body_100
        else:
            efficiency_ratio = 0.0
        length_eff_coef = efficiency_ratio ** 0.45

        raw_total = sum(sc.values())
        total = round(raw_total * length_eff_coef, 2)

        sc['eff_coef'] = round(length_eff_coef, 2)
        sc['TOTAL'] = total

        parts = [f"{label}"]
        for k, v in sc.items():
            parts.append(f"{k}: {v}")
        hover_str = '<br>'.join(parts)

        for idx in range(start, end + 1):
            detail_list[idx] = hover_str
        highlights.append((start, end, label))

    return detail_list, highlights


def _extract_swings_from_legs(df, leg_labels):
    """استخراج سوئینگ‌های بالا (سقف) و پایین (کف) از لگ‌ها."""
    swing_highs = []
    swing_lows = []
    n = len(df)
    i = 0
    while i < n:
        label = leg_labels[i]
        if label == 'Bullish Leg':
            start = i
            while i < n and leg_labels[i] == 'Bullish Leg':
                i += 1
            end = i - 1
            best_price = df['high'].iloc[start:end+1].max()
            best_idx = start + df['high'].iloc[start:end+1].values.argmax()
            j = end + 1
            while j < n and leg_labels[j] == '':
                if df['high'].iloc[j] > best_price:
                    best_price = df['high'].iloc[j]
                    best_idx = j
                j += 1
            swing_highs.append({'price': best_price, 'index': best_idx})
        elif label == 'Bearish Leg':
            start = i
            while i < n and leg_labels[i] == 'Bearish Leg':
                i += 1
            end = i - 1
            best_price = df['low'].iloc[start:end+1].min()
            best_idx = start + df['low'].iloc[start:end+1].values.argmin()
            j = end + 1
            while j < n and leg_labels[j] == '':
                if df['low'].iloc[j] < best_price:
                    best_price = df['low'].iloc[j]
                    best_idx = j
                j += 1
            swing_lows.append({'price': best_price, 'index': best_idx})
        else:
            i += 1
    return swing_highs, swing_lows


def prepare_sr_shapes_with_broken(df, leg_labels, break_factor=None, merge_factor=None, avg_body_lookback=100):
    """
    رسم خطوط/مستطیل حمایت و مقاومت از سوئینگ لگ‌ها.
    خطوط حتی اگر شکسته شوند، تا محل شکست کشیده می‌شوند.
    سطوح تنها در صورتی گروه (merge) می‌شوند که:
      1) هم‌نوع باشند (هر دو حمایت یا هر دو مقاومت)
      2) فاصله قیمتی آنها <= merge_factor * avg_range
      3) بازه‌های زمانی فعالیتشان همپوشانی داشته باشد
         (یعنی سطح جدید قبل از شکسته شدن سطح قبلی ظاهر شده باشد)
    انتهای ناحیه = اولین شکست در گروه (min broken_at)
    """
    if break_factor is None:
        break_factor = BREAK_FACTOR
    if merge_factor is None:
        merge_factor = MERGE_FACTOR

    # میانگین بدنه
    body = (df['close'] - df['open']).abs()
    if len(body) >= avg_body_lookback:
        avg_body = body.iloc[-avg_body_lookback:].mean()
    else:
        avg_body = body.mean()
    threshold = break_factor * avg_body

    swing_highs, swing_lows = _extract_swings_from_legs(df, leg_labels)

    # ساخت لیست سطح‌ها
    levels = []
    for h in swing_highs:
        if h['index'] < 20:
            continue
        levels.append({
            'index': h['index'],
            'price': h['price'],
            'type': 'resistance',
            'threshold': threshold,
            'avg_range': avg_body
        })
    for l in swing_lows:
        if l['index'] < 20:
            continue
        levels.append({
            'index': l['index'],
            'price': l['price'],
            'type': 'support',
            'threshold': threshold,
            'avg_range': avg_body
        })

    # پیدا کردن محل شکست هر سطح
    for lvl in levels:
        broken_idx = len(df) - 1
        l_type = lvl['type']
        for i in range(lvl['index'] + 1, len(df)):
            close = df['close'].iloc[i]
            if (l_type == 'resistance' and close > lvl['price'] + lvl['threshold']) or \
               (l_type == 'support' and close < lvl['price'] - lvl['threshold']):
                broken_idx = i - 1
                break
        lvl['broken_at'] = broken_idx

    # *** گروه‌بندی مبتنی بر زمان و همپوشانی ***
    # مرتب‌سازی بر اساس زمان (index)
    levels_sorted_by_time = sorted(levels, key=lambda x: x['index'])

    groups = []
    for lvl in levels_sorted_by_time:
        # آیا می‌تواند به گروه آخر اضافه شود؟
        added = False
        for group in groups:
            # بررسی هم‌نوع بودن
            if group[0]['type'] != lvl['type']:
                continue
            # بررسی فاصله قیمتی با میانگین گروه
            avg_price_group = np.mean([x['price'] for x in group])
            if abs(lvl['price'] - avg_price_group) > merge_factor * avg_body:
                continue
            # شرط کلیدی: همپوشانی زمانی
            # گروه تا min(broken_at) اعتبار دارد. lvl باید قبل از اتمام اعتبار گروه ظاهر شود
            group_end = min(x['broken_at'] for x in group)
            if lvl['index'] <= group_end:
                # همپوشانی دارد -> اضافه کن
                group.append(lvl)
                added = True
                break
        if not added:
            groups.append([lvl])

    # ساخت اشکال نهایی
    shapes = []
    for group in groups:
        l_type = group[0]['type']
        start_idx = min(x['index'] for x in group)
        # پایان ناحیه = اولین شکست در گروه (min broken_at)
        end_idx = min(x['broken_at'] for x in group)
        min_price = min(x['price'] for x in group)
        max_price = max(x['price'] for x in group)

        if len(group) > 1:
            color = "rgba(255, 0, 0, 0.2)" if l_type == 'resistance' else "rgba(0, 255, 0, 0.2)"
            shapes.append({
                'type': 'rect',
                'x0': start_idx, 'x1': end_idx,
                'y0': min_price, 'y1': max_price,
                'fillcolor': color,
                'line_width': 0
            })
        else:
            lvl = group[0]
            color = 'red' if l_type == 'resistance' else 'green'
            shapes.append({
                'type': 'line',
                'x0': lvl['index'], 'y0': lvl['price'],
                'x1': lvl['broken_at'], 'y1': lvl['price'],
                'line_color': color,
                'line_width': 2,
                'line_dash': 'dot'
            })
    return shapes


def plot_leg_chart(df, detail_list, highlights, sr_shapes):
    """رسم نمودار با هایلایت لگ‌ها، جزئیات هاور و خطوط حمایت/مقاومت"""
    df = df.copy()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=[0.8, 0.2])

    y_low = df['low'].min()
    y_high = df['high'].max()

    # هایلایت لگ‌ها (عمودی کامل)
    for start, end, label in highlights:
        color = "rgba(0,255,0,0.05)" if label == 'Bullish Leg' else "rgba(255,0,0,0.05)"
        fig.add_shape(
            type="rect",
            x0=start - 0.5, x1=end + 0.5,
            y0=y_low, y1=y_high,
            fillcolor=color,
            line=dict(width=0),
            row=1, col=1
        )

    # خطوط حمایت/مقاومت
    for shape in sr_shapes:
        if shape['type'] == 'rect':
            fig.add_shape(
                type='rect',
                x0=shape['x0'], y0=shape['y0'],
                x1=shape['x1'], y1=shape['y1'],
                fillcolor=shape['fillcolor'],
                line=dict(width=0),
                row=1, col=1
            )
        else:  # line
            fig.add_shape(
                type='line',
                x0=shape['x0'], y0=shape['y0'],
                x1=shape['x1'], y1=shape['y1'],
                line=dict(color=shape['line_color'], width=shape['line_width'],
                          dash=shape.get('line_dash', 'solid')),
                row=1, col=1
            )

    # کندل استیک با هاور
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name='Price',
        text=detail_list,
        hovertext=detail_list,
        hoverinfo='x+y+text'
    ), row=1, col=1)

    # حجم
    colors = ['green' if row['close'] >= row['open'] else 'red' for _, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['volume'], marker_color=colors, name='Volume'), row=2, col=1)

    fig.update_layout(
        template='plotly_dark',
        height=800,
        title='Leg Strength + Support/Resistance (including broken levels, drawn until break)',
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    fig.show()


# ======================== اجرا ========================
if __name__ == "__main__":
    raw_csv = "SI=F_5m.csv"
    import os
    if not os.path.exists(raw_csv):
        print(f"File {raw_csv} not found.")
        sys.exit(1)

    df = pd.read_csv(raw_csv)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])

    df = enrich_candles(df)

    n_candles = 400
    df_chart = df.tail(n_candles).reset_index(drop=True)

    leg_labels = detect_legs(df_chart, min_leg_candles=3, merge_gap=8)
    micro_channels, _ = find_channels(df_chart)
    detail_list, highlights = compute_detailed_strength(df_chart, leg_labels, micro_channels)
    sr_shapes = prepare_sr_shapes_with_broken(df_chart, leg_labels)
    plot_leg_chart(df_chart, detail_list, highlights, sr_shapes)