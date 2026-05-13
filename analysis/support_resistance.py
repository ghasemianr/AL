# analysis/support_resistance.py
import numpy as np
from config import MERGE_FACTOR, BREAK_FACTOR

def find_levels_from_legs(df, leg_labels, avg_body_lookback=100, break_factor=None):
    """
    یافتن سطوح حمایت/مقاومت فقط از نقاط اکسترمم هر لگ.
    (نسخه اصلی - بدون تغییر)
    """
    if break_factor is None:
        break_factor = BREAK_FACTOR

    body = (df['close'] - df['open']).abs()
    if len(body) >= avg_body_lookback:
        avg_body = body.iloc[-avg_body_lookback:].mean()
    else:
        avg_body = body.mean()

    swing_highs, swing_lows = _extract_swings_from_legs(df, leg_labels)

    levels = []
    for h in swing_highs:
        if h['index'] < 20:
            continue
        levels.append({
            'index': h['index'],
            'price': h['price'],
            'type': 'resistance',
            'threshold': break_factor * avg_body,
            'avg_range': avg_body
        })
    for l in swing_lows:
        if l['index'] < 20:
            continue
        levels.append({
            'index': l['index'],
            'price': l['price'],
            'type': 'support',
            'threshold': break_factor * avg_body,
            'avg_range': avg_body
        })
    return levels


def _extract_swings_from_legs(df, leg_labels):
    """استخراج سوئینگ‌های بالا (سقف) و پایین (کف) از لگ‌ها (بدون تغییر)"""
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


def prepare_sr_shapes_advanced(df, leg_labels, break_factor=None, merge_factor=None, avg_body_lookback=100):
    """
    رسم خطوط/مستطیل حمایت و مقاومت با منطق پیشرفته:
      - سطوح حتی اگر شکسته شوند، تا محل شکست کشیده می‌شوند.
      - گروه‌بندی بر اساس زمان و همپوشانی (نه فقط قیمت).
      - انتهای ناحیه = اولین شکست در گروه (min broken_at).
    """
    if break_factor is None:
        break_factor = BREAK_FACTOR
    if merge_factor is None:
        merge_factor = MERGE_FACTOR

    body = (df['close'] - df['open']).abs()
    if len(body) >= avg_body_lookback:
        avg_body = body.iloc[-avg_body_lookback:].mean()
    else:
        avg_body = body.mean()
    threshold = break_factor * avg_body

    swing_highs, swing_lows = _extract_swings_from_legs(df, leg_labels)

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

    # گروه‌بندی مبتنی بر زمان و همپوشانی
    levels_sorted_by_time = sorted(levels, key=lambda x: x['index'])
    groups = []

    for lvl in levels_sorted_by_time:
        added = False
        for group in groups:
            if group[0]['type'] != lvl['type']:
                continue
            avg_price_group = np.mean([x['price'] for x in group])
            if abs(lvl['price'] - avg_price_group) > merge_factor * avg_body:
                continue
            group_end = min(x['broken_at'] for x in group)
            if lvl['index'] <= group_end:
                group.append(lvl)
                added = True
                break
        if not added:
            groups.append([lvl])

    shapes = []
    for group in groups:
        l_type = group[0]['type']
        start_idx = min(x['index'] for x in group)
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
                'line_width': 0,
                'zone_type': l_type          # اضافه شد
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
                'line_dash': 'dot',
                'zone_type': l_type          # اضافه شد
            })
    return shapes