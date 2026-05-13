# analysis/channels_and_trendlines.py
import numpy as np
from config import TOUCH_TOLERANCE, TOUCH_REQUIREMENT

def find_channels(df):
    # ... (همان کد قبلی بدون تغییر) ...
    micro_channels = []
    tight_channels = []
    # --- میکروکانال ---
    bullish_count = bearish_count = 0
    start_idx = 0
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['open'].iloc[i]:
            if bullish_count == 0:
                start_idx = i
            bullish_count += 1
            bearish_count = 0
        elif df['close'].iloc[i] < df['open'].iloc[i]:
            if bearish_count == 0:
                start_idx = i
            bearish_count += 1
            bullish_count = 0
        else:
            bullish_count = bearish_count = 0

        if bullish_count >= 5 and (i == len(df)-1 or df['close'].iloc[i+1] <= df['open'].iloc[i+1]):
            valid = True
            for k in range(start_idx+1, i+1):
                if df['low'].iloc[k] < df['low'].iloc[k-1]:
                    valid = False
                    break
            if valid:
                micro_channels.append({'type': 'bullish', 'start': start_idx, 'end': i})

        if bearish_count >= 5 and (i == len(df)-1 or df['close'].iloc[i+1] >= df['open'].iloc[i+1]):
            valid = True
            for k in range(start_idx+1, i+1):
                if df['high'].iloc[k] > df['high'].iloc[k-1]:
                    valid = False
                    break
            if valid:
                micro_channels.append({'type': 'bearish', 'start': start_idx, 'end': i})

    # --- کانال فشرده ---
    i = 0
    while i < len(df) - 8:
        j = i + 1
        while j < len(df) and df['low'].iloc[j] >= df['low'].iloc[j-1]:
            j += 1
        if j - i >= 8:
            tight_channels.append({'type': 'bullish', 'start': i, 'end': j-1})
            i = j
            continue
        k = i + 1
        while k < len(df) and df['high'].iloc[k] <= df['high'].iloc[k-1]:
            k += 1
        if k - i >= 8:
            tight_channels.append({'type': 'bearish', 'start': i, 'end': k-1})
            i = k
            continue
        i = max(j, k)
    return micro_channels, tight_channels

def build_channel_info_array(df, micro_channels, tight_channels):
    # ... (همان کد قبلی) ...
    channel_info = [''] * len(df)
    for ch in tight_channels:
        ch_type = 'کانال فشرده صعودی' if ch['type'] == 'bullish' else 'کانال فشرده نزولی'
        for idx in range(ch['start'], ch['end'] + 1):
            if channel_info[idx]:
                channel_info[idx] += f' | {ch_type}'
            else:
                channel_info[idx] = ch_type
    for ch in micro_channels:
        ch_type = 'میکروکانال صعودی' if ch['type'] == 'bullish' else 'میکروکانال نزولی'
        for idx in range(ch['start'], ch['end'] + 1):
            if channel_info[idx]:
                channel_info[idx] += f' | {ch_type}'
            else:
                channel_info[idx] = ch_type
    return channel_info

def find_advanced_trendlines(df, raw_levels, leg_labels=None, dy=None, touch_requirement=None):
    """
    خطوط روند معتبر را پیدا می‌کند.
    
    اگر leg_labels داده شود، شرط اضافی اعمال می‌شود:
      - برای خط صعودی (حمایت): اگر نقطه دوم خط (p2) در آخرین کندل باشد،
        و آخرین لگ نزولی باشد و پایین‌ترین قیمت آن لگ در همان کندل آخر رخ داده باشد،
        آن خط رسم نمی‌شود.
      - برای خط نزولی (مقاومت): اگر نقطه دوم خط در آخرین کندل باشد،
        و آخرین لگ صعودی باشد و بالاترین قیمت آن لگ در همان کندل آخر رخ داده باشد،
        آن خط رسم نمی‌شود.
    """
    if touch_requirement is None:
        touch_requirement = TOUCH_REQUIREMENT

    supports = sorted([l for l in raw_levels if l['type'] == 'support'], key=lambda x: x['index'])
    resistances = sorted([l for l in raw_levels if l['type'] == 'resistance'], key=lambda x: x['index'])

    # محاسبه dy
    if dy is None:
        close_prices = df['close'].values
        lookback = 20
        std_dev = np.std(close_prices[:lookback])
        for i in range(lookback, len(close_prices)):
            std_dev = np.std(close_prices[i-lookback:i])
        dy = TOUCH_TOLERANCE * std_dev

    # ---- استخراج اطلاعات آخرین لگ (در صورت وجود leg_labels) ----
    last_leg_type = None       # 'bullish' یا 'bearish'
    last_leg_extreme_at_last = False  # آیا نقطه افراطی (قله/دره) در آخرین کندل است؟
    if leg_labels is not None:
        # یافتن آخرین لگ غیر تهی
        last_label = ''
        last_end = -1
        for idx, lbl in enumerate(leg_labels):
            if lbl != '':
                last_label = lbl
                last_end = idx
        if last_label != '':
            # یافتن شروع لگ
            start_idx = last_end
            while start_idx >= 0 and leg_labels[start_idx] == last_label:
                start_idx -= 1
            start_idx += 1
            if last_label == 'Bullish Leg':
                last_leg_type = 'bullish'
                sub_df = df.iloc[start_idx:last_end+1]
                max_idx = sub_df['high'].idxmax()
                if max_idx == len(df)-1:
                    last_leg_extreme_at_last = True
            elif last_label == 'Bearish Leg':
                last_leg_type = 'bearish'
                sub_df = df.iloc[start_idx:last_end+1]
                min_idx = sub_df['low'].idxmin()
                if min_idx == len(df)-1:
                    last_leg_extreme_at_last = True

    valid_support_lines = []
    valid_resistance_lines = []

    # ----- خطوط صعودی (حمایت) -----
    i = 0
    while i < len(supports) - 1:
        p1 = supports[i]
        found = False
        for j in range(i+1, len(supports)):
            p2 = supports[j]
            slope = (p2['price'] - p1['price']) / (p2['index'] - p1['index'])
            if slope <= 0:
                continue
            # بررسی تماس‌ها و شکست‌ها
            touches = 0
            breakouts = 0
            for k in range(len(supports)):
                pk = supports[k]
                if pk['index'] < p1['index']:
                    continue
                line_y = p1['price'] + slope * (pk['index'] - p1['index'])
                if line_y - dy <= pk['price'] <= line_y + dy:
                    touches += 1
                elif pk['price'] < line_y - dy:
                    breakouts += 1
            if touches >= touch_requirement and breakouts == 0:
                # شرط اضافی برای خط صعودی
                reject = False
                if leg_labels is not None:
                    # اگر نقطه دوم در آخرین کندل باشد
                    if p2['index'] == len(df) - 1:
                        # و آخرین لگ نزولی باشد و پایین‌ترین قیمت آن در آخرین کندل رخ داده باشد
                        if last_leg_type == 'bearish' and last_leg_extreme_at_last:
                            reject = True
                if not reject:
                    end_y = p1['price'] + slope * (len(df) - 1 - p1['index'])
                    valid_support_lines.append({
                        'x1': p1['index'], 'y1': p1['price'],
                        'x2': len(df)-1, 'y2': end_y,
                        'slope': slope, 'touches': touches
                    })
                    i = j
                    found = True
                    break
        if not found:
            i += 1

    # ----- خطوط نزولی (مقاومت) -----
    i = 0
    while i < len(resistances) - 1:
        p1 = resistances[i]
        found = False
        for j in range(i+1, len(resistances)):
            p2 = resistances[j]
            slope = (p2['price'] - p1['price']) / (p2['index'] - p1['index'])
            if slope >= 0:
                continue
            touches = 0
            breakouts = 0
            for k in range(len(resistances)):
                pk = resistances[k]
                if pk['index'] < p1['index']:
                    continue
                line_y = p1['price'] + slope * (pk['index'] - p1['index'])
                if line_y - dy <= pk['price'] <= line_y + dy:
                    touches += 1
                elif pk['price'] > line_y + dy:
                    breakouts += 1
            if touches >= touch_requirement and breakouts == 0:
                reject = False
                if leg_labels is not None:
                    # اگر نقطه دوم در آخرین کندل باشد
                    if p2['index'] == len(df) - 1:
                        # و آخرین لگ صعودی باشد و بالاترین قیمت آن در آخرین کندل رخ داده باشد
                        if last_leg_type == 'bullish' and last_leg_extreme_at_last:
                            reject = True
                if not reject:
                    end_y = p1['price'] + slope * (len(df) - 1 - p1['index'])
                    valid_resistance_lines.append({
                        'x1': p1['index'], 'y1': p1['price'],
                        'x2': len(df)-1, 'y2': end_y,
                        'slope': slope, 'touches': touches
                    })
                    i = j
                    found = True
                    break
        if not found:
            i += 1

    return valid_support_lines, valid_resistance_lines

def find_channel_lines(df, support_lines, resistance_lines, raw_levels, leg_labels,
                       avg_body_lookback=100, dy=None, touch_requirement=2):
    """
    پیدا کردن خطوط کانال موازی با خطوط روند.
    خط کانال از اولین نقطه تماس معتبر (حتی قبل از شروع خط روند) شروع می‌شود،
    به شرطی که در آن ناحیه شکسته نشده باشد.
    """
    # محاسبه میانگین بدنه کندل‌ها
    body = (df['close'] - df['open']).abs()
    if len(body) >= avg_body_lookback:
        avg_body = body.iloc[-avg_body_lookback:].mean()
    else:
        avg_body = body.mean()
    
    touch_tolerance = 1.5 * avg_body
    breakout_tolerance = 2.0 * avg_body
    
    resist_points = [{'index': l['index'], 'price': l['price']} for l in raw_levels if l['type'] == 'resistance']
    support_points = [{'index': l['index'], 'price': l['price']} for l in raw_levels if l['type'] == 'support']
    resist_points.sort(key=lambda x: x['index'])
    support_points.sort(key=lambda x: x['index'])
    
    channel_lines = []
    
    # ========== کانال بالایی (روند صعودی) ==========
    for base in support_lines:
        slope = base['slope']
        x0 = base['x1']
        y0 = base['y1']
        
        # کاندیدهای offset از نقاط مقاومت بعد از x0
        candidates = []
        for pt in resist_points:
            if pt['index'] <= x0:
                continue
            line_y_base = y0 + slope * (pt['index'] - x0)
            offset = pt['price'] - line_y_base
            if offset > 0:
                candidates.append({'index': pt['index'], 'price': pt['price'], 'offset': offset})
        
        if len(candidates) < touch_requirement:
            continue
        
        # بهترین offset (بیشترین تماس با در نظر گرفتن همه نقاط مقاومت)
        best_touches = 0
        best_offset = None
        for cand in candidates:
            offset = cand['offset']
            touches = 0
            for pt in resist_points:
                line_y = y0 + offset + slope * (pt['index'] - x0)
                if abs(pt['price'] - line_y) <= touch_tolerance:
                    touches += 1
            if touches >= touch_requirement:
                if touches > best_touches:
                    best_touches = touches
                    best_offset = offset
                elif touches == best_touches and best_offset is not None:
                    if offset > best_offset:
                        best_offset = offset
        
        if best_offset is None:
            continue
        
        # تابع خط کانال
        line_func = lambda idx: y0 + best_offset + slope * (idx - x0)
        
        # ---- تعیین نقطه شروع خط کانال (اولین تماس معتبر قبل از x0) ----
        # پیدا کردن همه نقاط تماس (همه نقاط مقاومت)
        touch_indices = []
        for pt in resist_points:
            if abs(pt['price'] - line_func(pt['index'])) <= touch_tolerance:
                touch_indices.append(pt['index'])
        if not touch_indices:
            start_idx = x0
        else:
            min_touch = min(touch_indices)
            # بررسی شکست‌ها در بازه [min_touch, x0]
            break_before = False
            for i in range(min_touch, x0 + 1):
                close_price = df['close'].iloc[i]
                line_val = line_func(i)
                if close_price > line_val + breakout_tolerance:
                    break_before = True
                    break
            if break_before:
                # اگر شکستی قبل از x0 رخ داده، از همان x0 شروع کن (یا از آخرین شکست+1)
                # برای سادگی از x0 شروع می‌کنیم
                start_idx = x0
            else:
                start_idx = min_touch
        
        # ---- تعیین نقطه پایان (اولین شکست بعد از start_idx) ----
        break_index = len(df) - 1
        for i in range(start_idx + 1, len(df)):
            close_price = df['close'].iloc[i]
            line_val = line_func(i)
            if close_price > line_val + breakout_tolerance:
                break_index = i - 1
                break
        
        # اگر break_index از start_idx کوچکتر شد، خط رسم نمی‌شود
        if break_index < start_idx:
            continue
        
        x1 = start_idx
        y1 = line_func(start_idx)
        x2 = break_index
        y2 = line_func(break_index)
        
        channel_lines.append({
            'type': 'channel_resistance',
            'x1': x1, 'y1': y1,
            'x2': x2, 'y2': y2,
            'slope': slope,
            'touches': best_touches,
            'base_line': base
        })
    
    # ========== کانال پایینی (روند نزولی) ==========
    for base in resistance_lines:
        slope = base['slope']   # منفی
        x0 = base['x1']
        y0 = base['y1']
        
        candidates = []
        for pt in support_points:
            if pt['index'] <= x0:
                continue
            line_y_base = y0 + slope * (pt['index'] - x0)
            offset = pt['price'] - line_y_base
            if offset < 0:
                candidates.append({'index': pt['index'], 'price': pt['price'], 'offset': offset})
        
        if len(candidates) < touch_requirement:
            continue
        
        best_touches = 0
        best_offset = None
        for cand in candidates:
            offset = cand['offset']
            touches = 0
            for pt in support_points:
                line_y = y0 + offset + slope * (pt['index'] - x0)
                if abs(pt['price'] - line_y) <= touch_tolerance:
                    touches += 1
            if touches >= touch_requirement:
                if touches > best_touches:
                    best_touches = touches
                    best_offset = offset
                elif touches == best_touches and best_offset is not None:
                    if offset < best_offset:
                        best_offset = offset
        
        if best_offset is None:
            continue
        
        line_func = lambda idx: y0 + best_offset + slope * (idx - x0)
        
        # ---- شروع از اولین تماس قبل از x0 (برای نقاط حمایت) ----
        touch_indices = []
        for pt in support_points:
            if abs(pt['price'] - line_func(pt['index'])) <= touch_tolerance:
                touch_indices.append(pt['index'])
        if not touch_indices:
            start_idx = x0
        else:
            min_touch = min(touch_indices)
            break_before = False
            for i in range(min_touch, x0 + 1):
                close_price = df['close'].iloc[i]
                line_val = line_func(i)
                if close_price < line_val - breakout_tolerance:
                    break_before = True
                    break
            if break_before:
                start_idx = x0
            else:
                start_idx = min_touch
        
        # ---- نقطه پایان (شکست) ----
        break_index = len(df) - 1
        for i in range(start_idx + 1, len(df)):
            close_price = df['close'].iloc[i]
            line_val = line_func(i)
            if close_price < line_val - breakout_tolerance:
                break_index = i - 1
                break
        
        if break_index < start_idx:
            continue
        
        x1 = start_idx
        y1 = line_func(start_idx)
        x2 = break_index
        y2 = line_func(break_index)
        
        channel_lines.append({
            'type': 'channel_support',
            'x1': x1, 'y1': y1,
            'x2': x2, 'y2': y2,
            'slope': slope,
            'touches': best_touches,
            'base_line': base
        })
    
    return channel_lines