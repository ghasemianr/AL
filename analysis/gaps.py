# analysis/gaps.py
import numpy as np
import pandas as pd

def find_gaps(df, support_lines, resistance_lines, leg_labels,
              avg_body_lookback=100, break_mult=2.0,
              close_mult=1.5, negative_wick_mult=3.0):
    """
    تشخیص گپ‌های صعودی/نزولی با مستطیل پویا که با نفوذ قیمت کوچک می‌شود.
    تنها در امتداد اصلی‌ترین خط روند صعودی/نزولی (کمترین شیب) بررسی می‌کند.

    پارامترها
    ----------
    df : DataFrame شامل open, high, low, close
    support_lines : خطوط روند صعودی (حمایت)
    resistance_lines : خطوط روند نزولی (مقاومت)
    leg_labels : خروجی detect_legs
    avg_body_lookback : تعداد کندل برای میانگین بدنه (پیش‌فرض ۱۰۰)
    break_mult : ضریب بدنه برای تأیید شکست (پیش‌فرض ۲)
    close_mult : ضریب بدنه برای بسته شدن گپ (پیش‌فرض ۱.۵)
    negative_wick_mult : ضریب نفوذ سایه برای گپ منفی (پیش‌فرض ۳)

    خروجی
    ------
    rectangles : list of dict
    gap_details : list of dict
    """
    # ۱. میانگین اندازه بدنه
    body = (df['close'] - df['open']).abs()
    if len(body) >= avg_body_lookback:
        avg_body = body.iloc[-avg_body_lookback:].mean()
    else:
        avg_body = body.mean()

    break_threshold = break_mult * avg_body          # ۲ برابر
    close_threshold = close_mult * avg_body          # ۱.۵ برابر
    neg_wick_threshold = negative_wick_mult * avg_body  # ۳ برابر

    # ۲. استخراج سوئینگ‌ها از لگ‌ها (با احتساب محدوده‌های خنثی)
    swing_highs, swing_lows = _extract_swings_from_legs(df, leg_labels)

    rectangles = []
    gap_details = []

    # --- انتخاب غالب‌ترین خط روند در هر جهت ---
    main_resistance = _select_main_trendline(resistance_lines, 'bearish')
    main_support = _select_main_trendline(support_lines, 'bullish')

    # ۳. گپ‌های نزولی (امتداد خط مقاومت غالب)
    if main_resistance is not None:
        line = main_resistance
        x_start, x_end = int(line['x1']), int(line['x2'])

        lows_in_trend = [l for l in swing_lows if x_start <= l['index'] <= x_end]
        highs_in_trend = [h for h in swing_highs if x_start <= h['index'] <= x_end]

        lows_in_trend.sort(key=lambda x: x['index'])
        highs_in_trend.sort(key=lambda x: x['index'])

        i = 0
        while i < len(lows_in_trend):
            valley = lows_in_trend[i]

            # --- شکست valley به سمت پایین ---
            breakout_idx = None
            for idx in range(valley['index'] + 1, min(x_end, len(df) - 1) + 1):
                if df['close'].iloc[idx] < valley['price'] - break_threshold:
                    breakout_idx = idx
                    break
            if breakout_idx is None:
                i += 1
                continue

            # --- دره جدید بعد از شکست ---
            next_valley = None
            for j in range(i + 1, len(lows_in_trend)):
                if lows_in_trend[j]['index'] >= breakout_idx:
                    next_valley = lows_in_trend[j]
                    break
            if next_valley is None:
                i += 1
                continue

            # --- قله‌های معتبر (کاملاً زیر دره شکسته شده) ---
            initial_highs = [h for h in highs_in_trend
                             if h['index'] >= next_valley['index']
                             and h['price'] < valley['price']]
            if not initial_highs:
                i = lows_in_trend.index(next_valley)
                continue
            initial_peak = max(initial_highs, key=lambda x: x['price'])

            # اسکن تا آخرین کندل
            scan_end = min(len(df) - 1, x_end)

            # --- اسکن پویا ---
            status, close_idx, final_peak_price, final_peak_idx = _bearish_gap_dynamic(
                df, valley['price'], next_valley['index'],
                initial_peak['price'], initial_peak['index'],
                avg_body, close_threshold, neg_wick_threshold,
                scan_end
            )

            # --- مستطیل ---
            rect_x0 = valley['index']
            rect_y0 = valley['price']          # بالای مستطیل
            rect_y1 = final_peak_price        # پایین مستطیل
            rect_x1 = min(len(df) - 1, x_end) # تا انتهای نمودار

            rectangles.append({
                'x0': rect_x0, 'x1': rect_x1,
                'y0': rect_y0, 'y1': rect_y1,
                'color': 'red', 'status': status
            })
            gap_details.append({
                'direction': 'bearish',
                'level': valley['price'],
                'pullback': final_peak_price,
                'pullback_idx': final_peak_idx,
                'close_idx': close_idx,
                'status': status
            })

            i = lows_in_trend.index(next_valley)

    # ۴. گپ‌های صعودی (امتداد خط حمایت غالب)
    if main_support is not None:
        line = main_support
        x_start, x_end = int(line['x1']), int(line['x2'])

        highs_in_trend = [h for h in swing_highs if x_start <= h['index'] <= x_end]
        lows_in_trend = [l for l in swing_lows if x_start <= l['index'] <= x_end]

        highs_in_trend.sort(key=lambda x: x['index'])
        lows_in_trend.sort(key=lambda x: x['index'])

        i = 0
        while i < len(highs_in_trend):
            peak = highs_in_trend[i]

            # --- شکست peak به سمت بالا ---
            breakout_idx = None
            for idx in range(peak['index'] + 1, min(x_end, len(df) - 1) + 1):
                if df['close'].iloc[idx] > peak['price'] + break_threshold:
                    breakout_idx = idx
                    break
            if breakout_idx is None:
                i += 1
                continue

            # --- قله جدید بعد از شکست ---
            next_peak = None
            for j in range(i + 1, len(highs_in_trend)):
                if highs_in_trend[j]['index'] >= breakout_idx:
                    next_peak = highs_in_trend[j]
                    break
            if next_peak is None:
                i += 1
                continue

            # --- کف‌های معتبر (کاملاً بالای قله شکسته شده) ---
            initial_lows = [l for l in lows_in_trend
                            if l['index'] >= next_peak['index']
                            and l['price'] > peak['price']]
            if not initial_lows:
                i = highs_in_trend.index(next_peak)
                continue
            initial_valley = min(initial_lows, key=lambda x: x['price'])

            # اسکن تا آخرین کندل
            scan_end = min(len(df) - 1, x_end)

            # --- اسکن پویا ---
            status, close_idx, final_valley_price, final_valley_idx = _bullish_gap_dynamic(
                df, peak['price'], next_peak['index'],
                initial_valley['price'], initial_valley['index'],
                avg_body, close_threshold, neg_wick_threshold,
                scan_end
            )

            # --- مستطیل ---
            rect_x0 = peak['index']
            rect_y0 = peak['price']            # پایین مستطیل
            rect_y1 = final_valley_price       # بالای مستطیل
            rect_x1 = min(len(df) - 1, x_end)  # تا انتهای نمودار

            rectangles.append({
                'x0': rect_x0, 'x1': rect_x1,
                'y0': rect_y0, 'y1': rect_y1,
                'color': 'green', 'status': status
            })
            gap_details.append({
                'direction': 'bullish',
                'level': peak['price'],
                'pullback': final_valley_price,
                'pullback_idx': final_valley_idx,
                'close_idx': close_idx,
                'status': status
            })

            i = highs_in_trend.index(next_peak)

    return rectangles, gap_details


# ======================================================================
# توابع کمکی
# ======================================================================

def _select_main_trendline(lines, direction):
    """
    انتخاب خط روند غالب: برای صعودی کمترین شیب مثبت، برای نزولی کمترین قدرمطلق شیب منفی.
    """
    if not lines:
        return None
    if direction == 'bullish':
        # شیب‌های مثبت (حمایت)
        valid = [l for l in lines if l['slope'] > 0]
        if not valid:
            return None
        return min(valid, key=lambda l: l['slope'])   # کمترین شیب (مسطح‌ترین)
    else:  # bearish
        valid = [l for l in lines if l['slope'] < 0]
        if not valid:
            return None
        return min(valid, key=lambda l: abs(l['slope']))  # کمترین قدرمطلق شیب


def _extract_swings_from_legs(df, leg_labels):
    """... (بدون تغییر، مطابق آخرین نسخه) ..."""
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


def _bearish_gap_dynamic(df, broken_level, scan_start, init_peak_price, init_peak_idx,
                         avg_body, close_threshold, neg_wick_threshold, end_idx):
    """... (بدون تغییر) ..."""
    current_peak = init_peak_price
    current_peak_idx = init_peak_idx
    negative_seen = False
    for i in range(scan_start + 1, end_idx + 1):
        c = df['close'].iloc[i]
        h = df['high'].iloc[i]
        if c >= broken_level + close_threshold:
            return 'closed', i, current_peak, current_peak_idx
        if h > current_peak and h < broken_level:
            current_peak = h
            current_peak_idx = i
        if h > broken_level and (h - broken_level) <= neg_wick_threshold \
                and c < broken_level + close_threshold:
            negative_seen = True
    if negative_seen:
        return 'negative', None, current_peak, current_peak_idx
    return 'open', None, current_peak, current_peak_idx


def _bullish_gap_dynamic(df, broken_level, scan_start, init_valley_price, init_valley_idx,
                         avg_body, close_threshold, neg_wick_threshold, end_idx):
    """... (بدون تغییر) ..."""
    current_valley = init_valley_price
    current_valley_idx = init_valley_idx
    negative_seen = False
    for i in range(scan_start + 1, end_idx + 1):
        c = df['close'].iloc[i]
        l = df['low'].iloc[i]
        if c <= broken_level - close_threshold:
            return 'closed', i, current_valley, current_valley_idx
        if l < current_valley and l > broken_level:
            current_valley = l
            current_valley_idx = i
        if l < broken_level and (broken_level - l) <= neg_wick_threshold \
                and c > broken_level - close_threshold:
            negative_seen = True
    if negative_seen:
        return 'negative', None, current_valley, current_valley_idx
    return 'open', None, current_valley, current_valley_idx