# analysis/breakouts.py
import pandas as pd
import numpy as np

def detect_breakouts(df, sr_shapes, range_zones, 
                     avg_body_lookback=100,
                     body_factor_good=2.5,
                     shadow_pct_good=0.15,
                     shadow_pct_bad=0.3,
                     close_extra_factor=1.5):
    """
    تشخیص کندل‌های بریک‌اوت و دسته‌بندی آن‌ها به سه نوع:
    1. Range Breakout
    2. Inside Range Breakout
    3. Normal Breakout
    
    اضافه شده:
    - quality: 'good', 'perfect', 'bad', یا None
    - follow_through: لیستی از دیکشنری برای کندل اول و دوم بعد از بریک‌اوت
    """
    body = (df['close'] - df['open']).abs()
    rolling_avg_body = body.rolling(window=avg_body_lookback, min_periods=1).mean()
    
    breakouts = [{'breakout': False, 'type': '', 'direction': '', 'quality': None, 'follow_through': []} 
                 for _ in range(len(df))]
    
    # --- گام اول: Range Breakouts ---
    if range_zones:
        for rz in range_zones:
            brk_idx = rz['x1'] + 1
            if brk_idx < len(df):
                close = df['close'].iloc[brk_idx]
                if close > rz['y1']:
                    breakouts[brk_idx] = _create_breakout_dict(
                        df, brk_idx, 'up', 'Range Breakout', rolling_avg_body,
                        body_factor_good, shadow_pct_good, shadow_pct_bad,
                        close_extra_factor, rz['y1']
                    )
                elif close < rz['y0']:
                    breakouts[brk_idx] = _create_breakout_dict(
                        df, brk_idx, 'down', 'Range Breakout', rolling_avg_body,
                        body_factor_good, shadow_pct_good, shadow_pct_bad,
                        close_extra_factor, rz['y0']
                    )
    
    # --- گام دوم: S/R Breakouts ---
    if sr_shapes:
        for sr in sr_shapes:
            brk_idx = sr['x1'] + 1
            if brk_idx < len(df):
                if breakouts[brk_idx]['breakout'] and breakouts[brk_idx]['type'] == 'Range Breakout':
                    continue
                direction = 'up' if sr['zone_type'] == 'resistance' else 'down'
                is_inside_range = False
                if range_zones:
                    for rz in range_zones:
                        if rz['x0'] <= brk_idx <= rz['x1']:
                            is_inside_range = True
                            break
                b_type = 'Inside Range Breakout' if is_inside_range else 'Normal Breakout'
                broken_level = sr['y1'] if direction == 'up' else sr['y0']
                breakouts[brk_idx] = _create_breakout_dict(
                    df, brk_idx, direction, b_type, rolling_avg_body,
                    body_factor_good, shadow_pct_good, shadow_pct_bad,
                    close_extra_factor, broken_level
                )
    
    # اضافه کردن فالوو ترو
    for i, b in enumerate(breakouts):
        if b['breakout']:
            b['follow_through'] = _get_follow_through(df, i, b['direction'], rolling_avg_body)
    
    return breakouts


def _create_breakout_dict(df, idx, direction, b_type, rolling_avg_body,
                          body_factor_good, shadow_pct_good, shadow_pct_bad,
                          close_extra_factor, broken_level):
    candle = df.iloc[idx]
    high = candle['high']
    low = candle['low']
    close = candle['close']
    open_ = candle['open']
    body_val = abs(close - open_)
    avg_body = rolling_avg_body.iloc[idx]
    
    candle_height = high - low
    if candle_height == 0:
        candle_height = 1e-8
    
    if direction == 'up':
        upper_shadow = high - max(close, open_)
        shadow_ratio = upper_shadow / candle_height
        close_above_level = close - broken_level
    else:
        lower_shadow = min(close, open_) - low
        shadow_ratio = lower_shadow / candle_height
        close_above_level = broken_level - close
    
    big_body = body_val > body_factor_good * avg_body if avg_body > 0 else False
    good_shadow = shadow_ratio < shadow_pct_good
    bad_shadow = shadow_ratio > shadow_pct_bad
    perfect_close = close_above_level > close_extra_factor * avg_body if avg_body > 0 else False
    
    quality = None
    if big_body and good_shadow:
        quality = 'perfect' if perfect_close else 'good'
    elif bad_shadow and not perfect_close:
        quality = 'bad'
    
    return {
        'breakout': True,
        'type': b_type,
        'direction': direction,
        'quality': quality,
        'follow_through': []
    }


def _get_follow_through(df, breakout_idx, direction, rolling_avg_body):
    """
    تشخیص Follow Through

    قوانین:

    FOLLOW THROUGH 1:
    -----------------
    - کندل هم‌جهت:
        trend bar -> good
        غیر trend -> normal

    - کندل مخالف:
        اگر شدوی مخالف > 2 برابر بدنه خودش باشد -> bad
        وگرنه -> normal

    - دوجی:
        اول شدو چک می‌شود
        اگر شدوی بزرگ داشته باشد -> bad
        وگرنه -> normal


    FOLLOW THROUGH 2:
    -----------------
    - کندل هم‌جهت:
        trend bar -> good
        غیر trend -> normal

    - کندل مخالف:
        اگر:
            شدوی بزرگ داشته باشد
            یا بدنه > میانگین بدنه باشد
        -> bad

        وگرنه -> normal
    """

    follow_list = []

    for offset in [1, 2]:

        idx = breakout_idx + offset

        if idx >= len(df):
            break

        candle = df.iloc[idx]

        candle_type = candle.get('candle_type', '')

        # --------------------------------
        # اطلاعات کندل
        # --------------------------------
        candle_open = candle['open']
        candle_close = candle['close']
        candle_high = candle['high']
        candle_low = candle['low']

        # --------------------------------
        # جهت واقعی کندل
        # --------------------------------
        if candle_close > candle_open:
            candle_dir = 'up'

        elif candle_close < candle_open:
            candle_dir = 'down'

        else:
            candle_dir = 'neutral'

        # --------------------------------
        # اندازه بدنه
        # --------------------------------
        body_val = abs(candle_close - candle_open)

        avg_body = rolling_avg_body.iloc[idx]

        # --------------------------------
        # دوجی
        # --------------------------------
        is_doji = 'دوجی' in candle_type

        is_small_body = body_val < avg_body

        is_doji_small_body = is_doji and is_small_body

        # --------------------------------
        # شدوی بزرگ
        # شدو با بدنه خود کندل مقایسه می‌شود
        # --------------------------------
        if direction == 'up':

            upper_shadow = (
                candle_high - max(candle_close, candle_open)
            )

            big_shadow = (
                upper_shadow > 2 * body_val
                if body_val > 0 else False
            )

        else:

            lower_shadow = (
                min(candle_close, candle_open) - candle_low
            )

            big_shadow = (
                lower_shadow > 2 * body_val
                if body_val > 0 else False
            )

        classification = None

        # ====================================================
        # FOLLOW THROUGH 1
        # ====================================================
        if offset == 1:

            # ================================================
            # BREAKOUT UP
            # ================================================
            if direction == 'up':

                # --------------------------------
                # کندل صعودی
                # --------------------------------
                if candle_dir == 'up':

                    classification = (
                        'good'
                        if 'ترندبار' in candle_type
                        else 'normal'
                    )

                # --------------------------------
                # کندل نزولی
                # --------------------------------
                elif candle_dir == 'down':

                    # اول شدو چک شود
                    if big_shadow:
                        classification = 'bad'

                    # بعد دوجی
                    elif is_doji_small_body:
                        classification = 'normal'

                    else:
                        classification = 'bad'

                else:
                    classification = 'normal'

            # ================================================
            # BREAKOUT DOWN
            # ================================================
            else:

                # --------------------------------
                # کندل نزولی
                # --------------------------------
                if candle_dir == 'down':

                    classification = (
                        'good'
                        if 'ترندبار' in candle_type
                        else 'normal'
                    )

                # --------------------------------
                # کندل صعودی
                # --------------------------------
                elif candle_dir == 'up':

                    # اول شدو چک شود
                    if big_shadow:
                        classification = 'bad'

                    # بعد دوجی
                    elif is_doji_small_body:
                        classification = 'normal'

                    else:
                        classification = 'bad'

                else:
                    classification = 'normal'

        # ====================================================
        # FOLLOW THROUGH 2
        # ====================================================
        else:

            # ================================================
            # BREAKOUT UP
            # ================================================
            if direction == 'up':

                # کندل هم‌جهت
                if candle_dir == 'up':

                    classification = (
                        'good'
                        if 'ترندبار' in candle_type
                        else 'normal'
                    )

                # کندل مخالف
                elif candle_dir == 'down':

                    if big_shadow or body_val > avg_body:
                        classification = 'bad'
                    else:
                        classification = 'normal'

                else:
                    classification = 'normal'

            # ================================================
            # BREAKOUT DOWN
            # ================================================
            else:

                # کندل هم‌جهت
                if candle_dir == 'down':

                    classification = (
                        'good'
                        if 'ترندبار' in candle_type
                        else 'normal'
                    )

                # کندل مخالف
                elif candle_dir == 'up':

                    if big_shadow or body_val > avg_body:
                        classification = 'bad'
                    else:
                        classification = 'normal'

                else:
                    classification = 'normal'

        # --------------------------------
        # ذخیره نتیجه
        # --------------------------------
        follow_list.append({
            'index': idx,
            'candle_type': candle_type,
            'direction': candle_dir,
            'classification': classification
        })

    return follow_list


def detect_breakout_phases(df, avg_body_lookback=100, factor=4):
    """تشخیص فازهای بریک‌اوت (بدون تغییر)"""
    if len(df) < 2:
        return []
    
    body = (df['close'] - df['open']).abs()
    rolling_avg = body.shift(1).rolling(window=avg_body_lookback, min_periods=1).mean()
    
    direction = np.where(df['close'] > df['open'], 1,
                         np.where(df['close'] < df['open'], -1, 0))
    
    phases = []
    i = 0
    n = len(df)
    while i < n:
        if direction[i] == 0:
            i += 1
            continue
        
        start = i
        dir_val = direction[i]
        i += 1
        while i < n and direction[i] == dir_val:
            i += 1
        
        has_large = False
        for j in range(start, i):
            if body.iloc[j] > factor * rolling_avg.iloc[j]:
                has_large = True
                break
        if has_large:
            y_low = df['low'].iloc[start:i].min()
            y_high = df['high'].iloc[start:i].max()
            phases.append({
                'start': start,
                'end': i - 1,
                'direction': 'up' if dir_val == 1 else 'down',
                'y0': y_low,
                'y1': y_high
            })
    
    return phases