# analysis/leg_strength.py
import numpy as np
import pandas as pd

def _get_leg_segments(leg_labels):
    """استخراج بازه‌های پیوسته هر لگ با نوع آن (Bullish Leg / Bearish Leg)"""
    segments = []
    i = 0
    n = len(leg_labels)
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


def _c1_micro_channel_score(leg_df, leg_label, micro_channels, leg_start_idx, leg_end_idx, leg_height):
    """
    امتیاز معیار ۱: میکروکانال‌های هم‌جهت (کاملاً یا جزئی درون لگ)
    و درصد پوشش ارتفاع لگ توسط بخش‌های مشترک.
    """
    if leg_label == 'Bullish Leg':
        target_type = 'bullish'
    else:
        target_type = 'bearish'

    total_coverage = 0
    has_micro = False

    for mc in micro_channels:
        if mc['type'] != target_type:
            continue

        # محدوده‌ی هم‌پوشانی
        overlap_start = max(mc['start'], leg_start_idx)
        overlap_end   = min(mc['end'], leg_end_idx)

        # اگر هم‌پوشانی وجود داشته باشد
        if overlap_start <= overlap_end:
            has_micro = True
            # کندل‌های داخل این بخش مشترک (با استفاده از leg_df که ایندکس اصلی را حفظ کرده)
            sub = leg_df.loc[overlap_start:overlap_end]
            ch_high = sub['high'].max()
            ch_low  = sub['low'].min()
            total_coverage += ch_high - ch_low

    if not has_micro:
        return 0

    coverage_pct = (total_coverage / leg_height) * 100
    if coverage_pct > 30:
        return 1.5
    elif coverage_pct > 20:
        return 1.4
    elif coverage_pct > 15:
        return 1.25
    else:
        return 1.0

def _c2_overlap_score(leg_df, leg_label, pre_candle=None):
    """
    امتیاز معیار ۲: درصد کندل‌های متوالی با هم‌پوشانی زیر ۲۰٪
    - اولین کندل لگ با pre_candle مقایسه می‌شود (اگر موجود باشد)
    """
    # ساخت DataFrame توسعه‌یافته
    frames = []
    if pre_candle is not None:
        frames.append(pre_candle.to_frame().T)
    frames.append(leg_df)
    ext_df = pd.concat(frames, ignore_index=True)
    
    n_pairs = len(ext_df) - 1  # تعداد جفت‌کندل‌ها
    if n_pairs <= 0:
        return 0
    
    low_overlap_count = 0  # تعداد جفت‌هایی که هم‌پوشانی زیر ۲۰٪ دارن
    
    for i in range(1, len(ext_df)):
        current = ext_df.iloc[i]
        prev = ext_df.iloc[i - 1]
        intersect_low = max(current['low'], prev['low'])
        intersect_high = min(current['high'], prev['high'])
        intersect = max(0, intersect_high - intersect_low)
        current_range = current['high'] - current['low']
        
        if current_range == 0:
            ratio = 1.0
        else:
            ratio = intersect / current_range
        
        # اگر کندل فعلی در جهت لگ حرکت نکرده باشد، اورلپ ۱۰۰٪
        if leg_label == 'Bullish Leg':
            if current['high'] <= prev['high']:
                ratio = 1.0
        else:  # Bearish Leg
            if current['low'] >= prev['low']:
                ratio = 1.0
        
        if ratio < 0.20:
            low_overlap_count += 1
    
    pct_low_overlap = (low_overlap_count / n_pairs) * 100
    
    if pct_low_overlap > 20:
        return 1.5
    elif pct_low_overlap > 15:
        return 1.0
    elif pct_low_overlap > 10:
        return 0.8
    else:
        return 0


def _c3_close_position_score(leg_df, leg_label):
    """امتیاز معیار ۳: میانگین جایگاه کلوز کندل‌های هم‌جهت لگ"""
    if leg_label == 'Bullish Leg':
        bullish = leg_df[leg_df['close'] > leg_df['open']]
        if len(bullish) == 0:
            return 0
        pos = (bullish['close'] - bullish['low']) / (bullish['high'] - bullish['low']) * 100
    else:
        bearish = leg_df[leg_df['close'] < leg_df['open']]
        if len(bearish) == 0:
            return 0
        pos = (bearish['high'] - bearish['close']) / (bearish['high'] - bearish['low']) * 100

    avg_pos = pos.mean()
    if avg_pos > 90:
        return 2
    elif avg_pos > 85:
        return 1.4
    elif avg_pos > 80:
        return 1.0
    elif avg_pos > 75:
        return 0.75
    elif avg_pos > 70:
        return 0.5
    else:
        return 0

def _c4_big_bodies_score(leg_df, leg_height, avg_body_100):
    """امتیاز معیار ۴: کندل‌های با بدنه بزرگ (≥۳× میانگین ۱۰۰ کندل گذشته) و نسبت آنها به ارتفاع لگ"""
    if avg_body_100 == 0:
        return 0
    total_big_body = 0
    for _, row in leg_df.iterrows():
        body = abs(row['close'] - row['open'])
        if body >= 3 * avg_body_100:
            total_big_body += body
    pct = (total_big_body / leg_height) * 100
    if pct >= 50:
        return 1.5
    elif pct >= 40:
        return 0.8
    elif pct >= 30:
        return 0.5
    else:
        return 0


def _c5_consecutive_opposite_score(leg_df, leg_label, leg_height):
    n_total = len(leg_df)
    if n_total < 5:
        return 0

    if n_total <= 7:
        mult = 0.6
    elif n_total <= 10:
        mult = 0.8
    elif n_total <= 14:
        mult = 1.0
    else:
        mult = 1.2

    if leg_height == 0:
        return 0

    opp_dir = '-' if leg_label == 'Bullish Leg' else '+'
    total_opp_body = 0

    i = 0
    while i < n_total:
        if leg_df.iloc[i]['direction'] == opp_dir:
            j = i
            while j < n_total and leg_df.iloc[j]['direction'] == opp_dir:
                j += 1
            seq_len = j - i
            if seq_len >= 2:
                seq_body = abs(leg_df.iloc[i:j]['close'] - leg_df.iloc[i:j]['open']).sum()
                total_opp_body += seq_body
            i = j
        else:
            i += 1

    pct = (total_opp_body / leg_height) * 100

    if pct <= 5:
        raw = 1.5
    elif pct <= 10:
        raw = 1.25
    elif pct <= 15:
        raw = 0.8
    elif pct <= 25:
        raw = 0.5
    else:
        raw = 0

    return round(raw * mult, 2)


def _c6_avg_body_ratio_score(leg_df, leg_label):
    """امتیاز معیار ۶: نسبت مجموع بدنه کندل‌های مخالف به مجموع بدنه کندل‌های موافق"""
    if leg_label == 'Bullish Leg':
        same_dir = leg_df[leg_df['direction'] == '+']
        opp_dir = leg_df[leg_df['direction'] == '-']
    else:
        same_dir = leg_df[leg_df['direction'] == '-']
        opp_dir = leg_df[leg_df['direction'] == '+']

    # اگر کندل موافق وجود نداشته باشد (نباید اتفاق بیفتد)
    if len(same_dir) == 0:
        return 0
    
    # اگر هیچ کندل مخالفی وجود نداشته باشد → عالی‌ترین حالت
    if len(opp_dir) == 0:
        raw_score = 1.5  # حداکثر امتیاز
    else:
        sum_same = (abs(same_dir['close'] - same_dir['open'])).sum()
        sum_opp = (abs(opp_dir['close'] - opp_dir['open'])).sum()
        
        if sum_same == 0:
            return 0
            
        ratio = sum_opp / sum_same
        
        if ratio < 0.1:
            raw_score = 2
        elif ratio < 0.2:
            raw_score = 1.5
        elif ratio < 0.3:
            raw_score = 1
        elif ratio < 0.4:
            raw_score = 0.8
        elif ratio < 0.5:
            raw_score = 0.5
        else:
            raw_score = 0
    
    # محاسبهٔ ضریب بر اساس طول لگ
    n = len(leg_df)
    if n < 7:
        coef = 0.5
    elif n <= 12:
        coef = 0.8
    else:
        coef = 1.0
    
    return raw_score * coef


def _c7_length_bonus_score(leg_df):
    """پاداش طول لگ (جایگزین معیار حذف شدهٔ C7)"""
    n_candles = len(leg_df)
    return max(0, n_candles - 15) * 0.05


def _c8_gap_score(leg_df, leg_label, leg_height, pre_candle=None, post_candle=None):
    """امتیاز معیار ۸: جمع گپ‌های صعودی (یا نزولی) بین سه‌کندل‌های متوالی داخل لگ.
    گپ‌هایی که بعداً با کندل‌های بعدی پر شوند (هم‌پوشانی) از مجموع کسر می‌شوند."""
    
    # ساخت DataFrame توسعه‌یافته با یک کندل قبل و یک کندل بعد (اگر موجود باشند)
    frames = []
    if pre_candle is not None:
        frames.append(pre_candle.to_frame().T)
    frames.append(leg_df)
    if post_candle is not None:
        frames.append(post_candle.to_frame().T)
    ext_df = pd.concat(frames, ignore_index=True)
    
    n = len(ext_df)
    total_gap = 0.0

    for i in range(n - 2):
        first = ext_df.iloc[i]
        last = ext_df.iloc[i + 2]
        if leg_label == 'Bullish Leg':
            gap = last['low'] - first['high']
            if gap > 0:
                gap_interval = (first['high'], last['low'])   # بازهٔ گپ (خلاء قیمتی)
            else:
                continue
        else:  # Bearish Leg
            gap = first['low'] - last['high']
            if gap > 0:
                gap_interval = (last['high'], first['low'])   # بازهٔ گپ
            else:
                continue

        # همهٔ کندل‌های بعد از کندل سوم این پنجره را بررسی کن
        intersections = []
        for k in range(i + 3, n):
            c = ext_df.iloc[k]
            c_low, c_high = c['low'], c['high']
            # اشتراک کندل با بازهٔ گپ
            inter_start = max(c_low, gap_interval[0])
            inter_end = min(c_high, gap_interval[1])
            if inter_start < inter_end:
                intersections.append((inter_start, inter_end))

        # محاسبهٔ طول یکپارچهٔ هم‌پوشانی‌ها (union)
        if intersections:
            intersections.sort(key=lambda x: x[0])
            union_len = 0.0
            cur_start, cur_end = intersections[0]
            for start, end in intersections[1:]:
                if start <= cur_end:
                    cur_end = max(cur_end, end)
                else:
                    union_len += cur_end - cur_start
                    cur_start, cur_end = start, end
            union_len += cur_end - cur_start
        else:
            union_len = 0.0

        # گپ مؤثر = طول اولیه منهای بخش پر شده
        effective_gap = max(0.0, (gap_interval[1] - gap_interval[0]) - union_len)
        total_gap += effective_gap

    if total_gap == 0:
        return 0

    pct = (total_gap / leg_height) * 100
    if pct > 45:
        return 1.5
    elif pct > 35:
        return 1.2
    elif pct > 25:
        return 0.8
    elif pct > 10:
        return 0.5
    else:
        return 0.3
        
def _c9_increasing_bodies_score(leg_df, leg_label, pre_candle, avg_body_100):
    """
    امتیاز معیار ۹: دنباله‌های با بدنهٔ فزاینده (حداقل ۲ کندل).
    - پنجرهٔ جستجو: [1 کندل قبل از لگ] + [کندل‌های داخل لگ].
    - کندل‌های بعدی هم‌جهت، غیردوجی، بدنهٔ بزرگ‌تر از قبلی و هم‌پوشانی ≤ 40٪.
    - امتیاز پایه بر اساس طول دنباله:
        2 کندل → 0.5 | 3 → 1.0 | 4 → 1.3 | ≥5 → 1.75
    - ضریب قدرت (پیوسته): multiplier = 1.0 + (ratio - 4) * 0.15  برای ratio ≥ 4
      (ratio = فاصلهٔ خالص بدنه / avg_body_100)
      حداکثر ضریب: 2.5
    """
    # ساخت DataFrame توسعه‌یافته
    frames = []
    if pre_candle is not None:
        frames.append(pre_candle.to_frame().T)
    frames.append(leg_df)
    ext_df = pd.concat(frames, ignore_index=True)

    n = len(ext_df)
    if n < 2:
        return 0

    target_dir = '+' if leg_label == 'Bullish Leg' else '-'
    bodies = abs(ext_df['close'] - ext_df['open'])
    is_doji = ext_df['candle_type'].str.startswith('دوجی')

    # محاسبهٔ هم‌پوشانی
    overlaps = [None] * n
    for i in range(1, n):
        cur = ext_df.iloc[i]
        prv = ext_df.iloc[i - 1]
        inter = max(0, min(cur['high'], prv['high']) - max(cur['low'], prv['low']))
        cur_range = cur['high'] - cur['low']
        overlaps[i] = (inter / cur_range) if cur_range > 0 else 1.0

    # مشخصات بهترین دنباله
    best_len = 0
    best_start_idx = -1
    best_end_idx = -1

    i = 0
    while i < n:
        if ext_df.iloc[i]['direction'] != target_dir or is_doji.iloc[i]:
            i += 1
            continue

        j = i + 1
        while j < n:
            if ext_df.iloc[j]['direction'] != target_dir or is_doji.iloc[j]:
                break
            if bodies.iloc[j] <= bodies.iloc[j - 1]:
                break
            if overlaps[j] is not None and overlaps[j] > 0.4:
                break
            j += 1

        length = j - i
        if length >= 2:
            if length > best_len:
                best_len = length
                best_start_idx = i
                best_end_idx = j - 1
            elif length == best_len and i > best_start_idx:
                best_start_idx = i
                best_end_idx = j - 1

        i = j

    if best_len == 0:
        return 0

    # امتیاز پایه بر اساس طول دنباله
    if best_len >= 5:
        base_score = 1.75
    elif best_len == 4:
        base_score = 1.3
    elif best_len == 3:
        base_score = 1.0
    else:  # 2
        base_score = 0.5

    # محاسبه ضریب قدرت پیوسته
    if avg_body_100 > 0:
        first_open = ext_df.iloc[best_start_idx]['open']
        last_close = ext_df.iloc[best_end_idx]['close']

        if leg_label == 'Bullish Leg':
            net_body_distance = last_close - first_open
        else:
            net_body_distance = first_open - last_close

        if net_body_distance < 0:
            net_body_distance = 0

        ratio = net_body_distance / avg_body_100
    else:
        ratio = 0

    if ratio < 4:
        multiplier = 1.0
    else:
        multiplier = 1.0 + (ratio - 4) * 0.15
        multiplier = min(multiplier, 2.5)   # سقف ضریب

    return round(base_score * multiplier, 2)

def _c10_god_candles_score(leg_df, leg_label, avg_body_100):
    """
    امتیاز معیار ۱۰: GOD Candles – کندل‌های انفجاری هم‌جهت با لگ.
    - تک‌کندل: بدنه > 3.5 * avg_body_100 → 1.2
    - دنباله ≥۲: هر کندل > 2.5 * avg_body_100 و کندل‌های بعدی هم‌پوشانی ≤ 20٪.
      امتیاز: 2 کندل → 1.5 | 3 کندل → 1.8 | 4+ کندل → 2.5
    """
    n = len(leg_df)
    if n == 0 or avg_body_100 == 0:
        return 0

    target_dir = '+' if leg_label == 'Bullish Leg' else '-'
    bodies = abs(leg_df['close'] - leg_df['open'])

    # محاسبه هم‌پوشانی
    overlaps = [None] * n
    for i in range(1, n):
        cur = leg_df.iloc[i]
        prv = leg_df.iloc[i - 1]
        intersect = max(0, min(cur['high'], prv['high']) - max(cur['low'], prv['low']))
        cur_range = cur['high'] - cur['low']
        overlaps[i] = (intersect / cur_range) if cur_range > 0 else 1.0

    max_len = 0
    i = 0
    while i < n:
        if leg_df.iloc[i]['direction'] != target_dir:
            i += 1
            continue
        if bodies.iloc[i] < 3.5 * avg_body_100:
            i += 1
            continue

        start_idx = i
        j = i + 1
        while j < n:
            if leg_df.iloc[j]['direction'] != target_dir:
                break
            if bodies.iloc[j] < 3.5 * avg_body_100:
                break
            # بعد از اولین کندل، هم‌پوشانی باید ≤ 20٪
            if j > start_idx and overlaps[j] is not None and overlaps[j] > 0.20:
                break
            j += 1

        length = j - i
        # اگر طول ۱ باشد، شرط بدنه سخت‌گیرانه‌تر است
        if length == 1 and bodies.iloc[i] <= 4.5 * avg_body_100:
            length = 0

        if length > max_len:
            max_len = length

        i = j

    if max_len >= 4:
        return 2.5
    elif max_len == 3:
        return 1.8
    elif max_len == 2:
        return 1.5
    elif max_len == 1:
        return 1.2
    return 0

def calculate_leg_strength(df, leg_labels, micro_channels, tight_channels=None):
    """
    محاسبه قدرت هر لگ (صعودی/نزولی) و برگرداندن لیست امتیازها برای تمام کندل‌ها.
    امتیاز فقط برای کندل‌های داخل لگ برگردانده می‌شود (عدد)، بقیه خالی.
    """
    strength_list = [''] * len(df)
    segments = _get_leg_segments(leg_labels)

    for start, end, label in segments:
        leg_df = df.iloc[start:end+1].copy()
        leg_height = leg_df['high'].max() - leg_df['low'].min()
        if leg_height == 0:
            continue

        n_candles = len(leg_df)

        # میانگین بدنه ۱۰۰ کندل گذشته (تا قبل از شروع لگ)
        lookback_start = max(0, start - 100)
        avg_body_100 = (abs(df.iloc[lookback_start:start]['close'] - df.iloc[lookback_start:start]['open'])).mean()
        pre_candle = df.iloc[start - 1] if start > 0 else None
        post_candle = df.iloc[end + 1] if end + 1 < len(df) else None

        # محاسبهٔ ضریب کارایی بر اساس طول و ارتفاع لگ (به‌صورت پیوسته)
        vertical_per_candle = leg_height / n_candles if n_candles > 0 else 0.0
        if avg_body_100 > 0:
            efficiency_ratio = vertical_per_candle / avg_body_100
        else:
            efficiency_ratio = 0.0


                # ضریب پیوسته با قابلیت پاداش: توان 0.5 جریمه و پاداش را متعادل می‌کند
        length_eff_coef = efficiency_ratio ** 0.35

        score = 0

        # 1. میکروکانال
        score += _c1_micro_channel_score(leg_df, label, micro_channels, start, end, leg_height)

        # 2. همپوشانی (با احتساب pre_candle)
        score += _c2_overlap_score(leg_df, label)

        # 4. بدنه‌های بزرگ (C4)
        c4_score = _c4_big_bodies_score(leg_df, leg_height, avg_body_100)
        score += c4_score

        # 3. جایگاه کلوز (C3) – فقط در صورتی که C4 امتیاز مثبت گرفته باشد
        c3_score = _c3_close_position_score(leg_df, label)
        if c4_score > 0:
            score += c3_score
        # اگر c4_score == 0 باشد، c3_score نادیده گرفته می‌شود

        # 5. توالی مخالف
        score += _c5_consecutive_opposite_score(leg_df, label, leg_height)

        # 6. نسبت بدنه‌ها
        score += _c6_avg_body_ratio_score(leg_df, label)

        # 7. طول لگ
        score += _c7_length_bonus_score(leg_df)
        
        # 8. گپ‌ها
        score += _c8_gap_score(leg_df, label, leg_height, pre_candle, post_candle)

        # 9. بدنه‌های افزایشی
        score += _c9_increasing_bodies_score(leg_df, label, None, avg_body_100)

        # 10. گاد کندل‌ها
        score += _c10_god_candles_score(leg_df, label, avg_body_100)

        # اعمال ضریب کارایی طول/حرکت
        score *= length_eff_coef

        strength_val = round(score, 2)
        for idx in range(start, end + 1):
            strength_list[idx] = strength_val

    return strength_list