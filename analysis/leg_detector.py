# analysis/leg_detector.py
import pandas as pd
import numpy as np

def heikin_ashi(df):
    """
    محاسبه کندل‌های هیکین آشی و برگرداندن DataFrame جدید با ستون‌های HA_open, HA_high, HA_low, HA_close.
    ورودی df باید شامل open, high, low, close باشد.
    """
    ha_df = pd.DataFrame(index=df.index)
    
    ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4.0
    ha_open = np.empty_like(ha_close)
    ha_open[0] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2.0

    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2.0

    ha_high = np.maximum(df['high'], np.maximum(ha_open, ha_close))
    ha_low = np.minimum(df['low'], np.minimum(ha_open, ha_close))

    ha_df['HA_open'] = ha_open
    ha_df['HA_high'] = ha_high
    ha_df['HA_low'] = ha_low
    ha_df['HA_close'] = ha_close

    return ha_df


def detect_legs(df, min_leg_candles=3, merge_gap=8):
    """
    تشخیص لگ‌های صعودی و نزولی بر اساس هیکین آشی با اعمال قوانین تنظیم انتهای لگ‌ها.
    
    پارامترها:
        df: دیتافریم شامل OHLC اصلی
        min_leg_candles: حداقل تعداد کندل‌های متوالی HA برای تشکیل لگ (پیش‌فرض ۳)
        merge_gap: حداکثر فاصله (تعداد کندل) بین دو لگ هم‌جهت برای ادغام (پیش‌فرض ۸)
    
    خروجی:
        یک لیست با طول len(df) شامل برچسب لگ برای هر کندل.
        مقادیر: 'Bullish Leg', 'Bearish Leg', یا '' (بدون لگ)
    """
    # محاسبه هیکین آشی
    ha = heikin_ashi(df)

    # تشخیص صعودی/نزولی بودن هر کندل HA (صعودی = بسته > باز)
    ha_bullish = ha['HA_close'] > ha['HA_open']

    leg_labels = [''] * len(df)

    # === مرحله ۱: پیدا کردن لگ‌های اولیه ===
    # لگ‌های صعودی
    i = 0
    while i < len(df):
        if ha_bullish.iloc[i]:
            start = i
            while i < len(df) and ha_bullish.iloc[i]:
                i += 1
            length = i - start
            if length >= min_leg_candles:
                for j in range(start, i):
                    leg_labels[j] = 'Bullish Leg'
        else:
            i += 1

    # لگ‌های نزولی
    i = 0
    while i < len(df):
        if not ha_bullish.iloc[i]:
            if ha['HA_close'].iloc[i] == ha['HA_open'].iloc[i]:
                i += 1
                continue
            start = i
            while i < len(df) and (not ha_bullish.iloc[i]) and (ha['HA_close'].iloc[i] != ha['HA_open'].iloc[i]):
                i += 1
            length = i - start
            if length >= min_leg_candles:
                for j in range(start, i):
                    leg_labels[j] = 'Bearish Leg'
        else:
            i += 1

    # === مرحله ۲: ادغام لگ‌های هم‌جهت با فاصله کم ===
    if merge_gap > 0:
        leg_labels = _merge_nearby_legs(leg_labels, merge_gap)

    # === مرحله ۳: تنظیم انتهای لگ‌ها با قوانین جدید ===
    # محاسبه میانگین بدنه ۱۰۰ کندل گذشته (برای استفاده در شرط بدنه بزرگ)
    body = (df['close'] - df['open']).abs()
    avg_body_100 = body.shift(1).rolling(window=100, min_periods=1).mean()
    leg_labels = _adjust_leg_tails(df, leg_labels, avg_body_100)
    
    return leg_labels


def _merge_nearby_legs(leg_labels, max_gap):
    """
    ادغام زنجیره‌ای لگ‌های هم‌جهت با فاصله <= max_gap.
    """
    merged = leg_labels.copy()
    n = len(merged)
    i = 0
    while i < n:
        if merged[i] != '':
            leg_type = merged[i]
            start = i
            j = i
            while j < n and merged[j] == leg_type:
                j += 1
            end = j - 1

            next_start = end + 1
            while next_start < n:
                if merged[next_start] != '' and merged[next_start] != leg_type:
                    break
                if merged[next_start] == leg_type:
                    gap = next_start - end - 1
                    if gap <= max_gap:
                        k = next_start
                        while k < n and merged[k] == leg_type:
                            k += 1
                        new_end = k - 1
                        for m in range(end + 1, new_end + 1):
                            merged[m] = leg_type
                        end = new_end
                        next_start = end + 1
                        continue
                    else:
                        break
                next_start += 1
            i = end + 1
        else:
            i += 1
    return merged


def _adjust_leg_tails(df, leg_labels, avg_body_100):
    """
    تنظیم انتهای لگ‌ها بر اساس رفتار کندل‌های مرزی:
    - کندل‌های انتهایی که در جهت لگ حرکت نکرده‌اند (سقف/کف جدید نزده‌اند)
      اگر بلافاصله لگ مخالف بعدی وجود داشته باشد به آن لگ اضافه می‌شوند،
      وگرنه خالی (بدون لگ) می‌مانند.
    - کندل‌های با بدنهٔ بزرگ مخالف که سقف/کف جدید زده‌اند و بلافاصله
      لگ مخالف بعدی شروع می‌شود، به عنوان کندل شروع لگ بعدی نیز علامت‌گذاری می‌شوند.
    """
    labels = leg_labels.copy()
    n = len(df)

    # یافتن بازه‌های لگ‌ها قبل از تغییر
    segments = []
    i = 0
    while i < n:
        if labels[i] != '':
            start = i
            lbl = labels[i]
            while i < n and labels[i] == lbl:
                i += 1
            segments.append((start, i - 1, lbl))
        else:
            i += 1

    for idx, (seg_start, seg_end, seg_type) in enumerate(segments):
        next_seg = segments[idx + 1] if idx + 1 < len(segments) else None

        if seg_type == 'Bullish Leg':
            j = seg_end
            while j >= seg_start:
                candle = df.iloc[j]
                is_bearish = candle['close'] < candle['open']
                prev_high = df['high'].iloc[seg_start:j].max() if j > seg_start else -np.inf

                if is_bearish and candle['high'] <= prev_high:
                    # کندل نزولی که سقف جدید نزده → از لگ صعودی حذف می‌شود
                    # اگر لگ نزولی بلافاصله بعد از این کندل باشد → به آن لگ اضافه می‌شود
                    if next_seg and next_seg[2] == 'Bearish Leg' and next_seg[0] == j + 1:
                        labels[j] = 'Bearish Leg'
                    else:
                        labels[j] = ''   # بدون لگ
                    j -= 1
                    continue

                elif is_bearish and candle['high'] > prev_high and abs(candle['close'] - candle['open']) > avg_body_100.iloc[j]:
                    # کندل نزولی بزرگ که سقف جدید زده → هم‌چنان صعودی می‌ماند
                    # و اگر لگ نزولی بلافاصله بعد باشد، شروع آن لگ می‌شود
                    if next_seg and next_seg[2] == 'Bearish Leg' and next_seg[0] == j + 1:
                        labels[j] = 'Bearish Leg'   # شروع لگ نزولی (روی صعودی بازنویسی می‌شود)
                    # در غیر این صورت، همان صعودی باقی می‌ماند
                    break
                else:
                    # کندل شرایطی برای حذف یا اشتراک ندارد
                    break

        elif seg_type == 'Bearish Leg':
            j = seg_end
            while j >= seg_start:
                candle = df.iloc[j]
                is_bullish = candle['close'] > candle['open']
                prev_low = df['low'].iloc[seg_start:j].min() if j > seg_start else np.inf

                if is_bullish and candle['low'] >= prev_low:
                    # کندل صعودی که کف جدید نزده → از لگ نزولی حذف می‌شود
                    if next_seg and next_seg[2] == 'Bullish Leg' and next_seg[0] == j + 1:
                        labels[j] = 'Bullish Leg'
                    else:
                        labels[j] = ''
                    j -= 1
                    continue

                elif is_bullish and candle['low'] < prev_low and abs(candle['close'] - candle['open']) > avg_body_100.iloc[j]:
                    # کندل صعودی بزرگ که کف جدید زده → نزولی می‌ماند و شروع لگ صعودی بعدی
                    if next_seg and next_seg[2] == 'Bullish Leg' and next_seg[0] == j + 1:
                        labels[j] = 'Bullish Leg'
                    break
                else:
                    break

    return labels