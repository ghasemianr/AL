# preprocessing/candle_classifier.py
import pandas as pd
import numpy as np

def classify_candle(df):
    """
    اضافه کردن ستون‌های 'direction' و 'candle_type' (دوجی / ترندبار)
    منطبق بر doji-trendbar.py
    """
    # محاسبه اجزای کندل
    body = abs(df['close'] - df['open'])
    upper_shadow = df['high'] - df[['close', 'open']].max(axis=1)
    lower_shadow = df[['close', 'open']].min(axis=1) - df['low']
    total_shadow = upper_shadow + lower_shadow

    # شرط‌ها
    is_doji = body < total_shadow
    is_trendbar = body >= total_shadow

    # جهت
    conditions = [
        df['open'] > df['close'],   # نزولی
        df['open'] < df['close'],   # صعودی
        df['open'] == df['close']   # خنثی
    ]
    directions = ['-', '+', '']
    df['direction'] = np.select(conditions, directions, default='')

    # نوع کندل
    df['candle_type'] = 'نامشخص'
    df.loc[is_doji, 'candle_type'] = 'دوجی' + df.loc[is_doji, 'direction']
    df.loc[is_trendbar, 'candle_type'] = 'ترندبار ' + df.loc[is_trendbar, 'direction'].map({
        '+': 'صعودی',
        '-': 'نزولی',
        '': 'خنثی'
    })
    return df


def classify_bar_patterns(df):
    """
    اضافه کردن ستون 'bar_type' (inside, outside, ii, oi, ...)
    منطبق بر in-out bar.py
    """
    # اطمینان از نوع عددی
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')

    prev_high = df['high'].shift(1)
    prev_low = df['low'].shift(1)

    # نوع پایه
    df['base_type'] = 'n'
    df.loc[(df['high'] <= prev_high) & (df['low'] >= prev_low), 'base_type'] = 'i'
    df.loc[(df['high'] > prev_high) & (df['low'] < prev_low), 'base_type'] = 'o'

    b0 = df['base_type']
    b1 = df['base_type'].shift(1)
    b2 = df['base_type'].shift(2)
    b3 = df['base_type'].shift(3)

    df['bar_type'] = df['base_type'].map({'n': 'normal', 'i': 'inside', 'o': 'outside'})

    # الگوهای ترکیبی
    df.loc[(b1 == 'i') & (b0 == 'i'), 'bar_type'] = 'ii'
    df.loc[(b2 == 'i') & (b1 == 'i') & (b0 == 'i'), 'bar_type'] = 'iii'
    df.loc[(b3 == 'i') & (b2 == 'i') & (b1 == 'i') & (b0 == 'i'), 'bar_type'] = 'iiii'
    df.loc[(b1 == 'o') & (b0 == 'i'), 'bar_type'] = 'oi'
    df.loc[(b2 == 'o') & (b1 == 'i') & (b0 == 'o'), 'bar_type'] = 'oio'

    df.drop(columns=['base_type'], inplace=True)
    return df


def add_breakout_score(df, avg_body_lookback=100):
    """
    اضافه کردن ستون 'bo_score' (امتیاز BO) برای هر کندل نسبت به کندل قبلی
    قوانین:
    1- دو کندل هم‌جهت:
       صعودی: اگر high > high قبلی -> +1، اگر close > high قبلی -> +1 اضافی
       نزولی: اگر low < low قبلی -> +1، اگر close < low قبلی -> +1 اضافی
       اگر کندل دوم دوجی باشد -> -1
       اگر شدوی سمت جهت (برای صعودی شدو بالا، برای نزولی شدو پایین) بیشتر از میانگین بدنه 100 تای قبلی
         و بزرگتر از نصف ارتفاع کندل باشد -> -1
    2- outside bar:
       اگر مخالف -> -1، و اگر کلوز فراتر از high/low قبلی رفت -> یک -1 دیگر
       اگر هم‌جهت و close > close قبلی (صعودی) یا close < close قبلی (نزولی) -> +1
    3- inside bar -> امتیاز 0
    4- دو کندل مخالف (normal bar) -> -1
    """
    if len(df) < 2:
        df['bo_score'] = 0
        return df

    body = (df['close'] - df['open']).abs()
    rolling_avg_body = body.rolling(window=avg_body_lookback, min_periods=1).mean()
    
    # تعیین جهت کندل: 1 صعودی، -1 نزولی، 0 خنثی
    direction = np.where(df['close'] > df['open'], 1,
                         np.where(df['close'] < df['open'], -1, 0))
    
    bo_scores = [0]  # برای اولین کندل
    
    for i in range(1, len(df)):
        score = 0
        curr = df.iloc[i]
        prev = df.iloc[i-1]
        curr_dir = direction[i]
        prev_dir = direction[i-1]
        
        # تشخیص inside / outside (با قیمت‌ها)
        is_inside = (curr['high'] <= prev['high'] and curr['low'] >= prev['low'])
        is_outside = (curr['high'] > prev['high'] and curr['low'] < prev['low'])
        
        # ===== داخل رنج (inside bar) =====
        if is_inside:
            score = 0   # قانون 3
        
        # ===== خارج رنج (outside bar) =====
        elif is_outside:
            # مخالف بودن بدنه
            if curr_dir * prev_dir < 0:   # مخالف
                score -= 1
                # اگر کلوز فراتر از high/low قبلی رفت، یک منفی دیگر
                if (curr_dir == 1 and curr['close'] > prev['high']) or \
                   (curr_dir == -1 and curr['close'] < prev['low']):
                    score -= 1
            else:  # هم‌جهت
                # امتیاز مثبت در صورتی که کلوز از کلوز قبلی بهتر شود
                if curr_dir == 1 and curr['close'] > prev['close']:
                    score += 1
                elif curr_dir == -1 and curr['close'] < prev['close']:
                    score += 1
                # در غیر این صورت امتیاز صفر
        
        # ===== نرمال (normal bar) =====
        else:
            if curr_dir == prev_dir:   # هم‌جهت (قانون 1)
                if curr_dir == 1:   # صعودی
                    if curr['high'] > prev['high']:
                        score += 1
                    if curr['close'] > prev['high']:
                        score += 1
                    # جریمه دوجی
                    if 'دوجی' in str(curr.get('candle_type', '')):
                        score -= 1
                    # جریمه شدوی بالایی بزرگ
                    upper_shadow = curr['high'] - max(curr['close'], curr['open'])
                    avg_body_val = rolling_avg_body.iloc[i]
                    candle_height = curr['high'] - curr['low']
                    if avg_body_val > 0 and candle_height > 0:
                        if upper_shadow > avg_body_val and upper_shadow > (candle_height / 2):
                            score -= 1
                else:   # نزولی
                    if curr['low'] < prev['low']:
                        score += 1
                    if curr['close'] < prev['low']:
                        score += 1
                    if 'دوجی' in str(curr.get('candle_type', '')):
                        score -= 1
                    lower_shadow = min(curr['close'], curr['open']) - curr['low']
                    avg_body_val = rolling_avg_body.iloc[i]
                    candle_height = curr['high'] - curr['low']
                    if avg_body_val > 0 and candle_height > 0:
                        if lower_shadow > avg_body_val and lower_shadow > (candle_height / 2):
                            score -= 1
            else:   # مخالف (قانون 4)
                score -= 1
        
        bo_scores.append(score)
    
    df['bo_score'] = bo_scores
    return df


def enrich_candles(df):
    """
    اعمال هر سه مرحله پیش‌پردازش روی داده خام:
    1. نوع کندل (دوجی/ترندبار)
    2. الگوهای inside/outside
    3. امتیاز BO
    """
    df = classify_candle(df)
    df = classify_bar_patterns(df)
    df = add_breakout_score(df)
    return df