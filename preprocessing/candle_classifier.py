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


def enrich_candles(df):
    """
    اعمال هر دو مرحله پیش‌پردازش روی داده خام
    """
    df = classify_candle(df)
    df = classify_bar_patterns(df)
    return df