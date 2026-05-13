# setup/ema_setup.py
import pandas as pd
import numpy as np

def detect_ema_signals(df, ema_period=20, max_confirmation_candles=5):
    """
    تشخیص سیگنال‌های خرید/فروش با EMA و ریست در برخوردها.
    برگرداندن سیگنال‌ها و مقادیر EMA.
    """
    df = df.copy()
    df['ema'] = df['close'].ewm(span=ema_period, adjust=False).mean()
    
    # تعریف وضعیت‌ها
    df['fully_above'] = df['low'] > df['ema']   # کندل کامل بالای EMA
    df['fully_below'] = df['high'] < df['ema']  # کندل کامل زیر EMA
    
    signals = []
    
    # اسکن خطی از ابتدا تا انتها با اشاره گر i
    i = ema_period  # از جای که EMA معتبر است شروع می‌کنیم
    n = len(df)
    
    while i < n:
        # تلاش برای یافتن سیگنال خرید (Long)
        # شرط: 20 کندل قبل از i (i-20 تا i-1) همگی fully_above باشند
        # اما باید حواست باشد که در بین آن کندل‌ها نباید ریست اتفاق افتاده باشد. برای سادگی، هر 20 کندل متوالی جدید چک می‌شود.
        
        # روش ساده‌تر: به جای شمارش مجدد هر بار، از پنجره کشویی استفاده کنیم.
        # اما برای جلوگیری از پیچیدگی، یک حلقه ساده برای هر i جداگانه بررسی می‌کنیم.
        
        # ----- خرید (Long) -----
        # بررسی 20 کندل قبل
        if i - ema_period >= 0:
            # آیا همه 20 کندل fully_above هستند؟
            window_above = df['fully_above'].iloc[i-ema_period:i].all()
            if window_above:
                # اکنون از i به بعد دنبال کندل سیگنال (fully_below) بگرد
                for k in range(i, n):
                    if df['fully_below'].iloc[k]:
                        # کندل سیگنال در k پیدا شد
                        signal_idx = k
                        # تا max_confirmation_candles کندل بعدی برای تأیید
                        for t in range(1, max_confirmation_candles+1):
                            if k + t >= n:
                                break
                            if df['close'].iloc[k+t] > df['high'].iloc[signal_idx]:
                                signals.append({
                                    'index': k+t,
                                    'type': 'buy',
                                    'entry_price': df['high'].iloc[signal_idx],
                                    'signal_candle_index': signal_idx
                                })
                                i = k + t + 1  # حرکت به بعد از سیگنال ثبت شده
                                break
                        else:
                            # تأیید نشد، i را به بعد از کندل سیگنال ببر
                            i = signal_idx + 1
                        break  # خارج شدن از حلقه جستجوی سیگنال
                else:
                    # هیچ سیگنالی یافت نشد، i را جلو ببر
                    i += 1
                continue  # برگرد به اول while (با i جدید)
        
        # ----- فروش (Short) -----
        if i - ema_period >= 0:
            window_below = df['fully_below'].iloc[i-ema_period:i].all()
            if window_below:
                for k in range(i, n):
                    if df['fully_above'].iloc[k]:
                        signal_idx = k
                        for t in range(1, max_confirmation_candles+1):
                            if k + t >= n:
                                break
                            if df['close'].iloc[k+t] < df['low'].iloc[signal_idx]:
                                signals.append({
                                    'index': k+t,
                                    'type': 'sell',
                                    'entry_price': df['low'].iloc[signal_idx],
                                    'signal_candle_index': signal_idx
                                })
                                i = k + t + 1
                                break
                        else:
                            i = signal_idx + 1
                        break
                else:
                    i += 1
                continue
        
        # اگر هیچ شرطی برقرار نبود، i را یکی جلو ببر
        i += 1
    
    return signals, df['ema'].values