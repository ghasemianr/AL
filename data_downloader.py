# data_downloader.py
"""
دانلودر داده با yfinance - جایگزین ccxt برای ایران
فرمت خروجی: time,open,high,low,close,volume (دقیقاً مثل فایل فعلی)
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
import os

def download_ohlcv(symbol='ETH-USD', timeframe='15m', days_back=30, save_csv=True, output_filename=None):
    """
    دانلود داده‌های OHLCV از Yahoo Finance
    
    پارامترها:
    - symbol: 'BTC-USD', 'ETH-USD', 'SOL-USD', ...
    - timeframe: '1m','5m','15m','30m','1h','4h','1d','1wk','1mo'
    - days_back: چند روز گذشته (برای 15m حداکثر ۶۰ روز)
    - save_csv: ذخیره خودکار فایل
    - output_filename: نام فایل خروجی (اختیاری)
    
    محدودیت‌های Yahoo Finance:
    - 1m: فقط ۷ روز
    - 5m/15m/30m: ۶۰ روز
    - 1h: ۷۳۰ روز
    - 1d به بالا: بدون محدودیت
    """
    
    # تنظیم period متناسب با تایم‌فریم
    period_map = {
        '1m': min(days_back, 7),
        '5m': min(days_back, 60),
        '15m': min(days_back, 60),
        '30m': min(days_back, 60),
        '1h': min(days_back, 730),
        '4h': days_back,
        '1d': days_back,
        '1wk': days_back,
        '1mo': days_back
    }
    
    max_days = period_map.get(timeframe, days_back)
    period = f"{max_days}d"
    
    print(f"📥 در حال دانلود {symbol} - تایم‌فریم {timeframe} - {max_days} روز گذشته...")
    
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=timeframe)
        
        if df.empty:
            print("❌ داده‌ای دریافت نشد. ممکن است تایم‌فریم نامعتبر باشد.")
            return None
        
        # حذف timezone
        df = df.reset_index()
        
        # تبدیل به فرمت پروژه
        df['time'] = df['Datetime'].dt.tz_localize(None).dt.strftime('%Y-%m-%d %H:%M:%S+00:00')
        
        # انتخاب و نام‌گذاری ستون‌ها
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
        
        # حذف ردیف‌های بدون قیمت
        df = df.dropna()
        df = df.reset_index(drop=True)
        
        print(f"✅ دانلود کامل شد. تعداد کندل‌ها: {len(df)}")
        
        # ذخیره CSV
        if save_csv:
            if output_filename is None:
                symbol_clean = symbol.replace('-', '')
                output_filename = f"{symbol_clean}_{timeframe}.csv"
            
            df.to_csv(output_filename, index=False, encoding='utf-8')
            print(f"💾 داده‌ها در '{output_filename}' ذخیره شدند.")
        
        return df
        
    except Exception as e:
        print(f"❌ خطا در دانلود: {str(e)}")
        print("\n🔍 راهنمایی:")
        print("1. اطمینان از اتصال اینترنت (VPN روشن باشد)")
        print("2. اطمینان از نصب yfinance: pip install yfinance")
        print("3. تایم‌فریم‌های معتبر: 1m, 5m, 15m, 30m, 1h, 4h, 1d")
        return None


def download_multiple(symbols, timeframes, days_back=30):
    """
    دانلود چند نماد و تایم‌فریم همزمان
    
    مثال:
    download_multiple(
        symbols=['BTC-USD', 'ETH-USD'],
        timeframes=['15m', '1h'],
        days_back=30
    )
    """
    for symbol in symbols:
        for tf in timeframes:
            try:
                download_ohlcv(symbol=symbol, timeframe=tf, days_back=days_back)
            except Exception as e:
                print(f"❌ {symbol} {tf}: {e}")
            print("-" * 50)


# ========== اجرای اصلی ==========
if __name__ == "__main__":
    
    # ============================================
    # 🔧 تنظیمات خود را اینجا تغییر دهید
    # ============================================
    
    SYMBOL = 'ETH-USD'       # Bitcoin: 'BTC-USD', Ethereum: 'ETH-USD', Solana: 'SOL-USD'
    TIMEFRAME = 'd'         # 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1wk
    DAYS_BACK = 100            # حداکثر برای 15m = ۶۰ روز
    
    # ============================================
    
    df = download_ohlcv(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        days_back=DAYS_BACK,
        save_csv=True
    )
    
    if df is not None:
        print(f"\n📊 نمونه داده‌ها:")
        print(df.head(5))
        print(f"\n📏 تعداد کندل: {len(df)}")
        print(f"📅 از: {df['time'].iloc[0]}")
        print(f"📅 تا: {df['time'].iloc[-1]}")