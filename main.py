# main.py
import pandas as pd
import os
import numpy as np
import re   # اضافه شده برای استخراج تیکر و تایم فریم
from config import BREAK_FACTOR, MERGE_FACTOR, TOUCH_TOLERANCE, TOUCH_REQUIREMENT
from preprocessing.candle_classifier import enrich_candles
from analysis.support_resistance import (
    prepare_sr_shapes_advanced, find_levels_from_legs
)
from analysis.channels_and_trendlines import (
    find_channels, build_channel_info_array, find_advanced_trendlines, find_channel_lines
)
from analysis.leg_detector import detect_legs
from analysis.leg_strength import calculate_leg_strength
from analysis.gaps import find_gaps
from chart.chart_drawer import plot_chart
from analysis.range_detector import find_range_zones_by_leg_overlap, find_probable_range_zones
from analysis.breakouts import detect_breakouts, detect_breakout_phases   # اضافه شد
from setup.ema_setup import detect_ema_signals


def prepare_data(csv_path):
    df = pd.read_csv(csv_path)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
    elif 'timestamp' in df.columns:
        df['time'] = pd.to_datetime(df['timestamp'])
    return df


if __name__ == "__main__":
    raw_csv = "BTCUSD_5m.csv"
    if not os.path.exists(raw_csv):
        print(f"File {raw_csv} not found.")
    else:
        # Load and preprocess data
        df_full = prepare_data(raw_csv)
        df_full = enrich_candles(df_full)

        # Last 400 candles
        n_candles = 400
        df = df_full.tail(n_candles).copy().reset_index(drop=True)

        # Channels
        micro_channels, tight_channels = find_channels(df)
        channel_info = build_channel_info_array(df, micro_channels, tight_channels)

        # Legs
        leg_labels = detect_legs(df, min_leg_candles=3)
        leg_strength = calculate_leg_strength(df, leg_labels, micro_channels)

        # Support/Resistance levels (new advanced version)
        sr_shapes = prepare_sr_shapes_advanced(
            df,
            leg_labels,
            break_factor=BREAK_FACTOR,
            merge_factor=MERGE_FACTOR,
            avg_body_lookback=100
        )

        # Range zones based on leg overlap
        range_zones = find_range_zones_by_leg_overlap(
            df, leg_labels, sr_shapes=sr_shapes, leg_strength=leg_strength
        )

        # ===== رنج‌های احتمالی (حداقل ۲ موج) =====
        probable_range_zones = find_probable_range_zones(
            df, leg_labels, sr_shapes=sr_shapes, leg_strength=leg_strength
        )

        # Compute dy for trendlines
        close_prices = df['close'].values
        lookback = 20
        if len(close_prices) >= lookback:
            std_dev = np.std(close_prices[:lookback])
            for i in range(lookback, len(close_prices)):
                std_dev = np.std(close_prices[i-lookback:i])
        else:
            std_dev = np.std(close_prices)
        dy = TOUCH_TOLERANCE * std_dev

        # Trendlines
        raw_levels = find_levels_from_legs(df, leg_labels, break_factor=BREAK_FACTOR)
        
        valid_support_lines, valid_resistance_lines = find_advanced_trendlines(
            df, raw_levels, leg_labels, dy, touch_requirement=TOUCH_REQUIREMENT
        )

        channel_lines = find_channel_lines(
            df=df,
            support_lines=valid_support_lines,
            resistance_lines=valid_resistance_lines,
            raw_levels=raw_levels,
            leg_labels=leg_labels,
            dy=dy,                     # همان dy که قبلاً محاسبه شده (مربوط به خط روند)
            touch_requirement=2
        )

        # Gaps
        gap_rectangles, gap_details = find_gaps(
            df,
            support_lines=valid_support_lines,
            resistance_lines=valid_resistance_lines,
            leg_labels=leg_labels,
            avg_body_lookback=100
        )

        # Breakouts detection
        breakouts = detect_breakouts(df, sr_shapes, range_zones)
        
        # ===== تشخیص فازهای بریک‌اوت (جدید) =====
        breakout_phases = detect_breakout_phases(
            df, avg_body_lookback=100, factor=4
        )
        
        ENABLE_EMA_SETUP = True   # قابلیت روشن/خاموش

        if ENABLE_EMA_SETUP:
            ema_signals, ema_values = detect_ema_signals(df, ema_period=20, max_confirmation_candles=5)
        else:
            ema_signals = []
            ema_values = None

        # ===== استخراج تیکر و تایم فریم از نام فایل =====
        filename = os.path.basename(raw_csv)   # مثلاً "GC=F_5m.csv"
        # الگو: هر چیزی قبل از اولین زیرخط به عنوان تیکر، و بین زیرخط تا .csv به عنوان تایم فریم
        match = re.match(r'([^_]+)_(.+?)\.csv', filename)
        if match:
            ticker = match.group(1)
            timeframe = match.group(2)
        else:
            ticker = "Unknown"
            timeframe = "Unknown"
    
        # Plot
        plot_chart(
            df=df,
            channel_info=channel_info,
            sr_shapes=sr_shapes,
            support_lines=valid_support_lines,
            resistance_lines=valid_resistance_lines,
            channel_lines=channel_lines,          # اضافه شد
            leg_labels=leg_labels,
            leg_strength=leg_strength,
            gap_rectangles=gap_rectangles,
            range_zones=range_zones,
            breakouts=breakouts,
            breakout_phases=breakout_phases,
            probable_zones=probable_range_zones,
            ema_values=ema_values,          # اضافه شده
            ema_signals=ema_signals,
            ticker=ticker,                  # اضافه شده
            timeframe=timeframe             # اضافه شده
        )