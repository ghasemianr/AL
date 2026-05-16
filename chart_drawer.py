# chart/chart_drawer.py
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_chart(df, channel_info, sr_shapes, support_lines, resistance_lines,
               leg_labels=None, leg_strength=None, gap_rectangles=None, range_zones=None,
               breakouts=None, channel_lines=None, breakout_phases=None, probable_zones=None,
               ema_values=None, ema_signals=None, ticker=None, timeframe=None):   # اضافه کردن channel_lines
    """
    رسم نمودار کندل‌استیک همراه با ولوم، سطوح، کانال‌ها، خطوط روند،
    گپ‌ها، محدوده‌های رنج و نمایش اطلاعات در هاور.
        ema_values: آرایه مقادیر EMA (هم‌طول df) یا None
    ema_signals: لیست سیگنال‌های dict شامل index, type, entry_price
    پارامتر breakouts: خروجی تابع detect_breakouts
    پارامتر channel_lines: خطوط کانال موازی با خطوط روند
    """
    
        # ===== ساخت دیکشنری follow-through برای هاور =====
    follow_map = {}  # key: index, value: متن توضیحی
    if breakouts:
        for i, b in enumerate(breakouts):
            if b.get('breakout') and b.get('follow_through'):
                for offset, ft in enumerate(b['follow_through'], start=1):
                    idx = ft['index']
                    classification = ft['classification']
                    if classification:
                        text = f"🔁 Follow-Through ({offset}): {classification}"
                    else:
                        text = f"🔁 Follow-Through ({offset}): normal"
                    # اگر چند بریک‌اوت روی یک کندل فالوو داشتیم، متن‌ها را جمع کنیم
                    if idx in follow_map:
                        follow_map[idx] += f" | {text}"
                    else:
                        follow_map[idx] = text
                        
    # ساخت متن هاور
    hover_texts = []
    for i in range(len(df)):
        parts = []
        if 'candle_type' in df.columns and pd.notna(df['candle_type'].iloc[i]):
            parts.append(str(df['candle_type'].iloc[i]))
        if 'bar_type' in df.columns and pd.notna(df['bar_type'].iloc[i]):
            parts.append(str(df['bar_type'].iloc[i]))
        if channel_info and i < len(channel_info) and channel_info[i] != '':
            parts.append(str(channel_info[i]))
        if leg_labels is not None and i < len(leg_labels) and leg_labels[i] != '':
            parts.append(str(leg_labels[i]))
            if leg_strength is not None and i < len(leg_strength) and leg_strength[i] not in ('', None):
                parts.append(f"Leg Strength: {leg_strength[i]}")
        if breakouts and i < len(breakouts) and breakouts[i]['breakout']:
            bt = breakouts[i]
            quality_text = f" [{bt['quality']}]" if bt.get('quality') else ""
            parts.append(f"🔥 {bt['type']} ({bt['direction']}){quality_text}")
        if i in follow_map:
            parts.append(follow_map[i])
        # اضافه کردن امتیاز BO (از ستون df)
        if 'bo_score' in df.columns and pd.notna(df['bo_score'].iloc[i]) and df['bo_score'].iloc[i] != 0:
            parts.append(f"BO Score: {df['bo_score'].iloc[i]:+d}")
        hover_texts.append('<br>'.join(parts) if parts else '')

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.8, 0.2]
    )

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'], high=df['high'],
        low=df['low'], close=df['close'],
        name='Price',
        text=hover_texts,
        hovertext=hover_texts,
        hoverinfo='x+y+text'
    ), row=1, col=1)

    colors = ['green' if row['close'] >= row['open'] else 'red' for _, row in df.iterrows()]
    fig.add_trace(go.Bar(
        x=df.index, y=df['volume'],
        marker_color=colors, name='Volume'
    ), row=2, col=1)

    # ===== رسم رنج‌های احتمالی (۲ موج) با رنگ روشن‌تر =====
    if probable_zones:
        for shape in probable_zones:
            fig.add_shape(
                type='rect',
                x0=shape['x0'], y0=shape['y0'],
                x1=shape['x1'], y1=shape['y1'],
                fillcolor=shape.get('fillcolor', 'rgba(169,169,169,0.2)'),
                line=dict(width=0),
                layer='below',          # زیر همه
                row=1, col=1
            )
            
    # محدوده‌های رنج
    if range_zones:
        for shape in range_zones:
            fig.add_shape(
                type='rect',
                x0=shape['x0'], y0=shape['y0'],
                x1=shape['x1'], y1=shape['y1'],
                fillcolor=shape['fillcolor'],
                line=dict(width=0),
                row=1, col=1
            )

    # سطوح حمایت و مقاومت (نقطه‌چین و مستطیل)
    for shape in sr_shapes:
        if shape['type'] == 'rect':
            fig.add_shape(
                type='rect',
                x0=shape['x0'], y0=shape['y0'],
                x1=shape['x1'], y1=shape['y1'],
                fillcolor=shape['fillcolor'],
                line=dict(width=0),
                row=1, col=1
            )
        elif shape['type'] == 'line':
            fig.add_shape(
                type='line',
                x0=shape['x0'], y0=shape['y0'],
                x1=shape['x1'], y1=shape['y1'],
                line=dict(color=shape['line_color'], width=shape['line_width'],
                          dash=shape.get('line_dash', 'solid')),
                row=1, col=1
            )

    # خطوط روند
    for tl in support_lines:
        color, width = ("darkblue",3) if tl['touches']>=4 else ("blue",2.5) if tl['touches']>=3 else ("lightblue",2)
        fig.add_shape(
            type='line',
            x0=tl['x1'], y0=tl['y1'], x1=tl['x2'], y1=tl['y2'],
            line=dict(color=color, width=width, dash='solid'),
            row=1, col=1
        )
        mid_x = (tl['x1']+tl['x2'])//2
        mid_y = (tl['y1']+tl['y2'])/2
        fig.add_annotation(x=mid_x, y=mid_y, text=str(tl['touches']),
                           showarrow=False, font=dict(size=8, color=color),
                           bgcolor="rgba(0,0,0,0.5)", row=1, col=1)

    for tl in resistance_lines:
        color, width = ("darkred",3) if tl['touches']>=4 else ("red",2.5) if tl['touches']>=3 else ("orange",2)
        fig.add_shape(
            type='line',
            x0=tl['x1'], y0=tl['y1'], x1=tl['x2'], y1=tl['y2'],
            line=dict(color=color, width=width, dash='solid'),
            row=1, col=1
        )
        mid_x = (tl['x1']+tl['x2'])//2
        mid_y = (tl['y1']+tl['y2'])/2
        fig.add_annotation(x=mid_x, y=mid_y, text=str(tl['touches']),
                           showarrow=False, font=dict(size=8, color=color),
                           bgcolor="rgba(0,0,0,0.5)", row=1, col=1)

    # ===== رسم خطوط کانال (اضافه شده) =====
    if channel_lines is not None:
        for ch_line in channel_lines:
            color = 'lightblue' if ch_line['type'] == 'channel_resistance' else 'lightcoral'
            fig.add_shape(
                type='line',
                x0=ch_line['x1'], y0=ch_line['y1'],
                x1=ch_line['x2'], y1=ch_line['y2'],
                line=dict(color=color, width=2, dash='solid'),
                row=1, col=1
            )

    # گپ‌ها
    if gap_rectangles:
        for gap in gap_rectangles:
            if gap['status'] == 'closed':
                continue
            if gap['status'] == 'open':
                fillcolor = 'green' if gap['color'] == 'green' else 'red'
                alpha = 0.8
            elif gap['status'] == 'negative':
                fillcolor = 'orange'
                alpha = 0.8
            else:
                fillcolor = gap['color']
                alpha = 0.8
            y_bottom = min(gap['y0'], gap['y1'])
            y_top = max(gap['y0'], gap['y1'])
            fig.add_shape(
                type='rect',
                x0=gap['x0'], y0=y_bottom,
                x1=gap['x1'], y1=y_top,
                fillcolor=fillcolor,
                opacity=alpha,
                line=dict(width=1, color=fillcolor),
                row=1, col=1
            )
    
    # مثلث‌های زرد رنگ برای بریک‌اوت‌ها
    if breakouts:
        up_x = []
        up_y = []
        down_x = []
        down_y = []
        for i, b in enumerate(breakouts):
            if not b['breakout']:
                continue
            if b['direction'] == 'up':
                down_x.append(i)
                down_y.append(df['low'].iloc[i] - (df['high'].iloc[i] - df['low'].iloc[i]) * 0.1)
            elif b['direction'] == 'down':
                up_x.append(i)
                up_y.append(df['high'].iloc[i] + (df['high'].iloc[i] - df['low'].iloc[i]) * 0.1)
        if up_x:
            fig.add_trace(go.Scatter(
                x=up_x, y=up_y,
                mode='markers',
                marker=dict(symbol='triangle-down', size=12, color='gold', line=dict(width=1, color='black')),
                name='Breakout Down (Support broken)',
                hoverinfo='skip'
            ), row=1, col=1)
        if down_x:
            fig.add_trace(go.Scatter(
                x=down_x, y=down_y,
                mode='markers',
                marker=dict(symbol='triangle-up', size=12, color='gold', line=dict(width=1, color='black')),
                name='Breakout Up (Resistance broken)',
                hoverinfo='skip'
            ), row=1, col=1)
        
        # ===== رسم فازهای بریک‌اوت (مستطیل‌های زردرنگ در پس‌زمینه) =====
    if breakout_phases:
        for phase in breakout_phases:
            fig.add_shape(
                type='rect',
                x0=phase['start'], y0=phase['y0'],
                x1=phase['end'],   y1=phase['y1'],
                fillcolor='rgba(255, 255, 0, 0.85)',   # زرد نیمه‌شفاف
                line=dict(width=0),
                layer='below',      # مهم: زیر کندل‌ها
                row=1, col=1
            )
        # اضافه کردن EMA line اگر وجود داشته باشد
    if ema_values is not None:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=ema_values,
            mode='lines',
            name=f'EMA 20',
            line=dict(color='orange', width=1.5, dash='dash'),
            hoverinfo='skip'
        ), row=1, col=1)

    # اضافه کردن فلش‌های سیگنال
    if ema_signals:
        buy_x, buy_y = [], []
        sell_x, sell_y = [], []
        for sig in ema_signals:
            idx = sig['index']
            if sig['type'] == 'buy':
                # فلش رو به بالا زیر کندل
                y_pos = df['low'].iloc[idx] - (df['high'].iloc[idx] - df['low'].iloc[idx]) * 0.3
                buy_x.append(idx)
                buy_y.append(y_pos)
            elif sig['type'] == 'sell':
                # فلش رو به پایین بالای کندل
                y_pos = df['high'].iloc[idx] + (df['high'].iloc[idx] - df['low'].iloc[idx]) * 0.3
                sell_x.append(idx)
                sell_y.append(y_pos)
        if buy_x:
            fig.add_trace(go.Scatter(
                x=buy_x, y=buy_y,
                mode='markers',
                marker=dict(symbol='triangle-up', size=15, color='lime', line=dict(width=1, color='black')),
                name='EMA Buy Signal',
                hoverinfo='skip'
            ), row=1, col=1)
        if sell_x:
            fig.add_trace(go.Scatter(
                x=sell_x, y=sell_y,
                mode='markers',
                marker=dict(symbol='triangle-down', size=15, color='red', line=dict(width=1, color='black')),
                name='EMA Sell Signal',
                hoverinfo='skip'
            ), row=1, col=1)
        
    # عنوان داینامیک (فقط نام تیکر و تایم فریم)
    if ticker and timeframe:
        title_text = f"{ticker} - {timeframe}"
    else:
        title_text = "Price Chart"   # یا هر عنوان ساده دیگر
        
    fig.update_layout(
        template='plotly_dark',
        height=800,
        title=title_text,
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        showlegend=False
    )
    fig.show()   # <------ این خط را اضافه کنید
    
