# analysis/range_detector.py
import numpy as np
from config import BREAK_FACTOR, MERGE_FACTOR
from analysis.support_resistance import find_levels_from_legs


def _get_sr_zones(df, leg_labels):
    """
    استخراج نواحی حمایت و مقاومت فقط از گروه‌های چند سطحی (حداقل ۲ سطح).
    این تابع برای رنج‌های اصلی (۳ موج) استفاده می‌شود.
    """
    raw_levels = find_levels_from_legs(df, leg_labels, break_factor=BREAK_FACTOR)

    for lvl in raw_levels:
        broken_idx = len(df) - 1
        l_type = lvl['type']
        for i in range(lvl['index'] + 1, len(df)):
            close = df['close'].iloc[i]
            if (l_type == 'resistance' and close > lvl['price'] + lvl['threshold']) or \
               (l_type == 'support' and close < lvl['price'] - lvl['threshold']):
                broken_idx = i - 1
                break
        lvl['broken_at'] = broken_idx

    levels_sorted = sorted(raw_levels, key=lambda x: x['index'])
    groups = []

    for lvl in levels_sorted:
        added = False
        for group in groups:
            if group[0]['type'] != lvl['type']:
                continue
            avg_price = np.mean([x['price'] for x in group])
            if abs(lvl['price'] - avg_price) > MERGE_FACTOR * lvl['avg_range']:
                continue
            group_end = min(x['broken_at'] for x in group)
            if lvl['index'] <= group_end:
                group.append(lvl)
                added = True
                break
        if not added:
            groups.append([lvl])

    support_zones = []
    resistance_zones = []

    for group in groups:
        if len(group) > 1:   # فقط گروه‌های چند سطحی
            start_idx = min(x['index'] for x in group)
            end_idx = min(x['broken_at'] for x in group)
            min_price = min(x['price'] for x in group)
            max_price = max(x['price'] for x in group)

            zone = {
                'start': start_idx,
                'end': end_idx,
                'bottom': min_price,
                'top': max_price,
                'threshold': group[0]['threshold']
            }
            if group[0]['type'] == 'support':
                support_zones.append(zone)
            else:
                resistance_zones.append(zone)

    return support_zones, resistance_zones


def _get_all_sr_zones(df, leg_labels):
    """
    استخراج تمام نواحی حمایت/مقاومت، حتی سطوح تکی.
    سطوح تکی یک باند باریک (۳۰٪ آستانه شکست) دریافت می‌کنند.
    """
    raw_levels = find_levels_from_legs(df, leg_labels, break_factor=BREAK_FACTOR)

    for lvl in raw_levels:
        broken_idx = len(df) - 1
        l_type = lvl['type']
        for i in range(lvl['index'] + 1, len(df)):
            close = df['close'].iloc[i]
            if (l_type == 'resistance' and close > lvl['price'] + lvl['threshold']) or \
               (l_type == 'support' and close < lvl['price'] - lvl['threshold']):
                broken_idx = i - 1
                break
        lvl['broken_at'] = broken_idx

    levels_sorted = sorted(raw_levels, key=lambda x: x['index'])
    groups = []

    for lvl in levels_sorted:
        added = False
        for group in groups:
            if group[0]['type'] != lvl['type']:
                continue
            avg_price = np.mean([x['price'] for x in group])
            if abs(lvl['price'] - avg_price) > MERGE_FACTOR * lvl['avg_range']:
                continue
            group_end = min(x['broken_at'] for x in group)
            if lvl['index'] <= group_end:
                group.append(lvl)
                added = True
                break
        if not added:
            groups.append([lvl])

    support_zones = []
    resistance_zones = []

    for group in groups:
        l_type = group[0]['type']
        threshold = group[0]['threshold']

        if len(group) > 1:
            start_idx = min(x['index'] for x in group)
            end_idx = min(x['broken_at'] for x in group)
            min_price = min(x['price'] for x in group)
            max_price = max(x['price'] for x in group)
        else:
            # تک سطح: باندی به عرض ۳۰٪ آستانه شکست
            lvl = group[0]
            start_idx = lvl['index']
            end_idx = lvl['broken_at']

            if l_type == 'support':
                min_price = lvl['price']
                max_price = lvl['price'] + threshold * 0.3
            else:  # resistance
                max_price = lvl['price']
                min_price = lvl['price'] - threshold * 0.3

        zone = {
            'start': start_idx,
            'end': end_idx,
            'bottom': min_price,
            'top': max_price,
            'threshold': threshold
        }
        if l_type == 'support':
            support_zones.append(zone)
        else:
            resistance_zones.append(zone)

    return support_zones, resistance_zones


def validate_3_waves(df, start_idx, end_idx, s_top, s_bottom, r_top, r_bottom, s_threshold, r_threshold, min_waves=3):
    """
    بررسی قانون N موج با حد مجاز نفوذ.
    پارامتر min_waves: تعداد موج‌های لازم برای تأیید (پیش‌فرض ۳)
    """
    wave_count = 0
    current_target = None
    w1_start = None
    final_wave_idx = None

    max_allowed_high = r_top + r_threshold
    min_allowed_low = s_bottom - s_threshold

    for idx in range(start_idx, end_idx + 1):
        high = df['high'].iloc[idx]
        low = df['low'].iloc[idx]

        if high > max_allowed_high or low < min_allowed_low:
            return False, None

        if current_target is None:
            if low <= s_top:
                current_target = 'R'
                w1_start = idx
            elif high >= r_bottom:
                current_target = 'S'
                w1_start = idx
            continue

        if current_target == 'R':
            if high >= r_bottom:
                wave_count += 1
                current_target = 'S'
                if wave_count >= min_waves:
                    final_wave_idx = idx
                    break
        elif current_target == 'S':
            if low <= s_top:
                wave_count += 1
                current_target = 'R'
                if wave_count >= min_waves:
                    final_wave_idx = idx
                    break

    if wave_count >= min_waves and final_wave_idx is not None:
        check_end = min(final_wave_idx + 5, len(df) - 1)
        for i in range(final_wave_idx + 1, check_end + 1):
            h = df['high'].iloc[i]
            l = df['low'].iloc[i]
            if h > max_allowed_high or l < min_allowed_low:
                return False, None
        return True, w1_start

    return False, None


def find_range_zones_by_leg_overlap(df, leg_labels, min_waves=3, **kwargs):
    """
    تابع اصلی تشخیص رنج‌ها بر پایه نواحی حمایت/مقاومت (فقط نواحی چند سطحی).
    پارامتر min_waves تعداد موج‌های لازم را تعیین می‌کند (پیش‌فرض ۳).
    """
    support_zones, resistance_zones = _get_sr_zones(df, leg_labels)
    range_shapes = []

    for s in support_zones:
        for r in resistance_zones:
            s_start, s_end = s['start'], s['end']
            r_start, r_end = r['start'], r['end']

            overlap_start = max(s_start, r_start)
            overlap_end = min(s_end, r_end)

            if overlap_start < overlap_end:
                s_bottom, s_top, s_thresh = s['bottom'], s['top'], s['threshold']
                r_bottom, r_top, r_thresh = r['bottom'], r['top'], r['threshold']

                if s_bottom >= r_top:
                    continue

                earliest_start = min(s_start, r_start)

                is_valid, w1_start = validate_3_waves(
                    df, earliest_start, overlap_end,
                    s_top, s_bottom, r_top, r_bottom,
                    s_thresh, r_thresh, min_waves
                )

                if not is_valid:
                    is_valid, w1_start = validate_3_waves(
                        df, overlap_start, overlap_end,
                        s_top, s_bottom, r_top, r_bottom,
                        s_thresh, r_thresh, min_waves
                    )

                if is_valid and w1_start is not None:
                    range_shapes.append({
                        'type': 'rect',
                        'x0': w1_start,
                        'x1': overlap_end,
                        'y0': s_bottom,
                        'y1': r_top,
                        'fillcolor': 'rgba(128, 128, 128, 0.25)',
                        'line_width': 0,
                        'layer': 'below'
                    })

    return range_shapes


def find_probable_range_zones(df, leg_labels, **kwargs):
    """
    تشخیص محدوده‌های رنج با حداقل ۲ موج (رنج‌های احتمالی).
    از تمام سطوح (شامل تک‌سطحی‌ها) استفاده می‌کند و رنگ کم‌رنگ‌تری دارد.
    """
    support_zones, resistance_zones = _get_all_sr_zones(df, leg_labels)
    range_shapes = []

    for s in support_zones:
        for r in resistance_zones:
            s_start, s_end = s['start'], s['end']
            r_start, r_end = r['start'], r['end']

            overlap_start = max(s_start, r_start)
            overlap_end = min(s_end, r_end)

            if overlap_start < overlap_end:
                s_bottom, s_top, s_thresh = s['bottom'], s['top'], s['threshold']
                r_bottom, r_top, r_thresh = r['bottom'], r['top'], r['threshold']

                if s_bottom >= r_top:
                    continue

                earliest_start = min(s_start, r_start)

                is_valid, w1_start = validate_3_waves(
                    df, earliest_start, overlap_end,
                    s_top, s_bottom, r_top, r_bottom,
                    s_thresh, r_thresh, min_waves=2
                )

                if not is_valid:
                    is_valid, w1_start = validate_3_waves(
                        df, overlap_start, overlap_end,
                        s_top, s_bottom, r_top, r_bottom,
                        s_thresh, r_thresh, min_waves=2
                    )

                if is_valid and w1_start is not None:
                    range_shapes.append({
                        'type': 'rect',
                        'x0': w1_start,
                        'x1': overlap_end,
                        'y0': s_bottom,
                        'y1': r_top,
                        'fillcolor': 'rgba(169, 169, 169, 0.15)',
                        'line_width': 0,
                        'layer': 'below'
                    })

    return range_shapes