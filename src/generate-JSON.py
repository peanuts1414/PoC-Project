# JSONファイル作成（正常・異常データ対応, 各windowごとのサイズを反映）

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd

# ===== 設定 =====
col_index_cycle = 5    # サイクル列（0〜36000000を繰り返す列, 0始まり）
skip_header_rows = 31  # データ開始行
input_json = "file_settings_normal_abnormal_multiple_file_2.json"  # 元のファイル設定
output_json = "file_settings_customize.json"  # 作成するJSONファイル

# ===== CSV 読込関数 =====
def load_csv_data(path, col_index, start_row=0, end_row=None, skip_header_rows=31):
    if not os.path.exists(path):
        print(f"[警告] ファイルが見つかりません: {path}")
        return np.array([])
    try:
        df = pd.read_csv(
            path,
            header=None,
            encoding='cp932',
            sep=None,
            engine='python',
            skiprows=skip_header_rows
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            path,
            header=None,
            encoding='utf-8',
            sep=None,
            engine='python',
            skiprows=skip_header_rows
        )
    if col_index >= df.shape[1]:
        print(f"[警告] 列数不足: {path}")
        return np.array([])
    if end_row is None:
        data = df.iloc[start_row:, col_index]
    else:
        data = df.iloc[start_row:end_row, col_index]
    return data.to_numpy()

# ===== サイクル列からウィンドウ検出 =====
def detect_windows(cycle_values):
    start_rows = [0]
    for i in range(1, len(cycle_values)):
        if cycle_values[i] < cycle_values[i-1]:
            start_rows.append(i)
    windows = []
    for i in range(len(start_rows)):
        start = start_rows[i]
        end = start_rows[i+1] if i+1 < len(start_rows) else len(cycle_values)
        size = end - start
        windows.append((start, end, size))
    return windows

def filter_windows(windows):
    if not windows:
        return []
    sizes = [w[2] for w in windows]
    avg_size = np.mean(sizes)
    min_size = avg_size * 0.8
    return [w for w in windows if w[2] >= min_size]

# ===== 元JSON読み込み =====
with open(input_json, encoding="utf-8") as f:
    file_settings = json.load(f)

# ===== 新しいJSON作成 =====
custom_settings = {"normal": {}, "abnormal": {}}

# ----- 正常データ ----- #
for path in file_settings.get("normal", {}).keys():
    cycle_data = load_csv_data(path, col_index_cycle, skip_header_rows=skip_header_rows)
    if len(cycle_data) == 0:
        continue
    windows = detect_windows(cycle_data)
    filtered = filter_windows(windows)
    if not filtered:
        continue
    for idx, (start, end, size) in enumerate(filtered):
        key = f"{Path(path).as_posix()}_win{idx}"
        custom_settings["normal"][key] = {
            "original_path": path,
            "start_row": start,
            "end_row": end,
            "window_size": size   # ← 各windowの本来のサイズをそのまま出力
        }

# ----- 異常データ ----- #
for path in file_settings.get("abnormal", {}).keys():
    cycle_data = load_csv_data(path, col_index_cycle, skip_header_rows=skip_header_rows)
    if len(cycle_data) == 0:
        continue
    windows = detect_windows(cycle_data)
    filtered = filter_windows(windows)
    if not filtered:
        continue
    for idx, (start, end, size) in enumerate(filtered):
        key = f"{Path(path).as_posix()}_win{idx}"
        custom_settings["abnormal"][key] = {
            "original_path": path,
            "start_row": start,
            "end_row": end,
            "window_size": size   # ← 平均ではなく各区間のサイズを保存
        }

# ===== JSON書き出し =====
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(custom_settings, f, ensure_ascii=False, indent=2)

print(f"[INFO] JSON作成完了 → {output_json}")
