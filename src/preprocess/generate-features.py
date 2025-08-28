# 特徴量作成 + CSV保存

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import json
import os

# ===== 設定 =====
col_index_current = 4   # 電流値（5列目）
col_index_speed   = 2   # モータ速度（3列目）
col_index_label   = 6   # ラベル列（7列目: 0=正常, 1=異常）
skip_header_rows  = 0
input_json        = "file_settings_customize.json"  # JSON作成済みファイル
anomaly_ratio_threshold = 0.5

# ===== CSV読込関数 =====
def load_csv_data(path, col_index, start_row=0, end_row=None, skip_header_rows=31):
    if not os.path.exists(path):
        print(f"[警告] ファイルが見つかりません: {path}")
        return np.array([])
    try:
        df = pd.read_csv(path, header=None, encoding='cp932', sep=None,
                         engine='python', skiprows=skip_header_rows)
    except UnicodeDecodeError:
        df = pd.read_csv(path, header=None, encoding='utf-8', sep=None,
                         engine='python', skiprows=skip_header_rows)

    if isinstance(col_index, list):
        data = df.iloc[start_row:end_row, col_index] if end_row else df.iloc[start_row:, col_index]
    else:
        data = df.iloc[start_row:end_row, col_index] if end_row else df.iloc[start_row:, col_index]
    return data.to_numpy()

# ===== 特徴量抽出関数 =====
def extract_features(current_data, speed_data, window_size):
    feats = []
    for i in range(0, len(current_data), window_size):
        cur_win = current_data[i:i+window_size]
        spd_win = speed_data[i:i+window_size]
        if len(cur_win) == window_size and len(spd_win) == window_size:
            ratio = cur_win / np.where(spd_win == 0, 1, spd_win)  # ゼロ割防止
            feats.append({
                # 電流特徴量
                "current_mean": np.mean(cur_win),
                "current_std": np.std(cur_win),
                "current_min": np.min(cur_win),
                "current_max": np.max(cur_win),
                # 速度特徴量
                "speed_mean": np.mean(spd_win),
                "speed_std": np.std(spd_win),
                "speed_min": np.min(spd_win),
                "speed_max": np.max(spd_win),
                # 電流/速度比
                "cur_spd_ratio_mean": np.mean(ratio),
                "cur_spd_ratio_std": np.std(ratio),
                "cur_spd_ratio_min": np.min(ratio),
                "cur_spd_ratio_max": np.max(ratio),
            })
    return pd.DataFrame(feats)

# ===== JSON読込 =====
with open(input_json, encoding="utf-8") as f:
    file_settings = json.load(f)

# ===== 正常データ（学習用） =====
train_features, train_labels = [], []

for key, setting in file_settings.get("normal", {}).items():
    path = setting["original_path"]
    start_row = setting["start_row"]
    end_row = setting["end_row"]
    window_size = setting["window_size"]

    current = load_csv_data(path, col_index_current, start_row, end_row)
    speed   = load_csv_data(path, col_index_speed, start_row, end_row)
    if len(current) == 0 or len(speed) == 0:
        continue

    feats = extract_features(current, speed, window_size)
    feats["source_file"] = Path(path).stem
    train_features.append(feats)
    train_labels.extend([1] * len(feats))  # 正常=1

train_df = pd.concat(train_features, ignore_index=True)

# ===== 異常データ（評価用） =====
eval_features, eval_labels = [], []

for key, setting in file_settings.get("abnormal", {}).items():
    path = setting["original_path"]
    start_row = setting["start_row"]
    end_row = setting["end_row"]
    window_size = setting["window_size"]

    data = load_csv_data(path, [col_index_current, col_index_speed, col_index_label], start_row, end_row)
    if len(data) == 0:
        continue
    current, speed, labels_csv = data[:,0], data[:,1], data[:,2]

    feats = extract_features(current, speed, window_size)
    feats["source_file"] = Path(path).stem
    eval_features.append(feats)

    # 多数決ラベル
    for i in range(0, len(labels_csv), window_size):
        window_labels = labels_csv[i:i+window_size]
        if len(window_labels) == window_size:
            abnormal_ratio = np.mean(window_labels == 1)
            eval_labels.append(-1 if abnormal_ratio > anomaly_ratio_threshold else 1)

eval_df = pd.concat(eval_features, ignore_index=True)
eval_df["label"] = eval_labels

# ===== CSV保存 =====
train_df.to_csv("train_features.csv", index=False, encoding="utf-8-sig")
eval_df.to_csv("eval_features.csv", index=False, encoding="utf-8-sig")

print("[INFO] train_features.csv / eval_features.csv を保存しました")
print("[INFO] train_features.csv の列:", train_df.columns.tolist())
print("[INFO] eval_features.csv の列:", eval_df.columns.tolist())
