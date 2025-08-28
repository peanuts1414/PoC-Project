# train_if.py
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib  # モデル保存用

# ===== 設定 =====
train_features_csv = "train_features_scaled.csv"
feature_cols = None
model_save_path = "isolation_forest_model.pkl"
scaler_save_path = "scaler_if.pkl"

# ===== データ読み込み =====
train_df = pd.read_csv(train_features_csv)
feature_cols = [col for col in train_df.columns if col not in ["source_file","label"]]

# ===== 標準化 =====
scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[feature_cols])

# ===== Isolation Forest 学習 =====
model_if = IsolationForest(contamination=0.3, random_state=42)
model_if.fit(X_train)

# ===== 保存 =====
joblib.dump(model_if, model_save_path)
joblib.dump(scaler, scaler_save_path)

print(f"[INFO] モデルとスケーラーを保存しました → {model_save_path}, {scaler_save_path}")
