import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import joblib

# ===== 設定 =====
train_features_csv = "train_features_scaled.csv"
feature_cols = None
model_save_path = "oneclass_svm_model.pkl"
scaler_save_path = "scaler_ocsvm.pkl"

# ハイパーパラメータ
nu_value = 0.0091
kernel_type = "rbf"
gamma_type = "scale"

# ===== データ読み込み =====
train_df = pd.read_csv(train_features_csv)
feature_cols = [c for c in train_df.columns if c not in ["source_file", "label"]]

# ===== 標準化 =====
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_df[feature_cols])

# ===== One-Class SVM 学習 =====
model_ocsvm = OneClassSVM(kernel=kernel_type, gamma=gamma_type, nu=nu_value)
model_ocsvm.fit(X_train_scaled)

# ===== 保存 =====
joblib.dump(model_ocsvm, model_save_path)
joblib.dump(scaler, scaler_save_path)

print(f"[INFO] One-Class SVM モデルとスケーラーを保存しました → {model_save_path}, {scaler_save_path}")
