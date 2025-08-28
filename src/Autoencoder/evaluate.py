import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import joblib

# ===== 設定 =====
eval_features_csv = "eval_features_scaled.csv"
feature_cols = ["current_mean","current_std","current_min","current_max",
                "speed_mean","speed_std","speed_min","speed_max",
                "cur_spd_ratio_mean","cur_spd_ratio_std","cur_spd_ratio_min","cur_spd_ratio_max"]

model_save_path = "autoencoder_model.pth"
scaler_save_path = "scaler_ae.pkl"
threshold_quantile = 0.986  # 正常データのパーセンタイル

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] 使用デバイス: {device}")

# ===== データ読み込み =====
eval_df = pd.read_csv(eval_features_csv)
X_eval = eval_df[feature_cols].to_numpy()
y_eval = eval_df["label"].to_numpy()  # 1=正常, -1=異常

# ===== 学習済みスケーラー読み込み & 標準化 =====
scaler = joblib.load(scaler_save_path)
X_eval_scaled = scaler.transform(X_eval)
X_eval_tensor = torch.tensor(X_eval_scaled, dtype=torch.float32).to(device)

# ===== Autoencoder モデル定義 & 読み込み =====
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

input_dim = X_eval_scaled.shape[1]
model = Autoencoder(input_dim).to(device)
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.eval()

# ===== 異常スコア（再構築誤差）計算 =====
with torch.no_grad():
    recon_eval = model(X_eval_tensor)
    errors = torch.mean((X_eval_tensor - recon_eval) ** 2, dim=1).cpu().numpy()

# ===== 閾値設定（正常データのパーセンタイル） =====
threshold = np.quantile(errors[y_eval==1], threshold_quantile)
preds = np.where(errors > threshold, -1, 1)  # -1=異常, 1=正常

# ===== 評価 =====
metrics = {
    "Accuracy": accuracy_score(y_eval, preds),
    "Precision": precision_score(y_eval, preds, pos_label=-1, zero_division=0),
    "Recall": recall_score(y_eval, preds, pos_label=-1, zero_division=0),
    "F1": f1_score(y_eval, preds, pos_label=-1, zero_division=0)
}
print("[INFO] Autoencoder 評価結果")
print(metrics)

# ===== 可視化 =====
plt.figure(figsize=(8,6))
plt.hist(errors[y_eval==1], bins=30, alpha=0.6, label="Normal")
plt.hist(errors[y_eval==-1], bins=30, alpha=0.6, label="Abnormal")
plt.axvline(threshold, color="red", linestyle="--", label=f"Threshold={threshold:.6f}")
plt.title("Autoencoder Reconstruction Errors")
plt.legend()
plt.tight_layout()
plt.savefig("autoencoder_scores.png")
print("[INFO] グラフ保存 → autoencoder_scores.png")

# ===== メトリクス保存 =====
pd.DataFrame([metrics]).to_csv("autoencoder_results.csv", index=False)
print("[INFO] 評価結果保存 → autoencoder_results.csv")
