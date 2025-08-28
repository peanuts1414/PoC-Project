import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import joblib

# ===== 設定 =====
eval_features_csv = "eval_features_scaled.csv"
threshold_ocsvm = -0.6  # decision_function 閾値
model_save_path = "oneclass_svm_model.pkl"
scaler_save_path = "scaler_ocsvm.pkl"

# ===== データ読み込み =====
eval_df = pd.read_csv(eval_features_csv)
feature_cols = [c for c in eval_df.columns if c not in ["source_file", "label"]]
y_eval = eval_df["label"].astype(int).to_numpy()  # 1=正常, -1=異常

# ===== 学習済みモデル・スケーラー読み込み =====
model_ocsvm = joblib.load(model_save_path)
scaler = joblib.load(scaler_save_path)

# ===== 標準化 =====
X_eval_scaled = scaler.transform(eval_df[feature_cols])

# ===== 予測 =====
scores = model_ocsvm.decision_function(X_eval_scaled)
preds = np.where(scores >= threshold_ocsvm, 1, -1)

# ===== 評価 =====
metrics = {
    "Accuracy": accuracy_score(y_eval, preds),
    "Precision": precision_score(y_eval, preds, pos_label=-1, zero_division=0),
    "Recall": recall_score(y_eval, preds, pos_label=-1, zero_division=0),
    "F1": f1_score(y_eval, preds, pos_label=-1, zero_division=0)
}

print("[INFO] One-Class SVM 評価結果")
print(metrics)

# ===== 混同行列と分類レポート =====
cm = confusion_matrix(y_eval, preds, labels=[1, -1])
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_eval, preds, labels=[1, -1], target_names=["Normal(1)", "Anomaly(-1)"]))

# ===== スコア分布可視化 =====
plt.figure(figsize=(8,6))
plt.hist(scores[y_eval==1], bins=30, alpha=0.6, label="Normal")
plt.hist(scores[y_eval==-1], bins=30, alpha=0.6, label="Abnormal")
plt.axvline(threshold_ocsvm, color="red", linestyle="--", label=f"Threshold={threshold_ocsvm}")
plt.title("One-Class SVM Scores (with threshold)")
plt.xlabel("Decision function score")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig("oneclass_svm_scores.png")
print("[INFO] グラフ保存 → oneclass_svm_scores.png")

# ===== メトリクス保存 =====
pd.DataFrame([metrics]).to_csv("oneclass_svm_results.csv", index=False)
print("[INFO] 評価結果保存 → oneclass_svm_results.csv")
