import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib

# ===== 設定 =====
train_features_csv = "train_features_scaled.csv"
feature_cols = ["current_mean","current_std","current_min","current_max",
                "speed_mean","speed_std","speed_min","speed_max",
                "cur_spd_ratio_mean","cur_spd_ratio_std","cur_spd_ratio_min","cur_spd_ratio_max"]

batch_size = 64
num_epochs = 40
learning_rate = 1e-3

model_save_path = "autoencoder_model.pth"
scaler_save_path = "scaler_ae.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] 使用デバイス: {device}")

# ===== データ読み込み =====
train_df = pd.read_csv(train_features_csv)
X_train = train_df[feature_cols].to_numpy()

# ===== 標準化 =====
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
joblib.dump(scaler, scaler_save_path)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
train_loader = DataLoader(TensorDataset(X_train_tensor), batch_size=batch_size, shuffle=True)

# ===== Autoencoder モデル =====
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

input_dim = X_train_scaled.shape[1]
model = Autoencoder(input_dim).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ===== 学習 =====
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        x_batch = batch[0]
        optimizer.zero_grad()
        x_hat = model(x_batch)
        loss = criterion(x_hat, x_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * x_batch.size(0)
    epoch_loss /= len(train_loader.dataset)
    if (epoch+1) % 10 == 0 or epoch == 0:
        print(f"[INFO] Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}")

# ===== モデル保存 =====
torch.save(model.state_dict(), model_save_path)
print(f"[INFO] Autoencoder モデル保存 → {model_save_path}")
print(f"[INFO] 標準化スケーラー保存 → {scaler_save_path}")
