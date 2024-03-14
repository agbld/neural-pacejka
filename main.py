#%%
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import CustomModel

# 檢查是否有可用的 CUDA 支持的 GPU
device = 'cpu' #torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

#%%
# 1. 加載和預處理數據
def transform_csv(input_csv):
    # 讀取 CSV 檔案
    data = input_csv

    # 預先定義新的 DataFrame，以便填充轉換後的數據
    transformed_data = pd.DataFrame(columns=['Fz', 'ia', 'a', 'Fy0'])

    # 對每一列（row）進行操作
    with tqdm(total=len(data)) as pbar:
        for index, row in data.iterrows():
            a_values = row[0:599].values  # 第0到598(含)是 a
            Fz_value = row[599]  # 第599個欄位是 Fz
            ia_value = row[600]  # 第600個欄位是 ia
            Fy0_values = row[601:].values  # 從第601個欄位開始到1199是 Fy0

            # 對於每個 a 值和對應的 Fy0 值，創建一個新的行
            for a, Fy0 in zip(a_values, Fy0_values):
                transformed_row = {'Fz': Fz_value, 'ia': ia_value, 'a': a, 'Fy0': Fy0}
                # transformed_data = transformed_data.append(transformed_row, ignore_index=True)
                transformed_data = pd.concat([transformed_data, pd.DataFrame(transformed_row, index=[0])], ignore_index=True)
            pbar.update(1)

    return transformed_data

if os.path.exists('transformed_data.csv'):
    data = pd.read_csv('transformed_data.csv')
else:
    raw_data = pd.read_csv('for_fit_csv.csv', header=None)
    data = transform_csv(raw_data)
    data.to_csv('transformed_data.csv', index=False)
X = data[['Fz', 'ia', 'a']].values
y = data['Fy0'].values

# 將數據轉換為 PyTorch 張量並移動到指定的設備（CPU 或 GPU）
X = torch.tensor(X, dtype=torch.float32).to(device)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(device)  # 調整形狀以匹配模型輸出

# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X.cpu().numpy(), y.cpu().numpy(), test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = map(lambda x: torch.tensor(x, device=device), (X_train, X_test, y_train, y_test))

# 封裝成 TensorDataset
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# 定義 DataLoader
batch_size = 64  # 或根據您的 GPU 記憶體大小選擇其他批量大小
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


#%%
# 2. 定義模型
model = CustomModel()

#%%
# 3. 定義損失函數和優化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

#%%
# 4. 訓練模型
epochs = 1000
for epoch in range(epochs):
    model.train()
    total_loss = 0
    total_abs_error = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch[:, 0], X_batch[:, 1], X_batch[:, 2])
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_abs_error += torch.mean(torch.abs(outputs - y_batch)).item()
    if epoch % 1 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader)}, Abs. Averaged Error: {total_abs_error / len(train_loader)}')



epochs = 1000
# with tqdm(total=epochs) as pbar:
for epoch in range(epochs):
    model.train()
    
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_train[:, 0], X_train[:, 1], X_train[:, 2])
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    if epoch % 1 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Abs. Averaged Error: {torch.mean(torch.abs(outputs - y_train)).item()}')
        # pbar.update(1)

#%%
# 5. 測試模型
model.eval()
with torch.no_grad():
    predictions = model(X_test[:, 0], X_test[:, 1], X_test[:, 2])
    test_loss = criterion(predictions, y_test)
    print(f'Test Loss: {test_loss.item()}, Test Abs. Averaged Error: {torch.mean(torch.abs(predictions - y_test)).item()}')

#%%