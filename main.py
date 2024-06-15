#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import SimpleNN

#%%
raw_data = pd.read_csv('for_fit_csv.csv', header=None)
raw_data = torch.tensor(raw_data.values, dtype=torch.float32)

A = raw_data[:, :599]
Fz = raw_data[:, 599]
ia = raw_data[:, 600]
Fy0 = raw_data[:, 601:]

#%%
A = A.reshape(-1, 1)
Fz = Fz.unsqueeze(1).repeat(1, 599).reshape(-1, 1)
ia = ia.unsqueeze(1).repeat(1, 599).reshape(-1, 1)
Fy0 = Fy0.reshape(-1, 1)

X = torch.cat((A, Fz, ia), dim=1)

y = Fy0

#%%
# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 封裝成 TensorDataset
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# 定義 DataLoader
batch_size = 1024  # 或根據您的 GPU 記憶體大小選擇其他批量大小
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#%%
# 2. 定義模型
model = SimpleNN(hidden_size=10)

#%%
# 3. 定義損失函數和優化器
loss_fn = nn.MSELoss()
mean_abs_error = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)

#%%
# 4. 訓練模型
epochs = 20
eval_steps = 100

step_count = 0
plot_train_x = []
plot_train_loss = []
plot_train_mean_abs_error = []
plot_test_x = []
plot_test_loss = []
plot_test_mean_abs_error = []

with tqdm(total=epochs) as pbar:
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            # monitor and visualize
            step_count += 1
            plot_train_x.append(step_count)
            plot_train_loss.append(loss.item())
            plot_train_mean_abs_error.append(mean_abs_error(outputs, y_batch).item())

            if step_count % eval_steps == 0:
                with torch.no_grad():
                    model.eval()
                    predictions = model(X_test)
                    test_loss = loss_fn(predictions, y_test)
                    plot_test_x.append(step_count)
                    plot_test_loss.append(test_loss.item())
                    plot_test_mean_abs_error.append(mean_abs_error(predictions, y_test).item())
        pbar.update(1)

    # plot
    plt.figure()
    plt.plot(plot_train_x, plot_train_loss, color='b', label='train_loss')
    plt.plot(plot_test_x, plot_test_loss, color='r', label='test_loss')
    plt.twinx()
    plt.plot(plot_train_x, plot_train_mean_abs_error, 'r', color='g', label='train_mean_abs_error')
    plt.plot(plot_test_x, plot_test_mean_abs_error, 'r', color='y')
    plt.show()

    # save model
    torch.save(model.state_dict(), 'model.pth')

#%%