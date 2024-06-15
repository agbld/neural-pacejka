#%%
import pandas as pd
import torch
import torch.nn as nn

from model import SimpleNN
from mpl_toolkits.mplot3d import Axes3D

#%%
raw_data = pd.read_csv('for_fit_csv.csv', header=None)
raw_data = torch.tensor(raw_data.values, dtype=torch.float32)

A = raw_data[:, :599]
Fz = raw_data[:, 599]
ia = raw_data[:, 600]
Fy0 = raw_data[:, 601:]

A = A.reshape(-1, 1)
Fz = Fz.unsqueeze(1).repeat(1, 599).reshape(-1, 1)
ia = ia.unsqueeze(1).repeat(1, 599).reshape(-1, 1)
Fy0 = Fy0.reshape(-1, 1)

X = torch.cat((A, Fz, ia), dim=1)

y = Fy0

#%%
# load model from model.pth
model = SimpleNN(hidden_size=10)
model.load_state_dict(torch.load('model.pth'))

#%%
# evaluate model
model.eval()
y_pred = model(X).detach().numpy()

loss_fn = nn.MSELoss()
mean_abs_error = nn.L1Loss()
loss = loss_fn(torch.tensor(y_pred), y)
mae = mean_abs_error(torch.tensor(y_pred), y)
print(f'Loss: {loss.item()}, MAE: {mae.item()}')

#%%
# plot a 3D graph A vs Fz vs y. use only ia == 0.0

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ia_0 = ia[:, 0] == 0
X_ia_0 = X[ia_0]
y_pred_ia_0 = y_pred[ia_0]

ax.scatter(X_ia_0[:, 0], X_ia_0[:, 1], y_pred_ia_0, c='r', marker='o')
ax.scatter(X_ia_0[:, 0], X_ia_0[:, 1], y[ia_0], c='b', marker='x')

ax.set_xlabel('A')
ax.set_ylabel('Fz')
ax.set_zlabel('y')

plt.show()

#%%