import torch
import torch.nn as nn
import math

class SimpleNN(nn.Module):
    def __init__(self, hidden_size=10):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)
        self.gelu = nn.GELU()

    def forward(self, x):
        # gelu as activation function
        x = self.gelu(self.fc1(x))
        x = self.gelu(self.fc2(x))
        x = self.gelu(self.fc3(x))
        x = self.fc4(x)
        return x

# class Pacejka2006_Fy0(nn.Module):
#     def __init__(self):
#         super(CustomModel, self).__init__()
#         # 定義可訓練參數
#         self.parameter_searcher = NNParameterSearcher()
#         # self.phy1 = nn.Parameter(torch.tensor(0.1))
#         # self.phy2 = nn.Parameter(torch.tensor(0.1))
#         # self.pky1 = nn.Parameter(torch.tensor(0.1))
#         # self.pky2 = nn.Parameter(torch.tensor(0.1))
#         # self.pky3 = nn.Parameter(torch.tensor(0.1))
#         # self.pky4 = nn.Parameter(torch.tensor(0.1))
#         # self.pky5 = nn.Parameter(torch.tensor(0.1))
#         # self.pky6 = nn.Parameter(torch.tensor(0.1))
#         # self.pky7 = nn.Parameter(torch.tensor(0.1))
#         # self.pvy1 = nn.Parameter(torch.tensor(0.1))
#         # self.pvy2 = nn.Parameter(torch.tensor(0.1))
#         # self.pvy3 = nn.Parameter(torch.tensor(0.1))
#         # self.pvy4 = nn.Parameter(torch.tensor(0.1))
#         # self.pcy1 = nn.Parameter(torch.tensor(0.1))
#         # self.pdy1 = nn.Parameter(torch.tensor(0.1))
#         # self.pdy2 = nn.Parameter(torch.tensor(0.1))
#         # self.pdy3 = nn.Parameter(torch.tensor(0.1))
#         # self.pey1 = nn.Parameter(torch.tensor(0.1))
#         # self.pey2 = nn.Parameter(torch.tensor(0.1))
#         # self.pey3 = nn.Parameter(torch.tensor(0.1))
#         # self.pey4 = nn.Parameter(torch.tensor(0.1))
#         # self.pey5 = nn.Parameter(torch.tensor(0.1))

#         # 定義常數
#         self.lambdaFzo = 1.0
#         self.lambdaMuyMi = 1.0
#         self.lambdyMuyPrime = 1.0
#         self.lambdaKya = 1.0
#         self.lambdaCy = 1.0
#         self.lambdaEy = 1.0
#         self.lambdaHy = 1.0
#         self.lambdaVy = 1.0
#         self.lambdaKyia = 1.0
#         self.zeta0 = 1.0
#         self.zeta2 = 1.0
#         self.zeta3 = 1.0
#         self.zeta4 = 1.0
#         self.epsilonK = 0.1
#         self.epsilonY = 0.1

#     def forward(self, A: torch.Tensor, Fz: torch.Tensor, ia: torch.Tensor):

#         x = self.parameter_searcher(A, Fz, ia)

#         phy2 = x[:, 0]
#         phy1 = x[:, 1]
#         pky1 = x[:, 2]
#         pky2 = x[:, 3]
#         pky3 = x[:, 4] 
#         pky4 = x[:, 5]
#         pky5 = x[:, 6]
#         pky6 = x[:, 7]
#         pky7 = x[:, 8]
#         pvy1 = x[:, 9]
#         pvy2 = x[:, 10]
#         pvy3 = x[:, 11]
#         pvy4 = x[:, 12]
#         pcy1 = x[:, 13]
#         pdy1 = x[:, 14]
#         pdy2 = x[:, 15]
#         pdy3 = x[:, 16]
#         pey1 = x[:, 17]
#         pey2 = x[:, 18]
#         pey3 = x[:, 19]
#         pey4 = x[:, 20]
#         pey5 = x[:, 21]

#         Fzo = Fz
#         FzoPrime = self.lambdaFzo * Fzo
#         dfz = (Fz - FzoPrime) / FzoPrime
#         ami = torch.tan(A * (math.pi / 180))
#         iaMi = torch.sin(ia * (math.pi / 180))

#         # 計算muy, Kya, Svyia, Kyiao, Shy, ay, Cy, Dy, Ey, By, Svy, Fy0
#         muy = ((pdy1 + pdy2 * dfz) / (1 + pdy3 * ia ** 2)) * self.lambdaMuyMi
#         Kya = pky1 * FzoPrime * torch.sin(pky4 * torch.atan((Fz / ((pky2 + pky5 * (iaMi ** 2)) * FzoPrime)))) / (1 + pky3 * (iaMi ** 2)) * self.zeta3 * self.lambdaKya
#         Svyia = Fz * (pvy3 + pvy4 * dfz) * iaMi * self.lambdaKyia * self.lambdyMuyPrime * self.zeta2
#         Kyiao = Fz * (pky6 + pky7 * dfz) * self.lambdaKyia
#         Shy = (phy1 + phy2 * dfz) * self.lambdaHy + ((Kyiao * iaMi - Svyia) * self.zeta0 / (Kya + self.epsilonK)) + self.zeta4 - 1
#         ay = ami + Shy
#         Cy = pcy1 * self.lambdaCy
#         Dy = muy * Fz * self.zeta2
#         Ey = (pey1 + pey2 * dfz) * (1 + pey5 * (iaMi ** 2) - (pey3 + pey4 * iaMi) * torch.sign(ay)) * self.lambdaEy
#         By = Kya / (Cy * Dy + self.epsilonY)
#         Svy = Fz * (pvy1 + pvy2 * dfz) * self.lambdaVy * self.lambdyMuyPrime * self.zeta2 + Svyia
#         Fy0 = Dy * torch.sin(Cy * torch.atan(By * ay - Ey * (By * ay - torch.atan(By * ay)))) + Svy

#         return Fy0
