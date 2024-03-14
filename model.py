import torch
import torch.nn as nn
import math

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        # 定義可訓練參數
        self.phy1 = nn.Parameter(torch.tensor(0.1))
        self.phy2 = nn.Parameter(torch.tensor(0.1))
        self.pky1 = nn.Parameter(torch.tensor(0.1))
        self.pky2 = nn.Parameter(torch.tensor(0.1))
        self.pky3 = nn.Parameter(torch.tensor(0.1))
        self.pky4 = nn.Parameter(torch.tensor(0.1))
        self.pky5 = nn.Parameter(torch.tensor(0.1))
        self.pky6 = nn.Parameter(torch.tensor(0.1))
        self.pky7 = nn.Parameter(torch.tensor(0.1))
        self.pvy1 = nn.Parameter(torch.tensor(0.1))
        self.pvy2 = nn.Parameter(torch.tensor(0.1))
        self.pvy3 = nn.Parameter(torch.tensor(0.1))
        self.pvy4 = nn.Parameter(torch.tensor(0.1))
        self.pcy1 = nn.Parameter(torch.tensor(0.1))
        self.pdy1 = nn.Parameter(torch.tensor(0.1))
        self.pdy2 = nn.Parameter(torch.tensor(0.1))
        self.pdy3 = nn.Parameter(torch.tensor(0.1))
        self.pey1 = nn.Parameter(torch.tensor(0.1))
        self.pey2 = nn.Parameter(torch.tensor(0.1))
        self.pey3 = nn.Parameter(torch.tensor(0.1))
        self.pey4 = nn.Parameter(torch.tensor(0.1))
        self.pey5 = nn.Parameter(torch.tensor(0.1))

        # 定義常數
        self.lambdaFzo = 1.0
        self.lambdaMuyMi = 1.0
        self.lambdyMuyPrime = 1.0
        self.lambdaKya = 1.0
        self.lambdaCy = 1.0
        self.lambdaEy = 1.0
        self.lambdaHy = 1.0
        self.lambdaVy = 1.0
        self.lambdaKyia = 1.0
        self.zeta0 = 1.0
        self.zeta2 = 1.0
        self.zeta3 = 1.0
        self.zeta4 = 1.0
        self.epsilonK = 0.1
        self.epsilonY = 0.1

    def forward(self, Fz, ia, a):
        Fzo = Fz
        FzoPrime = self.lambdaFzo * Fzo
        dfz = (Fz - FzoPrime) / FzoPrime
        ami = torch.tan(a * (math.pi / 180))
        iaMi = torch.sin(ia * (math.pi / 180))

        # 計算muy, Kya, Svyia, Kyiao, Shy, ay, Cy, Dy, Ey, By, Svy, Fy0
        muy = ((self.pdy1 + self.pdy2 * dfz) / (1 + self.pdy3 * ia ** 2)) * self.lambdaMuyMi
        Kya = self.pky1 * FzoPrime * torch.sin(self.pky4 * torch.atan((Fz / ((self.pky2 + self.pky5 * (iaMi ** 2)) * FzoPrime)))) / (1 + self.pky3 * (iaMi ** 2)) * self.zeta3 * self.lambdaKya
        Svyia = Fz * (self.pvy3 + self.pvy4 * dfz) * iaMi * self.lambdaKyia * self.lambdyMuyPrime * self.zeta2
        Kyiao = Fz * (self.pky6 + self.pky7 * dfz) * self.lambdaKyia
        Shy = (self.phy1 + self.phy2 * dfz) * self.lambdaHy + ((Kyiao * iaMi - Svyia) * self.zeta0 / (Kya + self.epsilonK)) + self.zeta4 - 1
        ay = ami + Shy
        Cy = self.pcy1 * self.lambdaCy
        Dy = muy * Fz * self.zeta2
        Ey = (self.pey1 + self.pey2 * dfz) * (1 + self.pey5 * (iaMi ** 2) - (self.pey3 + self.pey4 * iaMi) * torch.sign(ay)) * self.lambdaEy
        By = Kya / (Cy * Dy + self.epsilonY)
        Svy = Fz * (self.pvy1 + self.pvy2 * dfz) * self.lambdaVy * self.lambdyMuyPrime * self.zeta2 + Svyia
        Fy0 = Dy * torch.sin(Cy * torch.atan(By * ay - Ey * (By * ay - torch.atan(By * ay)))) + Svy

        return Fy0
