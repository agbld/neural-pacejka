% define trainable parameters
phy1, phy2, pky1, pky2, pky3, pky4, pky5, pky6, pky7, pvy1, pvy2, pvy3, pvy4, pcy1, pdy1, pdy2, pdy3, pey1, pey2, pey3, pey4, pey5, 

% define constant
lambdaFzo=1;        %λFzo   (nominal (rated) load) 
lambdaMuyMi=1;      %λμy*   (peak friction coefficient)
lambdyMuyPrime=1;   %λμy'   (peak friction coefficient)
lambdaKya=1;        %λKya   (cornering stiffness)
lambdaCy=1;         %λCy    (shape factor)
lambdaEy=1;         %λEy    (curvature factor)
lambdaHy=1;         %λHy    (horizontal shift)
lambdaVy=1;         %λVy    (vertical shift)
lambdaKyia=1;       %λKyγ   (camber force stiffness)

zeta0=1;
zeta2=1;
zeta3=1;
zeta4=1;

epsilonK=0.1;
epsilonY=0.1;

% define input, all of them are float scalar
Fz, ia, a

% forward passing
Fzo=Fz;
FzoPrime=lambdaFzo*Fzo;     %(4.E1)Fzo'
dfz=(Fz-FzoPrime)/FzoPrime; %(4.E2)dfz
%% 處理SA數據
ami=tan(a*(pi/180));       %(4.E3)α*
%% 處理IA數據
iaMi=sin(ia*(pi/180));     %(4.E4)γ*
%% Full formula set(lateral force pure side slip)
%(4.E23)
muy=((pdy1+pdy2*dfz)/(1+pdy3*ia^2))*lambdaMuyMi;

%(4.E25)
Kya=pky1*FzoPrime*sin(pky4*atan((Fz/((pky2+pky5*(iaMi^2))*FzoPrime)))) /(1+pky3*(iaMi^2))*zeta3*lambdaKya;

%(4.E28)
Svyia=Fz*(pvy3+pvy4*dfz)*iaMi*lambdaKyia*lambdyMuyPrime*zeta2;

%(4.E30)
Kyiao=Fz*(pky6+pky7*dfz)*lambdaKyia;

%(4.E27)
Shy=(phy1+phy2*dfz)*lambdaHy+((Kyiao*iaMi-Svyia)*zeta0/(Kya+epsilonK))+zeta4-1;

%(4.E20)
ay=ami+Shy;

%(4.E21)
Cy=pcy1*lambdaCy;

%(4.E22)
Dy=muy*Fz*zeta2;

%(4.E24)
Ey=(pey1+pey2*dfz)*(1+pey5*(iaMi^2)-(pey3+pey4*iaMi)*sign(ay))*lambdaEy;

%(4.E26)
By=Kya/(Cy*Dy+epsilonY);

%(4.E29)
Svy=Fz*(pvy1+pvy2*dfz)*lambdaVy*lambdyMuyPrime*zeta2+Svyia;

%(4.E19)final result=Fy0(pure side slip)Fy0=ANS，該項需趨近於FY(=TESTANS,CSV中第602:1200位)
Fy0=Dy*sin(Cy*atan(By*ay-Ey*(By*ay-atan(By*ay))))+Svy;