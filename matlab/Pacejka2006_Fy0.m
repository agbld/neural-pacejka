%Pacejka2006 formula set的Fy0部分(pure side slip)，程式特化用於擬合用，sa會以數組方式輸入，所以部分算式要用*
function Fy0 = Pacejka2006_Fy0(x, for_fit)
%% 定義需擬合係數
phy1=x(1);
phy2=x(2);

pky1=x(3);
pky2=x(4);
pky3=x(5);
pky4=x(6);%(usually=2)
pky5=x(7);
pky6=x(8);
pky7=x(9);

pvy1=x(10);
pvy2=x(11);
pvy3=x(12);
pvy4=x(13);

pcy1=x(14);

pdy1=x(15);
pdy2=x(16);
pdy3=x(17);

pey1=x(18);
pey2=x(19);
pey3=x(20);
pey4=x(21);
pey5=x(22);
%% 測試用
% a=linspace(-12,12,100);
% Fz=-400;
% x=[0,0,10,1.5,0,2,0,2.5,0,0,0,0.15,0,1.3,1,0,0,-1,0,0,2,0];
% phy1=x(1);
% phy2=x(2);
% 
% pky1=x(3);
% pky2=x(4);
% pky3=x(5);
% pky4=x(6);%(usually=2)
% pky5=x(7);
% pky6=x(8);
% pky7=x(9);
% 
% pvy1=x(10);
% pvy2=x(11);
% pvy3=x(12);
% pvy4=x(13);
% 
% pcy1=x(14);
% 
% pdy1=x(15);
% pdy2=x(16);
% pdy3=x(17);
% 
% pey1=x(18);
% pey2=x(19);
% pey3=x(20);
% pey4=x(21);
% pey5=x(22);
%% 定義使用者係數λ
%有問題的是λμy*和λμy'的定義，，應該要用4.E7和4.E8的公式，這裡直接用1
%但是4.E7和4.E8的目的是為了復現摩擦係數大幅變化，以測試結果與未來預期工況應該可忽略
lambdaFzo=1;        %λFzo   (nominal (rated) load) 
lambdaMuyMi=1;      %λμy*   (peak friction coefficient)
lambdyMuyPrime=1;   %λμy'   (peak friction coefficient)
lambdaKya=1;        %λKya   (cornering stiffness)
lambdaCy=1;         %λCy    (shape factor)
lambdaEy=1;         %λEy    (curvature factor)
lambdaHy=1;         %λHy    (horizontal shift)
lambdaVy=1;         %λVy    (vertical shift)
lambdaKyia=1;       %λKyγ   (camber force stiffness)


%% 定義ζ

%窩不知道這幹嘛用的，應該也是一個調整因子

zeta0=1;
zeta2=1;
zeta3=1;
zeta4=1;

%% 定義ε
%用意是不要讓分母為0
epsilonK=0.1;
epsilonY=0.1;
%% 將導入數據賦值給FZ,IA,A
Fz=for_fit.FZ;
ia=for_fit.IA;
a=for_fit.SA;
%% 處理FZ數據
Fzo=Fz;%(這裡理論上要做平均，但我已將浮動的FZ數據插值成單一值)
FzoPrime=lambdaFzo.*Fzo;     %(4.E1)Fzo'
dfz=(Fz-FzoPrime)./FzoPrime; %(4.E2)dfz
%% 處理SA數據
ami=tan(a.*(pi/180));       %(4.E3)α*
%% 處理IA數據
iaMi=sin(ia.*(pi/180));     %(4.E4)γ*
%% Full formula set(lateral force pure side slip)
%(4.E23)
muy=((pdy1+pdy2.*dfz)./(1+pdy3.*ia^2)).*lambdaMuyMi;

%(4.E25)
Kya=pky1.*FzoPrime.*sin(pky4.*atan((Fz./((pky2+pky5.*(iaMi^2)).*FzoPrime)))) ./(1+pky3.*(iaMi^2)).*zeta3.*lambdaKya;

%(4.E28)
Svyia=Fz.*(pvy3+pvy4.*dfz).*iaMi.*lambdaKyia.*lambdyMuyPrime.*zeta2;

%(4.E30)
Kyiao=Fz.*(pky6+pky7.*dfz).*lambdaKyia;

%(4.E27)
Shy=(phy1+phy2.*dfz).*lambdaHy+((Kyiao.*iaMi-Svyia).*zeta0./(Kya+epsilonK))+zeta4-1;

%(4.E20)
ay=ami+Shy;

%(4.E21)
Cy=pcy1.*lambdaCy;

%(4.E22)
Dy=muy.*Fz.*zeta2;

%(4.E24)
Ey=(pey1+pey2.*dfz).*(1+pey5.*(iaMi^2)-(pey3+pey4.*iaMi).*sign(ay)).*lambdaEy;

%(4.E26)
By=Kya./(Cy.*Dy+epsilonY);

%(4.E29)
Svy=Fz.*(pvy1+pvy2.*dfz).*lambdaVy.*lambdyMuyPrime.*zeta2+Svyia;

%(4.E19)final result=Fy0(pure side slip)
Fy0=Dy.*sin(Cy.*atan(By.*ay-Ey.*(By.*ay-atan(By.*ay))))+Svy;

%% 測試用
% figure;
% plot(a,Fy0)
end