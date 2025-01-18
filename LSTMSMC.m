clear all
close all
clc

%% Initial Conditions
x(1) = 0.25;
y(1) = 0.25;
z(1) = 0.25;

r = 0.3;
Gp = diag([0.1, 0.1, 0.1]);
Gs = 0.01 * eye(3, 3);
u = zeros(2, 1);
u1 = zeros(2, 1);
u2 = zeros(2, 1);
u3 = zeros(2, 1);

PHI = [0.8, 0, 0, 0;
       0, 0.5, 0, 0;
       0, 0, 0, 0];

mu = 0.5;
lambda = 25;
s = zeros(3, 1);

%% LSTM1 Initial Parameters
H = zeros(9, 1);
H1 = H;
bf = zeros(9, 1);
bI = zeros(9, 1);
bc = zeros(9, 1);
bo = zeros(9, 1);
bh = zeros(3, 1);
wf = zeros(9, 12);
wI = zeros(9, 12);
wc = zeros(9, 12);
wo = zeros(9, 12);
wh = rand(3, 9);

ETAL = 0.02;

%% Main Loop
for k = 1:1000
    C = [x(k); y(k); z(k)];

    if k == 1
        C1 = zeros(3, 1);
    else
        C1 = [x(k-1); y(k-1); z(k-1)];
    end

    Cd = [r*cos(2*pi*(sin(pi*k/250))^2)+0.2;
          r*cos(pi/4)*sin(2*pi*(sin(pi*k/250))^2)+0.25;
          r*sin(pi/4)*sin(2*pi*(sin(pi*k/250))^2)+0.25];

    E = Cd - C;
    e1(k) = E(1);
    e2(k) = E(2);
    e3(k) = E(3);
    d(k) = norm(E);

    ad(k+1) = ad(k) + d(k);

    %% LSTM1 Estimation
    x1(k) = d(k);
    if k == 1
        x2(k) = d(k);
        x3(k) = abs(d(k));
    else
        x2(k) = d(k) - d(k-1);
        x3(k) = x3(k-1) + abs(d(k));
    end
    X = [x1(k); x2(k); x3(k)];

    % LSTM1 Computation
    for i = 1:9
        netf(i) = wf(i, :) * [X; H1] + bf(i);
        f(i) = sigmoid(netf(i));

        netI(i) = wI(i, :) * [X; H1] + bI(i);
        I(i) = sigmoid(netI(i));

        netcb(i) = wc(i, :) * [X; H1] + bc(i);
        cb(i) = tanh(netcb(i));

        c(i) = c1(i) * f(i) + I(i) * cb(i);

        neto(i) = wo(i, :) * [X; H1] + bo(i);
        o(i) = sigmoid(neto(i));

        H(i) = o(i) * tanh(c(i));
    end

    if k == 1
        Gs = 0.5 * eye(3, 3);
    else
        Gs = diag(sigmoid(wh * H + bh));
    end

    GSD1(k) = Gs(1, 1);
    GSD2(k) = Gs(2, 2);
    GSD3(k) = Gs(3, 3);

    %% Update PHI
    if k > 1
        PHI = PHI - (Gp * (C - C1 - PHI * (U1 - U2)) * (U1 - U2)') / (mu + norm(U1 - U2)^2);
    end

    PHI1 = PHI(1:3, 1:2);
    PHI2 = PHI(1:3, 3:4);
    
    SS = inv(lambda * eye(2, 2) + PHI1' * PHI1) * PHI1';
    V = PHI1 * inv(lambda * eye(2, 2) + PHI1' * PHI1) * PHI1';

    Sign = sign(s);

    %% Update Control Inputs
    if k == 1
        U = [u; u1];
        U1 = [u1; u2];
        U2 = [u2; u3];
    else
        u = u1 + inv(lambda * eye(2, 2) + PHI1' * PHI1) * PHI1' * (Gs * Sign + C1 - C + 2 * E - s - PHI2 * (u1 - u2));
        u3 = u2;
        u2 = u1;
        u1 = u;
        U = [u; u1];
        U1 = [u1; u2];
        U2 = [u2; u3];
    end

    s = (eye(3, 3) - V) * (C1 - C + 2 * E - s - PHI2 * (u1 - u2)) + V * s - V * Gs * Sign;

    %% YOUR PLANT


    

    %% LSTM1 Update
    e = Cd - [x(k+1); y(k+1); z(k+1)];

    for i = 1 : 1 : 9
        Bp(i) =sign(s(1))*e(1)*(PHI(1,1)*SS(1,1) +PHI(1,2)*SS(2,1))*exp(-wh1(1,:)*H+bh1(1))*wh1(1,i)/(1+exp(-wh1(1,:)*H+bh1(1)))^2+sign(s(2))*e(2)*(PHI(2,1)*SS(1,2) +PHI(2,2)*SS(2,2))*exp(-wh1(2,:)*H+bh1(2))*wh1(2,i)/(1+exp(-wh1(2,:)*H+bh1(2)))^2+sign(s(3))*e(3)*(PHI(3,1)*SS(1,3) +PHI(3,2)*SS(2,3))*exp(-wh1(3,:)*H+bh1(3))*wh1(3,i)/(1+exp(-wh1(3,:)*H+bh1(3)))^2;

        wf(i,:) = wf1(i,:) - ETAL*Bp(i)*o(i)*(1-c(i)^2)*c1(i)*(exp(-(wf1(i,:)*[X;H1] + bf1(i)))/(1+exp(-(wf1(i,:)*[X;H1] + bf1(i))))^2)*[X;H1]';
        wI(i,:) = wI1(i,:) - ETAL*Bp(i)*o(i)*(1-c(i)^2)*cb(i)*(exp(-(wI1(i,:)*[X;H1] + bI1(i)))/(1+exp(-(wI1(i,:)*[X;H1] + bI1(i))))^2)*[X;H1]';
        wc(i,:) = wc1(i,:) - ETAL*Bp(i)*o(i)*(1-c(i)^2)*I(i)*(1-(wc1(i,:)*[X;H1] + bc1(i))^2)*[X;H1]';
        wo(i,:) = wo1(i,:) - (ETAL*Bp(i)*tanh(c(i))*exp(-(wo1(i,:)*[X;H1] + bo1(i)))/(1+exp(-(wo1(i,:)*[X;H1] + bo1(i))))^2)*[X;H1]';
    
        bf(i) = bf1(i) - ETAL*Bp(i)*o(i)*(1-c(i)^2)*c1(i)*exp(-(wf1(i,:)*[X;H1] + bf1(i)))/(1+exp(-(wf1(i,:)*[X;H1] + bf1(i))))^2;
        bI(i) = bI1(i) - ETAL*Bp(i)*o(i)*(1-c(i)^2)*cb(i)*exp(-(wI1(i,:)*[X;H1] + bI1(i)))/(1+exp(-(wI1(i,:)*[X;H1] + bI1(i))))^2;
        bc(i) = bc1(i) - ETAL*Bp(i)*o(i)*(1-c(i)^2)*I(i)*(1-(wc1(i,:)*[X;H1] + bc1(i))^2);
        bo(i) = bo1(i) - ETAL*Bp(i)*tanh(c(i))*exp(-(wo1(i,:)*[X;H1] + bo1(i)))/(1+exp(-(wo1(i,:)*[X;H1] + bo1(i))))^2;
    end
    wf1 = wf;
    wI1 = wI;
    wc1 = wc;
    wo1 = wo;
    bf1 = bf;
    bI1 = bI;
    bc1 = bc;
    bo1 = bo;

    for i = 1 : 1 : 3
        onet = wh1(i,:)*H+bh1(i);
        wh(i,:) = wh1(i,:) - (ETAL*sign(s(i))*e(i)*(PHI(i,1)*SS(1,i) +PHI(i,2)*SS(2,i))*exp(-onet)/(1+exp(-onet))^2)*H';
        bh(i) = bh1(i) - ETAL*sign(s(i))*e(i)*(PHI(i,1)*SS(1,i) +PHI(i,2)*SS(2,i))*exp(-onet)/(1+exp(-onet))^2;
    end
    wh1 = wh;
    bh1 = bh;

    c1 = c;
    H1 = H;