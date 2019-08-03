% Limpa ambiente
clc
clear all
clear

% Carrega dados
load xt2
load ydt2
load xv2
load ydv2

% Troca nomes das variaveis
x_treino = xt2;
y_treino = ydt2;
x_validacao = xv2;
y_validacao = ydv2;
clearvars xt ydt xv ydv;

% Variaveis do modelo
n_epocas = 50;
eta = 0.01; % taxa de aprendizado

n_pontos = size(x_treino, 1);
m = size(x_treino, 2); % número de features
k = 5; % número de regras

% Inicializacao dos parametros
xmax = max(x_treino);
xmin = min(x_treino);
delta = (xmax - xmin)/(k-1);
for j=1:k % regras
    for i=1:m % features
        c(i,j) = xmin(i) + (j-1)*delta(i);
        s(i,j) = delta(i)/(2*sqrt(2*log(2)));
        p(i,j) = 2*rand - 1;
    end
    q(j) = rand;
end
    
% Treino da rede
dphi_dqj = 1;
y_pred = zeros(n_pontos, 1);

for epoca=1:n_epocas
    for n=1:n_pontos %pontos
        [y_s, w, phi, b] = calcula_saida(m, k, x_treino(n,:), c, s, p, q);
        y_pred(n) = y_s;
        de_dypred = (y_pred(n) - y_treino(n));
        for j=1:k %regras
            dypred_dphij = w(j)/b;
            dypred_dwj = (phi(j)-y_pred(n))/b;
            for i=1:m %features
               dphij_dpij = x_treino(n,i);  
               dwj_dcij = w(j)*( (x_treino(n,i)-c(i,j))/s(i,j) );
               dwj_dsij = w(j)*( ((x_treino(n,i)-c(i,j))^2)/(s(i,j)^3) );

               de_dcij = de_dypred*dypred_dwj*dwj_dcij;
               de_dsij = de_dypred*dypred_dwj*dwj_dsij;
               de_dpij = de_dypred*dypred_dphij*dphij_dpij;

               c(i,j) = c(i,j) - eta*de_dcij;
               s(i,j) = s(i,j) - eta*de_dsij;
               p(i,j) = p(i,j) - eta*de_dpij;
            end
            de_dqj = de_dypred*dypred_dphij*dphi_dqj;
            q(j) = q(j) - eta*de_dqj;
        end
    end
end

% Predicao nos dados validacao
n_validacao = size(x_validacao, 1);
y_pred = zeros(n_validacao, 1);
for n=1:n_validacao
    
    [y_s, w, phi, b] = calcula_saida(m, k, x_validacao(n,:), c, s, p, q);
    y_pred(n) = y_s;
    
end

figure
subplot(2,2,1);
plot(y_validacao)
hold on
plot(y_pred)
legend('Validation Data', 'Sistema Fuzzy Adaptativo');

% Erro quadratico medio (MSE)
MSE = (sum((y_pred - y_validacao).^2))/n_validacao

% GENFIS1
trnData = [x_treino y_treino'];
numMFs = 5;
mfType = 'gaussmf';
epoch_n = 20;
in_fismat = genfis1(trnData, numMFs, mfType, 'linear');
out_fismat = anfis(trnData, in_fismat, epoch_n);
ys = evalfis(x_validacao, out_fismat);
subplot(2,2,2);
plot(y_validacao);
hold on
plot(ys);
legend('Validation Data', 'GENFIS1 Output');
MSE2 = (sum((ys - y_validacao).^2))/n_validacao

% GENFIS2
in_fismat2 = genfis2(x_treino, y_treino, 0.8);
out_fismat2 = anfis(trnData, in_fismat2, epoch_n);
ys2 = evalfis(x_validacao, out_fismat2);
subplot(2,2,3);
plot(y_validacao);
hold on
plot(ys2);
legend('Validation Data', 'GENFIS2 Output');
MSE3 = (sum((ys2 - y_validacao).^2))/n_validacao