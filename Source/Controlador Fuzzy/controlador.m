close all;
clear all;
clc;

% yreferencia
yref(1:80)=5;	
yref(81:160)=	-4;	
yref(161:240)=	3;	
yref(241:320)=	-2;	
yref(321:400)=	1;		

plot(yref)

% condicoes iniciais
y(1:2)=0;	
u(1:2)=0;
% 
% ke = 0.1;
% ks = 0.1;

ke = 0.0102;
ks = 0.235;

% param=readfis('controlador1.fis');
param=readfis('fuzzy_controller.fis');


for k=3:400
    % equacao do controlador
    y(k)=1.4*y(k-1) - 0.6*y(k-2) - 3*u(k-1)^3 + 2*u(k-1) - u(k-2)^3 + 2*u(k-2);
    erro(k) = yref(k)-y(k);
    u(k) = u(k-1) + ks*evalfis(ke*erro(k), param);
end

plot(yref);
hold on
plot(y);
legend('Referência', 'Resposta do controlador');


