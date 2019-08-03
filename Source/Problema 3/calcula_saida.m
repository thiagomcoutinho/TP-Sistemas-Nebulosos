function [y_pred, w, phi, b] = calcula_saida(m, k, x, c, s, p, q)
   a = zeros(1, 1);
   b = zeros(1, 1);
   w = ones(1, k);
   phi = q;

   for j=1:k % regras
       for i=1:m % features
           phi(j) = phi(j) + p(i,j)*x(i);
           w(j) = w(j)*exp(-1/2*(x(i)-c(i,j))^2/s(i,j)^2);
       end
       a = a + (w(j)*phi(j));
       b = b + w(j);
   end
   y_pred = a/b;
   
end  