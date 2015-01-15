function [ xp, v, h_x ] = cg(f, x0)
%CG: The CG method to nonlinear
    xk = x0;
    fk = f(xk);
    g_fk = gradient_of_function(f, xk);
    pk = -g_fk;
    e = 1e-4;
    h_x = xk;
    while norm(g_fk) > e
       %ak = wolfe(f, xk, pk);
       ak = strongwolfe(f, xk, pk, 2);
       tmp_xk = xk;
       xk = xk + ak * pk;
       tmp_g_fk = g_fk;
       g_fk = gradient_of_function(f, xk);
      
       bk = (g_fk' * g_fk) / (tmp_g_fk' * tmp_g_fk);
       pk = -g_fk + bk * pk;
       s = size(h_x);
       h_x(:, s(2)+1) = xk;
    end
    xp = xk ;
    v = f(xp);
    h_x = h_x;
end
