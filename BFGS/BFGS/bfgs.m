function [xp, v, h_x] = bfgs(f,x0)
    xk = x0;
    Im = eye(2);
    Hk = Im;
    df = gradient_of_function(f, xk);
    e = 1e-4;
    h_x = xk;
    while norm(df) > e
        pk = -Hk * df;
        alpha = wolfe(f, xk, pk);
        %alpha = 0.2
        sk = alpha * pk;
        xk = xk + sk;                          %x_k+1
        yk = gradient_of_function(f, xk) - df; %df_k+1 - df_k
        df = yk + df;                          %df_k+1 
        rok = 1 / (yk' * sk);
        Hk = (Im - rok*sk*yk') * Hk * (Im - rok * yk * sk') + rok * sk * sk';
        s = size(h_x);
        h_x(:, s(2)+1) = xk;
    end
    xp = xk;
    v = f(xp);
    h_x;
end