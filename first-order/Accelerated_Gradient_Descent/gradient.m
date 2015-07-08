function [ xp, v, h_x ] = gradient(f, x0)
    xk = x0;
    df = -gradient_of_function(f, xk);
    e = 1e-4;
    h_x = xk;
    total_iter = 0;
    while norm(df) > e && total_iter < 100
        total_iter = total_iter + 1;
        alpha = wolfe(f, xk, df);
        %alpha = 1e-4;
        %alpha = 1e-2;
        xk = xk + alpha * df;
        s = size(h_x);
        h_x(:, s(2)+1) = xk;
        df = -gradient_of_function(f, xk);
    end
    
    xp = xk;
    v = f(xp);
    h_x;
end

