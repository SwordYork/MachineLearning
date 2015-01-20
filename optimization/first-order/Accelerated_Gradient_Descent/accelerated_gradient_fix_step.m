function [ xp, v, h_x ] = accelerated_gradient_fix_step(f, x0 )
    yk = x0;
    xk = x0;
    df = -gradient_of_function(f, yk);
    dfx = -gradient_of_function(f, xk);
    e = 1e-4;
    h_x = xk;
    total_iter = 0;
    while norm(dfx) > e && total_iter < 100
        total_iter = total_iter + 1;
        alpha = 0.01;
        xk = yk + alpha * df;
        dfx = -gradient_of_function(f, xk);

        s = size(h_x);
        h_x(:, s(2)+1) = xk;
        s = size(h_x);
        yk = xk + (total_iter - 1) / (total_iter + 2) * (xk - h_x(:,s(2)-1));
        
        df = -gradient_of_function(f, yk);
    end
    
    xp = xk;
    v = f(xp);
    h_x;

end

