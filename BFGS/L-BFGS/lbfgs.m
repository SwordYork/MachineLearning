function [xp, v, h_x] = lbfgs(f, x0, m)
    xk = x0;
    Im = eye(2);
    Hk = Im;
    s = [];
    y = [];
    g_fk = gradient_of_function(f, xk);
    pk = -Hk * g_fk;
    alpha = wolfe(f, xk, pk);
    k = 0
    s(:,k+1) = alpha * pk;
    xk = xk + s(:,k+1)
    y(:,k+1) = gradient_of_function(f, xk) - g_fk;
    g_fk = y(:,k+1) + g_fk;
    e = 1e-4;
    h_x = xk;
    k = k + 1
    while norm(g_fk) > e      
        pk = - lbfgs_loop(s, y, k, m, g_fk);
        %alpha = strongwolfe(f, xk, pk,2);
        alpha = wolfe(f, xk, pk);
        s(:,k+1) = alpha * pk;
        xk = xk + s(:,k+1)
        y(:,k+1) = gradient_of_function(f, xk) - g_fk;
        g_fk = y(:,k+1) + g_fk;
        sh = size(h_x);
        h_x(:, sh(2)+1) = xk;
        k = k + 1
    end
    xp = xk;
    v = f(xp);
    
end

function [r] = lbfgs_loop(s, y, k, m, g_fk)
    q = g_fk;
    Im = eye(2);
    Hk = (s(:,k)' * y(:,k)) / (y(:,k)' * y(:,k)) * Im;
    if k >= m
        tail = k - m;
    else
        tail = 0
    end
    a = zeros(1, m);
    for i = k-1:-1:tail
        a(i+1) = (1/(y(:,i+1)' * s(:,i+1))) * s(:,i+1)' * q;
        q = q - a(i+1) * y(:,i+1)
    end
    r = Hk * q;
    for i = tail:k-1
        b = (1/(y(:,i+1)' * s(:,i+1))) * y(:,i+1)'*r;
        r = r + s(:,i+1) * (a(i+1) - b);
    end
end