function [xp, v, h_x] = vlbfgs(f, x0, m)
    xk = x0;
    Im = eye(2);
    Hk = Im;
    s = [];
    y = [];
    g_fk = gradient_of_function(f, xk);
    pk = -Hk * g_fk;
    alpha = wolfe(f, xk, pk);
    k = 0;
    h_x = xk;
    s(:,k+1) = alpha * pk;
    xk = xk + s(:,k+1);
    y(:,k+1) = gradient_of_function(f, xk) - g_fk;
    g_fk = y(:,k+1) + g_fk;
    e = 1e-4;
    sh = size(h_x);
    h_x(:, sh(2)+1) = xk;
    k = k + 1;
    while norm(g_fk) > e      
        pk =  vlbfgs_loop(s, y, k, m, g_fk);
        %alpha = strongwolfe(f, xk, pk,2);     
        alpha = wolfe(f, xk, pk);
        s(:,k+1) = alpha * pk;
        xk = xk + s(:,k+1);
        y(:,k+1) = gradient_of_function(f, xk) - g_fk;
        g_fk = y(:,k+1) + g_fk;
        sh = size(h_x);
        h_x(:, sh(2)+1) = xk;
        k = k + 1;        
    end
    xp = xk;
    v = f(xp);
    
end


function [r] = vlbfgs_loop(s, y, k, m, g_fk) 

    %initial b[]
    b = [];
    if k >= m
        tail = k - m;
    else
        tail = 0;
    end

    b(:,2*m + 1) = g_fk;
    j = 0;
    for i = k-1:-1:tail
        b(:,m - j) = s(:,i+1);
        b(:,2*m - j) = y(:,i+1);
        j = j + 1;
    end
    
    bb = zeros(2*m+1, 2*m+1);
    for i = 1:2*m+1
        for j = 1:2*m+1
            bb(i,j) = b(:,i)' * b(:,j);
        end
    end
    
    % vlbfgs
    kesi = zeros(1,2*m+1);
    kesi(2*m+1) = -1;
    size_of_f = size(g_fk);
    d= size_of_f(1);
    a = zeros(1, m);
   
    for i = k-1:-1:tail
        j = i - (k - m) + 1;
        a(i+1) = kesi * bb(:,j) / bb(j,j+m);
        kesi(m+j) = kesi(m+j) - a(i+1);
    end
    
    for i = 1:2*m+1
        kesi(i) = bb(m,2*m) / bb(2*m,2*m) * kesi(i);
    end
    
    for i = tail:k-1
        j = i - (k - m) + 1;
        beta = kesi * bb(:,m+j) / bb(j,j+m);
        kesi(j) = kesi(j) + a(i+1) - beta;
    end
    
    r = sum(repmat(kesi, d, 1) .* b, 2);
end