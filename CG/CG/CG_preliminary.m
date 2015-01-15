function [ xp, v, h_x ] = CG_preliminary( f, A, b, x0)
%CG_preliminary: The CG method to solve Ax=b
%   we could min 0.5xAx - bx
%   A must be symmetric positive definite
    xk = x0;
    rk = A * xk - b;
    pk = -rk;
    e = 1e-4;
    h_x = xk;
    while norm(rk) > e
       ak = - (rk' * pk) / (pk' * A * pk);
       xk = xk + ak * pk;
       rk = A * xk - b;
       bk = (rk' * A * pk) / (pk' * A * pk);
       pk = -rk + bk * pk;
       s = size(h_x);
       h_x(:, s(2)+1) = xk;
    end
    xp = xk ;
    v = f(xp);
    h_x = h_x;
end

