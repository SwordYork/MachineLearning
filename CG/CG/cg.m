function [ xp, v, h_x ] = cg(f, A, b, x0)
%CG_preliminary: The CG method to solve Ax=b
%   we could min 0.5xAx - bx
%   A must be symmetric positive definite
    xk = x0;
    rk = A * xk - b;
    pk = -rk;
    e = 1e-4;
    h_x = xk;
    while norm(rk) > e
       ak = (rk' * rk) / (pk' * A * pk);
       xk = xk + ak * pk;
       tmp_rk = rk;
       rk = ak * A * pk + rk;
       bk = (rk' * rk) / (tmp_rk' * tmp_rk);
       pk = -rk + bk * pk;
       s = size(h_x);
       h_x(:, s(2)+1) = xk;
    end
    xp = xk ;
    v = f(xp);
    h_x = h_x;
end
