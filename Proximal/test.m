f = @(x)abs(x-9)

tic, fminunc(f, 0), toc



tic
xkp = 0;
pf = @(x) f(x) + 1/2 * (x-xkp)^2 ;
xk = fminunc(pf, 0);

while abs(xkp-xk) > 1e-4
    pf = @(x) f(x) + 1/2 * (x-xk)^2;
    xkp = xk;
    xk = fminunc(pf,0);
end

toc