function g = gradient_of_function(f, xk)
    len = length(xk);
    g = zeros(len, 1);
    delta = 1e-5;
    for i=1:len
        t = zeros(len, 1);
        t(i) = delta;
        df = f(xk+t) - f(xk);
        g(i) = df / delta;
    end
end