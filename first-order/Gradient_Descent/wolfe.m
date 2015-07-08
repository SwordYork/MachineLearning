function alpha = wolfe(f, xk, pk)
% f target 
% xk the point
% pk descent direction
    alpha = 1;
    ro = 0.2;
    c = 1e-4;
    g = gradient_of_function(f, xk);
    while (f(xk + alpha*pk) > (f(xk) + c * alpha * g' * pk))
        alpha = ro * alpha;
    end     
end