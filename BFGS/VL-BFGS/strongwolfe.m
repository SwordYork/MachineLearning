function alphas = strongwolfe(f, x0, pk, alpham)
    alphap = 0;
    c1 = 1e-4;
    c2 = 0.9;
    alphax = alpham*rand(1);
    phi = @(a)f(x0 + a * pk);
    fx0 = phi(0);
    g_fx0 = gradient_of_function(phi, 0);
    fxp = fx0;
    i = 1;
    while 1
        fxx = phi(alphax);      
        if (fxx > fx0 + c1 * alphax * g_fx0) || ((i > 1) && (fxx > fxp))
            alphas = zoom(f,x0,pk,alphap,alphax);
            return;
        end
        g_fxx = gradient_of_function(phi, alphax);
        
        if abs(g_fxx) <= -c2 * g_fx0
            alphas = alphax;
            return;
        end
        
        if g_fxx >= 0
            alphas = zoom(f,x0,pk,alphax,alphap);
            return;
        end
        alphap = alphax;
        fxp = fxx;
        alphax = alphax + (alpham-alphax)*rand(1);
        i = i + 1;
    end
end


function alphas = zoom(f,x0,d,alphal,alphah)

    c1 = 1e-4;
    c2 = 0.9;

    phi = @(a)f(x0 + a * d);
    fx0 = phi(0);
    g_fx0 = gradient_of_function(phi, 0);
    while 1
       alphax = 1/2*(alphal+alphah);
       xx = x0 + alphax*d;
       fxx = f(xx);
       g_fxx = gradient_of_function(phi, alphax);
       
       xl = x0 + alphal*d;
       fxl = f(xl);
       if ((fxx > fx0 + c1*alphax*g_fx0) || (fxx >= fxl)),
          alphah = alphax;
       else
          if abs(g_fxx) <= -c2*g_fx0
            alphas = alphax;
            return;
          end
          if g_fxx*(alphah-alphal) >= 0,
            alphah = alphal;
          end
          alphal = alphax;
       end
    end
end 