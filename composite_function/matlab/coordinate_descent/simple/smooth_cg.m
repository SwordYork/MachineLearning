function [ x_opt, hist ] = smooth_cg( f, x0, dfx, dfy)
%smooth_cg minimize a smooth objective according to coordinate descent
%   Detailed explanation goes here

hist = x0;
x = x0;
while 1
    x_old = x;
%     for i=1:length(x0)
%         g = coordinate_g(f, x, i);
%         x(i) = fminunc(g, x(i));
%         hist = [hist x];
%     end
    x(1) = dfx(x(2));
    hist = [hist x];
    x(2) = dfy(x(1));
    hist = [hist x];
    if norm(x-x_old,2) < 1e-12
        x_opt = x;
        break
    end
end

end


% function [ x_opt, hist ] = smooth_cg( f, x0, dfx, dfy)
% %smooth_cg minimize a smooth objective according to coordinate descent
% %   Detailed explanation goes here
% 
% hist = x0;
% x = x0;
% while 1
%     x_old = x;
%     for i=1:length(x0)
%         g = coordinate_g(f, x, i);
%         x(i) = fminunc(g, x(i));
%         hist = [hist x];
%     end
%     if norm(x-x_old,2) < 1e-12
%         x_opt = x;
%         break
%     end
% end
% 
% end

% function hg = coordinate_g(f, x0, i)
%     hg = @(v) real_g(v,f,x0,i);
% end
%  
% function r = real_g(v,f,x0,i)
%     x0(i) = v;
%     r = f(x0);
% end
