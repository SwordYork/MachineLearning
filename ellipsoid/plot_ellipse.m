function [ a, b, alpha ] = plot_ellipse(P, x0)
%
% Ref: http://www.maa.org/external_archive/joma/Volume8/Kalman/General.html

%%
% P is sysmmetric
% the ellipse is defined as $$(z-x_0)^{T}P^{-1}(z-x0) = 1$
% From origin ellipse $\frac{x^2}{a^2} + \frac{y^2}{b^2} = 1$
% rotate an angle $\alpha$, transfered to
% $\frac{(xcos(\alpha)+ysin(\alpha))^2}{a^2} +
% \frac{(xsin(\alpha)-ycos(\alpha))^2}{b^2}=1$
% Then we can find $a, b, \alpha$
%
%%%%%


PT = (P^-1);
AA = PT(1,1);
BA = PT(1,2);
DA = PT(2,2);

% solve to find a, b, alpha
syms z
ra = double(sqrt( 1 / (max(solve(z^2-(AA+DA)*z+AA*DA-BA^2,z)))));
rb = sqrt(1/double(AA+DA-1/ra^2));

if abs(1/ra^2 - 1/rb^2) < 1e-5
    cosa2 = 0;
else
    cosa2 = (AA - 1/rb^2) / (1/ra^2 - 1/rb^2);
end

sina2 = 1 - cosa2;
cosa = sqrt(cosa2);
sina = sqrt(sina2);

% change rotation
if sign(1/ra^2 - 1/rb^2) ~= sign(BA)
    cosa = -cosa;  
end

t = 0:0.01:2*pi;
xx = ra * cos(t) * cosa + rb * sin(t) * sina + x0(1);
yy = ra * cos(t) * sina - rb * sin(t) * cosa + x0(2);

plot(xx,yy);
hold on


end

