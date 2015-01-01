f = (@(X) (exp(X(1,:)-1) + exp(1-X(2,:)) + (X(1,:) - X(2,:)).^2));

%f = (@(X) (sin(0.5*X(1,:).^2 - 0.25 * X(2,:).^2 + 3) .* cos(2*X(1,:) + 1 - exp(X(2,:))) ))
%f = (@(X) (1-X(1,:)).^2 + 100 * (X(2,:) - X(1,:).^2).^2);
[X, Y] = meshgrid([-2:0.1:2]);

XX = [reshape(X, 1, numel(X)); reshape(Y, 1, numel(Y))];
%surf(X, Y, reshape(f(XX), length(X), length(X)))
contour(X, Y, reshape(f(XX), length(X), length(X)), 50)

hold on;

% for i=1:length(XX)
%     tmp = XX(:,i);
%     g = gradient_of_function(f, tmp);
%     plot([tmp(1),tmp(1)+g(1)*0.02],[tmp(2),tmp(2)+g(2)*0.02]);
% end

%quiver(X,Y,DX,DY)

%wolfe(f, [1;1], 1)
x0 = [-1; -1];
[x, v, h] = bfgs(f, x0)
[x, v] = fminunc(f, x0)


for i=2:length(h)
    tmp1 = h(:,i-1);
    tmp2 = h(:,i);
    quiver(tmp1(1),tmp1(2),tmp2(1)-tmp1(1),tmp2(2)-tmp1(2), 0, 'r','LineWidth',2)   
end