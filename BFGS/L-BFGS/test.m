%f = (@(X) (exp(X(1,:)-1) + exp(1-X(2,:)) + (X(1,:) - X(2,:)).^2));

%f = (@(X) (sin(0.5*X(1,:).^2 - 0.25 * X(2,:).^2 + 3) .* cos(2*X(1,:) + 1 - exp(X(2,:))) ))
%f = (@(X) (1-X(1,:)).^2 + 100 * (X(2,:) - X(1,:).^2).^2);
f = (@(X) (abs(X(1,:)) + abs(X(2,:)-1)));
plot_type = 0;
[X, Y] = meshgrid([-2:0.1:2]);
XX = [reshape(X, 1, numel(X)); reshape(Y, 1, numel(Y))];
%surf(X, Y, reshape(f(XX), length(X), length(X)))
contour(X, Y, reshape(f(XX), length(X), length(X)), 50)

hold on;


x0 = [-1; -1];
[x, v, h] = lbfgs(f, x0, 2)

% plot step
for i=2:length(h)
    tmp1 = h(:,i-1);
    tmp2 = h(:,i);
    if plot_type == 0
        quiver(tmp1(1),tmp1(2),tmp2(1)-tmp1(1),tmp2(2)-tmp1(2), 0, 'r','LineWidth',2)   
    else
        quiver3(tmp1(1),tmp1(2),(f(tmp1) + 0.5)*1.1, tmp2(1)-tmp1(1),tmp2(2)-tmp1(2), (f(tmp2) - f(tmp1))*1.1 , 0, 'r','LineWidth',3)   
    end
end


[x, v, h] = bfgs(f, x0)

% plot step
for i=2:length(h)
    tmp1 = h(:,i-1);
    tmp2 = h(:,i);
    if plot_type == 0
        quiver(tmp1(1),tmp1(2),tmp2(1)-tmp1(1),tmp2(2)-tmp1(2), 0, 'g','LineWidth',2)   
    else
        quiver3(tmp1(1),tmp1(2),(f(tmp1) + 0.5)*1.1, tmp2(1)-tmp1(1),tmp2(2)-tmp1(2), (f(tmp2) - f(tmp1))*1.1 , 0, 'r','LineWidth',3)   
    end
end