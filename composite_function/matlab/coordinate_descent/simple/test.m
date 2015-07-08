f = @(x) (5*x(1,:).^2 - 6*x(1,:) .* x(2,:) + 5*(x(2,:)-1).^2 - 0.0259)  + (abs(x(1,:)) + abs(x(2,:))) 

fminunc(f, [1;1])
[X, Y] = meshgrid(linspace(0.4,1.4,200), linspace(0.8,1.8,200));

vX = reshape(X, numel(X),1);
vY = reshape(Y, numel(Y),1);

vZ = f([vX, vY]');
Z = reshape(vZ, size(X));


contour(X, Y, Z, 80);
[x_opt, hist] = smooth_cg(f, [1;1], @(y) (0.6*y), @(x) (1+0.6*x))
f(x_opt)

hold on;
%plot(hist(1,:), hist(2,:), 'r-')

% plot step
for i=2:length(hist)
    tmp1 = hist(:,i-1);
    tmp2 = hist(:,i);

    quiver(tmp1(1),tmp1(2),tmp2(1)-tmp1(1),tmp2(2)-tmp1(2), 0, 'r','LineWidth',2)   
end