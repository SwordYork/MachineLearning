clear;

epsilon = 1e-4;

n = 10;
plot_type = 0;

A = rand(n, n);
b = rand(n, 1);

A = A'*A;


% A = [0.557302084233416,0.284240528163439;0.284240528163439,0.219852483398487];
% b = [0.482022061031856;0.120611613297162];
f = (@(x) (0.5 * x' * A * x - b' * x));
x0 = rand(1,n)

%% naive computing
r = A \ b
assert( norm(A * r - b) < epsilon, 'naive error result');

if n == 2
    [X, Y] = meshgrid( linspace(-2*abs(min(r)+1), 2*abs(max(r)+1), 20));
    XX = [reshape(X, 1, numel(X)); reshape(Y, 1, numel(Y))];
    xcell = num2cell(XX,1);
    v = cellfun(f, xcell);
    if plot_type == 0
        contour(X, Y, reshape(v, length(X), length(X)), 50);
        hold on
        plot(x0(1), x0(2),'or')
    else
        surf(X, Y, reshape(v, length(X), length(X)));
        hold on
        plot3(x0(1), x0(2), f(x0'), 'or')
    end
    
end


%% built-in search function
%[x_b, v] = fminunc(f, x0');
%assert( norm(A * x_b - b) < epsilon, 'built-in error result');


%% my CG preliminary method
[x_m, v, h] = CG_preliminary(f, A, b, x0')
assert( norm(A * x_m - b) < epsilon, 'cg-preliminary error result');


%% my CG method
[x_c, v, h] = cg(f, A, b, x0')
assert( norm(A * x_c - b) < epsilon, 'cg-preliminary error result');

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

