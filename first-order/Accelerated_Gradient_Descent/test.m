%% Accelerated Gradient Descent

%% Gradient
% Gradient descent method is based on gradient
%
% $$ \nabla f  = \frac{\partial f}{\partial x_1 }\mathbf{e}_1 + \cdots +
% \frac{\partial f}{\partial x_n }\mathbf{e}_n $$
%
% gradient always point to the asent direction

%% Gradient Descent
% f is object function, and this is unconstrained
%
% $$\min_{x} f $$
%


%% Accelerated Gradient Descent
% 
% for t = 1,2,...
%
% $$x^{(t)} = y^{(t-1)} - alpha \nabla f(y^{t-1})$$
%
% $$y^{(t)} = x^{(t)} + (t-1) / (t+2) * (x^{(t)} - x^{(t-1)})$$ 
%

f = (@(X) (exp(X(1,:)-1) + exp(1-X(2,:)) + (X(1,:) - X(2,:)).^2));
%f = (@(X) (sin(0.5*X(1,:).^2 - 0.25 * X(2,:).^2 + 3) .* cos(2*X(1,:) + 1 - exp(X(2,:))) ))



%% Plot contour
[X, Y] = meshgrid(-2:0.1:2);
XX = [reshape(X, 1, numel(X)); reshape(Y, 1, numel(Y))];
%surf(X, Y, reshape(f(XX), length(X), length(X)))
contour(X, Y, reshape(f(XX), length(X), length(X)), 50);

hold on;

%%
% plot gradient of function
for i=1:5:length(XX)
    tmp = XX(:,i);
    g = gradient_of_function(f, tmp);
    %plot([tmp(1),tmp(1)+g(1)*0.02],[tmp(1),tmp(2)+g(1)*0.02]);
    quiver(tmp(1),tmp(2),g(1)*0.02,g(2)*0.02);
end


%%
% calculation
x0 = [-1; -1];

%%
% without wolfe step, fix step as alpha = 0.01, $x_k = x_{k-1} + alpha *
% (-\nabla f)$
%
[x_gf, v_gf, h_gf] = gradient_fix_step(f, x0)
[x_af, v_af, h_af] = accelerated_gradient_fix_step(f, x0)

%%
% find suitable step size
%
[x_g, v_g, h_g] = gradient(f, x0)
[x_a, v_a, h_ax, h_ay] = accelerated_gradient(f, x0)


% built-in method
[x_in, v_in] = fminunc(f, x0)



%%
% plot descent steps
for i=2:length(h_gf)
    tmp1 = h_gf(:,i-1);
    tmp2 = h_gf(:,i);
    quiver(tmp1(1),tmp1(2),tmp2(1)-tmp1(1),tmp2(2)-tmp1(2), 0, 'g','LineWidth',3)   
end


for i=2:length(h_af)
    tmp1 = h_af(:,i-1);
    tmp2 = h_af(:,i);
    quiver(tmp1(1),tmp1(2),tmp2(1)-tmp1(1),tmp2(2)-tmp1(2), 0, 'b','LineWidth',3)   
end


for i=2:length(h_g)
    tmp1 = h_g(:,i-1);
    tmp2 = h_g(:,i);
    quiver(tmp1(1),tmp1(2),tmp2(1)-tmp1(1),tmp2(2)-tmp1(2), 0, 'r','LineWidth',2)   
end

for i=2:length(h_ax)
    tmp1 = h_ax(:,i-1);
    tmp2 = h_ax(:,i);
    quiver(tmp1(1),tmp1(2),tmp2(1)-tmp1(1),tmp2(2)-tmp1(2), 0, 'c','LineWidth',2)   
end

for i=2:length(h_ay)
    tmp1 = h_ay(:,i-1);
    tmp2 = h_ay(:,i);
    quiver(tmp1(1),tmp1(2),tmp2(1)-tmp1(1),tmp2(2)-tmp1(2), 0, 'm','LineWidth',2)   
end



%% Reference
% # <http://stronglyconvex.com/blog/accelerated-gradient-descent.html>