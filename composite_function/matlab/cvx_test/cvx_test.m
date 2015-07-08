s = RandStream.create('mt19937ar','seed',0);
RandStream.setGlobalStream(s);

m = 500;       % number of examples
n = 400;      % number of features

x0 = sprandn(n,1,0.05);
A = randn(m,n);
A = A*spdiags(1./sqrt(sum(A.^2))',0,n,n); % normalize columns
v = sqrt(0.001)*randn(m,1);
b = A*x0 + v;

fprintf('solving instance with %d examples, %d variables\n', m, n);
fprintf('nnz(x0) = %d; signal-to-noise ratio: %.2f\n', nnz(x0), norm(A*x0)^2/norm(v)^2);

gamma_max = norm(A'*b,'inf');
gamma = 0.01*gamma_max


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                       Ridge Regression 
cvx_clear
% minimize    (1/2)*||Ax - b||^2 + gamma*||x||_2^2,
fprintf('minimize    (1/2)*||Ax - b||^2 + %f*||x||_2^2\n', gamma);
cvx_begin quiet 
    variable x(n)
    minimize(0.5*sum_square(A*x - b) + gamma*sum_square(x))
cvx_end

xs = x;
norm_xs = norm(xs,2)
fprintf('\n\n\n');

% minimize    (1/2)*||Ax - b||^2,
% s.t.        ||x||_2^2 <= ||xs||_2^2
fprintf('minimize    (1/2)*||Ax - b||^2,\n');
fprintf('s.t.        ||x||_2^2 <= %f \n', norm_xs^2);
cvx_begin quiet
    variable x(n)
    dual variable y
    minimize(0.5*sum_square(A*x - b))
    subject to
        y: sum_square(x) <= sum_square(xs);
cvx_end

diff = norm(xs - x)
fprintf('\n\n\n');

% minimize    (1/2)*||Ax - b||^2,
% s.t.        ||x||_2 <= ||xs||_2
fprintf('minimize    (1/2)*||Ax - b||^2,\n');
fprintf('s.t.        ||x||_2 <= %f\n', norm_xs);
cvx_begin quiet
    variable x(n)
    dual variable y
    minimize(0.5*sum_square(A*x - b))
    subject to
        y: norm(x,2) <= norm(xs,2);
cvx_end
dual_variable = y
diff = norm(xs - x)
fprintf('\n\n\n');




% minimize    (1/2)*||Ax - b||^2 + y*||x||_2,
fprintf('minimize    (1/2)*||Ax - b||^2 + %f*||x||_2\n', y);
cvx_begin quiet
    variable x(n)
    minimize(0.5*sum_square(A*x - b) + y * norm(x,2))
cvx_end

diff = norm(xs - x)
fprintf('\n\n\n');


% minimize    (1/2)*||Ax - b||_2,
% s.t.        ||x||_2 <= ||xs||_2
fprintf('minimize    (1/2)*||Ax - b||_2,\n');
fprintf('s.t.        ||x||_2 <= %f\n', norm_xs);
cvx_begin quiet
    variable x(n)
    dual variable y
    minimize(0.5*norm(A*x - b, 2))
    subject to
        y: norm(x,2) <= norm(xs,2);
cvx_end
dual_variable = y
diff = norm(xs - x)
fprintf('\n\n\n');


% minimize    (1/2)*||Ax - b||_2 + y*||x||_2,
fprintf('minimize    (1/2)*||Ax - b||_2 + %f*||x||_2\n', y);
cvx_begin quiet
    variable x(n)
    minimize(0.5*norm(A*x - b,2) + y * norm(x,2))
cvx_end

diff = norm(xs - x)
fprintf('\n\n\n');
fprintf('\n\n\n');

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                        LASSO
% minimize    (1/2)*||Ax - b||^2 + gamma*||x||_1,
fprintf('minimize    (1/2)*||Ax - b||^2 + %f*||x||_1\n', gamma);
cvx_begin quiet
    variable x(n)
    minimize(0.5*sum_square(A*x - b) + gamma*norm(x,1))
cvx_end

xs = x;
norm_xs = norm(xs)
fprintf('\n\n\n');


% minimize    (1/2)*||Ax - b||^2,
% s.t.        ||x||_1 <= ||xs||_1
fprintf('minimize    (1/2)*||Ax - b||^2,\n');
fprintf('s.t.        ||x||_1 <= %f\n', norm_xs);
cvx_begin quiet
    variable x(n);
    dual variable y;
    minimize(0.5*sum_square(A*x - b));
    subject to
        y: norm(x,1) <= norm(xs,1);
cvx_end

dual_variable = y
diff = norm(xs - x)

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                 Logistic Regression

randn('state',0);
rand('state',0);

m = 100;       % number of examples
n = 50;      % number of features

% Generate data
x0 = sprandn(n,1,0.05);
A = randn(m,n);
v = sqrt(0.001)*randn(m,1);
y = (0.5 < exp(A*x0+v)./(1+exp(A*x0+v)));

% Solve problem
%
% minimize  -(sum_(y_i=1) ui)*a - b + sum log (1+exp(a*ui+b)

pA = [ones(m,1) A];
cvx_expert true
cvx_begin
    variables x(n+1)
    minimize(-y'*pA*x+sum(log_sum_exp([zeros(1,m); x'*pA'])) + 0.01 * norm(x,1))
cvx_end

