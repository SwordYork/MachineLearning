[y, X] = libsvmread('/home/york/Program/dataset/small/mushrooms.txt');
X = [ones(size(y)) X];
y = (y-1)*2 - 1;
[n, p] = size(X);
train_n = ceil(n * 0.5);
%r = randperm(n);
r = 1:n;
train_X = X(r(1:train_n),:);
train_y = y(r(1:train_n));

test_X = X(r(train_n+1:n),:);
test_y = y(r(train_n+1:n));

[n, p] = size(train_X);

lambda = 0.5;
% minimize    (1/2)*||Ax - b||_2 + lambda*||x||_2,
fprintf('minimize    (1/2)*||Ax - b||_2 + %f*||x||_2\n', lambda);
tic
cvx_begin
    variable x(p)
    minimize(0.5*norm(train_X*x - train_y, 2) + lambda * norm(x, 2))
cvx_end
toc

py = (test_X * x > 0.5)*2 - 1;
sum(py ~=  test_y) / length(py)

fprintf('\n\n\n');
fprintf('\n\n\n');


cvx_begin
    variables x(p)
    minimize(-train_y'*train_X*x+sum(log_sum_exp([zeros(1,n); x'*train_X'])) + lambda * norm(x,1))
cvx_end
py = (test_X * x > 0.5)*2 - 1;
sum(py ~=  test_y) / length(py)

fprintf('\n\n\n');
fprintf('\n\n\n');
