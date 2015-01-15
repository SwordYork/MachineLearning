x = -50:0.1:50;
x = x';
y = exp(sin(x));

n = length(y);

ny = y + randn(n,1);


figure;
subplot(2,1,1)
plot(x, ny);

D = zeros(n-1, n);

for i=1:n-1
    D(i,i) = -1;
    D(i,i+1) = 1;
end

lambda = 0.01;
cvx_begin
    variable xh(n)
    minimize norm(xh-y) + lambda*norm(D*xh)
cvx_end
subplot(2,1,2)
plot(x,xh);
hold on

cvx_begin
    variable xh(n)
    minimize norm(xh-y) + lambda*norm(D*xh,1)
cvx_end

plot(x,xh,'go');
