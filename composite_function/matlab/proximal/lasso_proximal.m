clear

% generate random data
s = RandStream.create('mt19937ar','seed',0);
RandStream.setGlobalStream(s);

n = 600;       % number of examples
p = 500;      % number of features

x0 = sprandn(p,1,0.1);
A = randn(n,p);
A = A*spdiags(1./sqrt(sum(A.^2))',0,p,p); % normalize columns
v = sqrt(0.001)*randn(n,1);
y = A*x0 + v;

fprintf('solving instance with %d examples, %d variables\n', n, p);
fprintf('nnz(x0) = %d; signal-to-noise ratio: %.2f\n', nnz(x0), norm(A*x0)^2/norm(v)^2);


lambda = 0.1;

% cached computations for all methods
AtA = A'*A;
Atb = A'*y;


f = @(u) 0.5*(A*u-y)'*(A*u-y);
objective = @(u) f(u) + lambda * norm(u, 1);
prox_l1 = @(a, kappa) max(0, a-kappa) - max(0, -a-kappa);

MAX_ITER = 100;
ABSTOL   = 1e-7;
RELTOL   = 1e-4;


gamma = 1;
beta = 0.5;

tic;

x = zeros(p,1);
xprev = x;
prox_optval = [];
for k = 1:MAX_ITER
    while 1
        grad_x = AtA*x - Atb;
        z = prox_l1(x - gamma*grad_x, lambda*lambda);
        if f(z) <= f(x) + grad_x'*(z - x) + (1/(2*gamma))*sum_square(z - x)
            break;
        end
        gamma = beta*gamma;
    end
    xprev = x;
    x = z;

    prox_optval(k) = objective(x);
    if k > 1 && abs(prox_optval(k) - prox_optval(k-1)) < ABSTOL
        break;
    end
end

x_opt = x;
toc;

subplot(2,1,1)
plot(1:length(prox_optval),prox_optval,'o-') 

subplot(2,1,2)
plot(x,'o') 
hold on
plot(x0,'r.') 
legend('fit', 'real')
