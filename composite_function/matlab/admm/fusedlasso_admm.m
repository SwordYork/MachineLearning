clear

% generate random data
s = RandStream.create('mt19937ar','seed',0);
RandStream.setGlobalStream(s);

n = 600;       % number of examples
p = 500;      % number of features

x0 = sprandn(p-2,1,0.1);
x0(p-1) = 0;
x0(p) = 0;

tx0 = nonzeros(x0);
ni = find(x0);

x0(ni+1) = tx0;
x0(ni+2) = tx0;


A = randn(n,p);
%A = A*spdiags(1./sqrt(sum(A.^2))',0,p,p); % normalize columns
v = sqrt(0.001)*randn(n,1);
y = A*x0 + v;

fprintf('solving instance with %d examples, %d variables\n', n, p);
fprintf('nnz(x0) = %d; signal-to-noise ratio: %.2f\n', nnz(x0), norm(A*x0)^2/norm(v)^2);


lambda = 0.1;

% cached computations for all methods
AtA = A'*A;
Atb = A'*y;


f = @(u) 0.5*(A*u-y)'*(A*u-y);
objective = @(x, z) f(x) + lambda * norm(z, 1);
prox_l1 = @(a, kappa) max(0, a-kappa) - max(0, -a-kappa);

MAX_ITER = 100;
ABSTOL   = 1e-7;
RELTOL   = 1e-4;


F = eye(p-1,p);
for i=1:p-1
    F(i,i+1) = -1;
end
F = sparse(F);


rho = 1;

tic;

x = zeros(p,1);
z = zeros(p-1,1);
u = zeros(p-1,1);

AtApFtF = inv(AtA + rho * F' * F);

for k = 1:MAX_ITER

    % x-update
    x = AtApFtF * ( Atb + rho * F' * (z - u));

    % z-update
    zold = z;
    z = prox_l1(F*x + u, lambda / rho);

    % u-update
    u = u + F*x - z;

    % diagnostics, reporting, termination checks
    admm_optval(k)   = objective(x, z);
    r_norm(k)   = norm(F*x - z);
    s_norm(k)   = norm(-rho*(z - zold));
    eps_pri(k)  = sqrt(p)*ABSTOL + RELTOL*max(norm(F*x), norm(-z));
    eps_dual(k) = sqrt(p)*ABSTOL + RELTOL*norm(rho*u);

    if r_norm(k) < eps_pri(k) && s_norm(k) < eps_dual(k)
         break;
    end
end
toc;


subplot(2,1,1)
plot(1:length(admm_optval),admm_optval,'o-') 

subplot(2,1,2)
plot(x,'ko') 
hold on
plot(x0,'r.') 
legend('fit', 'real')
 
