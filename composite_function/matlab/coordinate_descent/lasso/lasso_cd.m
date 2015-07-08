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


f = @(u) 0.5*(A*u-y)'*(A*u-y);
objective = @(x) f(x) + lambda * norm(x, 1);
prox_l1 = @(a, kappa) max(0, a-kappa) - max(0, -a-kappa);

MAX_ITER = 100;
ABSTOL   = 1e-7;
RELTOL   = 1e-4;


tic;


x = zeros(p,1);
all_Ax = A * x;
for k = 1:MAX_ITER
    for i = 1:p
        Ai = A(:,i);
        aa = Ai' * Ai;
       
        part_Ax = all_Ax - Ai*x(i);
        
        x(i) = prox_l1( Ai'*(y - part_Ax) / aa, lambda / aa);
 
        all_Ax = part_Ax + Ai * x(i); 
    end    
    
    cd_optval(k) = objective(x);
    if k > 1 && norm(cd_optval(k-1) - cd_optval(k)) < ABSTOL
        break
    end
end
toc;


subplot(2,1,1)
plot(1:length(cd_optval),cd_optval,'o-') 

subplot(2,1,2)
plot(x,'ko') 
hold on
plot(x0,'r.') 
legend('fit', 'real')
 
 
