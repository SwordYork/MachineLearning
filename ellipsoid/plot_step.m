% number of variables n and linear functions m
n = 2; 
m = 5;

%********************************************************************
% generate an example
%********************************************************************
randn('state',2);
A = randn(m,n);
b = randn(m,1);

x_range = -2:0.1:2;
[X,Y] = meshgrid(x_range);


for i=1:m
  Z = A(i,1) * X + A(i,2) * Y + b(i);
  contour(X, Y, Z, 20);
  %surf(X,Y,Z)
  hold on;
end
%axis([-2,2,-2,2]);

%********************************************************************
% compute pwl optimal point using CVX
%********************************************************************
cvx_begin
    variable x(n)  
    minimize max(A*x + b)
cvx_end
f_star = cvx_optval
x
plot(x(1), x(2), 'r*');


%********************************************************************
% ellipsoid method
%********************************************************************
eps = 0.001;
% number of iterations 
niter = 20;
% initial ellipsoid
P = [1 0;0 1/2];
% initial point 
x = zeros(n,1); 


for iter = 1:niter 
    % plot step
    plot_ellipse(P, x);
     
    % find active functions at current x
    [f, idx] = max(A*x + b);      
    % subgradient at current x
    
    g = A(idx(1),:)';
    py = -g(1)/g(2) * x_range + x(2) + g(1)/g(2)*x(1);

   % if 0 < iter & iter < 3
        plot(x_range, py)
        plot(x(1), x(2), 'bs');
   % end
    
	% convergence test
	if sqrt(g'*P*g) <= eps
		disp('converged')
		break
    end
   

    % update ellipsoid with shallow cut.
    gt = g/sqrt(g'*P*g);
    x = x - 1/(n+1)*P*gt;
    P = n^2/(n^2-1)*(P-2/(n+1)*P*gt*gt'*P);
    
    %pause
    %if (mod(iter,2) == 0)
    %    clf;
    %end
end

iter
x
f
plot(x(1), x(2), 'go');



