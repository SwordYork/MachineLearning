c = 3;
n = 5;
X = [];
%random data generation
for j = 1:c
	for i=1:n
		X = [X; [normrnd(30 * j * cos(1.7*j/pi) + 5 +i, j ^ 1.2 + 1), normrnd(20*j*sin(1.4 * j/pi) + 10, j^1.1 + 2), normrnd(j*15,4) ] ];
	end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%        matrix method
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
H = (eye(c*n) - 1/n/c*(ones(c*n,1) * ones(1,n*c) ));
St = X'*H*X;
E = zeros(c,c*n);
E(1,1:n) = 1;
E(2,n+1:2*n) = 1;
E(3,2*n+1:3*n) = 1;
E = E';
PI = diag([1/n,1/n,1/n]);
Sb = X'*H*E*PI*E'*H*X;
Sw = St - Sb;
[u,lambda,v] = svd( pinv(Sw) * Sb);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  old method
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%mean
m = zeros(c,3);
for j=1:c
	m(j,:) = sum(X( ((j-1)*n+1):(j*n) ,:)) / n;
end

% with-in class covariance
Sw = zeros(3,3);
for j=1:c
	Sw = Sw + (X( ((j-1)*n+1):(j*n) ,:) - repmat( m(j,:),n,1) )' * (X( ((j-1)*n+1):(j*n) ,:) - repmat( m(j,:),n,1)); 
end

% between-class covariance
total_m = sum(X,1) / size(X,1);
Sb = zeros(3,3);
for j=1:c
	Sb = Sb + n * (m(j,:) - total_m)' * ( m(j,:) - total_m); 
end
% Error !!!!!!!!!!!
%[u,lambda,v] = svd( pinv(Sw) * Sb);
[u,lambda,v] = svd( pinv(St) * Sb)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  qz factorization for generalized eigenvalues.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%D = diag([1/n^0.5,1/n^0.5,1/n^0.5]);
%T = eye(n*c)  - E*PI*E';
%F = chol(T);
%A = D*E'*H*X;
%B = F*H*X;

%A'*A;
%Sb;
%B'*B;
%Sw;
%[U,V,X,SigmaA,SigmaB] = gsvd(A,B)
%[Q,Xr] = qr(X)
%C = [B;A];
%C = [C,C(:,1:3)]

% same with note
%[Q,R,E] = qr(C');
%Q
%[AA, BB, Q, Z, V, W] = qz(Sb,Sw);

[AA, BB, Q, Z, V, W] = qz(Sb,St);
alpha = diag(AA);
beta = diag(BB);



lambda;
u' * pinv(St) * Sb * v;  % = lambda
                        % St*u*lambda

%
% both equal
Sb*v;
St*u*lambda;
Sb*V;
St*V*diag(alpha./beta);

drawPlan(u(:,1:2))
dv = diag(alpha./beta);
[Y,I] = sort(diag(dv),'descend');
drawPlan(V(:,I(1:2)))


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     RDA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
D = diag([1/n^0.5,1/n^0.5,1/n^0.5]);
fi = inv(St)*X'*H*E*D;
ffi = D*E'*H*X*fi;
[u,s,v] =svd(ffi);
G = fi*v*s.^0.5;
drawPlan(G(:,1:2));



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% GSVD ?  Complete Orthogonal Decomposition
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
D = diag([1/n^0.5,1/n^0.5,1/n^0.5]);
A = D*E'*H*X;
B = H*X;
C = [A;B];


[U,V,X,SigmaA,SigmaB] = gsvd(A,B);

U'*A*inv(X'); %SigmaA
V'*B*inv(X'); %SigmaB
xxx = inv(X')    %that is!!!!
drawPlan(xxx(:,2:3))

[Q,R] = qr(C) %C = Q*R
Q1 = Q(1:3,1:3);
Q2 = Q(4:18,1:3);
Q1'*Q1 + Q2'*Q2;
[q_u,q_s,w] = svd(Q1);
[q2_q,q2_r] = qr(Q2*w);
q_s'*q_s + q2_r'*q2_r;
drawPlan(xxx(:,2:3));

xxxx = inv(R(1:3,:))*w;
drawPlan(xxxx(:,1:2));



