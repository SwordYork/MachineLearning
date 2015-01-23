% Movielens 1M	
%	Given Y
%	minimize_{A,X} (1/2) ||P_{omega}(Y-X)||F^2 + (lambda1) * || A - X ||F^2 + (lambda2) * ||A||*
%	 = (1/2)||
%	Iterative Algorithm:	Given random X0,A0,
%	%
% alternating update A and X
Iter = 9000;
step = 9;

movie = 3706;
user = 6040;
Yomega = zeros(user,movie);

lambda1 = 0.1;
lambda = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Yomega2 = load('r3train');
%omega = (Yomega ~= 0);
%omega_c = (Yomega == 0 );
[row,col] = size(Yomega);
a = 1
for i=1:row
	vec = Yomega2(i,:);
	Yomega(vec(1),vec(2)) = vec(3);
end
row = user;
col = movie;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
omega = (Yomega ~= 0);
omega_c = (Yomega == 0);
X = zeros(row,col);
%A = zeros(row,col);
a = 2
for i=1:row
	for j=1:col
		X(i,j) = rand()*7 + 1;
		%X(i,j) = rand()/10;
	end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Test = zeros(user,movie);
Test2 = load('r3test');
[rrow,ccol] = size(Test2);
for i=1:rrow
	vec = Test2(i,:);
	Test(vec(1),vec(2)) = vec(3);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Tomega = ( Test ~= 0 ); %omega
T_size = sum(sum(Tomega));
a = 3
record = [];
for i=1:Iter
	tic;
	if (floor(i/step)*step==i)
		%Obj = (1/2) * norm(Yomega-(omega.*X),'fro')^2 + (lambda1) * norm(X-A,'fro')^2 + (lambda2) * sum(svd(A));
		%fprintf('%d -th iter:   Obj  %d\n',i,Obj);
		%fprintf('RMSE  ');	%%%%	TEST	RMSE	%%%%%
		fprintf('%d Iter',i);
		Sum = sqrt(sum(sum((Test - X.*Tomega).^2))/T_size)
		record = [record , Sum]
		%fprintf('\n');
	end
	Y_star = Yomega + omega_c .* X;
	%	Update X;
	%X = (1/(1+2*lambda1)) * (Y_star + 2 * lambda1 * A);
	%	Update A
	%[U,sigma,V] = svd(X);
	%A = U * max(sigma - lambda2/(2*lambda1) , 0) * V';
	[U,D,V] = svd(Y_star);
	X = U * max(D-lambda,0) * V';
	toc;
end
name = ['record_',num2str(lambda1),'_',num2str(lambda2)];
save -ascii name record;
%%%%%%%%%%%		Test		%%%%%%%%%%%
%Test = load('u1test');
%Tomega = ( Test ~= 0 );	%omega
%T_size = sum(sum(Tomega));
%disp('RMSE');
%Sum = sqrt(sum(sum((Test - X.*Tomega).^2))/T_size)

