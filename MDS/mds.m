pkg load statistics;


D = [0 2^0.5 2^0.5 5^0.5; 2^0.5 0 2^0.5*2 1; 2^0.5 2*2^0.5 0 13^0.5;5^0.5 1 13^0.5 0];
n = size(D)(1);
[Y,e] = cmdscale(D)

A = D.^2 * (-0.5);
H = eye(n) - ones(n,1)*ones(1,n)/n;
B = H*A*H ;

[V, L] = eigs(B);


%%%%%%%%%%
% select positive eigenvalues
% and corressbonding eigenvectors
%%%%%%%%%%
diagL = diag(L);
useL_pos = diagL > 10^-10;

L1 = zeros(sum(useL_pos),sum(useL_pos));
V1 = [];
j = 1;
for i=1:n
    if(useL_pos(i) == 1 )
	L1(j,j) = L(i,i);
	V1 = [V1,V(:,i)];
	j += 1;
    end
    
end

printf('cordinate:\n')
X = V1*L1^0.5;
disp(X)


%%%%%%%%%%
%  recalculate distance
%%%%%%%%%%
CD = zeros(n,n);
for i=1:n
    for j=1:n
	CD(i,j) = sum(  (X(i,:)-X(j,:)).^2  );
    end
end


printf('D.^2:\n')
disp(D.^2)


printf('calculate distance:\n')
disp(CD)







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      add new distance
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

