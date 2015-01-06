data = gendata(1000,1,10,0.3,2);
[M,N] = size(data);
H = eye(M) - ones(M,1) * ones(1,M) / M;
zero_data = H * data ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%			 Method 1
%			 S = X^THX
%			 Convert Coordinates: HXG
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
S = transpose(zero_data) * zero_data; 
[G,L] = eig(S);
Test = zero_data*G;
zero_data(1:10,:)
Recover = Test*G';
Recover(1:10,:)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%			 Method 2: PCO
%			 T = HX^TXH
%			 Convert Coordinates: UL.^0.5
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T = zero_data * transpose(zero_data);
[U,L1] = eig(T);
p = find((L1 > 1000) == 1);	% useful eigenvalues
cL = ([L1(:,(p(1)+1000)/1001),L1(:,(p(2)+1000)/1001)].^0.5);
Test = U*cL;
zero_data(1:10,:)
Recover = Test * G';		% must use G
Recover(1:10,:)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%			 Method 3: SVD 
%			 HX = UlV^T l^2 = L
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
HX = zero_data;
[U,l,V] = svd(HX);
Test = HX*V;
zero_data(1:10,:)
Recover = Test*V';
Recover(1:10,:)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%			 Method 4: Direct SVD 
%			 X^THX = ULV^T  % eig
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[U,s,V] = svd(S);
Test = zero_data*V;
zero_data(1:10,:)
Recover = Test*V';
Recover(1:10,:)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       Reduce Dimision
% 		SVD
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[U,s,V] = svd(S)
Test = zero_data*V(:,1);
zero_data(1:10,:)
Recover = Test*V(:,1)';
Recover(1:10,:)
sum(sum((Recover - zero_data).^2)) %ofcourse is the small singular value



% resort eig
V = diag(L);
[junk, rindice] = sort(-1*V);
V = V(rindice);
G = G(:,rindice);


% first pcn component
pcn = 2;
pcg = G(:,1:pcn);

%X = [1:100];
%Y = X * G(2,2) / G(1,2);
%plot(X,Y);
%hold on
signals = transpose(pcg) * transpose(zero_data);
signals = transpose(signals);

%covariance calculation
cov_orign = zero_data' * zero_data / N
cov_pca = signals' * signals / N
pcg

subplot(2,3,1)
plot(zero_data(:,1),'.')  %because sort rindice
grid on;
title("origin x");

subplot(2,3,2)
plot(zero_data(:,2),'.')  %because sort rindice
grid on;
title("origin y");


subplot(2,3,3)
%princeple component
for i=1:length(V)
	X = -50:50;
	Y = X * G(2,i) / G(1,i);
	plot(X,Y,'g','linewidth',6)
	hold on;
end
grid on;

%axis([-110,110,-110,110])

plot(zero_data(:,1),zero_data(:,2),'.')
title("origin and pca");

subplot(2,3,4)
plot(signals(:,1),'.g')  %because sort rindice
grid on;
title("pca 1");


subplot(2,3,5)
plot(signals(:,2),'.g')  %because sort rindice
grid on;
title("pca 2");


subplot(2,3,6)
plot(signals(:,1),signals(:,2),'.r')  %because sort rindice
grid on;
title("pca covariance");
