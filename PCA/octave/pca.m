data = gendata(1000,1,10,0.3,2);
[M,N] = size(data);
H = eye(M) - ones(M,1) * ones(1,M) / M;
zero_data = H * data ;

S = transpose(zero_data) * zero_data / N;
[G,L] = eig(S);
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
