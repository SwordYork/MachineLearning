n = 500;
data = gendata(n,1,10,0.3,2); 
[M,N] = size(data);
H = eye(M) - ones(M,1) * ones(1,M) / M;
zero_data = H * data;
S = transpose(zero_data) * zero_data / M;
S

%S = zeros(2,2)
u = sum(data) / n;
%for i=1:n
%	S += (data(i,:) - u)' * (data(i,:) - u);
%end 
%S = S / n

% 2d -> 1d
p = 2;
q = 1;
[U,L,V] = svd(S);
%S*U
%U*L
dL = diag(L);
t = sum(dL(q+1:p)) / (p - q);
W = U(:,1:q) * ((L(1:q,1:q) - t * eye(q)).^0.5);

C = W*W' + t * eye(p);
minx = min(zero_data(:,1)) * 1.2;
miny = min(zero_data(:,2)) * 1.2;
maxx = max(zero_data(:,1)) * 1.2;
maxy = max(zero_data(:,2)) * 1.2;
plotgauss(zeros(1,p),C,linspace(minx,maxx,47),linspace(miny,maxy,47))

hold on
plot3(zero_data(:,1),zero_data(:,2),zeros(n),'r.');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  use z to generate x
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m = 250;
mu = zeros(q,1);
sigma = eye(q);
hz = mvnrnd(mu,sigma,m)';

mu = zeros(p,1);
sigma = eye(p);
he = mvnrnd(mu,sigma,m)';
hx = W*hz + repmat(u',1,m) + t^0.5*he;
hx = hx';
plot3(hx(:,1),hx(:,2),zeros(m),'g.');
axis([minx,maxx,miny,maxy]);

