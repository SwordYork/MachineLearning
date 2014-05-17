c = 3;
n = 70;
X = [];
color = ['*r';'sg';'.b';];
color2 = ['-.*r';'--sg';'.-.b';];
facecolor = ['r';'g';'b'];
%random data generation
for j = 1:c
	for i=1:n
		X = [X; [normrnd(30 * j * cos(1.7*j/pi) + 5 +i, j ^ 1.2 + 1), normrnd(20*j*sin(1.4 * j/pi) + 10, j^1.1 + 2), normrnd(j*15,4) ] ];
	end
end
%plot origin

%plot3(X(:,1),X(:,2),X(:,3),'.')
for j=1:c 
	plot3( X( ((j-1)*n +1):(j*n) ,1), X( ((j-1)*n+1):(j*n) ,2), X( ((j-1)*n+1):(j*n) ,3) , color(j,:) )
	hold on;
end
grid on;
pause;
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

[u,lambda,v] = svd( pinv(Sw) * Sb)



%plot
project = u(:,1:2);

p = project';
A = p(:,1:2);
B = p(:,3);
normv = [A\B;-1]
point = projPointOnPlane( X,[0,0,0,project(:)'] );

mx = max(point(:,1)) * 1.2;
my = max(point(:,2)) * 1.2;
[xx,yy]=ndgrid(0:mx,0:my);
z = (-normv(1)*xx - normv(2)*yy)/normv(3);
surf(xx,yy,z)
colormap(winter)
shading interp

for i = 1:size(X)
	plot3([point(i,1),X(i,1)], [point(i,2),X(i,2)], [point(i,3),X(i,3)] , color2(ceil(i/n),:) ,'MarkerFaceColor',facecolor(ceil(i/n),:));
end


%pca
Covar = X' * X / m;
[U, S, V] = svd(Covar);
pca_c = U(:,1:2);

p = pca_c';
A = p(:,1:2);
B = p(:,3);
normv = [A\B;-1]

[xx,yy]=ndgrid(0:mx/2,0:my/2);
z = (-normv(1)*xx - normv(2)*yy)/normv(3);
surf(xx,yy,z)
shading interp
