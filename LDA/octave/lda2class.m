% two class date
c = 2;
X1 = [4,1;2,4;2,3;3,6;4,4;];
X2 = [9,10;6,8;9,5;8,7;10,8;];
%X2 = [9,10;6,8;9,5;8,7;10,8;9,9];
plot(X1(:,1),X1(:,2),".or")
hold on;
grid on;
plot(X2(:,1),X2(:,2),".*g")

% mean
m1 = sum(X1) / size(X1,1)
m2 = sum(X2) / size(X2,1)

% with-in class covariance
%
% method 1
%Sw = 0;
%for i=1:size(X1,1)
%	Sw += (X1(i,:) - m1)' * (X1(i,:) - m1);
%end
%for i=1:size(X2,1)
%	Sw += (X2(i,:) - m2)' * (X2(i,:) - m2);
%end

%method 2
Sw = (X1 - m1)' * (X1 - m1) + (X2 - m2)' * (X2 - m2);

% method 3
%Sw = cov(X1) + cov(X2);

% between-class covariance
% Sb
m = ( sum(X1) + sum(X2) ) / (size(X1,1) + size(X2,1));
Sb = 0;
Sb = Sb + size(X1,1) * (m1 - m)' * (m1 - m);
Sb = Sb + size(X2,1) * (m2 - m)' * (m2 - m);


% the vector
[u,lambda,v] = svd( pinv(Sw) * Sb);
w = u(:,1)

%plot w
x = 0:14;
y =  x * w(2) / w(1);
plot(x,y)
projectPlot2D(X1,w,'-.dm')
projectPlot2D(X2,w,'-.sc')
axis([0,14,0,14]);

