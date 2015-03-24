% Generate "two spirals" dataset with N instances.
% degrees controls the length of the spirals
% start determines how far from the origin the spirals start, in degrees
% noise displaces the instances from the spiral. 
%  0 is no noise, at 1 the spirals will start overlapping
N = 800;

degrees = 650;

start = 20;

noise = 0.1;

    
deg2rad = (2*pi)/360;
start = start * deg2rad;

N1 = floor(N/2);
N2 = N-N1;

n = start + sqrt(rand(N1,1)) * degrees * deg2rad;
d1 = [-cos(n).*n + rand(N1,1)*noise sin(n).*n+rand(N1,1)*noise zeros(N1,1)];

n = start + sqrt(rand(N1,1)) * degrees * deg2rad;
d2 = [cos(n).*n+rand(N2,1)*noise -sin(n).*n+rand(N2,1)*noise ones(N2,1)];

%data = [d1;d2];

ndata = data((data(:,3) == 0),:);
pdata = data((data(:,3) == 1),:);
plot(ndata(:,1),ndata(:,2),'ro');
hold on;
plot(pdata(:,1),pdata(:,2),'b*');


fid = fopen('gen_train.txt', 'w');

% print a title, followed by a blank line
for i=1:N
    fprintf(fid, '%f %f %d\n', data(i,1), data(i,2), data(i,3));
end
