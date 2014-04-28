M = [ 2 3 3.2 4;4 2 1 2;3 4 7 8];


[u, s, v] =svd( M'*M ) ;
nv = v;
nv(:,4) = [];
nv(:,3) = [];
q = nv'*M'*M*nv;

h = eye(3) - ones(3,1) * ones(1,3) / 3;
%v
z = h*M*nv;
%
%
z'*ones(3,1)
%
zv = z *  nv';

[uz,sz,vz] = svd( z'*z );


sum( s(:)) - sum( sz(:) )

sum(sum(((M - zv).^2)) )

plotdiff(M,zv)
