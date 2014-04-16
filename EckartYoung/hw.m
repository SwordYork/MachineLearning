M = [ 2 3 3.2 4;4 2 1 2;3 4 7 8];


[u, s, v] =svd( M'*M ) ;
nv = v;
nv(:,4) = [];
nv(:,3) = [];
q = nv'*M'*M*nv;


%v
z = M* nv 
z = M*nv - ones(3,1) * (ones(1,3) *  M * nv / 3)
%
%
z'*ones(3,1)
%
zv = z *  nv';

sum(  ((M - zv).^2)(:) ) 


