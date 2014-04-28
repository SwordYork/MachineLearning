M = [ 2 3 3.2 4;4 2 1 2;3 4 7 8];
n = 2;
%[u, s, v] = svd(M);
%nv = v;
%nv(:,3) = 0;
%nv(:,4) = 0;
%u*s
%nM = u*s*nv' ;
%rank(nM);
%sum(  ((M - nM).^2)(:) ) 


[v,s] = eigs( M'*M ) 
ns = s(1:n,1:n);
nv = v(:,1:n);
%nv(:,4) = [];
%nv(:,3) = 0;
q = nv'*M'*M*nv;


%v(:,3) = -v(:,2)-v(:,1);
%v(:,4) = [];

z = M* nv;
%
zv = z *  nv';

sum(sum(  ((M - zv).^2) ) )
plotdiff(M,zv)
%  