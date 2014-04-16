M = [ 2 3 3.2 4;4 2 1 2;3 4 7 8];
%[u, s, v] = svd(M);
%nv = v;
%nv(:,3) = 0;
%nv(:,4) = 0;
%u*s
%nM = u*s*nv' ;
%rank(nM);
%sum(  ((M - nM).^2)(:) ) 



[u, s, v] =svd( M'*M ) ;
nv = v;
nv(:,4) = [];
nv(:,3) = 0;
q = nv'*M'*M*nv;


v'*M';
M*v;

v(:,3) = -v(:,2)-v(:,1);
v(:,4) = [];


%v
z = M* v;
%
%
z*ones(3,1);
%
zv = z *  nv';


nv'*nv;




sum(  ((M - zv).^2)(:) ) ;

