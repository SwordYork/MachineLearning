A = reshape(1:15,5,3);
B = magic(3);
[U,V,X,SigmaA,SigmaB] = gsvd(A,B);
U*SigmaA*X';  %A
V*SigmaB*X';  %B
% same with note
[Q,Xr] = qr(X);

Xr;
%U'*A*Q = C*Xr' = \Sigma_{A}
U'*A*Q;
SigmaA*Xr';
%V'*B*Q = S*Xr' = \Sigma_{B}
V'*B*Q;
SigmaB*Xr';

size(A) %5*3
size(B) %3*3
%Complete Orthogonal Decomposition
C = [A;B] 
%C = [C,C(:,1:3)]

% same with note
[Q,R,E] = qr(C')
E'*C*Q ;
E'
E(1:5,1:3) %P_{11}
E(6:8,1:3) %P_{21}
U*SigmaA*Xr' %P_{11}*W
V*SigmaB*Xr' %P_{21}*W

A*Q
B*Q
