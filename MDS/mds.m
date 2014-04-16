function [V1,L1,X] = mds(D)

  n = size(D)(1);

  A = D.^2 * (-0.5);
  H = eye(n) - ones(n,1)*ones(1,n)/n;
  B = H*A*H ;

  [V, L] = eigs(B,n);
  
  %%%%%%%%%%
  % select positive eigenvalues
  % and corressbonding eigenvectors
  %%%%%%%%%%
  diagL = diag(L);
  useL_pos = diagL > 10^-10;
  L1 = zeros(sum(useL_pos),sum(useL_pos));
  V1 = [];
  j = 1;
  for i=1:length(diagL)
      if(useL_pos(i) == 1 )
	  L1(j,j) = L(i,i);
	  V1 = [V1,V(:,i)];
	  j += 1;
      end
    
  end

  X = V1*L1^0.5;


end
